"""Command-line interface for channel head coupling analysis."""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import topotoolbox as tt3

from .coupling_analysis import CouplingAnalyzer
from .first_meet_pairs_for_outlet import first_meet_pairs_for_outlet
from .stream_utils import outlet_node_ids_from_streampoi
from .logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze channel head coupling in drainage networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single DEM
  ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif -o results/inyo_coupling.csv

  # Mask low elevations and use custom threshold
  ch-analyze dem.tif -o results.csv --threshold 500 --mask-below 1200

  # Analyze specific outlets only
  ch-analyze dem.tif -o results.csv --outlets 5,12,18
        """,
    )

    parser.add_argument("dem", type=Path, help="Path to DEM file (GeoTIFF)")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output CSV path for coupling results"
    )
    parser.add_argument(
        "--threshold", type=int, default=300, help="Stream network area threshold (default: 300)"
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Connectivity for coupling detection: 4 or 8 (default: 8)",
    )
    parser.add_argument(
        "--mask-below",
        type=float,
        metavar="ELEVATION",
        help="Mask DEM elevations below this threshold (optional)",
    )
    parser.add_argument(
        "--outlets",
        type=str,
        metavar="IDS",
        help="Comma-separated outlet IDs to analyze (default: all outlets)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, console=True)

    # Validate DEM path
    if not args.dem.exists():
        logger.error("DEM file not found: %s", args.dem)
        return 1

    try:
        # Load DEM
        logger.info("Loading DEM: %s", args.dem)
        dem = tt3.read_tif(str(args.dem))

        # Apply elevation mask if requested
        if args.mask_below is not None:
            logger.info("Masking elevations below %s m", args.mask_below)
            dem.z[dem.z < args.mask_below] = float("nan")

        # Derive flow and stream networks
        logger.info("Deriving flow direction...")
        fd = tt3.FlowObject(dem)

        logger.info("Deriving stream network (threshold=%d)...", args.threshold)
        s = tt3.StreamObject(fd, threshold=args.threshold)

        # Select outlets to analyze
        if args.outlets:
            outlet_ids = [int(x.strip()) for x in args.outlets.split(",")]
            logger.info("Analyzing %d specified outlets: %s", len(outlet_ids), outlet_ids)
        else:
            outlet_ids = outlet_node_ids_from_streampoi(s)
            logger.info("Analyzing all %d outlets", len(outlet_ids))

        # Initialize analyzer
        an = CouplingAnalyzer(fd, s, dem, connectivity=args.connectivity)

        # Process each outlet
        all_results = []
        for i, outlet_id in enumerate(outlet_ids, 1):
            logger.debug("[%d/%d] Processing outlet %d...", i, len(outlet_ids), outlet_id)

            pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)
            df = an.evaluate_pairs_for_outlet(outlet_id, pairs)

            if not df.empty:
                all_results.append(df)
                n_pairs = len(df)
                n_touching = df["touching"].sum()
                logger.debug("  Outlet %d: %d pairs (%d touching)", outlet_id, n_pairs, n_touching)
            else:
                logger.debug("  Outlet %d: no pairs", outlet_id)

            # Clear cache between outlets to prevent unbounded memory growth
            an.clear_cache()

        # Save results
        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)

            # Create output directory if needed
            args.output.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df_all.to_csv(args.output, index=False)

            # Print summary
            total_pairs = len(df_all)
            touching_pairs = df_all["touching"].sum()
            touching_pct = df_all["touching"].mean() * 100

            logger.info("Results saved to %s", args.output)
            logger.info("  Total pairs: %d", total_pairs)
            logger.info("  Touching pairs: %d (%.1f%%)", touching_pairs, touching_pct)
            logger.info(
                "  Non-touching pairs: %d (%.1f%%)",
                total_pairs - touching_pairs,
                100 - touching_pct,
            )

            logger.debug("Columns: %s", ", ".join(df_all.columns))
        else:
            logger.warning("No pairs found in any outlet")
            return 2

        return 0

    except Exception as e:
        logger.error("Error: %s", e)
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
