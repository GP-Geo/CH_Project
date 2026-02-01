"""Command-line interface for channel head coupling analysis."""

import argparse
from pathlib import Path
import sys
import pandas as pd
import topotoolbox as tt3

from .coupling_analysis import CouplingAnalyzer
from .first_meet_pairs_for_outlet import first_meet_pairs_for_outlet
from .stream_utils import outlet_node_ids_from_streampoi


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
        """
    )

    parser.add_argument(
        "dem",
        type=Path,
        help="Path to DEM file (GeoTIFF)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output CSV path for coupling results"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=300,
        help="Stream network area threshold (default: 300)"
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Connectivity for coupling detection: 4 or 8 (default: 8)"
    )
    parser.add_argument(
        "--mask-below",
        type=float,
        metavar="ELEVATION",
        help="Mask DEM elevations below this threshold (optional)"
    )
    parser.add_argument(
        "--outlets",
        type=str,
        metavar="IDS",
        help="Comma-separated outlet IDs to analyze (default: all outlets)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Validate DEM path
    if not args.dem.exists():
        print(f"Error: DEM file not found: {args.dem}", file=sys.stderr)
        return 1

    try:
        # Load DEM
        if args.verbose:
            print(f"Loading DEM: {args.dem}")
        dem = tt3.read_tif(str(args.dem))

        # Apply elevation mask if requested
        if args.mask_below is not None:
            if args.verbose:
                print(f"Masking elevations below {args.mask_below} m")
            dem.z[dem.z < args.mask_below] = float('nan')

        # Derive flow and stream networks
        if args.verbose:
            print("Deriving flow direction...")
        fd = tt3.FlowObject(dem)

        if args.verbose:
            print(f"Deriving stream network (threshold={args.threshold})...")
        s = tt3.StreamObject(fd, threshold=args.threshold)

        # Select outlets to analyze
        if args.outlets:
            outlet_ids = [int(x.strip()) for x in args.outlets.split(",")]
            if args.verbose:
                print(f"Analyzing {len(outlet_ids)} specified outlets: {outlet_ids}")
        else:
            outlet_ids = outlet_node_ids_from_streampoi(s)
            if args.verbose:
                print(f"Analyzing all {len(outlet_ids)} outlets")

        # Initialize analyzer
        an = CouplingAnalyzer(fd, s, dem, connectivity=args.connectivity)

        # Process each outlet
        all_results = []
        for i, outlet_id in enumerate(outlet_ids, 1):
            if args.verbose:
                print(f"  [{i}/{len(outlet_ids)}] Processing outlet {outlet_id}...", end="")

            pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)
            df = an.evaluate_pairs_for_outlet(outlet_id, pairs)

            if not df.empty:
                all_results.append(df)
                if args.verbose:
                    n_pairs = len(df)
                    n_touching = df['touching'].sum()
                    print(f" {n_pairs} pairs ({n_touching} touching)")
            else:
                if args.verbose:
                    print(" no pairs")

        # Save results
        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)

            # Create output directory if needed
            args.output.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df_all.to_csv(args.output, index=False)

            # Print summary
            total_pairs = len(df_all)
            touching_pairs = df_all['touching'].sum()
            touching_pct = df_all['touching'].mean() * 100

            print(f"\nResults saved to {args.output}")
            print(f"  Total pairs: {total_pairs}")
            print(f"  Touching pairs: {touching_pairs} ({touching_pct:.1f}%)")
            print(f"  Non-touching pairs: {total_pairs - touching_pairs} ({100-touching_pct:.1f}%)")

            if args.verbose:
                print(f"\nColumns: {', '.join(df_all.columns)}")
        else:
            print("\nWarning: No pairs found in any outlet", file=sys.stderr)
            return 2

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
