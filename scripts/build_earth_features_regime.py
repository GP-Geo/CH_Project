#!/usr/bin/env python
"""Step 2 (Mars calibration) — Build per-basin Earth features under a
specific pruning regime.

Headless replay of ``notebooks/training/00_full_pipeline.ipynb`` with two
changes:

  1. Stream threshold is given in km^2 and converted to pixel cells per
     basin using each DEM's pixel size (geographic CRS aware).
  2. The global ``StreamObject`` is pruned with
     :func:`channel_heads.apply_strategy` (``pre_remove_max_order`` then
     ``order_gap_to_prune``) before any pair analysis runs.

Regime presets:
  - regA: T=0.05 km^2, pre_remove<=2 (drop 1st+2nd order), order_gap>=4
  - regB: T=0.25 km^2, pre_remove<=1 (drop 1st order only), order_gap>=4

Outputs (per regime, side-by-side with existing production artifacts):
  data/results/{basin}/full_features_{regime}.csv
  data/results/master_dataset_{regime}.csv

The production ``full_features.csv`` and ``master_dataset_v2.csv`` are
never touched. Existing per-basin caches are skipped unless ``--force``.

Run::

    python scripts/build_earth_features_regime.py --regime regA
    python scripts/build_earth_features_regime.py --regime regB --force
    python scripts/build_earth_features_regime.py --regime regA --basins inyo taiwan
"""

from __future__ import annotations

import argparse
import logging

from channel_heads.io.paths import RESULTS_DIR
from channel_heads.logging_config import setup_logging
from channel_heads.regimes import REGIMES
from channel_heads.training.regime import build_regime_feature_dataset

log = logging.getLogger("build_earth_features_regime")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime",
        required=True,
        choices=sorted(REGIMES.keys()),
        help="Pruning regime preset.",
    )
    parser.add_argument(
        "--basins",
        nargs="+",
        default=None,
        help="Restrict to a subset of basin names (default: all on disk).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute per-basin full_features_<regime>.csv even if cached.",
    )
    parser.add_argument(
        "--no-master",
        action="store_true",
        help="Skip the concat + filter + subsample + master CSV write.",
    )
    parser.add_argument(
        "--min-basin-px",
        type=int,
        default=500,
        help=(
            "Drop outlet basins smaller than this many DEM pixels. "
            "Matches earth_network_pruning notebook default of 500."
        ),
    )
    parser.add_argument(
        "--max-outlets",
        type=int,
        default=40,
        help=(
            "Cap per-basin outlet count by largest basin pixels (default 40, "
            "matches earth_network_pruning notebook). 0 = no cap."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level=logging.INFO if args.verbose else logging.INFO)
    # The package's setup_logging only attaches to the `channel_heads` logger;
    # our script's own logger needs its own basic config so log.info() prints.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    regime = REGIMES[args.regime]
    log.info(
        "Regime %s: T=%.3f km^2, pre_remove<=%d, order_gap>=%d",
        regime.name,
        regime.threshold_km2,
        regime.pre_remove_max_order,
        regime.order_gap_to_prune,
    )

    return build_regime_feature_dataset(
        regime,
        results_dir=RESULTS_DIR,
        requested_basins=args.basins,
        force=args.force,
        no_master=args.no_master,
        min_basin_px=args.min_basin_px,
        max_outlets=args.max_outlets,
        log_override=log,
    )


if __name__ == "__main__":
    raise SystemExit(main())
