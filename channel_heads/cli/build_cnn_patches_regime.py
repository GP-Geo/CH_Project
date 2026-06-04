#!/usr/bin/env python
"""Step 3 (Mars calibration) — Build 128x128 5-class CNN patches against
the regime-specific pruned Earth networks.

Wraps :func:`channel_heads.training.regime.build_regime_patch_dataset`, which
uses a regime-aware stream loader that:

  1. Loads the basin DEM with its ``z_th`` elevation mask.
  2. Builds a StreamObject at the regime's km^2 threshold (converted to
     pixel cells per basin's pixel size).
  3. Applies the same Strahler-strip + order-gap pruning as Step 2 so the
     ``head_1`` / ``head_2`` / ``confluence`` / ``outlet`` node IDs in
     ``master_dataset_<regime>.csv`` resolve correctly.

Patches are written under
``data/results/_rasters_<regime>/{basin}/rasters/{outlet}_{h1}_{h2}.npy``
to avoid clobbering the production ``data/results/{basin}/rasters/`` cache.
A manifest CSV (``master_dataset_<regime>.csv`` with an added
``raster_path`` column) is written to
``data/results/raster_manifest_<regime>.csv``.

Run::

    python scripts/build_cnn_patches_regime.py --regime regA
    python scripts/build_cnn_patches_regime.py --regime regB
"""

from __future__ import annotations

import argparse
import logging

from channel_heads.io.paths import RESULTS_DIR
from channel_heads.regimes import REGIMES
from channel_heads.training.regime import (
    build_regime_patch_dataset,
    regime_patch_paths,
)

log = logging.getLogger("build_cnn_patches_regime")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime",
        required=True,
        choices=sorted(REGIMES.keys()),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=128,
        help="Patch H x W in pixels (default: 128).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Threads for per-basin pair rasterization (default: 1 = serial, "
            "bit-identical to prior behavior). >1 parallelizes pairs within "
            "each basin."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

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

    master_csv, _output_root, _manifest_path = regime_patch_paths(
        regime,
        results_dir=RESULTS_DIR,
    )
    if not master_csv.exists():
        log.error("Missing master dataset: %s — run Step 2 first.", master_csv)
        return 1

    build_regime_patch_dataset(
        regime,
        results_dir=RESULTS_DIR,
        target_size=args.target_size,
        n_workers=args.workers,
        log_override=log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
