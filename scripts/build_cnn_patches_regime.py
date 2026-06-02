#!/usr/bin/env python
"""Step 3 (Mars calibration) — Build 128x128 5-class CNN patches against
the regime-specific pruned Earth networks.

Wraps :func:`channel_heads.rasterizer.precompute_raster_dataset` with a
regime-aware ``dem_loader`` that:

  1. Loads the basin DEM with its ``z_th`` elevation mask.
  2. Builds a StreamObject at the regime's km^2 threshold (converted to
     pixel cells per basin's pixel size).
  3. Applies the same Strahler-strip + order-gap pruning as Step 2 so the
     ``head_1`` / ``head_2`` / ``confluence`` / ``outlet`` node IDs in
     ``master_dataset_<regime>.csv`` resolve correctly.

Patches are written under
``data/results/{basin}_<regime>/rasters/{outlet}_{h1}_{h2}.npy`` to
avoid clobbering the production ``data/results/{basin}/rasters/`` cache.
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
from pathlib import Path

import numpy as np
import topotoolbox as tt3

from channel_heads import apply_strategy
from channel_heads.basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
from channel_heads.config import RESULTS_DIR, resolve_dem_path
from channel_heads.dd_calibration import (
    compute_pixel_size_m_from_dem,
    compute_threshold_cells,
)
from channel_heads.rasterizer import precompute_raster_dataset

# Re-use the regime presets from the Step 2 script.
from channel_heads.regimes import REGIMES, Regime

log = logging.getLogger("build_cnn_patches_regime")


def make_regime_stream_loader(regime: Regime):
    """Return a ``precompute_raster_dataset`` compatible loader for a regime.

    Signature matches ``default_stream_loader(basin, lat, z_th, threshold)``
    — the trailing ``threshold`` arg is ignored because the regime supplies
    its own km^2 threshold and pruning recipe.
    """

    def loader(basin: str, lat: float, z_th: float, threshold: int):
        dem_path = resolve_dem_path(basin)
        if dem_path is None or not Path(dem_path).exists():
            log.warning("DEM not found for basin '%s'", basin)
            return None
        try:
            dem = tt3.read_tif(str(dem_path))
            if z_th is not None and not np.isnan(z_th):
                dem.z[dem.z < z_th] = np.nan
            pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=lat)
            cells = compute_threshold_cells(regime.threshold_km2, pixel_size_m)
            fd = tt3.FlowObject(dem)
            s_full = tt3.StreamObject(fd, threshold=cells)
            s = apply_strategy(
                s_full,
                pre_remove_max_order=regime.pre_remove_max_order,
                order_gap_to_prune=regime.order_gap_to_prune,
            )
            if s is None:
                log.warning(
                    "[%s] regime %s pruning removed all nodes",
                    basin,
                    regime.name,
                )
                return None
            return s, dem
        except Exception:  # noqa: BLE001 — loader contract returns None on failure
            log.exception("[%s] regime %s loader failed", basin, regime.name)
            return None

    return loader


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

    master_csv = RESULTS_DIR / f"master_dataset_{regime.name}.csv"
    if not master_csv.exists():
        log.error("Missing master dataset: %s — run Step 2 first.", master_csv)
        return 1

    # Use a regime-suffixed output root so production rasters/ are untouched.
    output_root = RESULTS_DIR / f"_rasters_{regime.name}"
    output_root.mkdir(parents=True, exist_ok=True)
    log.info("Rasters output root: %s", output_root)

    loader = make_regime_stream_loader(regime)
    df = precompute_raster_dataset(
        master_csv=master_csv,
        output_dir=output_root,
        dem_loader=loader,
        target_size=args.target_size,
        # threshold is forwarded to loader but ignored there; kept to satisfy signature.
        threshold=0,
    )

    n_total = len(df)
    if "raster_status" in df.columns:
        status_counts = df["raster_status"].value_counts(dropna=False).to_dict()
        n_ok = int((df["raster_status"] == "ok").sum())
        log.info("Raster status counts: %s", status_counts)
    else:
        n_ok = int(df["raster_path"].notna().sum())
    log.info("Rasterized %d / %d pairs", n_ok, n_total)

    manifest_path = RESULTS_DIR / f"raster_manifest_{regime.name}.csv"
    df.to_csv(manifest_path, index=False)
    log.info("Wrote manifest -> %s", manifest_path)

    # Cheap sanity check: which basins failed entirely?
    failed_basins = [
        str(name)
        for name, sub in df.groupby("basin")
        if (
            int((sub["raster_status"] == "ok").sum())
            if "raster_status" in sub.columns
            else sub["raster_path"].notna().sum()
        )
        == 0
    ]
    if failed_basins:
        log.warning("Basins with zero rasters: %s", failed_basins)
    return 0


if __name__ == "__main__":
    # Avoid unused-import nag on get_basin_config / LOCAL_TO_PAPER_BASIN
    _ = (get_basin_config, LOCAL_TO_PAPER_BASIN)
    raise SystemExit(main())
