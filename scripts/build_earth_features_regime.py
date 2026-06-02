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
import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import topotoolbox as tt3

from channel_heads import (
    CouplingAnalyzer,
    GeometricFeaturesAnalyzer,
    LengthwiseAsymmetryAnalyzer,
    apply_strategy,
    filter_hard_negatives,
    first_meet_pairs_for_outlet,
    generate_labeled_dataset,
    outlet_node_ids_from_streampoi,
)
from channel_heads.basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
from channel_heads.config import EXAMPLE_DEMS, RESULTS_DIR
from channel_heads.dd_calibration import (
    compute_pixel_size_m_from_dem,
    compute_threshold_cells,
)
from channel_heads.logging_config import setup_logging
from channel_heads.regimes import REGIMES, Regime

log = logging.getLogger("build_earth_features_regime")


# Mirror nb00 cell 20.
NEGATIVE_RATIO = 3.0
RANDOM_SEED = 42
HARD_NEG_MAX_L_RATIO = 3.0
HARD_NEG_MAX_DIST_RATIO = 5.0
CONNECTIVITY = 8


# ---------------------------------------------------------------------------
# Per-basin processing
# ---------------------------------------------------------------------------
def _process_outlet(
    outlet_id: int,
    s,
    coupling_an: CouplingAnalyzer,
    asym_an: LengthwiseAsymmetryAnalyzer,
    geom_an: GeometricFeaturesAnalyzer,
    n_workers: int = 4,
) -> pd.DataFrame | None:
    pairs_at_confluence, _heads = first_meet_pairs_for_outlet(s, outlet_id)
    if not pairs_at_confluence:
        return None
    if sum(len(p) for p in pairs_at_confluence.values()) == 0:
        return None

    coupling_df = coupling_an.evaluate_pairs_for_outlet_parallel(
        outlet_id, pairs_at_confluence, n_workers=n_workers
    )
    if coupling_df.empty:
        return None

    asym_df = asym_an.evaluate_pairs_for_outlet(outlet_id, pairs_at_confluence)
    geom_df = geom_an.evaluate_pairs_for_outlet(
        outlet_id, pairs_at_confluence, asymmetry_df=asym_df
    )
    return generate_labeled_dataset(coupling_df, asym_df, geom_df)


def process_basin(
    basin_name: str,
    dem_path: Path,
    regime: Regime,
    min_basin_px: int,
    max_outlets: int | None,
) -> tuple[pd.DataFrame | None, dict]:
    """Build the regime-pruned StreamObject and run the full pair pipeline."""
    stats: dict = {
        "basin": basin_name,
        "n_outlets": 0,
        "n_outlets_kept": 0,
        "n_outlets_with_pairs": 0,
        "n_pairs": 0,
        "n_touching": 0,
        "n_not_touching": 0,
        "threshold_cells": 0,
        "n_nodes_full": 0,
        "n_nodes_pruned": 0,
        "min_basin_px": min_basin_px,
        "max_outlets": max_outlets,
        "time_s": 0.0,
        "error": None,
    }
    t0 = time.time()

    paper_name = LOCAL_TO_PAPER_BASIN.get(basin_name, basin_name)
    cfg = get_basin_config(paper_name)
    z_th = cfg["z_th"]
    lat = float(cfg["lat"])

    try:
        dem = tt3.read_tif(str(dem_path))
        dem.z[dem.z < z_th] = np.nan

        pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=lat)
        cells = compute_threshold_cells(regime.threshold_km2, pixel_size_m)
        stats["threshold_cells"] = cells

        fd = tt3.FlowObject(dem)
        s_full = tt3.StreamObject(fd, threshold=cells)
        n_nodes_full = int(np.asarray(s_full.node_indices[0]).size)
        stats["n_nodes_full"] = n_nodes_full

        s = apply_strategy(
            s_full,
            pre_remove_max_order=regime.pre_remove_max_order,
            order_gap_to_prune=regime.order_gap_to_prune,
        )
        if s is None:
            stats["error"] = "Pruning removed entire network"
            stats["time_s"] = time.time() - t0
            return None, stats
        n_nodes_pruned = int(np.asarray(s.node_indices[0]).size)
        stats["n_nodes_pruned"] = n_nodes_pruned

        log.info(
            "[%s] pixel=%.1fm  T=%.3fkm^2 (%d cells)  full=%d nodes -> pruned=%d nodes",
            basin_name,
            pixel_size_m,
            regime.threshold_km2,
            cells,
            n_nodes_full,
            n_nodes_pruned,
        )

        outlets = outlet_node_ids_from_streampoi(s)
        stats["n_outlets"] = int(len(outlets))
        if len(outlets) == 0:
            stats["error"] = "No outlets after pruning"
            stats["time_s"] = time.time() - t0
            return None, stats

        # Runtime bounds: drop tiny artifact basins (< min_basin_px pixels)
        # and cap to the N largest by basin pixel count. Without these the
        # pruning regimes blow up on dense initial networks (Taiwan at
        # T=0.05 km^2 has 647k stream nodes and thousands of outlets).
        if min_basin_px > 1 or max_outlets is not None:
            valid_dem = ~np.isnan(dem.z)
            n_rows = dem.z.shape[0]
            from channel_heads.dd_calibration import (  # local to avoid heavy import path
                _node_rowcol,
                linear_index_fortran,
            )

            rows_all, cols_all = _node_rowcol(s)
            outlet_lin = np.array(
                [
                    linear_index_fortran(rows_all[o], cols_all[o], n_rows)
                    for o in outlets
                ],
                dtype=np.int64,
            )
            labels = np.asarray(fd.drainagebasins(outlet_lin).z)
            valid_labels = labels[valid_dem].ravel().astype(np.int64)
            counts = np.bincount(valid_labels, minlength=outlets.size + 1)
            sizes = counts[1 : outlets.size + 1]
            keep_idx = np.flatnonzero(sizes >= min_basin_px)
            if max_outlets is not None and keep_idx.size > max_outlets:
                keep_idx = keep_idx[
                    np.argsort(sizes[keep_idx])[::-1][:max_outlets]
                ]
            outlets_kept = outlets[keep_idx]
            log.info(
                "[%s] %d outlets -> %d kept (min_px=%d, max=%s)",
                basin_name,
                len(outlets),
                len(outlets_kept),
                min_basin_px,
                "all" if max_outlets is None else max_outlets,
            )
        else:
            outlets_kept = outlets
        stats["n_outlets_kept"] = int(len(outlets_kept))
        if len(outlets_kept) == 0:
            stats["error"] = "No outlets pass size filter"
            stats["time_s"] = time.time() - t0
            return None, stats

        # Prefilter sizing is computed per OUTLET inside the loop below,
        # using that outlet's actual drainage area in pixels. Initialise
        # with threshold=1 (effectively no-op floor); we override
        # ``_prefilter_distance`` per outlet before each evaluate call.
        coupling_an = CouplingAnalyzer(
            fd, s, dem, connectivity=CONNECTIVITY, threshold=1
        )
        asym_an = LengthwiseAsymmetryAnalyzer(s, dem, lat=lat)
        node_orders = s.streamorder(method="strahler")
        geom_an = GeometricFeaturesAnalyzer(s, dem, lat=lat, node_orders=node_orders)

        # Per-outlet prefilter from drainage area: outlet_basin_px is already
        # in `sizes[i]` from the size filter. The prefilter distance is
        # `2 * sqrt(basin_area_px)` pixels — the same formula production
        # used, just with the OUTLET'S actual basin as the "threshold"
        # instead of a fixed 300-cell guess. Floor at min_prefilter_px so
        # pathologically small basins still get a sane lookup window.
        sizes_kept = sizes[keep_idx]
        import math as _math

        outlet_results: list[pd.DataFrame] = []
        for oid, basin_px in zip(outlets_kept, sizes_kept):
            try:
                # Override the analyzer's prefilter for this outlet.
                pf_px = max(
                    regime.min_prefilter_px,
                    2.0 * _math.sqrt(float(basin_px)),
                )
                coupling_an._prefilter_distance = pf_px
                df_o = _process_outlet(
                    int(oid), s, coupling_an, asym_an, geom_an,
                    n_workers=regime.coupling_n_workers,
                )
            except Exception:  # noqa: BLE001 — one bad outlet shouldn't kill the basin
                log.exception("[%s] outlet=%d failed; skipping", basin_name, int(oid))
                continue
            finally:
                coupling_an.clear_cache()
            if df_o is not None and not df_o.empty:
                outlet_results.append(df_o)
                stats["n_outlets_with_pairs"] += 1

        if not outlet_results:
            stats["error"] = "No pairs found in any outlet"
            stats["time_s"] = time.time() - t0
            return None, stats

        df_basin = pd.concat(outlet_results, ignore_index=True)
        df_basin["basin"] = basin_name
        stats["n_pairs"] = int(len(df_basin))
        stats["n_touching"] = int(df_basin["y"].sum())
        stats["n_not_touching"] = stats["n_pairs"] - stats["n_touching"]
        stats["time_s"] = time.time() - t0
        return df_basin, stats

    except Exception as exc:  # noqa: BLE001
        stats["error"] = str(exc)
        stats["time_s"] = time.time() - t0
        log.exception("[%s] processing failed", basin_name)
        return None, stats


# ---------------------------------------------------------------------------
# Stratified negative subsampling — port of nb00 cell 24
# ---------------------------------------------------------------------------
def stratified_subsample_negatives(
    df: pd.DataFrame,
    target_ratio: float = 3.0,
    random_state: int = 42,
) -> pd.DataFrame:
    if df.empty:
        return df
    rng = np.random.default_rng(random_state)
    positives = df[df["y"] == 1].copy()
    negatives = df[df["y"] == 0].copy()
    n_pos = len(positives)
    n_neg_target = int(n_pos * target_ratio)
    if len(negatives) <= n_neg_target:
        return df

    basin_neg_counts = negatives.groupby("basin").size()
    basin_proportions = basin_neg_counts / basin_neg_counts.sum()
    basin_targets = (basin_proportions * n_neg_target).round().astype(int)
    diff = n_neg_target - basin_targets.sum()
    if diff != 0:
        order = basin_neg_counts.sort_values(ascending=False).index
        for basin in order:
            if diff == 0:
                break
            if diff > 0:
                basin_targets[basin] += 1
                diff -= 1
            elif diff < 0 and basin_targets[basin] > 0:
                basin_targets[basin] -= 1
                diff += 1

    sampled = []
    for basin, n_target in basin_targets.items():
        basin_negs = negatives[negatives["basin"] == basin]
        if len(basin_negs) <= n_target:
            sampled.append(basin_negs)
        else:
            idx = rng.choice(len(basin_negs), size=int(n_target), replace=False)
            sampled.append(basin_negs.iloc[idx])
    out = pd.concat([positives] + sampled, ignore_index=True)
    return out.sort_values(["basin", "outlet", "confluence"], ignore_index=True)


# ---------------------------------------------------------------------------
# DEM discovery — same DEM→basin mapping as nb00
# ---------------------------------------------------------------------------
DEM_TO_BASIN: dict[str, str] = {
    "CalnAlpine_strm_crop.tif": "calnalpine",
    "Daqing_strm_crop.tif": "daqing",
    "Finisterre_strm_crop.tif": "finisterre",
    "Humboldt_strm_crop.tif": "humboldt",
    "Inyo_strm_crop.tif": "inyo",
    "Kammanasie_strm_crop.tif": "kammanasie",
    "Luliang_strm_crop.tif": "luliang",
    "Panamint_strm_crop.tif": "panamint",
    "Sakhalin_strm_crop.tif": "sakhalin",
    "SierraMadre_strm_crop.tif": "sierramadre",
    "SierraNevadaSpain_strm_crop.tif": "sierranevadaspain",
    "SierradelValleFertil_strm_crop.tif": "vallefertil",
    "Taiwan_strm_crop.tif": "taiwan",
    "Toano_strm_crop.tif": "toano",
    "Troodos_strm_crop.tif": "troodos",
    "Tsugaru_strm_crop.tif": "tsugaru",
    "Yoro_strm_crop.tif": "yoro",
}


def _resolve_basins(requested: list[str] | None) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for dem_file, basin_name in sorted(DEM_TO_BASIN.items()):
        if requested and basin_name not in requested:
            continue
        dem_path = EXAMPLE_DEMS.get(basin_name)
        if dem_path is None or not dem_path.exists():
            log.warning("[%s] DEM missing on disk; skipping", basin_name)
            continue
        pairs.append((basin_name, dem_path))
    return pairs


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

    basins = _resolve_basins(args.basins)
    if not basins:
        log.error("No basins available; aborting.")
        return 1
    log.info("Processing %d basins", len(basins))

    all_results: list[pd.DataFrame] = []
    all_stats: list[dict] = []
    t_total = time.time()

    for basin_name, dem_path in basins:
        cache_path = RESULTS_DIR / basin_name / f"full_features_{regime.name}.csv"
        if cache_path.exists() and not args.force:
            log.info("[%s] loading cache %s", basin_name, cache_path)
            df_basin = pd.read_csv(cache_path)
            all_results.append(df_basin)
            all_stats.append(
                {
                    "basin": basin_name,
                    "n_pairs": int(len(df_basin)),
                    "n_touching": int((df_basin["y"] == 1).sum()),
                    "n_not_touching": int((df_basin["y"] == 0).sum()),
                    "from_cache": True,
                    "time_s": 0.0,
                }
            )
            continue

        df_basin, stats = process_basin(
            basin_name,
            dem_path,
            regime,
            min_basin_px=args.min_basin_px,
            max_outlets=(None if args.max_outlets == 0 else args.max_outlets),
        )
        all_stats.append({**stats, "from_cache": False})

        if df_basin is None or df_basin.empty:
            log.error(
                "[%s] FAILED in %.1fs: %s",
                basin_name,
                stats.get("time_s", 0.0),
                stats.get("error"),
            )
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_basin.to_csv(cache_path, index=False)
            log.info(
                "[%s] %d pairs (%d touching) in %.1fs -> %s",
                basin_name,
                stats["n_pairs"],
                stats["n_touching"],
                stats["time_s"],
                cache_path,
            )
            all_results.append(df_basin)

        gc.collect()

    log.info("Per-basin loop done in %.1fs", time.time() - t_total)

    # Per-basin stats CSV (always written, even on partial runs).
    stats_csv = RESULTS_DIR / f"build_earth_features_{regime.name}_stats.csv"
    pd.DataFrame(all_stats).to_csv(stats_csv, index=False)
    log.info("Wrote per-basin stats -> %s", stats_csv)

    if args.no_master or not all_results:
        log.info("Skipping master dataset assembly (no-master=%s).", args.no_master)
        return 0

    df_combined = pd.concat(all_results, ignore_index=True)
    log.info(
        "Combined: %d rows from %d basins (touching=%d, not=%d)",
        len(df_combined),
        df_combined["basin"].nunique(),
        int((df_combined["y"] == 1).sum()),
        int((df_combined["y"] == 0).sum()),
    )

    df_filtered = filter_hard_negatives(
        df_combined,
        max_L_ratio=HARD_NEG_MAX_L_RATIO,
        max_dist_ratio=HARD_NEG_MAX_DIST_RATIO,
    )
    log.info(
        "After filter_hard_negatives: %d rows (touching=%d, not=%d)",
        len(df_filtered),
        int((df_filtered["y"] == 1).sum()),
        int((df_filtered["y"] == 0).sum()),
    )

    df_master = stratified_subsample_negatives(
        df_filtered, target_ratio=NEGATIVE_RATIO, random_state=RANDOM_SEED
    )
    log.info(
        "After subsample (target neg:pos = %.1f:1): %d rows (touching=%d, not=%d)",
        NEGATIVE_RATIO,
        len(df_master),
        int((df_master["y"] == 1).sum()),
        int((df_master["y"] == 0).sum()),
    )

    master_path = RESULTS_DIR / f"master_dataset_{regime.name}.csv"
    df_master.to_csv(master_path, index=False)
    log.info("Wrote master dataset -> %s (%.2f MB)", master_path, master_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
