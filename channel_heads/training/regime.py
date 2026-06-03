"""Regime Earth-feature / patch workflow helpers.

Canonical home for the regime-specific Earth feature-generation and patch-stream
orchestration that was inline in ``scripts/build_earth_features_regime.py`` and
``scripts/build_cnn_patches_regime.py``:

* per-basin regime feature build (``process_basin`` / ``_process_outlet``),
* stratified negative subsampling for the master dataset,
* the DEM→basin map and basin resolution helper, and
* the regime patch stream loader factory.

The heavy stream-building / pair-analysis logic is moved **verbatim** and reuses
the existing analyzers (``geometric_analysis`` / pairing / pruning / rasterizer);
no scientific behaviour is changed here. TopoToolbox is imported lazily so the
pure helpers (``stratified_subsample_negatives``, ``resolve_regime_basins``)
stay importable without it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
from channel_heads.coupling_analysis import CouplingAnalyzer
from channel_heads.dd_calibration import (
    compute_pixel_size_m_from_dem,
    compute_threshold_cells,
)
from channel_heads.features.asymmetry import LengthwiseAsymmetryAnalyzer
from channel_heads.features.earth_geometry import GeometricFeaturesAnalyzer
from channel_heads.io.paths import EXAMPLE_DEMS, resolve_dem_path
from channel_heads.pairing.earth import first_meet_pairs_for_outlet
from channel_heads.pruning import apply_strategy
from channel_heads.rasterization.earth_batch import precompute_raster_dataset
from channel_heads.regimes import Regime
from channel_heads.stream_utils import outlet_node_ids_from_streampoi
from channel_heads.training.labeling import generate_labeled_dataset

log = logging.getLogger(__name__)

# Mirror nb00 cell 20 / build_earth_features_regime constants.
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
    import topotoolbox as tt3

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


def resolve_regime_basins(requested: list[str] | None) -> list[tuple[str, Path]]:
    """Resolve ``(basin_name, dem_path)`` pairs that exist on disk.

    Iterates the DEM→basin map in sorted order, optionally restricting to
    ``requested`` basin names, and skips basins whose DEM is missing.
    """
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
# Regime patch stream loader (Step 3)
# ---------------------------------------------------------------------------
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
            import topotoolbox as tt3

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


def regime_patch_paths(
    regime: Regime,
    *,
    results_dir: Path,
) -> tuple[Path, Path, Path]:
    """Return master CSV, output root, and manifest path for regime patches."""
    return (
        results_dir / f"master_dataset_{regime.name}.csv",
        results_dir / f"_rasters_{regime.name}",
        results_dir / f"raster_manifest_{regime.name}.csv",
    )


def build_regime_patch_dataset(
    regime: Regime,
    *,
    results_dir: Path,
    target_size: int = 128,
    stream_loader=None,
    precompute_func=precompute_raster_dataset,
    log_override=None,
) -> tuple[pd.DataFrame, Path]:
    """Rasterize CNN patches for a regime master dataset and write the manifest."""
    master_csv, output_root, manifest_path = regime_patch_paths(
        regime,
        results_dir=results_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    active_log = log if log_override is None else log_override
    active_log.info("Rasters output root: %s", output_root)

    loader = (
        make_regime_stream_loader(regime)
        if stream_loader is None
        else stream_loader
    )
    df = precompute_func(
        master_csv=master_csv,
        output_dir=output_root,
        dem_loader=loader,
        target_size=target_size,
        # threshold is forwarded to loader but ignored there; kept to satisfy signature.
        threshold=0,
    )

    n_total = len(df)
    if "raster_status" in df.columns:
        status_counts = df["raster_status"].value_counts(dropna=False).to_dict()
        n_ok = int((df["raster_status"] == "ok").sum())
        active_log.info("Raster status counts: %s", status_counts)
    else:
        n_ok = int(df["raster_path"].notna().sum())
    active_log.info("Rasterized %d / %d pairs", n_ok, n_total)

    df.to_csv(manifest_path, index=False)
    active_log.info("Wrote manifest -> %s", manifest_path)

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
        active_log.warning("Basins with zero rasters: %s", failed_basins)

    return df, manifest_path


__all__ = [
    "NEGATIVE_RATIO",
    "RANDOM_SEED",
    "HARD_NEG_MAX_L_RATIO",
    "HARD_NEG_MAX_DIST_RATIO",
    "CONNECTIVITY",
    "DEM_TO_BASIN",
    "process_basin",
    "stratified_subsample_negatives",
    "resolve_regime_basins",
    "make_regime_stream_loader",
    "regime_patch_paths",
    "build_regime_patch_dataset",
]
