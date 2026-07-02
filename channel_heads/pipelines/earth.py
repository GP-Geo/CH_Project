"""Earth training pipeline — produces the models Mars inference consumes.

Read top to bottom for the training flow. The production artifacts
(``models/xgb_touching_classifier.json``, ``models/cnn_outlet_final.pt``) are
preserved as-is; a rebuild writes research/regime variants unless explicitly
intended (see ``docs/modeling.md``).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, NamedTuple

from channel_heads.io import paths
from channel_heads.logging_config import get_logger

log = get_logger("pipelines.earth")


class EarthBasinNetwork(NamedTuple):
    """A built Earth basin network and the objects needed to plot/evaluate it."""

    basin: str
    dem: Any  # GridObject
    fd: Any  # FlowObject
    s: Any  # StreamObject (regime-pruned)
    analyzer: Any  # CouplingAnalyzer | None
    threshold_cells: int
    pixel_size_m: float


def build_earth_basin_network(
    basin: str,
    regime: Any | str | None = None,
    connectivity: int = 8,
    build_analyzer: bool = True,
) -> EarthBasinNetwork:
    """Build the regime-pruned StreamObject (and analyzer) for one Earth basin.

    Factors the exact build steps used by the regime feature pipeline
    (:func:`channel_heads.training.regime.process_basin`) so notebooks/figures can
    reconstruct a basin's network without duplicating that logic. ``regime`` may
    be a :class:`~channel_heads.regimes.Regime`, a key like ``"regC"`` (the
    default), or ``None``. Requires TopoToolbox (imported lazily).
    """
    import numpy as np
    import topotoolbox as tt3

    from channel_heads.basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
    from channel_heads.coupling_analysis import CouplingAnalyzer
    from channel_heads.dd_calibration import (
        compute_pixel_size_m_from_dem,
        compute_threshold_cells,
    )
    from channel_heads.pruning import apply_strategy
    from channel_heads.regimes import REGIMES

    if regime is None:
        regime = REGIMES["regC"]
    elif isinstance(regime, str):
        regime = REGIMES[regime]

    dem_path = paths.resolve_dem_path(basin) or paths.EXAMPLE_DEMS.get(basin)
    if dem_path is None or not Path(dem_path).exists():
        raise FileNotFoundError(f"DEM for basin {basin!r} not found")

    paper_name = LOCAL_TO_PAPER_BASIN.get(basin, basin)
    cfg = get_basin_config(paper_name)
    z_th = cfg["z_th"]
    lat = float(cfg["lat"])

    dem = tt3.read_tif(str(dem_path))
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
        raise RuntimeError(f"Pruning removed the entire network for basin {basin!r}")

    analyzer = (
        CouplingAnalyzer(fd, s, dem, connectivity=connectivity, threshold=cells)
        if build_analyzer
        else None
    )
    log.info(
        "[%s] built regime=%s T=%.3fkm^2 (%d cells) px=%.1fm",
        basin, regime.name, regime.threshold_km2, cells, pixel_size_m,
    )
    return EarthBasinNetwork(
        basin=basin, dem=dem, fd=fd, s=s, analyzer=analyzer,
        threshold_cells=cells, pixel_size_m=pixel_size_m,
    )


def _run_cli(script_name: str, argv: list[str] | None = None) -> None:
    module = importlib.import_module(f"channel_heads.cli.{script_name}")
    log.info("running channel-heads %s", script_name.replace("_", "-"))
    module.main(list(argv or []))


def train_earth_cnn() -> None:
    """Train + persist the Earth outlet CNN (models/cnn_outlet_final.pt)."""
    _run_cli("train_cnn_baseline", ["-v"])


def train_earth_xgb_variants() -> None:
    """Train the 3 Earth XGBoost variants (geom-only / +emb / +logit)."""
    _run_cli("train_combined_xgb_phase6b")


def train_earth_models() -> None:
    """Full Earth training: CNN then the XGBoost variants."""
    train_earth_cnn()
    train_earth_xgb_variants()


class NetworkVariant(NamedTuple):
    """One extraction setup applied to a basin (for the regime comparison)."""

    label: str
    s: Any  # StreamObject (or None if pruning emptied it)
    threshold_cells: int
    threshold_km2: float
    pre_remove_max_order: int
    order_gap_to_prune: int


def trim_description(pre_remove_max_order: int) -> str:
    """Human-readable trimming description for a ``pre_remove_max_order`` value."""
    if pre_remove_max_order <= 0:
        return "none"
    if pre_remove_max_order == 1:
        return "drop 1st-order"
    return f"drop ≤{pre_remove_max_order}-order"


def build_earth_network_variants(
    basin: str,
    regimes: Any | None = None,
    baseline_threshold_cells: int = 145,
) -> tuple[Any, float, list[NetworkVariant]]:
    """Build the baseline + regime networks for one basin on a shared FlowObject.

    Returns ``(dem, pixel_size_m, variants)`` where ``variants`` is the ordered
    list ``[Baseline, regA, regB, regC]`` (label, StreamObject and the exact
    threshold/trim/order-gap parameters used). The baseline uses a fixed
    ``baseline_threshold_cells`` (145 px) with **no** pruning; each regime uses
    its ``threshold_km2`` (converted to cells for this basin's pixel size) and
    :func:`channel_heads.pruning.apply_strategy`. Requires TopoToolbox.
    """
    import numpy as np
    import topotoolbox as tt3

    from channel_heads.basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
    from channel_heads.dd_calibration import (
        compute_pixel_size_m_from_dem,
        compute_threshold_cells,
    )
    from channel_heads.pruning import apply_strategy
    from channel_heads.regimes import REGIMES

    if regimes is None:
        regimes = REGIMES

    dem_path = paths.resolve_dem_path(basin) or paths.EXAMPLE_DEMS.get(basin)
    if dem_path is None or not Path(dem_path).exists():
        raise FileNotFoundError(f"DEM for basin {basin!r} not found")

    paper_name = LOCAL_TO_PAPER_BASIN.get(basin, basin)
    cfg = get_basin_config(paper_name)
    dem = tt3.read_tif(str(dem_path))
    dem.z[dem.z < cfg["z_th"]] = np.nan
    pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=float(cfg["lat"]))
    pixel_area_km2 = (pixel_size_m * pixel_size_m) / 1e6
    fd = tt3.FlowObject(dem)

    variants: list[NetworkVariant] = [
        NetworkVariant(
            "Baseline",
            tt3.StreamObject(fd, threshold=baseline_threshold_cells),
            baseline_threshold_cells,
            baseline_threshold_cells * pixel_area_km2,
            0,
            0,
        )
    ]
    for name, reg in regimes.items():
        cells = compute_threshold_cells(reg.threshold_km2, pixel_size_m)
        s_full = tt3.StreamObject(fd, threshold=cells)
        s = apply_strategy(s_full, reg.pre_remove_max_order, reg.order_gap_to_prune)
        variants.append(
            NetworkVariant(name, s, cells, reg.threshold_km2,
                           reg.pre_remove_max_order, reg.order_gap_to_prune)
        )
    log.info("[%s] built %d network variants (px=%.1fm)", basin, len(variants), pixel_size_m)
    return dem, pixel_size_m, variants


__all__ = [
    "EarthBasinNetwork",
    "NetworkVariant",
    "build_earth_basin_network",
    "build_earth_network_variants",
    "trim_description",
    "train_earth_cnn",
    "train_earth_xgb_variants",
    "train_earth_models",
]
