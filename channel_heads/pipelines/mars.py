"""High-level Mars cross-planet pipeline — the readable top-to-bottom flow.

Each function is one pipeline stage. Read this module top to bottom to
understand the whole Earth→Mars inference pipeline without opening ``scripts/``.

Stage order (see ``docs/pipeline.md``)::

    build_mars_topology        # Phase 1   valley vectors -> graph GeoPackage
    -> extract_mars_pairs       # Phase 2B  first-meet channel-head pairs
    -> build_mars_features       # Phase 3A  5 dimensionless tabular features
    -> run_mars_xgb_inference     # Phase 3B  production XGBoost (tabular)
    -> build_mars_cnn_patches      # Phase 4   5-class 128x128 patches
    -> extract_mars_cnn_embeddings  # Phase 5  CNN embeddings (cnn_outlet_final.pt)
    -> run_mars_combined_inference   # Phase 6C combined geom+CNN variants
    -> compare_mars_model_outputs     # Phase 6C variant comparison

Migrated into the package: topology, pairs (:mod:`channel_heads.mars`),
features (:mod:`channel_heads.features.mars_features`), tabular XGBoost inference
(:mod:`channel_heads.models.mars_inference`) and CNN patches
(:mod:`channel_heads.rasterization.mars_patches`).
TRANSITIONAL (logic still in ``scripts/``, scheduled for extraction):
embeddings, combined inference, comparison.
"""

from __future__ import annotations

from pathlib import Path

from channel_heads import mars
from channel_heads.io import paths
from channel_heads.pipelines._delegate import run_script


# --------------------------------------------------------------------------- #
# Phase 1 — topology  (MIGRATED)
# --------------------------------------------------------------------------- #
def build_mars_topology(
    valleys_path=paths.MARS_VALLEYS,
    mola_path=paths.MARS_DEM,
    output_path=paths.MARS_TOPOLOGY_GPKG,
    *,
    snap_tolerance_m: float = 200.0,
) -> Path:
    """Phase 1: build the Mars valley-network topology GeoPackage."""
    return mars.build_and_write_topology(
        valleys_path, mola_path, output_path, snap_tolerance_m=snap_tolerance_m
    )


# --------------------------------------------------------------------------- #
# Phase 2B — first-meet pairs  (MIGRATED)
# --------------------------------------------------------------------------- #
def extract_mars_pairs(
    topology_gpkg=paths.MARS_TOPOLOGY_GPKG,
    output_gpkg=paths.MARS_PAIRS_GPKG,
) -> Path:
    """Phase 2B: extract first-meet channel-head pairs."""
    return mars.extract_first_meet_pairs(topology_gpkg, output_gpkg)


# --------------------------------------------------------------------------- #
# Phase 3A — tabular features  (MIGRATED)
# --------------------------------------------------------------------------- #
def build_mars_features(
    topology_gpkg=paths.MARS_TOPOLOGY_GPKG,
    pairs_gpkg=paths.MARS_PAIRS_GPKG,
    output_dir=paths.MARS_MODEL_INPUTS_DIR,
) -> dict:
    """Phase 3A: build the Mars 5-feature tables (all + model-ready) + audit.

    Calls :func:`channel_heads.features.build_mars_features` directly. Returns
    ``{"all", "model_ready", "audit", "paths"}``.
    """
    from channel_heads.features import build_mars_features as _build

    return _build(topology_gpkg, pairs_gpkg, output_dir)


# --------------------------------------------------------------------------- #
# Phase 3B — production XGBoost inference  (MIGRATED)
# --------------------------------------------------------------------------- #
def run_mars_xgb_inference(
    features_parquet=paths.MARS_MODEL_INPUTS_DIR / "mars_pair_features_5feat_model_ready.parquet",
    output_dir=paths.MARS_MODEL_OUTPUTS_DIR,
) -> dict:
    """Phase 3B: production XGBoost on the Mars 5-feature model-ready table.

    Calls :func:`channel_heads.models.run_mars_tabular_inference` directly.
    Returns ``{"predictions", "summary", "by_network", "threshold", "paths"}``.
    """
    from channel_heads.models import run_mars_tabular_inference

    return run_mars_tabular_inference(
        features_parquet=features_parquet, output_dir=output_dir
    )


# --------------------------------------------------------------------------- #
# Phase 4 — CNN patches  (MIGRATED)
# --------------------------------------------------------------------------- #
def build_mars_cnn_patches(output_dir=paths.MARS_CNN_PATCHES_DIR) -> dict:
    """Phase 4: 5-class 128x128 CNN patches (must stay 5-class).

    Calls :func:`channel_heads.rasterization.build_mars_cnn_patches` directly.
    Returns ``{"manifest", "n_ok", "n_invalid", "n_failed", "paths"}``.
    """
    from channel_heads.rasterization import build_mars_cnn_patches as _build

    return _build(output_dir=output_dir)


# --------------------------------------------------------------------------- #
# Phase 5 — CNN embeddings  (TRANSITIONAL)
# --------------------------------------------------------------------------- #
def extract_mars_cnn_embeddings() -> None:
    """Phase 5: CNN embeddings via models/cnn_outlet_final.pt.

    TRANSITIONAL — runs ``scripts/extract_mars_cnn_embeddings.py``.
    """
    run_script("extract_mars_cnn_embeddings.py")


# --------------------------------------------------------------------------- #
# Phase 6C — combined inference + comparison  (TRANSITIONAL)
# --------------------------------------------------------------------------- #
def run_mars_combined_inference() -> None:
    """Phase 6C: combined geom+CNN XGBoost variants on Mars.

    TRANSITIONAL — runs ``scripts/run_mars_combined_xgb_inference.py``.
    """
    run_script("run_mars_combined_xgb_inference.py")


def compare_mars_model_outputs() -> None:
    """Phase 6C: summarise/compare the Mars model variant outputs.

    TRANSITIONAL — the comparison is produced by the combined-inference script;
    see :func:`run_mars_combined_inference`. Use
    :func:`channel_heads.models.comparison.compare_predictions` for ad-hoc
    comparisons in notebooks.
    """
    run_script("run_mars_combined_xgb_inference.py")


def run_full_mars_pipeline() -> None:
    """Run every Mars stage in order (Phase 1 → 6C)."""
    build_mars_topology()
    extract_mars_pairs()
    build_mars_features()
    run_mars_xgb_inference()
    build_mars_cnn_patches()
    extract_mars_cnn_embeddings()
    run_mars_combined_inference()


__all__ = [
    "build_mars_topology",
    "extract_mars_pairs",
    "build_mars_features",
    "run_mars_xgb_inference",
    "build_mars_cnn_patches",
    "extract_mars_cnn_embeddings",
    "run_mars_combined_inference",
    "compare_mars_model_outputs",
    "run_full_mars_pipeline",
]
