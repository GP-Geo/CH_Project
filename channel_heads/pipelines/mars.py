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
features (:mod:`channel_heads.features.mars_features`), tabular XGBoost
inference (:mod:`channel_heads.models.mars_inference`), CNN patches
(:mod:`channel_heads.rasterization.mars_patches`), CNN embeddings
(:mod:`channel_heads.models.embeddings`) and combined Mars inference
(:mod:`channel_heads.models.mars_combined`).
"""

from __future__ import annotations

from pathlib import Path

from channel_heads import mars
from channel_heads.io import paths


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
# Phase 5 — CNN embeddings  (MIGRATED)
# --------------------------------------------------------------------------- #
def extract_mars_cnn_embeddings(
    patch_index_parquet=paths.MARS_CNN_PATCH_INDEX,
    features_parquet=paths.MARS_MODEL_INPUTS_DIR / "mars_pair_features_5feat_model_ready.parquet",
    predictions_parquet=paths.MARS_MODEL_OUTPUTS_DIR / "mars_xgb_predictions_5feat.parquet",
    model_path=paths.CNN_PRODUCTION,
    output_embeddings=paths.MARS_MODEL_INPUTS_DIR / "mars_cnn_embeddings.parquet",
    output_combined=paths.MARS_MODEL_INPUTS_DIR / "mars_model_input_tabular_plus_cnn.parquet",
    figures_dir=paths.MARS_CNN_PATCHES_DIR / "figures",
    *,
    batch_size: int = 64,
    device: str | None = None,
    write: bool = True,
    make_figures: bool = True,
) -> dict:
    """Phase 5: CNN embeddings via models/cnn_outlet_final.pt.

    Calls :func:`channel_heads.models.extract_mars_cnn_embeddings` directly.
    Returns ``{"embeddings", "combined", "paths"}``.
    """
    from channel_heads.models import extract_mars_cnn_embeddings as _extract

    return _extract(
        patch_index_parquet=patch_index_parquet,
        features_parquet=features_parquet,
        predictions_parquet=predictions_parquet,
        model_path=model_path,
        output_embeddings=output_embeddings,
        output_combined=output_combined,
        figures_dir=figures_dir,
        batch_size=batch_size,
        device=device,
        write=write,
        make_figures=make_figures,
    )


# --------------------------------------------------------------------------- #
# Phase 6C — combined inference + comparison  (MIGRATED)
# --------------------------------------------------------------------------- #
def run_mars_combined_inference(
    input_parquet=paths.MARS_MODEL_INPUTS_DIR / "mars_model_input_tabular_plus_cnn.parquet",
    tabular_predictions_parquet=paths.MARS_MODEL_OUTPUTS_DIR / "mars_xgb_predictions_5feat.parquet",
    patch_index_parquet=paths.MARS_CNN_PATCH_INDEX,
    cnn_model_path=paths.CNN_PRODUCTION,
    output_dir=paths.MARS_MODEL_OUTPUTS_DIR,
    pairs_gpkg=paths.MARS_PAIRS_GPKG,
    *,
    batch_size: int = 64,
    device: str | None = None,
    write: bool = True,
    write_gpkg: bool = True,
    make_figures: bool = True,
) -> dict:
    """Phase 6C: combined geom+CNN XGBoost variants on Mars.

    Calls :func:`channel_heads.models.run_mars_combined_inference` directly.
    Returns ``{"predictions", "summary", "by_network", "thresholds",
    "model_paths", "paths"}``.
    """
    from channel_heads.models import run_mars_combined_inference as _run

    return _run(
        input_parquet=input_parquet,
        tabular_predictions_parquet=tabular_predictions_parquet,
        patch_index_parquet=patch_index_parquet,
        cnn_model_path=cnn_model_path,
        output_dir=output_dir,
        pairs_gpkg=pairs_gpkg,
        batch_size=batch_size,
        device=device,
        write=write,
        write_gpkg=write_gpkg,
        make_figures=make_figures,
    )


def compare_mars_model_outputs(
    predictions_parquet=paths.MARS_MODEL_OUTPUTS_DIR / "mars_combined_model_predictions.parquet",
):
    """Phase 6C: summarise/compare the Mars model variant outputs.

    Reads the Phase-6C combined prediction table and returns the package
    comparison summary.
    """
    import pandas as pd

    from channel_heads.models import compare_mars_model_variants

    return compare_mars_model_variants(pd.read_parquet(predictions_parquet))


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
