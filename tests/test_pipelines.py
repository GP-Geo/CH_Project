"""Smoke tests for the high-level channel_heads.pipelines public API.

These assert the package is explainable top-to-bottom through `pipelines`
(import + callable surface). They do not execute heavy stages.
"""

from __future__ import annotations

import inspect

import pytest

from channel_heads import pipelines

MARS_STAGES = [
    "build_mars_topology",
    "extract_mars_pairs",
    "build_mars_features",
    "run_mars_xgb_inference",
    "build_mars_cnn_patches",
    "extract_mars_cnn_embeddings",
    "run_mars_combined_inference",
    "compare_mars_model_outputs",
]
EARTH_STAGES = ["train_earth_cnn", "train_earth_xgb_variants", "train_earth_models"]
OTHER = ["generate_poster_figures", "run_full_mars_pipeline"]


@pytest.mark.parametrize("name", MARS_STAGES + EARTH_STAGES + OTHER)
def test_pipeline_stage_is_callable(name):
    fn = getattr(pipelines, name)
    assert callable(fn)
    assert inspect.getdoc(fn), f"{name} must document its behaviour"


def test_migrated_stages_do_not_delegate():
    # Migrated stages must call the package directly, not shell out to scripts.
    for name in MARS_STAGES + ["run_full_mars_pipeline"]:
        src = inspect.getsource(getattr(pipelines, name))
        assert "run_script" not in src, f"{name} should call channel_heads.* directly"


def test_earth_and_mars_stages_are_callable():
    for name in ("train_earth_cnn", "train_earth_xgb_variants", "generate_poster_figures",
                 "extract_mars_cnn_embeddings", "run_mars_combined_inference"):
        assert callable(getattr(pipelines, name))


def test_trim_description_matches_pruning_semantics():
    # pre_remove_max_order: 0 keeps all, 1 drops 1st-order, n drops <= n order.
    assert pipelines.trim_description(0) == "none"
    assert pipelines.trim_description(1) == "drop 1st-order"
    assert "2" in pipelines.trim_description(2)


def test_network_variant_and_builders_exposed():
    assert callable(pipelines.build_earth_network_variants)
    assert callable(pipelines.build_earth_basin_network)
    # NetworkVariant carries the exact extraction parameters per setup.
    assert pipelines.NetworkVariant._fields == (
        "label", "s", "threshold_cells", "threshold_km2",
        "pre_remove_max_order", "order_gap_to_prune",
    )
