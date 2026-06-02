"""Tests for channel_heads.rasterization.mars_patches + manifest (Phase 4).

The broad Earth-parity smoke test lives in tests/test_rasterizer.py
(TestMarsRasterizationSmoke, now loading the package module); here we cover the
patch contract details and the manifest schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channel_heads.rasterization import manifest
from channel_heads.rasterization import mars_patches as mp

shapely = pytest.importorskip("shapely")
LineString = shapely.geometry.LineString


def _simple_pair():
    conf_xy = (0.0, 0.0)
    h1_xy = (-2000.0, 8000.0)
    h2_xy = (2000.0, 8000.0)
    path_a = LineString([h1_xy, conf_xy])
    path_b = LineString([h2_xy, conf_xy])
    network_segments = [LineString([conf_xy, (0.0, -6000.0)])]
    return h1_xy, h2_xy, conf_xy, path_a, path_b, network_segments


def test_patch_shape_dtype_and_classes():
    h1, h2, conf, pa, pb, segs = _simple_pair()
    patch = mp.render_pair_patch(h1, h2, conf, pa, pb, segs)
    assert patch.shape == (mp.TARGET_SIZE, mp.TARGET_SIZE) == (128, 128)
    assert patch.dtype == np.uint8
    assert set(np.unique(patch)).issubset({0, 1, 2, 3, 4})


def test_class_labels_and_confluence_marker_present():
    h1, h2, conf, pa, pb, segs = _simple_pair()
    patch = mp.render_pair_patch(h1, h2, conf, pa, pb, segs)
    assert mp.BACKGROUND == 0 and mp.BRANCH_A == 1 and mp.BRANCH_B == 2
    assert mp.OTHER_STREAMS == 3 and mp.CONFLUENCE_MARKER == 4
    # exactly one confluence marker pixel
    assert int((patch == mp.CONFLUENCE_MARKER).sum()) == 1
    assert (patch == mp.BRANCH_A).any()
    assert (patch == mp.BRANCH_B).any()


def test_branch_ab_protection_keeps_both_branches():
    # Branches that share the confluence must both survive (mutual protection).
    h1, h2, conf, pa, pb, segs = _simple_pair()
    patch = mp.render_pair_patch(h1, h2, conf, pa, pb, segs)
    flags = mp.raster_quality_flags(patch)
    assert flags["has_branch_a"] and flags["has_branch_b"]
    assert flags["branch_a_connected"] and flags["branch_b_connected"]
    assert flags["branches_connected"]


def test_rasterize_mars_pair_alias():
    assert mp.rasterize_mars_pair is mp.render_pair_patch


def test_manifest_schema_and_validation():
    rows = [
        {
            "network_id": 0, "pair_id": "p", "head_id_1": 1, "head_id_2": 2,
            "head_node_id_1": 1, "head_node_id_2": 2, "confluence_id": 3,
            "confluence_node_id": 3, "patch_path": "data/.../p.npy",
            "patch_shape": "128,128", "patch_dtype": "uint8",
            "patch_encoding": "earth_5class", "patch_status": "ok",
            "patch_qa_reason": "", **manifest.empty_flags(),
        }
    ]
    df = manifest.build_patch_manifest(rows)
    assert list(df.columns) == manifest.PATCH_INDEX_COLUMNS
    manifest.validate_patch_manifest(df)  # should not raise


def test_validate_patch_manifest_rejects_bad_schema():
    df = pd.DataFrame({"network_id": [0]})
    with pytest.raises(ValueError):
        manifest.validate_patch_manifest(df)

    rows = [
        {
            "network_id": 0, "pair_id": "p", "head_id_1": 1, "head_id_2": 2,
            "head_node_id_1": 1, "head_node_id_2": 2, "confluence_id": 3,
            "confluence_node_id": 3, "patch_path": "", "patch_shape": "64,64",
            "patch_dtype": "uint8", "patch_encoding": "earth_5class",
            "patch_status": "ok", "patch_qa_reason": "", **manifest.empty_flags(),
        }
    ]
    with pytest.raises(ValueError):  # wrong patch_shape
        manifest.validate_patch_manifest(manifest.build_patch_manifest(rows))
