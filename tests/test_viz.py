"""Tests for channel_heads.viz — vector contact-sheet rendering.

Headless (Agg backend). These pin the shared rendering primitives extracted from
the Mars rendering scripts: a single pair panel, and the contact-sheet grid that
either saves to disk or returns the figure for inline display.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

pytest.importorskip("geopandas")
pytest.importorskip("shapely")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from channel_heads.viz import (  # noqa: E402
    make_palette,
    render_contact_sheet,
    render_outlet_touching_pairs,
    render_pair_panel,
    roc_curve_panel,
)


def _segments():
    return gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 0), (0, 1)])])


def test_render_pair_panel_sets_title_and_limits():
    fig, ax = plt.subplots()
    render_pair_panel(
        ax,
        segs_n=_segments(),
        path_a=LineString([(0.0, 0.0), (1.0, 1.0)]),
        path_b=LineString([(0.0, 0.0), (-1.0, 1.0)]),
        h1_xy=(1.0, 1.0),
        h2_xy=(-1.0, 1.0),
        conf_xy=(0.0, 0.0),
        outlet_xy=(0.0, -1.0),
        title="pair-x",
    )
    assert ax.get_title() == "pair-x"
    assert ax.get_aspect() == 1.0  # set_aspect("equal")
    plt.close(fig)


def _contact_inputs():
    nodes = gpd.GeoDataFrame(
        {"node_id": [0, 1, 2]},
        geometry=gpd.points_from_xy([0.0, 1.0, 0.5], [0.0, 0.0, 1.0]),
    )
    rows = pd.DataFrame(
        [
            {
                "network_id": 7,
                "pair_id": "p",
                "head_node_id_1": 0,
                "head_node_id_2": 1,
                "confluence_node_id": 2,
            }
        ]
    )
    paths = {
        ("p", "A"): LineString([(0.0, 0.0), (0.5, 1.0)]),
        ("p", "B"): LineString([(1.0, 0.0), (0.5, 1.0)]),
    }
    return rows, {7: nodes}, {7: _segments()}, {7: (0.0, -1.0)}, paths


def test_render_contact_sheet_returns_figure_when_no_output():
    rows, nodes_by, segs_by, outlets_by, paths = _contact_inputs()
    fig = render_contact_sheet(
        rows,
        nodes_by,
        segs_by,
        outlets_by,
        paths,
        title="sheet",
        subtitle_fn=lambda r: str(r["pair_id"]),
        output=None,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_render_contact_sheet_saves_when_output_given(tmp_path):
    rows, nodes_by, segs_by, outlets_by, paths = _contact_inputs()
    out = tmp_path / "sheet.png"
    ret = render_contact_sheet(
        rows,
        nodes_by,
        segs_by,
        outlets_by,
        paths,
        title="sheet",
        subtitle_fn=lambda r: str(r["pair_id"]),
        output=out,
    )
    assert ret is None
    assert out.exists() and out.stat().st_size > 0


def test_render_contact_sheet_empty_returns_none():
    rows, nodes_by, segs_by, outlets_by, paths = _contact_inputs()
    assert (
        render_contact_sheet(
            rows.iloc[0:0],
            nodes_by,
            segs_by,
            outlets_by,
            paths,
            title="x",
            subtitle_fn=lambda r: "",
            output=None,
        )
        is None
    )


def test_make_palette_length():
    assert len(make_palette(50)) == 50
    assert len(make_palette(3)) == 3


def test_render_outlet_touching_pairs_returns_fig_and_handles_empty():
    nodes = gpd.GeoDataFrame(
        {"node_id": [0, 1, 2], "node_type": ["channel_head", "channel_head", "confluence"]},
        geometry=gpd.points_from_xy([0.0, 1.0, 0.5], [0.0, 0.0, 1.0]),
    )
    paths = {
        ("p", "A"): LineString([(0.0, 0.0), (0.5, 1.0)]),
        ("p", "B"): LineString([(1.0, 0.0), (0.5, 1.0)]),
    }
    df_net = pd.DataFrame([{
        "pair_id": "p", "pred_touching_emb": 1, "prob_touching_emb": 0.9,
        "head_node_id_1": 0, "head_node_id_2": 1,
    }])
    fig = render_outlet_touching_pairs(
        9,
        df_net,
        _segments(),
        nodes,
        (0.5, -1.0),
        paths,
        output=None,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
    # number_heads: unique C# head labels + "Pair k: Ca – Cb" legend, rect frame
    fig = render_outlet_touching_pairs(
        9, df_net, _segments(), nodes, (0.5, -1.0), paths, output=None,
        number_heads=True,
    )
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert all(sp.get_visible() for sp in ax.spines.values())
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t.startswith("Pair 1:") and "C1" in t and "C2" in t for t in legend_texts)
    plt.close(fig)
    # no touching pairs -> returns None
    df_empty = pd.DataFrame([{"pair_id": "p", "pred_touching_emb": 0, "prob_touching_emb": 0.1}])
    assert (
        render_outlet_touching_pairs(
            9,
            df_empty,
            _segments(),
            nodes,
            None,
            paths,
            output=None,
        )
        is None
    )


def test_roc_curve_panel_legend_title_and_aspect():
    fig, ax = plt.subplots()
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    roc_curve_panel(ax, [("model", y, p)], "ROC")
    assert ax.get_title() == "ROC"
    assert ax.get_aspect() == 1.0
    leg = ax.get_legend()
    assert leg is not None and "AUC=" in leg.get_texts()[0].get_text()
    plt.close(fig)
