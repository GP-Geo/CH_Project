"""Tests for channel_heads.viz.poster — poster figure primitives.

Headless (Agg backend). These pin the reusable poster helpers used by
``notebooks/presentation/14_poster_figure_inventory.ipynb``: shared style,
dual-format export, the synthetic concept/flowchart figures, 5-class patch
rendering, feature/model/Mars summary panels, and representative-network
selection.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from channel_heads.rasterization.schema import NUM_CLASSES  # noqa: E402
from channel_heads.viz import poster  # noqa: E402


def test_apply_poster_style_sets_white_background():
    poster.apply_poster_style()
    assert plt.rcParams["figure.facecolor"] in ("white", (1.0, 1.0, 1.0, 1.0))
    assert plt.rcParams["svg.fonttype"] == "none"


def test_save_figure_writes_both_formats(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = poster.save_figure(fig, tmp_path, "demo", formats=("svg", "png"))
    assert set(written) == {"svg", "png"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


def test_pipeline_flowchart_returns_figure():
    fig = poster.pipeline_flowchart()
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_pipeline_flowchart_includes_calibration_stage():
    labels = [lbl for lbl, _ in poster.DEFAULT_PIPELINE_STAGES]
    assert any("calibration" in lbl.lower() for lbl in labels)
    assert "calibration" in poster.PHASE_COLORS


def test_frame_only_shows_all_spines_and_hides_ticks():
    fig, ax = plt.subplots()
    poster.apply_poster_style()  # hides top/right spines globally
    poster.frame_only(ax)
    assert all(sp.get_visible() for sp in ax.spines.values())
    assert ax.get_xticks().size == 0 and ax.get_yticks().size == 0
    plt.close(fig)


def test_colored_hillshade_draws_two_layers():
    fig, ax = plt.subplots()
    z = np.random.default_rng(3).random((12, 12)) * 50
    poster.colored_hillshade(ax, z, extent=(0, 12, 0, 12))
    assert len(ax.images) == 2  # terrain + hillshade overlay
    plt.close(fig)


def test_format_degree_axes_labels_have_degree_sign():
    fig, ax = plt.subplots()
    ax.set_xlim(-118.1, -117.8)
    ax.set_ylim(36.6, 36.9)
    poster.format_degree_axes(ax)
    assert ax.get_xticklabels()
    assert all("°" in t.get_text() for t in ax.get_xticklabels())
    assert any("W" in t.get_text() for t in ax.get_xticklabels())
    assert any("N" in t.get_text() for t in ax.get_yticklabels())
    plt.close(fig)


def test_assign_channel_head_labels_only_participating():
    nodes = pd.DataFrame({
        "node_id": [5, 2, 9, 4],
        "node_type": ["channel_head", "channel_head", "channel_head", "confluence"],
    })
    label_map, heads = poster.assign_channel_head_labels(nodes, only_node_ids=[9, 2])
    # only the two requested heads, numbered by ascending node_id
    assert label_map == {2: "C1", 9: "C2"}
    assert set(heads["node_id"]) == {2, 9}


def test_predicted_pair_list_sorted_by_probability():
    touch = pd.DataFrame({
        "head_node_id_1": [2, 9],
        "head_node_id_2": [9, 2],
        "prob_touching": [0.80, 0.99],
    })
    label_map = {2: "C1", 9: "C2"}
    rows = poster.predicted_pair_list(touch, label_map, prob_col="prob_touching")
    assert rows[0] == (1, "C2", "C1", 0.99)  # highest prob first
    assert rows[1][0] == 2 and rows[1][3] == 0.80


def test_pair_definition_concept_two_panels():
    fig = poster.pair_definition_concept()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_patch_cmap_norm_and_legend_match_num_classes():
    cmap, _ = poster.patch_cmap_norm()
    assert cmap.N == NUM_CLASSES
    assert len(poster.patch_legend_handles()) == NUM_CLASSES


def test_plot_5class_patch_uses_nearest_interpolation():
    fig, ax = plt.subplots()
    patch = np.random.randint(0, NUM_CLASSES, (16, 16)).astype("uint8")
    poster.plot_5class_patch(ax, patch, title="p")
    assert ax.images and ax.images[0].get_interpolation() == "nearest"
    plt.close(fig)


def _labelled_features(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "orientation_diff_deg": rng.random(n) * 90,
            "headhead_dist_norm": rng.random(n),
            "apex_angle_deg": rng.random(n) * 120,
            "strahler_order_diff": rng.integers(0, 4, n),
            "proximity_profile_norm": rng.random(n),
        }
    )


def test_feature_distribution_panel_with_labels():
    fig = poster.feature_distribution_panel(_labelled_features())
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_feature_distribution_panel_without_labels():
    df = _labelled_features().drop(columns=["y"])
    fig = poster.feature_distribution_panel(df, label_col="y")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_feature_distribution_panel_without_schematic():
    fig = poster.feature_distribution_panel(_labelled_features(), include_schematic=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_feature_schematic_panel_covers_all_features():
    fig = poster.feature_schematic_panel()
    assert isinstance(fig, Figure)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    for feat in poster.PRODUCTION_GEOM_FEATURES:
        assert any(feat in t for t in titles), feat
    plt.close(fig)


def test_model_comparison_bars_skips_absent_metrics():
    df = pd.DataFrame({"model": ["a", "b"], "roc_auc": [0.7, 0.9], "F1": [0.5, 0.6]})
    fig = poster.model_comparison_bars(df, metric_cols=("roc_auc", "pr_auc", "F1"))
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_mars_prediction_summary_bars_one_and_two_panels():
    per = pd.DataFrame(
        {
            "model_variant": ["tab", "emb", "logit"],
            "n_pairs": [3785, 3785, 3785],
            "n_touching": [2183, 566, 2604],
            "n_high_confidence_prob_ge_0.80": [898, 642, 2795],
        }
    )
    fig1 = poster.mars_prediction_summary_bars(per)
    assert len(fig1.axes) == 1
    plt.close(fig1)

    ag = pd.DataFrame(
        {
            "model_variant": ["agreement_tabular_vs_emb", "agreement_emb_vs_logit"],
            "agreement_fraction": [0.57, 0.46],
            "n_changes": [1619, 2042],
        }
    )
    fig2 = poster.mars_prediction_summary_bars(per, ag)
    assert len(fig2.axes) == 2
    plt.close(fig2)


def test_select_representative_networks_categories():
    rng = np.random.default_rng(1)
    n_per = {1: 40, 2: 6, 3: 18, 4: 25}
    rows = []
    for nid, k in n_per.items():
        for _ in range(k):
            rows.append(
                {
                    "network_id": nid,
                    "pred_touching_emb": rng.integers(0, 2),
                    "prob_touching_emb": rng.random(),
                    "emb_logit_disagreement": rng.integers(0, 2),
                }
            )
    preds = pd.DataFrame(rows)
    picks = poster.select_representative_networks(preds, min_pairs=6)
    assert picks["dense"] == 1  # most pairs
    assert picks["simple"] == 2  # fewest pairs (>= min_pairs)
    # categories are de-duplicated: no network id appears twice
    assert len(set(picks.values())) == len(picks)


def test_hillshade_range_and_shape():
    z = np.random.default_rng(2).random((20, 20)) * 100
    hs = poster.hillshade(z)
    assert hs.shape == z.shape
    assert 0.0 <= float(hs.min()) and float(hs.max()) <= 1.0


def test_drainage_density_regime_panel():
    df = pd.DataFrame(
        {"threshold_km2": [0.05, 0.1, 0.5, 1.0], "dd_med": [3.1, 2.2, 1.5, 1.4]}
    )
    fig = poster.drainage_density_regime_panel(
        df,
        threshold_col="threshold_km2",
        earth_dd_col="dd_med",
        mars_dd_median=1.46,
        regime_thresholds={"regA": 0.2, "regB": 0.25},
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
