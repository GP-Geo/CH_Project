#!/usr/bin/env python
"""Generate result figures for the rebuilt models:

  1. ROC curves on the Earth held-out test split (baseline 3 variants + regimes).
  2. A multi-panel view of real Mars networks and their channels, annotated with
     the combined-model touching predictions.

Outputs -> data/results/figures_models/  and  data/Mars/model_outputs/figures_combined/

Primary interface: ``notebooks/presentation/result_figures.ipynb`` renders the
ROC comparison inline via ``channel_heads.viz.roc_curve_panel`` /
``channel_heads.eval.outlet_group_holdout``; this script is the headless wrapper
that writes the PNG figures (ROC + Mars networks).
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from channel_heads.eval import outlet_group_holdout  # noqa: E402
from channel_heads.io.paths import PROJECT_ROOT as ROOT  # noqa: E402
from channel_heads.viz import roc_curve_panel  # noqa: E402

FIG_M = ROOT / "data/results/figures_models"
FIG_MARS = ROOT / "data/Mars/model_outputs/figures_combined"

GEOM = ["orientation_diff_deg", "headhead_dist_norm", "apex_angle_deg",
        "strahler_order_diff", "proximity_profile_norm"]
EMB = [f"emb_{i}" for i in range(4)]


def test_split(df: pd.DataFrame) -> pd.DataFrame:
    _, test_idx = outlet_group_holdout(df)
    return df.iloc[test_idx]


def make_roc():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    base = pd.read_csv(ROOT / "data/results/master_dataset_v4_cnn_full.csv")
    bt = test_split(base)
    y = bt["y"].astype(int).to_numpy()
    entries = []
    for name, mp, feats in [
        ("geom_only", "xgb_geom_only.json", GEOM),
        ("geom+cnn_emb", "xgb_geom_plus_cnn_emb.json", GEOM + EMB),
        ("geom+cnn_logit", "xgb_geom_plus_cnn_logit.json", GEOM + ["cnn_logit"]),
    ]:
        m = xgb.XGBClassifier()
        m.load_model(str(ROOT / "models" / mp))
        entries.append((name, y, m.predict_proba(bt[feats])[:, 1]))
    roc_curve_panel(axes[0], entries, "Baseline (Earth held-out test)")

    reg_entries = []
    for R in ["regA", "regB", "regC"]:
        d = pd.read_csv(ROOT / f"data/results/master_dataset_{R}_with_emb.csv")
        dt = test_split(d)
        m = xgb.XGBClassifier()
        m.load_model(str(ROOT / "models" / f"xgb_geom_plus_cnn_emb_{R}.json"))
        reg_entries.append(
            (R, dt["y"].astype(int).to_numpy(), m.predict_proba(dt[GEOM + EMB])[:, 1])
        )
    roc_curve_panel(axes[1], reg_entries, "Regimes geom+cnn_emb (Earth held-out test)")

    fig.suptitle("ROC — rebuilt models on fixed rasters", fontsize=13)
    fig.tight_layout()
    out = FIG_M / "roc_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def make_networks(n_panels: int = 6):
    """Plot real Mars networks (their channels) coloured by touching prediction."""
    topo = ROOT / "data/Mars/topology/mars_vn_topology_model_ready.gpkg"
    segs = gpd.read_file(topo, layer="mars_segments")
    nodes = gpd.read_file(topo, layer="mars_nodes")
    preds = pd.read_parquet(
        ROOT / "data/Mars/model_outputs/mars_combined_model_predictions.parquet"
    )
    pred_col = "pred_touching_emb"
    pairs = gpd.read_file(ROOT / "data/Mars/topology/mars_vn_pairs.gpkg",
                          layer="mars_pair_paths")

    # Pick reasonably-sized networks with a MIX of touching / non-touching
    # pairs so the colouring is informative (not all-red, not all-blue).
    g = preds.groupby("network_id")
    summary = pd.DataFrame({
        "n_pairs": g.size(),
        "frac_touch": g[pred_col].mean(),
        "n_seg": segs.groupby("network_id").size(),
    }).dropna()
    cand = summary[(summary["n_pairs"].between(8, 40))
                   & (summary["frac_touch"].between(0.3, 0.7))]
    cand = cand.sort_values("n_seg", ascending=False)
    nids = list(cand.head(n_panels).index)
    if len(nids) < n_panels:  # fallback: just the largest networks
        extra = summary.sort_values("n_seg", ascending=False).index
        nids += [n for n in extra if n not in nids][: n_panels - len(nids)]

    cols = 3
    rows = int(np.ceil(len(nids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, nid in zip(axes, nids):
        net_segs = segs[segs["network_id"] == nid]
        net_segs.plot(ax=ax, color="0.55", lw=0.8, zorder=1)
        net_nodes = nodes[nodes["network_id"] == nid]
        net_nodes.plot(ax=ax, color="0.2", markersize=4, zorder=2)

        npred = preds[preds["network_id"] == nid]
        ptouch = set(npred.loc[npred[pred_col] == 1, "pair_id"].astype(str))
        pnot = set(npred.loc[npred[pred_col] == 0, "pair_id"].astype(str))
        # Non-touching pair branches in blue, touching in red, over gray base.
        p_no = pairs[pairs["pair_id"].astype(str).isin(pnot)]
        p_yes = pairs[pairs["pair_id"].astype(str).isin(ptouch)]
        if len(p_no):
            p_no.plot(ax=ax, color="#1f77b4", lw=1.3, zorder=3, alpha=0.8)
        if len(p_yes):
            p_yes.plot(ax=ax, color="#d62728", lw=1.6, zorder=4, alpha=0.9)

        n_t = int((npred[pred_col] == 1).sum())
        ax.set_title(
            f"network {nid}\n{len(net_segs)} channels · {len(npred)} pairs · "
            f"{n_t} touching",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    for ax in axes[len(nids):]:
        ax.set_visible(False)

    fig.suptitle(
        "Mars networks & channels — gray = channels · red = predicted-touching "
        "pairs · blue = non-touching (geom+cnn_emb)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_MARS / "mars_networks_channels.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="channel-heads make-result-figures",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)
    FIG_M.mkdir(parents=True, exist_ok=True)
    FIG_MARS.mkdir(parents=True, exist_ok=True)
    make_roc()
    make_networks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
