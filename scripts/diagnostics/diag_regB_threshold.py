#!/usr/bin/env python
"""Quick diagnostic: regB Earth test PR curve + Mars prob distribution.

Goal: decide whether the current "max P at recall>=0.5" protocol is the
right knee to anchor the threshold at, by looking at:

  1. RegB Earth test PR curve (with current threshold marked)
  2. Production combined-emb Earth test PR curve (for sharpness comparison)
  3. Mars prob histogram under regB
  4. Mars prob histogram under production (for shape comparison)

Outputs:
  data/results/diag_regB_threshold_pr_curve.png
  data/results/diag_regB_threshold_mars_hist.png
  stdout summary

Primary interface: ``notebooks/diagnostics/regB_threshold.ipynb`` runs the same
comparison inline read-only (calls ``channel_heads.eval`` /
``channel_heads.inference``); this script is the headless wrapper that writes the
PNG diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from channel_heads.eval.diagnostics import holdout_split_predict, pr_curve_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/diagnostics/ -> repo root
RESULTS_DIR = PROJECT_ROOT / "data/results"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = RESULTS_DIR


def main() -> int:
    # ----- regB -----
    regB_csv = RESULTS_DIR / "master_dataset_regB_with_emb.csv"
    regB_model = MODELS_DIR / "xgb_geom_plus_cnn_emb_regB.json"
    regB_feats = MODELS_DIR / "feature_columns_geom_plus_cnn_emb_regB.txt"
    regB_thr = float(
        (MODELS_DIR / "optimal_threshold_geom_plus_cnn_emb_regB.txt")
        .read_text().strip().splitlines()[0]
    )
    yB, pB, _ = holdout_split_predict(regB_csv, regB_model, regB_feats)
    metB = pr_curve_metrics(yB, pB, regB_thr)

    # ----- production combined-emb (Phase 6B) -----
    prod_csv = RESULTS_DIR / "master_dataset_v4_cnn_full.csv"
    prod_model = MODELS_DIR / "xgb_geom_plus_cnn_emb.json"
    prod_feats = MODELS_DIR / "feature_columns_geom_plus_cnn_emb.txt"
    prod_thr_path = MODELS_DIR / "optimal_threshold_geom_plus_cnn_emb.txt"
    prod_thr = float(prod_thr_path.read_text().strip().splitlines()[0])
    yP, pP, _ = holdout_split_predict(prod_csv, prod_model, prod_feats)
    metP = pr_curve_metrics(yP, pP, prod_thr)

    print("=" * 70)
    print("Earth test set metrics (GroupShuffleSplit by basin__outlet, seed=42)")
    print("=" * 70)
    for label, met, thr in [("regB", metB, regB_thr), ("production_emb", metP, prod_thr)]:
        print(f"\n[{label}]")
        print(f"  test size           : {len(yB if label=='regB' else yP)}")
        print(f"  pos fraction        : {(yB if label=='regB' else yP).mean():.3f}")
        print(f"  ROC AUC             : {met['roc_auc']:.4f}")
        print(f"  PR AUC              : {met['pr_auc']:.4f}")
        print(f"  current threshold   : {thr:.4f}")
        print(f"    -> P={met['P_at_thr']:.3f}  R={met['R_at_thr']:.3f}  F1={met['F1_at_thr']:.3f}")
        print(f"  F1-optimal threshold: {met['F1_optimal_threshold']:.4f}  (F1={met['F1_optimal_max']:.3f})")

    # ----- Earth test PR curves -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, met, thr, label, color in [
        (axes[0], metB, regB_thr, f"regB (PR AUC={metB['pr_auc']:.3f})", "#1f6fb4"),
        (axes[1], metP, prod_thr, f"production_emb (PR AUC={metP['pr_auc']:.3f})", "#b5533f"),
    ]:
        ax.plot(met["recalls"], met["precisions"], color=color, lw=2)
        if thr is not None:
            mask = met["thresholds"] >= thr
            if mask.any():
                idx = int(np.argmax(mask))
                ax.scatter(
                    met["recalls"][idx], met["precisions"][idx],
                    s=80, c="black", zorder=5,
                    label=f"current thr={thr:.3f}\n(P={met['P_at_thr']:.2f}, R={met['R_at_thr']:.2f})",
                )
        f1_thr = met["F1_optimal_threshold"]
        mask = met["thresholds"] >= f1_thr
        if mask.any():
            idx = int(np.argmax(mask))
            ax.scatter(
                met["recalls"][idx], met["precisions"][idx],
                s=80, marker="x", c="red", zorder=5,
                label=f"F1-opt thr={f1_thr:.3f}\n(F1={met['F1_optimal_max']:.2f})",
            )
        ax.axvline(0.5, color="gray", ls=":", lw=1, label="recall=0.5")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Earth test PR curves — regB (current) vs production combined-emb")
    fig.tight_layout()
    pr_out = OUTPUT_DIR / "diag_regB_threshold_pr_curve.png"
    fig.savefig(pr_out, dpi=140)
    plt.close(fig)
    print(f"\nWrote PR curves -> {pr_out}")

    # ----- Mars prob histograms -----
    mars_regB = pd.read_parquet(
        PROJECT_ROOT / "data/Mars/model_outputs/mars_combined_regB_predictions.parquet"
    )
    mars_base = pd.read_parquet(
        PROJECT_ROOT / "data/Mars/model_outputs/mars_combined_model_predictions.parquet"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(mars_regB["prob_touching"], bins=bins, alpha=0.55,
            color="#1f6fb4", label=f"regB (n={len(mars_regB)})")
    ax.hist(mars_base["prob_touching_emb"], bins=bins, alpha=0.55,
            color="#b5533f", label=f"production emb (n={len(mars_base)})")
    ax.axvline(regB_thr, color="#1f6fb4", ls="--", lw=1.5,
               label=f"regB thr={regB_thr:.3f}")
    ax.axvline(prod_thr, color="#b5533f", ls="--", lw=1.5,
               label=f"prod thr={prod_thr:.3f}")
    ax.set_xlabel("Predicted probability of touching")
    ax.set_ylabel("# Mars pairs")
    ax.set_title("Mars prob_touching distribution — regB vs production")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    mars_out = OUTPUT_DIR / "diag_regB_threshold_mars_hist.png"
    fig.savefig(mars_out, dpi=140)
    plt.close(fig)
    print(f"Wrote Mars hist  -> {mars_out}")

    print("\nMars prob_touching distribution:")
    for label, probs in [("regB", mars_regB["prob_touching"]),
                          ("production_emb", mars_base["prob_touching_emb"])]:
        print(
            f"  {label:18s}  median={probs.median():.3f}  "
            f"p25={probs.quantile(0.25):.3f}  p75={probs.quantile(0.75):.3f}  "
            f"frac >= 0.5: {(probs >= 0.5).mean():.2%}  "
            f"frac >= 0.9: {(probs >= 0.9).mean():.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
