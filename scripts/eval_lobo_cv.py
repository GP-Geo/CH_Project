#!/usr/bin/env python
"""Leave-one-basin-out CV for the geom+cnn_emb XGBoost — honest, low-variance
metrics that don't depend on a single GroupShuffleSplit draw.

For each config (baseline + regA/regB/regC) every basin is held out in turn;
the model trains on the others and predicts the held-out basin. We report the
per-fold ROC-AUC mean±std and the pooled out-of-fold AUC / F1 / precision /
recall / accuracy.

Run::  python scripts/eval_lobo_cv.py
Writes models/lobo_cv_metrics.csv

Primary interface: ``notebooks/diagnostics/lobo_cv.ipynb`` runs and displays the
same LOBO comparison read-only (calls ``channel_heads.eval`` —
``max_precision_threshold`` / ``classification_metrics``); this script is the
headless wrapper that persists ``models/lobo_cv_metrics.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from channel_heads.eval import (
    classification_metrics,
    leave_one_group_out_oof,
    max_precision_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
GEOM = ["orientation_diff_deg", "headhead_dist_norm", "apex_angle_deg",
        "strahler_order_diff", "proximity_profile_norm"]
EMB = [f"emb_{i}" for i in range(4)]
FEATURES = GEOM + EMB
N_ESTIMATORS, MAX_DEPTH, LR, SEED = 200, 4, 0.1, 42

DATASETS = {
    "baseline": ROOT / "data/results/master_dataset_v4_cnn_full.csv",
    "regA": ROOT / "data/results/master_dataset_regA_with_emb.csv",
    "regB": ROOT / "data/results/master_dataset_regB_with_emb.csv",
    "regC": ROOT / "data/results/master_dataset_regC_with_emb.csv",
}


def _fit_predict(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame) -> np.ndarray:
    """Train the geom+emb XGBoost on a fold and predict P(touching)."""
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LR,
        scale_pos_weight=spw, eval_metric="logloss", random_state=SEED,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def lobo(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=FEATURES + ["y", "basin"]).reset_index(drop=True)
    oof, y, groups, fold_aucs = leave_one_group_out_oof(df, FEATURES, _fit_predict)
    thr = max_precision_threshold(y, oof)
    m = classification_metrics(y, oof, thr)
    return {
        "n": len(df),
        "n_basins": groups.nunique(),
        "pooled_auc": m["roc_auc"],
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "threshold": thr,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
    }


def main() -> int:
    rows = []
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        r = lobo(pd.read_csv(path))
        r = {"config": name, **r}
        rows.append(r)
        print(
            f"{name:9} LOBO: pooled_AUC={r['pooled_auc']:.3f}  "
            f"fold_AUC={r['fold_auc_mean']:.3f}±{r['fold_auc_std']:.3f}  "
            f"F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
            f"Acc={r['accuracy']:.3f}  (n={r['n']}, {r['n_basins']} basins)"
        )
    if rows:
        out = ROOT / "models/lobo_cv_metrics.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
