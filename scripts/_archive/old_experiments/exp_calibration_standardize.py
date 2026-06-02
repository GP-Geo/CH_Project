#!/usr/bin/env python
"""Measure whether per-basin feature standardization fixes the cross-basin
calibration problem (regC's pooled-AUC collapse) without hurting per-basin AUC.

Each basin/network z-scores its OWN features (mean/std computed within the
basin). This is leakage-free and mirrors how a Mars network would self-
normalize at inference. We compare LOBO-CV with vs without standardization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
GEOM = ["orientation_diff_deg", "headhead_dist_norm", "apex_angle_deg",
        "strahler_order_diff", "proximity_profile_norm"]
EMB = [f"emb_{i}" for i in range(4)]
FEATURES = GEOM + EMB
N_EST, DEPTH, LR, SEED = 200, 4, 0.1, 42

DATASETS = {
    "baseline": ROOT / "data/results/master_dataset_v4_cnn_full.csv",
    "regA": ROOT / "data/results/master_dataset_regA_with_emb.csv",
    "regB": ROOT / "data/results/master_dataset_regB_with_emb.csv",
    "regC": ROOT / "data/results/master_dataset_regC_with_emb.csv",
}


def per_basin_standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """z-score each column within each basin (self-normalization)."""
    out = df.copy()
    g = out.groupby("basin")
    for c in cols:
        mu = g[c].transform("mean")
        sd = g[c].transform("std").replace(0, 1.0).fillna(1.0)
        out[c] = (out[c] - mu) / sd
    return out


def lobo(df: pd.DataFrame, standardize_cols: list[str] | None) -> dict:
    df = df.dropna(subset=FEATURES + ["y", "basin"]).reset_index(drop=True)
    if standardize_cols:
        df = per_basin_standardize(df, standardize_cols)
    X, y, groups = df[FEATURES], df["y"].astype(int).to_numpy(), df["basin"]
    oof = np.full(len(df), np.nan)
    fold = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = XGBClassifier(n_estimators=N_EST, max_depth=DEPTH, learning_rate=LR,
                          scale_pos_weight=spw, eval_metric="logloss",
                          random_state=SEED)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        if len(np.unique(y[te])) > 1:
            fold.append(roc_auc_score(y[te], p))
    prec, rec, thr = precision_recall_curve(y, oof)
    ok = rec[:-1] >= 0.5
    t = float(thr[np.argmax(np.where(ok, prec[:-1], -1))]) if ok.any() else 0.5
    return {
        "pooled_auc": roc_auc_score(y, oof),
        "fold_auc_mean": float(np.mean(fold)),
        "fold_auc_std": float(np.std(fold)),
        "f1": f1_score(y, (oof >= t).astype(int)),
    }


def main() -> int:
    rows = []
    for name, path in DATASETS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        base = lobo(df, None)
        std_emb = lobo(df, EMB)          # standardize embeddings only
        std_all = lobo(df, FEATURES)     # standardize everything
        print(f"\n=== {name} ===")
        print(f"  raw          : pooled={base['pooled_auc']:.3f}  "
              f"fold={base['fold_auc_mean']:.3f}±{base['fold_auc_std']:.3f}  F1={base['f1']:.3f}")
        print(f"  std(emb)     : pooled={std_emb['pooled_auc']:.3f}  "
              f"fold={std_emb['fold_auc_mean']:.3f}±{std_emb['fold_auc_std']:.3f}  F1={std_emb['f1']:.3f}")
        print(f"  std(all feat): pooled={std_all['pooled_auc']:.3f}  "
              f"fold={std_all['fold_auc_mean']:.3f}±{std_all['fold_auc_std']:.3f}  F1={std_all['f1']:.3f}")
        rows.append({"config": name, **{f"raw_{k}": v for k, v in base.items()},
                     **{f"stdemb_{k}": v for k, v in std_emb.items()},
                     **{f"stdall_{k}": v for k, v in std_all.items()}})
    pd.DataFrame(rows).to_csv(ROOT / "models/calibration_experiment.csv", index=False)
    print("\nWrote models/calibration_experiment.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
