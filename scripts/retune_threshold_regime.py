#!/usr/bin/env python
"""Re-tune a regime's combined-emb XGBoost threshold to the F1-optimal point.

Reproduces the same GroupShuffleSplit as
``scripts/train_combined_xgb_regime.py``, predicts on the test set, finds the
threshold maximising F1, and overwrites the regime's
``optimal_threshold_geom_plus_cnn_emb_<regime>.txt``.

Use this when the standard ``max precision at recall >= 0.5`` protocol
collapses on a small test set (e.g. regA: test n=209, tuned precision=1.000,
threshold=0.9915 -> only 3.8% Mars touching).

Primary interface
-----------------
``notebooks/regime/02_threshold_retune.ipynb`` is the primary, documented way to
explore this step (read-only — it does not overwrite the threshold file); it
calls the same shared package functions (``channel_heads.eval`` —
``outlet_group_holdout``, ``f1_optimal_threshold``, ``classification_metrics``;
``channel_heads.inference``). This script is the headless wrapper that persists
the re-tuned threshold.

Run::

    python scripts/retune_threshold_regime.py --regime regA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from channel_heads.eval import (
    classification_metrics,
    f1_optimal_threshold,
    outlet_group_holdout,
)
from channel_heads.inference import load_feature_columns
from channel_heads.regimes import REGIMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEOM_FEATURES = [
    "orientation_diff_deg", "headhead_dist_norm", "apex_angle_deg",
    "strahler_order_diff", "proximity_profile_norm",
]
EMB_FEATURES = [f"emb_{i}" for i in range(4)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=sorted(REGIMES.keys()))
    args = parser.parse_args()
    regime = REGIMES[args.regime]

    csv = PROJECT_ROOT / "data/results" / f"master_dataset_{regime.name}_with_emb.csv"
    model_path = PROJECT_ROOT / "models" / f"xgb_geom_plus_cnn_emb_{regime.name}.json"
    thr_path = PROJECT_ROOT / "models" / f"optimal_threshold_geom_plus_cnn_emb_{regime.name}.txt"
    feat_path = PROJECT_ROOT / "models" / f"feature_columns_geom_plus_cnn_emb_{regime.name}.txt"
    metrics_csv = PROJECT_ROOT / "models" / f"xgb_geom_plus_cnn_emb_{regime.name}_metrics.csv"

    for p in (csv, model_path, thr_path, feat_path):
        if not p.exists():
            print(f"Missing: {p}")
            return 1

    feats = load_feature_columns(feat_path)
    df = pd.read_csv(csv)
    _, test_idx = outlet_group_holdout(df)
    y = df["y"].astype(int).to_numpy()
    X = df[feats].to_numpy(dtype=float)
    X_test, y_test = X[test_idx], y[test_idx]

    model = XGBClassifier()
    model.load_model(str(model_path))
    proba = model.predict_proba(X_test)[:, 1]

    f1_threshold, f1_max = f1_optimal_threshold(y_test, proba)
    m = classification_metrics(y_test, proba, f1_threshold)
    new_metrics = {
        "regime": regime.name,
        "threshold_source": "F1_optimal_retune",
        "optimal_threshold_new": f1_threshold,
        "f1_at_new_threshold": f1_max,
        "precision_at_new": m["precision"],
        "recall_at_new": m["recall"],
        "accuracy_at_new": m["accuracy"],
        "roc_auc_test": m["roc_auc"],
        "pr_auc_test": m["pr_auc"],
    }

    old_threshold = float(thr_path.read_text().strip().splitlines()[0])
    old_pred = (proba >= old_threshold).astype(int)
    print(f"Regime: {regime.name}")
    print(f"  test n               : {len(y_test)} (pos={int(y_test.sum())})")
    print(f"  ROC AUC              : {new_metrics['roc_auc_test']:.4f}")
    print(f"  PR AUC               : {new_metrics['pr_auc_test']:.4f}")
    print(f"  OLD threshold        : {old_threshold:.4f}")
    print(f"    P={precision_score(y_test, old_pred):.3f}  "
          f"R={recall_score(y_test, old_pred):.3f}  "
          f"F1={f1_score(y_test, old_pred):.3f}")
    print(f"  NEW threshold (F1opt): {f1_threshold:.4f}")
    print(f"    P={new_metrics['precision_at_new']:.3f}  "
          f"R={new_metrics['recall_at_new']:.3f}  "
          f"F1={f1_max:.3f}")

    # Persist new threshold and append re-tune metrics to the CSV.
    thr_path.write_text(f"{f1_threshold:.6f}\n")
    print(f"\nOverwrote {thr_path}")

    if metrics_csv.exists():
        existing = pd.read_csv(metrics_csv)
        # Add new columns to the existing metrics row.
        for col, val in new_metrics.items():
            if col not in existing.columns:
                existing[col] = pd.NA
            existing.loc[0, col] = val
        existing.to_csv(metrics_csv, index=False)
        print(f"Updated  {metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
