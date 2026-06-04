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

import pandas as pd

from channel_heads.eval import lobo as lobo_mod
from channel_heads.io.paths import PROJECT_ROOT as ROOT

GEOM = lobo_mod.GEOM
EMB = lobo_mod.EMB
FEATURES = lobo_mod.FEATURES
N_ESTIMATORS = lobo_mod.N_ESTIMATORS
MAX_DEPTH = lobo_mod.MAX_DEPTH
LR = lobo_mod.LR
SEED = lobo_mod.SEED
DATASETS = lobo_mod.lobo_dataset_paths(ROOT)


lobo = lobo_mod.lobo_xgb_report


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
