"""Tests for channel_heads.eval.lobo — LOBO XGBoost report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from channel_heads.eval import lobo


def _synthetic(single_class_basin: bool):
    """Multi-basin frame; optionally make one basin single-class (y all 1)."""
    rng = np.random.default_rng(0)
    rows = []
    basins = ["b1", "b2", "b3", "b4"]
    for b in basins:
        for i in range(40):
            feat = {f: float(rng.random()) for f in lobo.FEATURES}
            if single_class_basin and b == "b4":
                y = 1
            else:
                y = int((feat["orientation_diff_deg"] + rng.normal(0, 0.1)) > 0.5)
            rows.append({**feat, "y": y, "basin": b})
    return pd.DataFrame(rows)


class TestLoboDatasetPaths:
    def test_keys_and_suffixes(self, tmp_path):
        m = lobo.lobo_dataset_paths(tmp_path)
        assert set(m) == {"baseline", "regA", "regB", "regC"}
        assert m["baseline"].name == "master_dataset_v4_cnn_full.csv"
        assert m["regA"].name == "master_dataset_regA_with_emb.csv"
        assert str(m["regB"]).endswith("data/results/master_dataset_regB_with_emb.csv")


class TestLoboReport:
    def test_report_schema(self):
        df = _synthetic(single_class_basin=False)
        r = lobo.lobo_xgb_report(df)
        assert set(r) == {
            "n", "n_basins", "pooled_auc", "fold_auc_mean", "fold_auc_std",
            "threshold", "precision", "recall", "f1", "accuracy",
        }
        assert r["n"] == len(df)
        assert r["n_basins"] == 4

    def test_single_class_fold_is_skipped_without_error(self):
        # b4 is single-class; roc_auc on that fold would raise if not skipped.
        df = _synthetic(single_class_basin=True)
        r = lobo.lobo_xgb_report(df)
        # The report still completes and yields a finite fold-AUC mean computed
        # over the multi-class folds only.
        assert np.isfinite(r["fold_auc_mean"])
        assert r["n_basins"] == 4

    def test_features_constant_order(self):
        assert lobo.FEATURES == [
            "orientation_diff_deg", "headhead_dist_norm", "apex_angle_deg",
            "strahler_order_diff", "proximity_profile_norm",
            "emb_0", "emb_1", "emb_2", "emb_3",
        ]
