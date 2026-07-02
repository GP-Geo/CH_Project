"""Tests for the true leave-one-basin-out engine in channel_heads.eval.lobo.

These cover the leakage-critical guarantees: whole-basin holdout, train/test id
disjointness assertions, train-only threshold tuning, embedding-provenance
flagging, and single-class-fold tolerance. The engine is model-agnostic and
torch-free, so these run with a plain XGBoost geom factory.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from channel_heads.eval import lobo


def _synthetic(n_basins: int = 5, per_outlet: int = 4, outlets: int = 8, seed: int = 0,
               single_class_basin: str | None = None) -> pd.DataFrame:
    """Multi-basin pair frame with the geom features + id columns the engine needs."""
    rng = np.random.default_rng(seed)
    rows = []
    for bi in range(n_basins):
        basin = f"b{bi}"
        for o in range(outlets):
            for k in range(per_outlet):
                feats = {f: float(rng.random()) for f in lobo.GEOM}
                signal = feats["orientation_diff_deg"] + rng.normal(0, 0.15)
                y = 1 if (single_class_basin == basin) else int(signal > 0.5)
                rows.append(
                    {
                        **feats,
                        "basin": basin,
                        "outlet": o,
                        "head_1": 100 + k,
                        "head_2": 200 + k,
                        "raster_path": f"/r/{basin}/{o}/{k}.npy",
                        "y": y,
                    }
                )
    return pd.DataFrame(rows)


class TestPairId:
    def test_head_order_invariant(self):
        a = pd.DataFrame({"basin": ["x"], "outlet": [1], "head_1": [7], "head_2": [3]})
        b = pd.DataFrame({"basin": ["x"], "outlet": [1], "head_1": [3], "head_2": [7]})
        assert lobo.make_pair_id(a).iloc[0] == lobo.make_pair_id(b).iloc[0] == "x__1__3_7"


class TestAuditFold:
    def test_clean_fold_passes(self):
        df = _synthetic()
        tr = df[df["basin"] != "b2"]
        te = df[df["basin"] == "b2"]
        audit = lobo.audit_fold(tr, te, test_basin="b2", embedding_provenance=lobo.PROV_NO_CNN)
        assert audit.clean
        assert audit.basins_disjoint and audit.pairs_disjoint and audit.rasters_disjoint

    def test_basin_overlap_raises(self):
        df = _synthetic()
        te = df[df["basin"] == "b2"]
        # Train set still contains b2 rows -> must trip the assertion.
        with pytest.raises(AssertionError):
            lobo.audit_fold(df, te, test_basin="b2", embedding_provenance=lobo.PROV_NO_CNN)

    def test_raster_overlap_raises(self):
        df = _synthetic()
        tr = df[df["basin"] != "b2"].copy()
        te = df[df["basin"] == "b2"].copy()
        # Force one shared raster record while keeping basin ids disjoint.
        tr.iloc[0, tr.columns.get_loc("raster_path")] = te.iloc[0]["raster_path"]
        with pytest.raises(AssertionError, match="raster"):
            lobo.audit_fold(tr, te, test_basin="b2", embedding_provenance=lobo.PROV_NO_CNN)


class TestRunLobo:
    def test_per_basin_and_summary_schema(self):
        df = _synthetic()
        res = lobo.run_lobo(
            df, lobo.GEOM, lobo.make_xgb_factory(lobo.GEOM),
            embedding_provenance=lobo.PROV_NO_CNN,
        )
        # Every basin appears exactly once as a held-out fold.
        assert sorted(res.per_basin["basin"]) == sorted(df["basin"].unique())
        for col in ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy",
                    "prevalence", "n_pairs", "threshold"]:
            assert col in res.per_basin.columns
        for key in ["roc_auc_mean", "roc_auc_std", "roc_auc_median",
                    "pooled_roc_auc", "pooled_f1", "n_pairs", "n_basins"]:
            assert key in res.summary
        # Out-of-fold predictions cover every input row once.
        assert res.summary["n_pairs"] == len(df)
        assert res.all_folds_clean

    def test_threshold_uses_train_only(self, monkeypatch):
        """The df handed to threshold tuning must never contain the held-out basin."""
        df = _synthetic()
        basins_sorted = sorted(df["basin"].astype(str).unique())
        seen_train_basins: list[set[str]] = []
        real = lobo.tune_threshold_train_only

        def spy(df_train, factory, **kw):
            seen_train_basins.append(set(df_train["basin"].astype(str)))
            return real(df_train, factory, **kw)

        monkeypatch.setattr(lobo, "tune_threshold_train_only", spy)
        lobo.run_lobo(df, lobo.GEOM, lobo.make_xgb_factory(lobo.GEOM),
                      embedding_provenance=lobo.PROV_NO_CNN)
        # Basins are processed in sorted order; the i-th tuning call is fold i.
        assert len(seen_train_basins) == len(basins_sorted)
        for held_out, train_basins in zip(basins_sorted, seen_train_basins):
            assert held_out not in train_basins

    def test_cheap_threshold_halves_fits_and_keeps_auc(self):
        """refit_for_threshold=False -> one factory fit per fold, identical AUC."""
        df = _synthetic()
        n_basins = df["basin"].nunique()

        def counting_factory():
            calls = {"n": 0}
            base = lobo.make_xgb_factory(lobo.GEOM)

            def factory(df_train):
                calls["n"] += 1
                return base(df_train)

            return factory, calls

        f_rig, c_rig = counting_factory()
        r_rig = lobo.run_lobo(df, lobo.GEOM, f_rig, embedding_provenance=lobo.PROV_NO_CNN,
                              refit_for_threshold=True)
        f_cheap, c_cheap = counting_factory()
        r_cheap = lobo.run_lobo(df, lobo.GEOM, f_cheap, embedding_provenance=lobo.PROV_NO_CNN,
                                refit_for_threshold=False)

        assert c_cheap["n"] == n_basins              # one fit per fold
        assert c_rig["n"] == 2 * n_basins            # extra inner fit per fold
        # AUC is threshold-free -> unchanged by the threshold strategy.
        assert r_cheap.summary["pooled_roc_auc"] == pytest.approx(
            r_rig.summary["pooled_roc_auc"])

    def test_single_class_fold_tolerated(self):
        df = _synthetic(single_class_basin="b1")
        res = lobo.run_lobo(df, lobo.GEOM, lobo.make_xgb_factory(lobo.GEOM),
                            embedding_provenance=lobo.PROV_NO_CNN)
        row = res.per_basin.set_index("basin").loc["b1"]
        assert np.isnan(row["roc_auc"])           # AUC undefined on one class
        assert np.isfinite(res.summary["roc_auc_mean"])  # nan-safe aggregate
        assert res.summary["n_basins_scored"] < res.summary["n_basins"]


class TestSaveAndProvenance:
    def test_precomputed_mode_is_flagged(self, tmp_path):
        df = _synthetic()
        res = lobo.run_lobo(df, lobo.GEOM, lobo.make_xgb_factory(lobo.GEOM),
                            embedding_provenance=lobo.PROV_PRECOMPUTED)
        out = lobo.save_lobo_result(res, tmp_path, dataset_label="regX", features=lobo.GEOM)
        audit = json.loads((out / "leakage_audit.json").read_text())
        assert audit["embedding_leakage_risk"] is True
        assert audit["all_folds_row_disjoint"] is True  # rows still disjoint
        for fname in ["fold_assignments.csv", "predictions.csv",
                      "metrics_per_basin.csv", "metrics_summary.csv", "leakage_audit.md"]:
            assert (out / fname).exists()

    def test_geom_only_not_flagged(self, tmp_path):
        df = _synthetic()
        res = lobo.run_lobo(df, lobo.GEOM, lobo.make_xgb_factory(lobo.GEOM),
                            embedding_provenance=lobo.PROV_NO_CNN)
        out = lobo.save_lobo_result(res, tmp_path, dataset_label="regX", features=lobo.GEOM)
        audit = json.loads((out / "leakage_audit.json").read_text())
        assert audit["embedding_leakage_risk"] is False
