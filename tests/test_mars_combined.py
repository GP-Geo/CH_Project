"""Tests for channel_heads.models.mars_combined (Phase 6C Mars inference)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channel_heads.models import mars_combined as mc


class _FakeBooster:
    def __init__(self, feature_names):
        self.feature_names = feature_names


class FakeModel:
    """Minimal XGBClassifier stand-in returning preset touching probabilities."""

    def __init__(self, proba, feature_names=None):
        self._proba = np.asarray(proba, dtype=float)
        self._feature_names = feature_names

    def get_booster(self):
        return _FakeBooster(self._feature_names)

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._proba, self._proba])


def _features():
    return pd.DataFrame(
        {
            "network_id": [1, 1, 2, 2],
            "pair_id": ["a", "b", "c", "d"],
            "geom": [0.1, np.nan, 0.3, 0.4],
            "emb_0": [1.0, 1.1, 1.2, 1.3],
            "cnn_logit": [0.2, 0.4, 0.6, 0.8],
        }
    )


def _tabular_predictions():
    return pd.DataFrame(
        {
            "pair_id": ["a", "b", "c", "d"],
            "prob_touching_tabular_only": [0.8, 0.3, 0.6, 0.1],
            "pred_touching_tabular_only": [1, 0, 1, 0],
            "threshold_tabular_only": [0.577406] * 4,
        }
    )


def _combined_output():
    df = _features()
    emb_result = mc.run_model_variant_inference(
        df,
        name="emb",
        model=FakeModel([0.9, 0.3, 0.81, 0.2], feature_names=["geom", "emb_0"]),
        feature_cols=["geom", "emb_0"],
        threshold=0.5,
        model_path="models/emb.json",
    )
    logit_result = mc.run_model_variant_inference(
        df,
        name="logit",
        model=FakeModel([0.95, 0.6, 0.1, 0.2], feature_names=["geom", "cnn_logit"]),
        feature_cols=["geom", "cnn_logit"],
        threshold=0.4,
        model_path="models/logit.json",
    )
    out = mc.append_variant_predictions(df, df, emb_result)
    out = mc.append_variant_predictions(out, df, logit_result)
    out = out.merge(_tabular_predictions(), on="pair_id", how="left")
    return mc.add_combined_derived_columns(out)


def test_variant_feature_validation_allows_nan_but_rejects_missing_and_inf():
    df = _features()
    stats = mc.verify_model_variant_inputs(
        df,
        FakeModel([0.1, 0.2, 0.3, 0.4], feature_names=["geom", "emb_0"]),
        ["geom", "emb_0"],
        "emb",
    )
    assert stats["n_rows"] == 4
    assert stats["n_nan_cells"] == 1

    with pytest.raises(RuntimeError, match="missing model feature columns"):
        mc.verify_model_variant_inputs(
            df,
            FakeModel([0.1, 0.2, 0.3, 0.4]),
            ["geom", "missing"],
            "bad",
        )

    bad = df.copy()
    bad.loc[0, "geom"] = np.inf
    with pytest.raises(RuntimeError, match="inf values"):
        mc.verify_model_variant_inputs(
            bad,
            FakeModel([0.1, 0.2, 0.3, 0.4]),
            ["geom", "emb_0"],
            "bad",
        )


def test_run_model_variant_inference_applies_variant_threshold():
    df = _features()
    result = mc.run_model_variant_inference(
        df,
        name="emb",
        model=FakeModel([0.2, 0.7, 0.55, 0.6], feature_names=["geom", "emb_0"]),
        feature_cols=["geom", "emb_0"],
        threshold=0.6,
        model_path="models/emb.json",
    )

    assert result["threshold"] == 0.6
    assert result["pred"].tolist() == [0, 1, 0, 1]
    assert result["proba"].tolist() == [0.2, 0.7, 0.55, 0.6]


def test_append_variant_predictions_schema_and_nan_status():
    df = _features()
    result = mc.run_model_variant_inference(
        df,
        name="emb",
        model=FakeModel([0.9, 0.3, 0.81, 0.2], feature_names=["geom", "emb_0"]),
        feature_cols=["geom", "emb_0"],
        threshold=0.5,
        model_path="models/emb.json",
    )

    out = mc.append_variant_predictions(df, df, result)

    for col in [
        "prob_touching_emb",
        "pred_touching_emb",
        "threshold_emb",
        "model_path_emb",
        "inference_status_emb",
    ]:
        assert col in out.columns
    assert out["pred_touching_emb"].tolist() == [1, 0, 1, 0]
    assert out["inference_status_emb"].tolist() == [
        "ok",
        "ok_with_nan_features",
        "ok",
        "ok",
    ]


def test_agreement_disagreement_and_high_confidence_flags():
    out = _combined_output()

    assert out["agreement_touching"].tolist() == [1, 0, 0, 0]
    assert out["emb_logit_disagreement"].tolist() == [0, 1, 1, 0]
    assert out["max_combined_prob"].tolist() == [0.95, 0.6, 0.81, 0.2]
    assert out["high_confidence_any_combined"].tolist() == [1, 0, 1, 0]


def test_comparison_summary_and_by_network_outputs():
    out = _combined_output()
    summary = mc.build_comparison_summary(out)
    counts = summary[summary["metric_kind"] == "per_model_counts"].set_index(
        "model_variant"
    )
    agreements = summary[summary["metric_kind"] == "agreements"].set_index(
        "model_variant"
    )

    assert counts.loc["tabular_only", "n_touching"] == 2
    assert counts.loc["geom_plus_cnn_emb", "n_touching"] == 2
    assert counts.loc["geom_plus_cnn_logit", "n_touching"] == 2
    assert counts.loc["geom_plus_cnn_emb", "n_high_confidence_prob_ge_0.80"] == 2
    assert agreements.loc["agreement_emb_vs_logit", "n_changes"] == 2
    assert agreements.loc["agreement_emb_vs_logit", "agreement_fraction"] == 0.5

    by_network = mc.build_by_network(out).set_index("network_id")
    assert by_network.loc[1, "n_pairs_scored"] == 2
    assert by_network.loc[1, "n_disagreements_between_combined_models"] == 1
    assert by_network.loc[2, "pct_touching_geom_plus_cnn_emb"] == 0.5


def test_attach_cnn_logit_uses_existing_patches_without_real_cnn(tmp_path, monkeypatch):
    patch_index = pd.DataFrame(
        {
            "pair_id": ["a", "b", "c"],
            "patch_path": ["a.npy", "b.npy", "c.npy"],
            "patch_status": ["ok", "failed", "ok"],
        }
    )
    df = pd.DataFrame({"pair_id": ["a", "b", "c"]})

    def fake_extract_logits(patch_paths, **kwargs):
        assert [p.name for p in patch_paths] == ["a.npy", "c.npy"]
        return np.array([1.2, -0.4])

    monkeypatch.setattr(pd, "read_parquet", lambda path: patch_index.copy())
    monkeypatch.setattr(mc, "extract_logits", fake_extract_logits)
    out = mc.attach_cnn_logit(
        df,
        patch_index_parquet=tmp_path / "patch_index.parquet",
        model_path=tmp_path / "cnn.pt",
        device="cpu",
        project_root=tmp_path,
    )

    assert out["cnn_logit"].tolist()[0] == 1.2
    assert pd.isna(out["cnn_logit"].tolist()[1])
    assert out["cnn_logit"].tolist()[2] == -0.4
