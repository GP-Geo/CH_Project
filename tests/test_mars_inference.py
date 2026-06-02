"""Tests for channel_heads.models.mars_inference (Phase 3B tabular inference).

Uses a fake predictor + synthetic table — no real model or GeoPackage needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channel_heads.models import mars_inference as mi
from channel_heads.models.xgboost import predict_with_threshold

FEATS = ["f1", "f2"]


class FakeModel:
    """Minimal XGBClassifier stand-in returning preset touching probabilities."""

    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._proba, self._proba])


def _synthetic_table():
    return pd.DataFrame(
        {
            "network_id": [0, 0, 1],
            "pair_id": ["a", "b", "c"],
            "f1": [0.1, np.nan, 0.3],
            "f2": [1.0, 2.0, 3.0],
        }
    )


def test_apply_operating_threshold():
    out = mi.apply_operating_threshold([0.2, 0.6, 0.5], 0.5)
    assert out.tolist() == [0, 1, 1]


def test_validate_feature_columns_ok_and_missing():
    df = _synthetic_table()
    stats = mi.validate_feature_columns(df, FEATS)
    assert stats["n_rows"] == 3 and stats["n_cols"] == 2
    assert stats["n_nan_cells"] == 1  # the single NaN in f1
    with pytest.raises(RuntimeError):
        mi.validate_feature_columns(df, ["f1", "f3_missing"])


def test_fake_predictor_then_assemble_schema():
    df = _synthetic_table()
    model = FakeModel([0.9, 0.2, 0.6])
    proba, pred = predict_with_threshold(model, df, FEATS, 0.5)
    assert pred.tolist() == [1, 0, 1]

    df_pred = mi.assemble_predictions(df, proba, pred, 0.5, FEATS, "models/x.json")
    # Prediction block present and at the end, in canonical order.
    assert list(df_pred.columns[-len(mi._PRED_COLS):]) == mi._PRED_COLS
    assert df_pred["xgb_decision_threshold"].unique().tolist() == [0.5]
    assert df_pred["model_feature_set"].unique().tolist() == [mi.MODEL_FEATURE_SET]
    # NaN bookkeeping
    assert df_pred["n_nan_features"].tolist() == [0, 1, 0]
    assert df_pred["inference_status"].tolist() == ["ok", "ok_with_nan_features", "ok"]


def test_summarize_predictions_counts():
    df = _synthetic_table()
    model = FakeModel([0.9, 0.2, 0.6])
    proba, pred = predict_with_threshold(model, df, FEATS, 0.5)
    df_pred = mi.assemble_predictions(df, proba, pred, 0.5, FEATS, "x")
    summary = mi.summarize_predictions(df_pred, 0.5, "x").set_index("metric")["value"]
    assert summary["n_pairs_total"] == 3
    assert summary["n_predicted_touching"] == 2
    assert summary["predicted_touching_fraction"] == pytest.approx(2 / 3)
    assert summary["n_high_confidence_touching_prob_ge_0.80"] == 1  # only 0.9


def test_summarize_by_network():
    df = _synthetic_table()
    model = FakeModel([0.9, 0.2, 0.6])
    proba, pred = predict_with_threshold(model, df, FEATS, 0.5)
    df_pred = mi.assemble_predictions(df, proba, pred, 0.5, FEATS, "x")
    by_net = mi.summarize_by_network(df_pred).set_index("network_id")
    assert by_net.loc[0, "n_pairs"] == 2 and by_net.loc[0, "n_touching"] == 1
    assert by_net.loc[1, "n_pairs"] == 1 and by_net.loc[1, "touching_fraction"] == 1.0
