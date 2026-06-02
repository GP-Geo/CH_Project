"""Tests for channel_heads.models.embeddings (Phase 5 Mars CNN embeddings)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channel_heads.models import embeddings as emb


def _patch_index(tmp_path):
    ok_a = tmp_path / "a.npy"
    ok_b = tmp_path / "b.npy"
    np.save(ok_a, np.zeros((128, 128), dtype=np.uint8))
    np.save(ok_b, np.ones((128, 128), dtype=np.uint8))
    return pd.DataFrame(
        {
            "network_id": [10, 10, 11],
            "pair_id": ["a", "b", "c"],
            "patch_path": [str(ok_a), str(ok_b), str(tmp_path / "c.npy")],
            "patch_status": ["ok", "ok", "failed"],
            "patch_qa_reason": ["", "", "missing branch"],
        }
    )


def _extracted_embeddings(tmp_path):
    return pd.DataFrame(
        {
            "network_id": [10, 10],
            "pair_id": ["a", "b"],
            "patch_path": [str(tmp_path / "a.npy"), str(tmp_path / "b.npy")],
            "emb_0": [0.1, 0.2],
            "emb_1": [1.1, 1.2],
            "emb_2": [2.1, 2.2],
            "emb_3": [3.1, 3.2],
        }
    )


def test_prepare_manifest_and_patch_sample_validation(tmp_path):
    patch_index = _patch_index(tmp_path)
    manifest = emb.prepare_embedding_manifest(patch_index)

    assert manifest["pair_id"].tolist() == ["a", "b"]
    assert manifest["raster_path"].map(lambda p: p.endswith(".npy")).all()
    emb.validate_patch_sample(manifest)


def test_assemble_embedding_table_schema_and_skipped_rows(tmp_path):
    df_emb = emb.assemble_embedding_table(
        _patch_index(tmp_path), _extracted_embeddings(tmp_path)
    )

    assert list(df_emb.columns) == [
        "network_id",
        "pair_id",
        "patch_path",
        "emb_0",
        "emb_1",
        "emb_2",
        "emb_3",
        "embedding_status",
        "embedding_qa_reason",
    ]
    assert len(df_emb) == 3
    assert df_emb["embedding_status"].tolist() == ["ok", "ok", "skipped"]
    skipped = df_emb[df_emb["pair_id"] == "c"].iloc[0]
    assert np.isnan(skipped["emb_0"])
    assert skipped["embedding_qa_reason"] == "patch_status=failed;missing branch"


def test_validate_embedding_table_rejects_nan_or_inf_on_ok_rows(tmp_path):
    df_emb = emb.assemble_embedding_table(
        _patch_index(tmp_path), _extracted_embeddings(tmp_path)
    )

    # Skipped rows may carry NaN embeddings; ok rows must be finite.
    emb.validate_embedding_table(df_emb, expected_n=3)

    bad = df_emb.copy()
    bad.loc[bad["pair_id"] == "a", "emb_0"] = np.inf
    with pytest.raises(RuntimeError, match="Inf"):
        emb.validate_embedding_table(bad, expected_n=3)

    bad = df_emb.copy()
    bad.loc[bad["pair_id"] == "b", "emb_1"] = np.nan
    with pytest.raises(RuntimeError, match="NaN"):
        emb.validate_embedding_table(bad, expected_n=3)


def test_merge_embeddings_with_tabular_features_preserves_rows(tmp_path):
    df_emb = emb.assemble_embedding_table(
        _patch_index(tmp_path), _extracted_embeddings(tmp_path)
    )
    tabular = pd.DataFrame(
        {
            "pair_id": ["a", "b", "missing"],
            "network_id": [10, 10, 12],
            "f1": [1.0, 2.0, 3.0],
        }
    )

    merged = emb.merge_embeddings_with_tabular_features(tabular, df_emb)

    assert len(merged) == len(tabular)
    assert merged.loc[merged["pair_id"] == "a", "emb_0"].iloc[0] == 0.1
    assert merged.loc[merged["pair_id"] == "b", "embedding_status"].iloc[0] == "ok"
    missing = merged[merged["pair_id"] == "missing"].iloc[0]
    assert pd.isna(missing["embedding_status"])
    assert pd.isna(missing["emb_3"])
