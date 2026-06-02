"""Tests for channel_heads.inference.regime — regime embedding-attach glue.

The CNN forward pass (``extract_regime_embeddings``) needs a trained model, so
it is monkeypatched here; these tests pin the *merge / drop / column-assignment*
logic of ``attach_regime_embeddings`` that was extracted from
``scripts/run_mars_combined_regime.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import channel_heads.inference.regime as regime


def _write_patch_index(tmp_path):
    idx = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3"],
            "patch_status": ["ok", "ok", "fail"],  # p3 excluded (not "ok")
            "patch_path": ["a.npy", "b.npy", "c.npy"],
        }
    )
    p = tmp_path / "patch_index.parquet"
    idx.to_parquet(p)
    return p


def test_attach_merges_drops_and_overrides_embeddings(tmp_path, monkeypatch):
    pidx = _write_patch_index(tmp_path)
    df_in = pd.DataFrame({"pair_id": ["p1", "p2", "p3"], "emb_0": [9.0, 9.0, 9.0]})

    def fake_extract(model_path, patch_paths, device, batch_size=64, embedding_dim=4):
        # deterministic per-row embedding [0, 1, ..., embedding_dim-1]
        return np.tile(np.arange(embedding_dim, dtype=float), (len(patch_paths), 1))

    monkeypatch.setattr(regime, "extract_regime_embeddings", fake_extract)

    out = regime.attach_regime_embeddings(
        df_in,
        cnn_model_path=tmp_path / "m.pt",
        patch_index_path=pidx,
        project_root=tmp_path,
        device="cpu",
        embedding_dim=4,
    )

    # p3 dropped (patch_status != "ok"); baseline emb_0 overridden
    assert set(out["pair_id"]) == {"p1", "p2"}
    assert (out["emb_0"] == 0.0).all()
    assert (out["emb_3"] == 3.0).all()
    assert "patch_path_abs" not in out.columns


def test_attach_raises_on_nonfinite_embeddings(tmp_path, monkeypatch):
    pidx = _write_patch_index(tmp_path)
    df_in = pd.DataFrame({"pair_id": ["p1", "p2"]})

    def bad_extract(model_path, patch_paths, device, batch_size=64, embedding_dim=4):
        arr = np.zeros((len(patch_paths), embedding_dim))
        arr[0, 0] = np.nan
        return arr

    monkeypatch.setattr(regime, "extract_regime_embeddings", bad_extract)

    with pytest.raises(RuntimeError, match="Non-finite"):
        regime.attach_regime_embeddings(
            df_in,
            cnn_model_path=tmp_path / "m.pt",
            patch_index_path=pidx,
            project_root=tmp_path,
            device="cpu",
            embedding_dim=4,
        )
