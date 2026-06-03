"""Tests for the regime embedding-attach glue.

The canonical implementation lives in :mod:`channel_heads.models.regime`.
The CNN forward pass (``extract_regime_embeddings``) needs a trained model,
so for the merge/drop/column-assignment tests it is monkeypatched on the
canonical module; a separate torch-guarded test exercises the real strict
state-dict load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import channel_heads.models.regime as regime


def _write_patch_index(tmp_path):
    pytest.importorskip("pyarrow")

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


class TestRegimeCanonicalSurface:
    """The canonical ``models.regime`` and ``models`` package expose the symbols."""

    def test_models_package_exposes_canonical(self):
        import channel_heads.models as models
        import channel_heads.models.regime as canonical

        # importorskip torch indirectly: models.regime requires torch
        assert models.attach_regime_embeddings is canonical.attach_regime_embeddings
        assert models.extract_regime_embeddings is canonical.extract_regime_embeddings


class TestStrictStateDictLoad:
    """Pin the strict ``load_state_dict(strict=True)`` forward-pass behavior."""

    def _write_rasters(self, tmp_path, n=2):
        paths = []
        rng = np.random.default_rng(0)
        for i in range(n):
            raster = rng.integers(0, 5, size=(64, 64), dtype=np.uint8)
            p = tmp_path / f"patch_{i}.npy"
            np.save(p, raster)
            paths.append(p)
        return paths

    def test_strict_load_succeeds_and_returns_finite_matrix(self, tmp_path):
        torch = pytest.importorskip("torch")
        from channel_heads.models.cnn import OutletCNN

        model = OutletCNN(embedding_dim=4)
        model_path = tmp_path / "regime_cnn.pt"
        torch.save(model.state_dict(), model_path)

        patch_paths = self._write_rasters(tmp_path, n=3)
        emb = regime.extract_regime_embeddings(
            model_path, patch_paths, device="cpu", embedding_dim=4
        )

        assert emb.shape == (3, 4)
        assert np.isfinite(emb).all()

    def test_strict_load_rejects_mismatched_state_dict(self, tmp_path):
        torch = pytest.importorskip("torch")
        from channel_heads.models.cnn import OutletCNN

        # A state dict for a different embedding head must not load under strict=True.
        mismatched = OutletCNN(embedding_dim=8)
        model_path = tmp_path / "mismatched_cnn.pt"
        torch.save(mismatched.state_dict(), model_path)

        patch_paths = self._write_rasters(tmp_path, n=1)
        with pytest.raises(RuntimeError):
            regime.extract_regime_embeddings(
                model_path, patch_paths, device="cpu", embedding_dim=4
            )
