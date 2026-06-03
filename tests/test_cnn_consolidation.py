"""CNN consolidation tests: architecture, training, and embedding canonical homes.

Pins behavior of channel_heads.models.cnn, channel_heads.training.cnn,
and channel_heads.models.cnn_features after the package-first refactor.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import channel_heads.models.cnn as canonical
from channel_heads.rasterization import NUM_CLASSES

# State-dict keys the trained artifact ``models/cnn_outlet_final.pt`` is keyed
# to. Pinned here so any accidental architecture drift is caught immediately.
EXPECTED_STATE_DICT_KEYS = {
    "conv1.weight",
    "conv1.bias",
    "bn1.weight",
    "bn1.bias",
    "bn1.running_mean",
    "bn1.running_var",
    "bn1.num_batches_tracked",
    "conv2.weight",
    "conv2.bias",
    "bn2.weight",
    "bn2.bias",
    "bn2.running_mean",
    "bn2.running_var",
    "bn2.num_batches_tracked",
    "conv3.weight",
    "conv3.bias",
    "bn3.weight",
    "bn3.bias",
    "bn3.running_mean",
    "bn3.running_var",
    "bn3.num_batches_tracked",
    "fc_embed.weight",
    "fc_embed.bias",
    "fc_head.weight",
    "fc_head.bias",
}


class TestCanonicalLocation:
    """Architecture lives in channel_heads.models.cnn."""

    def test_outletcnn_module_is_models_cnn(self):
        assert canonical.OutletCNN.__module__ == "channel_heads.models.cnn"

    def test_constants(self):
        assert canonical.DEFAULT_EMBEDDING_DIM == 4
        assert canonical.DEFAULT_TARGET_SIZE == 128


class TestArchitecturePreserved:
    """Constructor defaults, forward shapes, and state-dict keys are unchanged."""

    def test_instantiate(self):
        model = canonical.OutletCNN()
        assert isinstance(model, torch.nn.Module)

    def test_state_dict_keys_unchanged(self):
        model = canonical.OutletCNN()
        assert set(model.state_dict().keys()) == EXPECTED_STATE_DICT_KEYS

    def test_default_embedding_dim_constructor(self):
        model = canonical.OutletCNN()
        assert model.fc_embed.out_features == 4
        assert model.fc_head.in_features == 4
        assert model.conv1.in_channels == NUM_CLASSES

    def test_forward_shape(self):
        model = canonical.OutletCNN()
        model.eval()
        x = torch.randn(3, NUM_CLASSES, 128, 128)
        with torch.no_grad():
            out = model(x)
            emb = model.embed(x)
        assert out.shape == (3, 1)
        assert emb.shape == (3, 4)

    def test_strict_load_roundtrip(self):
        """state_dict from one instance loads strict into another (artifact path)."""
        src = canonical.OutletCNN()
        dst = canonical.OutletCNN()
        missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
        assert not missing and not unexpected


class TestDatasetPreserved:
    """OutletPairDataset behavior preserved through the canonical home."""

    def test_getitem_shapes_and_dtypes(self, tmp_path):
        raster = np.zeros((64, 64), dtype=np.uint8)
        np.save(tmp_path / "raster.npy", raster)
        ds = canonical.OutletPairDataset([tmp_path / "raster.npy"], [1])
        image, label = ds[0]
        assert image.shape == (NUM_CLASSES, 64, 64)
        assert image.dtype == torch.float32
        assert label.dtype == torch.float32
        assert len(ds) == 1

    def test_length_mismatch_raises(self, tmp_path):
        raster = np.zeros((8, 8), dtype=np.uint8)
        np.save(tmp_path / "raster.npy", raster)
        with pytest.raises(ValueError, match="Length mismatch"):
            canonical.OutletPairDataset([tmp_path / "raster.npy"], [0, 1])


class TestLazyTrainingReexport:
    """models.cnn re-exports the training core lazily without an import cycle."""

    def test_train_symbols_available_from_canonical(self):
        from channel_heads.models.cnn import pick_device, train_cnn

        assert callable(train_cnn)
        assert callable(pick_device)

    def test_training_defaults_match_training_module(self):
        from channel_heads.training import cnn as training_cnn

        assert canonical.DEFAULT_EPOCHS == training_cnn.DEFAULT_EPOCHS
        assert canonical.HOLDOUT_BASIN == training_cnn.HOLDOUT_BASIN

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = canonical.does_not_exist


class TestPickDeviceDeduplicated:
    """pick_device is canonical in models.device; training.cnn re-exports it."""

    def test_training_cnn_pick_device_is_canonical(self):
        from channel_heads.training.cnn import pick_device as training_pick
        from channel_heads.models.device import pick_device as canonical_pick

        assert training_pick is canonical_pick

    def test_models_cnn_lazy_pick_device_is_canonical(self):
        from channel_heads.models.cnn import pick_device as cnn_pick
        from channel_heads.models.device import pick_device as canonical_pick

        assert cnn_pick is canonical_pick


class TestCNNFeaturesConsolidated:
    """Earth/generic embedding helpers: canonical home in models.cnn_features."""

    def test_canonical_function_module(self):
        import channel_heads.models.cnn_features as canonical_feat

        assert canonical_feat.extract_embeddings.__module__ == "channel_heads.models.cnn_features"

    def test_cnn_feature_cols_unchanged(self):
        from channel_heads.models.cnn_features import CNN_FEATURE_COLS

        assert CNN_FEATURE_COLS == ["emb_0", "emb_1", "emb_2", "emb_3"]
        assert CNN_FEATURE_COLS == [f"emb_{i}" for i in range(4)]

    def test_extract_embeddings_new_path_smoke(self, tmp_path):
        """Lenient default state-dict load + embedding extraction on a fixture."""
        import numpy as np
        import pandas as pd

        from channel_heads.models.cnn import OutletCNN
        from channel_heads.models.cnn_features import CNN_FEATURE_COLS, extract_embeddings

        # Save a state dict from the same architecture (loads via default
        # load_state_dict, exactly as the historical extract_embeddings did).
        model_path = tmp_path / "model.pt"
        torch.save(OutletCNN(embedding_dim=4).state_dict(), model_path)

        raster_dir = tmp_path / "rasters"
        raster_dir.mkdir()
        n = 5
        for i in range(n):
            raster = np.random.default_rng(i).integers(0, NUM_CLASSES, size=(64, 64), dtype=np.uint8)
            np.save(raster_dir / f"r_{i}.npy", raster)

        manifest = pd.DataFrame(
            {
                "outlet": [1] * n,
                "confluence": list(range(n)),
                "head_1": list(range(10, 10 + n)),
                "head_2": list(range(20, 20 + n)),
                "raster_path": [f"r_{i}.npy" for i in range(n)],
            }
        )

        result = extract_embeddings(model_path, raster_dir, manifest)
        assert len(result) == n
        for col in CNN_FEATURE_COLS:
            assert col in result.columns
            assert np.all(np.isfinite(result[col].to_numpy()))
            assert (result[col] >= 0).all()  # ReLU embedding


class TestTrainingCoreConsolidated:
    """CNN training core: canonical home in training.cnn."""

    def test_train_cnn_module_is_training_cnn(self):
        from channel_heads.training.cnn import train_cnn

        assert train_cnn.__module__ == "channel_heads.training.cnn"

    def test_defaults_unchanged(self):
        from channel_heads.training import cnn as cnn_training

        assert cnn_training.DEFAULT_EPOCHS == 60
        assert cnn_training.DEFAULT_LR == 1e-3
        assert cnn_training.DEFAULT_WEIGHT_DECAY == 1e-4
        assert cnn_training.DEFAULT_BATCH_SIZE == 64
        assert cnn_training.DEFAULT_DROPOUT == 0.3
        assert cnn_training.DEFAULT_PATIENCE == 12

    def test_holdout_basin_unchanged(self):
        from channel_heads.training import cnn as cnn_training

        assert cnn_training.HOLDOUT_BASIN == "taiwan"

    def test_random_state_unchanged(self):
        from channel_heads.training import cnn as cnn_training

        assert cnn_training.RANDOM_STATE == 42

    def test_models_cnn_lazy_reexport_sources_from_training_cnn(self):
        """models.cnn lazy re-export resolves to training.cnn (no import cycle)."""
        import channel_heads.models.cnn as models_cnn
        from channel_heads.training.cnn import train_cnn as canonical_train

        assert models_cnn.train_cnn is canonical_train
        assert models_cnn.HOLDOUT_BASIN == "taiwan"
