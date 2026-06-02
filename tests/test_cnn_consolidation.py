"""Slice 3a consolidation tests: CNN architecture canonical home + shim.

Proves that moving the OutletCNN architecture/dataset into
``channel_heads.models.cnn`` preserved behavior exactly and that the historical
``channel_heads.cnn_model`` import path still resolves to the *same* objects.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import channel_heads.cnn_model as legacy
import channel_heads.models.cnn as canonical
from channel_heads.rasterizer import NUM_CLASSES

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


class TestImportPathsIdentical:
    """Old and new import paths must resolve to the same implementation."""

    def test_outletcnn_is_same_object(self):
        assert legacy.OutletCNN is canonical.OutletCNN

    def test_outletpairdataset_is_same_object(self):
        assert legacy.OutletPairDataset is canonical.OutletPairDataset

    def test_encode_raster_onehot_is_same_object(self):
        assert legacy.encode_raster_onehot is canonical.encode_raster_onehot

    def test_constants_match(self):
        assert legacy.DEFAULT_EMBEDDING_DIM == canonical.DEFAULT_EMBEDDING_DIM == 4
        assert legacy.DEFAULT_TARGET_SIZE == canonical.DEFAULT_TARGET_SIZE == 128

    def test_from_import_resolves_same(self):
        from channel_heads.cnn_model import OutletCNN as OldOutletCNN
        from channel_heads.models.cnn import OutletCNN as NewOutletCNN

        assert OldOutletCNN is NewOutletCNN

    def test_canonical_class_module_is_models_cnn(self):
        """Real definition lives in models.cnn; cnn_model only re-exports it."""
        assert canonical.OutletCNN.__module__ == "channel_heads.models.cnn"
        assert legacy.OutletCNN.__module__ == "channel_heads.models.cnn"

    def test_shim_reexports_num_classes(self):
        assert legacy.NUM_CLASSES == NUM_CLASSES


class TestArchitecturePreserved:
    """Constructor defaults, forward shapes, and state-dict keys are unchanged."""

    def test_instantiate_from_old_path(self):
        model = legacy.OutletCNN()
        assert isinstance(model, torch.nn.Module)

    def test_instantiate_from_new_path(self):
        model = canonical.OutletCNN()
        assert isinstance(model, torch.nn.Module)

    def test_state_dict_keys_unchanged(self):
        model = canonical.OutletCNN()
        assert set(model.state_dict().keys()) == EXPECTED_STATE_DICT_KEYS

    def test_state_dict_keys_same_from_both_paths(self):
        assert set(legacy.OutletCNN().state_dict().keys()) == set(
            canonical.OutletCNN().state_dict().keys()
        )

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
        dst = legacy.OutletCNN()
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
        from channel_heads import cnn_training

        assert canonical.DEFAULT_EPOCHS == cnn_training.DEFAULT_EPOCHS
        assert canonical.HOLDOUT_BASIN == cnn_training.HOLDOUT_BASIN

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = canonical.does_not_exist


class TestPickDeviceDeduplicated:
    """cnn_training.pick_device is the canonical models.device.pick_device."""

    def test_cnn_training_pick_device_is_canonical(self):
        from channel_heads.cnn_training import pick_device as training_pick
        from channel_heads.models.device import pick_device as canonical_pick

        assert training_pick is canonical_pick

    def test_models_cnn_lazy_pick_device_is_canonical(self):
        from channel_heads.models.cnn import pick_device as cnn_pick
        from channel_heads.models.device import pick_device as canonical_pick

        assert cnn_pick is canonical_pick


class TestCNNFeaturesConsolidated:
    """Earth/generic embedding helpers: canonical home + shim identity."""

    def test_old_and_new_paths_same_objects(self):
        import channel_heads.cnn_features as legacy_feat
        import channel_heads.models.cnn_features as canonical_feat

        assert legacy_feat.extract_embeddings is canonical_feat.extract_embeddings
        assert legacy_feat.merge_cnn_features is canonical_feat.merge_cnn_features
        assert legacy_feat.CNN_FEATURE_COLS is canonical_feat.CNN_FEATURE_COLS

    def test_from_import_resolves_same(self):
        from channel_heads.cnn_features import extract_embeddings as old_extract
        from channel_heads.models.cnn_features import extract_embeddings as new_extract

        assert old_extract is new_extract

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
    """CNN training core: canonical home in training.cnn + shim identity."""

    def test_old_and_new_train_cnn_same_object(self):
        from channel_heads.cnn_training import train_cnn as old_train
        from channel_heads.training.cnn import train_cnn as new_train

        assert old_train is new_train

    def test_train_cnn_module_is_training_cnn(self):
        from channel_heads.training.cnn import train_cnn

        assert train_cnn.__module__ == "channel_heads.training.cnn"

    def test_defaults_unchanged_and_identical_across_paths(self):
        from channel_heads import cnn_training as shim
        from channel_heads.training import cnn as canonical

        assert canonical.DEFAULT_EPOCHS == shim.DEFAULT_EPOCHS == 60
        assert canonical.DEFAULT_LR == shim.DEFAULT_LR == 1e-3
        assert canonical.DEFAULT_WEIGHT_DECAY == shim.DEFAULT_WEIGHT_DECAY == 1e-4
        assert canonical.DEFAULT_BATCH_SIZE == shim.DEFAULT_BATCH_SIZE == 64
        assert canonical.DEFAULT_DROPOUT == shim.DEFAULT_DROPOUT == 0.3
        assert canonical.DEFAULT_PATIENCE == shim.DEFAULT_PATIENCE == 12

    def test_holdout_basin_unchanged(self):
        from channel_heads import cnn_training as shim
        from channel_heads.training import cnn as canonical

        assert canonical.HOLDOUT_BASIN == shim.HOLDOUT_BASIN == "taiwan"

    def test_random_state_unchanged(self):
        from channel_heads import cnn_training as shim
        from channel_heads.training import cnn as canonical

        assert canonical.RANDOM_STATE == shim.RANDOM_STATE == 42

    def test_models_cnn_lazy_reexport_sources_from_training_cnn(self):
        """models.cnn lazy re-export resolves to training.cnn (no import cycle)."""
        import channel_heads.models.cnn as models_cnn
        from channel_heads.training.cnn import train_cnn as canonical_train

        assert models_cnn.train_cnn is canonical_train
        assert models_cnn.HOLDOUT_BASIN == "taiwan"
