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
