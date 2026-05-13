"""Tests for channel_heads.cnn_model module."""

import numpy as np
import pytest
import torch

from channel_heads.cnn_model import (
    DEFAULT_EMBEDDING_DIM,
    OutletCNN,
    OutletPairDataset,
    encode_raster_onehot,
)
from channel_heads.rasterizer import NUM_CLASSES

# =============================================================================
# One-Hot Encoding Tests
# =============================================================================


class TestEncodeRasterOnehot:
    """Tests for encode_raster_onehot."""

    def test_output_shape(self):
        """Output has shape (NUM_CLASSES, H, W)."""
        raster = np.zeros((64, 64), dtype=np.uint8)
        result = encode_raster_onehot(raster)
        assert result.shape == (NUM_CLASSES, 64, 64)

    def test_output_dtype(self):
        """Output is float32."""
        raster = np.zeros((32, 32), dtype=np.uint8)
        result = encode_raster_onehot(raster)
        assert result.dtype == np.float32

    def test_all_background(self):
        """All-zero raster: only channel 0 is active."""
        raster = np.zeros((16, 16), dtype=np.uint8)
        result = encode_raster_onehot(raster)
        assert result[0].sum() == 16 * 16
        for c in range(1, NUM_CLASSES):
            assert result[c].sum() == 0

    def test_single_class_per_pixel(self):
        """Each pixel is one-hot: exactly one channel is 1."""
        rng = np.random.default_rng(42)
        raster = rng.integers(0, NUM_CLASSES, size=(32, 32), dtype=np.uint8)
        result = encode_raster_onehot(raster)
        # Sum across channels should be 1 everywhere
        channel_sum = result.sum(axis=0)
        np.testing.assert_array_equal(channel_sum, np.ones((32, 32)))

    def test_correct_channel_assignment(self):
        """Values are placed in the correct channel."""
        raster = np.array([[0, 1], [2, 3], [4, 0]], dtype=np.uint8)
        result = encode_raster_onehot(raster)
        # Check specific pixels
        assert result[0, 0, 0] == 1.0  # pixel (0,0) = 0 → channel 0
        assert result[1, 0, 1] == 1.0  # pixel (0,1) = 1 → channel 1
        assert result[2, 1, 0] == 1.0  # pixel (1,0) = 2 → channel 2
        assert result[3, 1, 1] == 1.0  # pixel (1,1) = 3 → channel 3
        assert result[4, 2, 0] == 1.0  # pixel (2,0) = 4 → channel 4
        assert result[0, 2, 1] == 1.0  # pixel (2,1) = 0 → channel 0


# =============================================================================
# OutletCNN Tests
# =============================================================================


class TestOutletCNN:
    """Tests for OutletCNN architecture."""

    def test_forward_output_shape(self):
        """Forward pass returns (B, 1) logits."""
        model = OutletCNN()
        x = torch.randn(4, NUM_CLASSES, 128, 128)
        out = model(x)
        assert out.shape == (4, 1)

    def test_embed_output_shape(self):
        """Embed returns (B, embedding_dim) vectors."""
        model = OutletCNN(embedding_dim=4)
        x = torch.randn(4, NUM_CLASSES, 128, 128)
        emb = model.embed(x)
        assert emb.shape == (4, 4)

    def test_custom_embedding_dim(self):
        """Custom embedding dimension is respected."""
        model = OutletCNN(embedding_dim=8)
        x = torch.randn(2, NUM_CLASSES, 128, 128)
        emb = model.embed(x)
        assert emb.shape == (2, 8)

    def test_single_sample(self):
        """Works with batch size 1."""
        model = OutletCNN()
        x = torch.randn(1, NUM_CLASSES, 128, 128)
        out = model(x)
        assert out.shape == (1, 1)
        emb = model.embed(x)
        assert emb.shape == (1, DEFAULT_EMBEDDING_DIM)

    def test_different_input_sizes(self):
        """AdaptiveAvgPool handles non-128 input sizes."""
        model = OutletCNN()
        for size in [32, 64, 96, 128, 256]:
            x = torch.randn(1, NUM_CLASSES, size, size)
            out = model(x)
            assert out.shape == (1, 1)

    def test_gradient_flow(self):
        """Gradients flow through all layers."""
        model = OutletCNN()
        x = torch.randn(2, NUM_CLASSES, 128, 128, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check gradients exist for all conv and linear layers
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_embed_no_dropout(self):
        """Embed method produces consistent output in eval mode."""
        model = OutletCNN(dropout=0.5)
        model.eval()
        x = torch.randn(2, NUM_CLASSES, 64, 64)
        emb1 = model.embed(x)
        emb2 = model.embed(x)
        torch.testing.assert_close(emb1, emb2)

    def test_embedding_values_nonnegative(self):
        """Embedding values are non-negative (ReLU activation)."""
        model = OutletCNN()
        model.eval()
        x = torch.randn(4, NUM_CLASSES, 64, 64)
        with torch.no_grad():
            emb = model.embed(x)
        assert (emb >= 0).all()


# =============================================================================
# OutletPairDataset Tests
# =============================================================================


class TestOutletPairDataset:
    """Tests for OutletPairDataset."""

    def test_len(self, tmp_path):
        """Dataset length matches number of rasters."""
        # Create dummy rasters
        for i in range(5):
            raster = np.zeros((64, 64), dtype=np.uint8)
            np.save(tmp_path / f"raster_{i}.npy", raster)

        paths = [tmp_path / f"raster_{i}.npy" for i in range(5)]
        labels = [0, 1, 1, 0, 1]
        ds = OutletPairDataset(paths, labels)
        assert len(ds) == 5

    def test_getitem_shapes(self, tmp_path):
        """__getitem__ returns correct tensor shapes."""
        raster = np.zeros((64, 64), dtype=np.uint8)
        np.save(tmp_path / "raster.npy", raster)

        ds = OutletPairDataset([tmp_path / "raster.npy"], [1])
        image, label = ds[0]
        assert image.shape == (NUM_CLASSES, 64, 64)
        assert label.shape == ()

    def test_getitem_dtypes(self, tmp_path):
        """Tensors have correct dtypes."""
        raster = np.zeros((64, 64), dtype=np.uint8)
        np.save(tmp_path / "raster.npy", raster)

        ds = OutletPairDataset([tmp_path / "raster.npy"], [0])
        image, label = ds[0]
        assert image.dtype == torch.float32
        assert label.dtype == torch.float32

    def test_length_mismatch_raises(self, tmp_path):
        """Mismatched paths and labels raises ValueError."""
        raster = np.zeros((64, 64), dtype=np.uint8)
        np.save(tmp_path / "raster.npy", raster)

        with pytest.raises(ValueError, match="Length mismatch"):
            OutletPairDataset([tmp_path / "raster.npy"], [0, 1])

    def test_augmentation_flips(self, tmp_path):
        """Augmentation produces different outputs across calls."""
        # Create an asymmetric raster to detect flips
        raster = np.zeros((64, 64), dtype=np.uint8)
        raster[0, 0] = 1  # top-left only
        np.save(tmp_path / "raster.npy", raster)

        ds = OutletPairDataset([tmp_path / "raster.npy"], [1], augment=True)

        # Run many times — at least one should differ from the original
        np.random.seed(42)
        seen_different = False
        for _ in range(20):
            image, _ = ds[0]
            # Check if top-left pixel of channel 1 is still active
            if image[1, 0, 0] != 1.0:
                seen_different = True
                break
        assert seen_different, "Augmentation never produced a flip"
