"""Tests for channel_heads.cnn_features module."""

import numpy as np
import pandas as pd
import pytest
import torch

from channel_heads.cnn_features import (
    CNN_FEATURE_COLS,
    extract_embeddings,
    merge_cnn_features,
)
from channel_heads.cnn_model import DEFAULT_EMBEDDING_DIM, OutletCNN
from channel_heads.rasterizer import NUM_CLASSES

# =============================================================================
# CNN_FEATURE_COLS Tests
# =============================================================================


class TestCNNFeatureCols:
    """Tests for the CNN_FEATURE_COLS constant."""

    def test_length(self):
        """CNN_FEATURE_COLS has correct length."""
        assert len(CNN_FEATURE_COLS) == DEFAULT_EMBEDDING_DIM

    def test_naming(self):
        """Column names follow emb_N pattern."""
        for i, col in enumerate(CNN_FEATURE_COLS):
            assert col == f"emb_{i}"


# =============================================================================
# extract_embeddings Tests
# =============================================================================


class TestExtractEmbeddings:
    """Tests for extract_embeddings."""

    @pytest.fixture
    def model_and_rasters(self, tmp_path):
        """Create a saved model and dummy rasters."""
        # Save a randomly initialized model
        model = OutletCNN(embedding_dim=DEFAULT_EMBEDDING_DIM)
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Create dummy rasters
        raster_dir = tmp_path / "rasters"
        raster_dir.mkdir()
        n_samples = 6
        for i in range(n_samples):
            raster = np.random.default_rng(i).integers(
                0, NUM_CLASSES, size=(64, 64), dtype=np.uint8
            )
            np.save(raster_dir / f"raster_{i}.npy", raster)

        # Create manifest DataFrame
        manifest_df = pd.DataFrame(
            {
                "outlet": [1] * n_samples,
                "confluence": list(range(n_samples)),
                "head_1": list(range(10, 10 + n_samples)),
                "head_2": list(range(20, 20 + n_samples)),
                "raster_path": [f"raster_{i}.npy" for i in range(n_samples)],
            }
        )

        return model_path, raster_dir, manifest_df

    def test_embedding_columns_added(self, model_and_rasters):
        """Embedding columns emb_0..emb_3 are added."""
        model_path, raster_dir, manifest_df = model_and_rasters
        result = extract_embeddings(model_path, raster_dir, manifest_df)
        for col in CNN_FEATURE_COLS:
            assert col in result.columns

    def test_output_row_count(self, model_and_rasters):
        """Output has same number of rows as input."""
        model_path, raster_dir, manifest_df = model_and_rasters
        result = extract_embeddings(model_path, raster_dir, manifest_df)
        assert len(result) == len(manifest_df)

    def test_original_columns_preserved(self, model_and_rasters):
        """Original columns from manifest_df are preserved."""
        model_path, raster_dir, manifest_df = model_and_rasters
        result = extract_embeddings(model_path, raster_dir, manifest_df)
        for col in manifest_df.columns:
            assert col in result.columns

    def test_embeddings_are_finite(self, model_and_rasters):
        """Embedding values are finite (no NaN/Inf)."""
        model_path, raster_dir, manifest_df = model_and_rasters
        result = extract_embeddings(model_path, raster_dir, manifest_df)
        for col in CNN_FEATURE_COLS:
            assert np.all(np.isfinite(result[col].values))

    def test_embeddings_nonnegative(self, model_and_rasters):
        """Embeddings are non-negative (due to ReLU)."""
        model_path, raster_dir, manifest_df = model_and_rasters
        result = extract_embeddings(model_path, raster_dir, manifest_df)
        for col in CNN_FEATURE_COLS:
            assert (result[col] >= 0).all()

    def test_batch_size_variations(self, model_and_rasters):
        """Different batch sizes produce identical results."""
        model_path, raster_dir, manifest_df = model_and_rasters
        result_bs2 = extract_embeddings(model_path, raster_dir, manifest_df, batch_size=2)
        result_bs4 = extract_embeddings(model_path, raster_dir, manifest_df, batch_size=4)
        for col in CNN_FEATURE_COLS:
            np.testing.assert_allclose(result_bs2[col].values, result_bs4[col].values, atol=1e-6)


# =============================================================================
# merge_cnn_features Tests
# =============================================================================


class TestMergeCNNFeatures:
    """Tests for merge_cnn_features."""

    def test_merge_on_keys(self):
        """Merge correctly joins on (outlet, confluence, head_1, head_2)."""
        base = pd.DataFrame(
            {
                "outlet": [1, 1, 2],
                "confluence": [10, 11, 12],
                "head_1": [100, 101, 102],
                "head_2": [200, 201, 202],
                "existing_feature": [0.5, 0.6, 0.7],
            }
        )
        embedding = pd.DataFrame(
            {
                "outlet": [1, 1, 2],
                "confluence": [10, 11, 12],
                "head_1": [100, 101, 102],
                "head_2": [200, 201, 202],
                "emb_0": [0.1, 0.2, 0.3],
                "emb_1": [0.4, 0.5, 0.6],
                "emb_2": [0.7, 0.8, 0.9],
                "emb_3": [1.0, 1.1, 1.2],
            }
        )
        result = merge_cnn_features(base, embedding)
        assert len(result) == 3
        assert "existing_feature" in result.columns
        assert "emb_0" in result.columns
        assert "emb_3" in result.columns
        assert result["emb_0"].iloc[0] == 0.1

    def test_left_join_preserves_base_rows(self):
        """Unmatched base rows get NaN embeddings (left join)."""
        base = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [10, 11],
                "head_1": [100, 101],
                "head_2": [200, 201],
            }
        )
        embedding = pd.DataFrame(
            {
                "outlet": [1],
                "confluence": [10],
                "head_1": [100],
                "head_2": [200],
                "emb_0": [0.5],
            }
        )
        result = merge_cnn_features(base, embedding)
        assert len(result) == 2
        assert result["emb_0"].iloc[0] == 0.5
        assert pd.isna(result["emb_0"].iloc[1])

    def test_extra_columns_in_embedding_ignored(self):
        """Non-emb columns in embedding_df (besides keys) are dropped."""
        base = pd.DataFrame(
            {
                "outlet": [1],
                "confluence": [10],
                "head_1": [100],
                "head_2": [200],
            }
        )
        embedding = pd.DataFrame(
            {
                "outlet": [1],
                "confluence": [10],
                "head_1": [100],
                "head_2": [200],
                "emb_0": [0.5],
                "raster_path": ["/some/path.npy"],
                "extra_col": [42],
            }
        )
        result = merge_cnn_features(base, embedding)
        assert "emb_0" in result.columns
        assert "extra_col" not in result.columns
        assert "raster_path" not in result.columns
