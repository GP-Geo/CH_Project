#!/usr/bin/env python
"""Phase 5 — Mars CNN embedding extraction (thin wrapper).

The real logic now lives in :mod:`channel_heads.models.embeddings`. Prefer
``python scripts/cli/run_mars_pipeline.py --stage embeddings`` or calling
``channel_heads.pipelines.extract_mars_cnn_embeddings()`` directly.

Outputs (under ``data/Mars/model_inputs/``):
  mars_cnn_embeddings.{parquet,csv}
  mars_model_input_tabular_plus_cnn.{parquet,csv}
  cnn_patches_5class/figures/
"""

from __future__ import annotations

from channel_heads import pipelines


def main() -> None:
    pipelines.extract_mars_cnn_embeddings()


if __name__ == "__main__":
    main()
