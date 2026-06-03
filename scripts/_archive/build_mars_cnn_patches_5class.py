#!/usr/bin/env python
"""Phase 4 — Mars CNN patch generation (thin wrapper).

The real logic now lives in
:mod:`channel_heads.rasterization.mars_patches`
(``build_mars_cnn_patches`` / ``render_pair_patch``). Prefer
``python scripts/cli/run_mars_pipeline.py --stage patches`` or calling
``channel_heads.pipelines.build_mars_cnn_patches()`` directly.

Outputs (under ``data/Mars/model_inputs/``):
  cnn_patches_5class/{network_id}/{pair_id}.npy
  mars_cnn_patch_index.{parquet,csv}
  cnn_patches_5class/figures/
"""

from __future__ import annotations

from channel_heads import pipelines


def main() -> None:
    pipelines.build_mars_cnn_patches()


if __name__ == "__main__":
    main()
