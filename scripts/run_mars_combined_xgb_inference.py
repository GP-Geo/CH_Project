#!/usr/bin/env python
"""Phase 6C — Mars combined XGBoost inference (thin wrapper).

The real logic now lives in :mod:`channel_heads.models.mars_combined`. Prefer
``python scripts/cli/run_mars_pipeline.py --stage combined`` or calling
``channel_heads.pipelines.run_mars_combined_inference()`` directly.

Outputs (under ``data/Mars/model_outputs/``):
  mars_combined_model_predictions.{parquet,csv,gpkg}
  mars_model_comparison_summary.csv
  mars_predictions_by_network_combined.csv
  figures_combined/
"""

from __future__ import annotations

from channel_heads import pipelines


def main() -> None:
    pipelines.run_mars_combined_inference()


if __name__ == "__main__":
    main()
