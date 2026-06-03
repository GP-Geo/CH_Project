#!/usr/bin/env python
"""Phase 3B — Mars tabular XGBoost inference (thin wrapper).

The real logic now lives in
:func:`channel_heads.models.mars_inference.run_mars_tabular_inference`. Prefer
``python scripts/cli/run_mars_pipeline.py --stage xgb`` or calling
``channel_heads.pipelines.run_mars_xgb_inference()`` directly.

Outputs (under ``data/Mars/model_outputs/``):
  mars_xgb_predictions_5feat.{parquet,csv,gpkg}
  mars_xgb_predictions_5feat_summary.csv
  mars_xgb_predictions_by_network.csv
  figures/
"""

from __future__ import annotations

from channel_heads import pipelines


def main() -> None:
    pipelines.run_mars_xgb_inference()


if __name__ == "__main__":
    main()
