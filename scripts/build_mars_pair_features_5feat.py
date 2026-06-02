#!/usr/bin/env python
"""Phase 3A — Mars 5-feature table generation (thin wrapper).

The real logic now lives in
:func:`channel_heads.features.mars_features.build_mars_features`. This script is
kept only as a headless entry point; prefer
``python scripts/cli/run_mars_pipeline.py --stage features`` or calling
``channel_heads.pipelines.build_mars_features()`` directly.

Outputs (under ``data/Mars/model_inputs/``):
  mars_pair_features_5feat_{all,model_ready}.{parquet,csv}
  mars_pair_filtering_audit.csv
  mars_features_validation.gpkg
"""

from __future__ import annotations

from channel_heads import pipelines


def main() -> None:
    pipelines.build_mars_features()


if __name__ == "__main__":
    main()
