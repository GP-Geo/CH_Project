# Pipeline

End-to-end Earth→Mars flow, with inputs/outputs per stage and which artifacts
are source (never delete) vs regenerable. Path constants are
`channel_heads.io.paths.*`. Run the whole thing with:

```python
from channel_heads import pipelines
pipelines.train_earth_models()       # Earth training (produces the models)
pipelines.run_full_mars_pipeline()   # Mars inference (consumes them)
```
…or per stage via `channel_heads/cli/run_mars_pipeline.py --stage <name>`.

## Dependency graph

```
                 Earth DEMs (RAW_KEEP)
                        │  train_earth_models()
                        ▼
        cnn_outlet_final.pt + xgb_*.json (models, kept)
                        │
 Mars valley vectors ───┤
 + MOLA DEM (RAW_KEEP)  │
        │ build_mars_topology()            [MIGRATED]
        ▼
 mars_vn_topology_model_ready.gpkg
        │ extract_mars_pairs()             [MIGRATED]
        ▼
 mars_vn_pairs.gpkg
        │ build_mars_features()
        ▼
 mars_pair_features_5feat_*.parquet ──► run_mars_xgb_inference()  ► tabular preds
        │ build_mars_cnn_patches()
        ▼
 cnn_patches_5class/ ─► extract_mars_cnn_embeddings() ─► mars_cnn_embeddings.parquet
        │ run_mars_combined_inference()
        ▼
 mars_combined_*_predictions.{gpkg,parquet,csv}  ► compare_mars_model_outputs()
```

## Stages

| # | `pipelines.*` | Input | Output | Output class |
|---|---------------|-------|--------|--------------|
| 1 | `build_mars_topology` | `MARS_VALLEYS`, `MARS_DEM` | `MARS_TOPOLOGY_GPKG` (7 layers) | regenerable |
| 2B | `extract_mars_pairs` | `MARS_TOPOLOGY_GPKG` | `MARS_PAIRS_GPKG` (pairs + paths) | regenerable |
| 3A | `build_mars_features` | `MARS_PAIRS_GPKG` | `mars_pair_features_5feat_*.parquet` | regenerable (tabular) |
| 3B | `run_mars_xgb_inference` | features + `XGB_PRODUCTION` | `mars_xgb_predictions_5feat.*` | regenerable |
| 4 | `build_mars_cnn_patches` | `MARS_PAIRS_GPKG` | `MARS_CNN_PATCHES_DIR/` (5-class) | regenerable (raster) |
| 5 | `extract_mars_cnn_embeddings` | patches + `CNN_PRODUCTION` | `MARS_CNN_EMBEDDINGS` | regenerable (raster) |
| 6C | `run_mars_combined_inference` | features + embeddings + `xgb_geom_plus_cnn_*` | `mars_combined_*_predictions.*` | regenerable |

## Source vs regenerable

- **Source (RAW_KEEP — never delete):** `data/cropped_DEMs/` (Earth DEMs),
  `data/raw/`, `MARS_VALLEYS` (`data/final_valleys/`), `MARS_DEM`,
  `MARS_HILLSHADE`.
- **Kept models:** `models/` (production `xgb_touching_classifier.json` @
  threshold 0.577406 and `cnn_outlet_final.pt` are frozen; variants carry
  `_geom_*` / `_reg{A,B,C}` suffixes).
- **Regenerable:** everything under `data/Mars/topology|model_inputs|model_outputs`,
  `data/results/<basin>/`, CNN patches/embeddings, predictions, figures. The
  raster chain built before the direct-final-grid rasterizer rewrite is
  `STALE_AFTER_RASTER_FIX` (regenerate before trusting) — see
  [data_management.md](data_management.md).

The regime-calibration variants (regA/B/C) reuse the same stages with pruned
Earth networks; see `scripts/run_regime_pipeline.sh` and
[modeling.md](modeling.md).
