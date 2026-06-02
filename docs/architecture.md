# Architecture (package-first)

The project is understandable through two things: the **`channel_heads/`
package** and the **notebooks**. `scripts/` is no longer part of the
architecture — it holds only thin CLI wrappers plus transitional stage
implementations pending extraction.

> Read `channel_heads/pipelines/mars.py` top to bottom and you have the whole
> Earth→Mars inference pipeline without opening a single script.

## Layered package

```
channel_heads/
├── io/                  # where files live + how we read/write them
│   ├── paths.py         #   single source of truth: Earth + Mars + models paths
│   ├── tables.py        #   write_table/read_table (parquet canonical, csv sidecar)
│   ├── geopackage.py    #   read_gpkg/write_gpkg (lazy geopandas)
│   └── cleanup.py       #   generated-data manifest + dry-run (never raw/models)
│
├── mars/                # Mars cross-planet logic (was scripts/build_mars_*, extract_mars_*)
│   ├── topology.py      #   Phase 1: valley vectors -> graph GeoPackage   [MIGRATED]
│   └── pairs.py         #   Phase 2B: first-meet channel-head pairs        [MIGRATED]
│
├── pairing/             # graph-agnostic first-meet core (Earth + Mars share this)
├── features/            # dimensionless feature math (geometry, paths)
├── rasterization/       # 5-class patch generation + drawing primitives (over rasterizer.py)
├── models/              # xgboost.py, thresholds.py, comparison.py, cnn.py, embeddings.py
├── viz/                 # vector figures (contact sheets, ROC, per-outlet)
│
├── pipelines/           # the readable top layer — one function per stage
│   ├── mars.py          #   build_mars_topology -> ... -> compare_mars_model_outputs
│   ├── earth.py         #   train_earth_cnn -> train_earth_xgb_variants
│   └── poster.py        #   generate_poster_figures
│
└── (core: coupling_analysis, geometric_analysis, units, regimes, dd_calibration,
         pruning, basin_config, config, logging_config, cli)
```

`models/`, `rasterization/`, `io.paths` are **curated surfaces** over existing
implementation modules (`inference/`, `eval/`, `cnn_*`, `rasterizer.py`,
`config.py`); the implementations are preserved as-is so the frozen production
artifacts (`xgb_touching_classifier.json` @ 0.577406, `cnn_outlet_final.pt`) and
the 5-class raster contract are untouched.

## Public API entry points

```python
from channel_heads import pipelines

pipelines.build_mars_topology()         # Phase 1   (channel_heads.mars.topology)
pipelines.extract_mars_pairs()          # Phase 2B  (channel_heads.mars.pairs)
pipelines.build_mars_features()         # Phase 3A  (transitional)
pipelines.run_mars_xgb_inference()      # Phase 3B  (transitional)
pipelines.build_mars_cnn_patches()      # Phase 4   (transitional)
pipelines.extract_mars_cnn_embeddings() # Phase 5   (transitional)
pipelines.run_mars_combined_inference() # Phase 6C  (transitional)
pipelines.run_full_mars_pipeline()      # all of the above, in order

from channel_heads.io import paths, read_table, write_table, read_gpkg, write_gpkg
from channel_heads import models   # models.xgboost / thresholds / comparison / cnn / embeddings
```

## Migration status & extraction backlog

| Stage | Status | Logic location |
|-------|--------|----------------|
| Mars topology (Phase 1) | ✅ migrated | `channel_heads/mars/topology.py` |
| Mars first-meet pairs (2B) | ✅ migrated | `channel_heads/mars/pairs.py` |
| io / paths / tables / geopackage / cleanup | ✅ new | `channel_heads/io/` |
| models / rasterization curated surfaces | ✅ new | `channel_heads/{models,rasterization}/` |
| Mars features (3A) | ⏳ transitional | `scripts/build_mars_pair_features_5feat.py` |
| Mars XGB inference (3B) | ⏳ transitional | `scripts/run_mars_xgb_inference_5feat.py` |
| Mars CNN patches (4) | ⏳ transitional | `scripts/build_mars_cnn_patches_5class.py` |
| Mars CNN embeddings (5) | ⏳ transitional | `scripts/extract_mars_cnn_embeddings.py` |
| Mars combined inference (6C) | ⏳ transitional | `scripts/run_mars_combined_xgb_inference.py` |
| Earth training | ⏳ transitional | `scripts/train_*` |
| Regime calibration | ⏳ transitional | `scripts/*_regime.py`, `run_regime_pipeline.sh` |

**Transitional** = the `pipelines.*` function runs the script in-process
(`channel_heads/pipelines/_delegate.py`). The next batch of work moves each
script body into the matching package module (`channel_heads/features/`,
`channel_heads/mars/`, `channel_heads/models/`) and turns the script into a thin
`cli/` wrapper. The architecture and public API do not change when that happens.

See also: [pipeline.md](pipeline.md) · [modeling.md](modeling.md) ·
[data_management.md](data_management.md) · [notebooks.md](notebooks.md).
