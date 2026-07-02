# Architecture (package-first)

The project is understandable through two things: the **`channel_heads/`
package** and the **notebooks**. The CLI is part of the package
(`channel_heads/cli/`, run via `python -m channel_heads <command>`). `scripts/`
now holds only shell orchestrators (`run_regime_pipeline.sh`,
`run_full_rebuild.sh`), diagnostics, and figure-render helpers.

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
├── rasterization/       # 5-class patch generation + drawing primitives
├── models/              # xgboost.py, thresholds.py, comparison.py, cnn.py, embeddings.py
├── viz/                 # vector figures (contact sheets, ROC, per-outlet)
│
├── pipelines/           # the readable top layer — one function per stage
│   ├── mars.py          #   build_mars_topology -> ... -> compare_mars_model_outputs
│   ├── earth.py         #   train_earth_cnn -> train_earth_xgb_variants
│   └── poster.py        #   generate_poster_figures
│
├── eval/                # threshold tuning, metrics, true-LOBO engine + leakage audit (lobo.py, lobo_cnn.py; CLI lobo-validate)
├── training/            # Earth CNN/XGBoost training loops + dataset/regime builders
├── cli/                 # CLI package: dispatcher + one module per command
│
└── (core modules: coupling_analysis, units, regimes, dd_calibration,
         pruning, basin_config, stream_utils, logging_config)
```

The model/raster/feature logic lives directly in `models/`, `rasterization/`,
`features/`, and `training/`; the frozen production artifacts
(`xgb_touching_classifier.json` @ 0.577406, `cnn_outlet_final.pt`) and the 5-class
raster contract are preserved as-is, with research/regime variants alongside them
under explicit suffixes.

## Public API entry points

```python
from channel_heads import pipelines

pipelines.build_mars_topology()         # Phase 1   (channel_heads.mars.topology)
pipelines.extract_mars_pairs()          # Phase 2B  (channel_heads.mars.pairs)
pipelines.build_mars_features()         # Phase 3A  (channel_heads.features.mars_features)
pipelines.run_mars_xgb_inference()      # Phase 3B  (channel_heads.models.mars_inference)
pipelines.build_mars_cnn_patches()      # Phase 4   (channel_heads.rasterization.mars_patches)
pipelines.extract_mars_cnn_embeddings() # Phase 5   (channel_heads.models.embeddings)
pipelines.run_mars_combined_inference() # Phase 6C  (channel_heads.models.mars_combined)
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
| Mars features (3A) | ✅ migrated | `channel_heads/features/mars_features.py` |
| Mars XGB inference (3B) | ✅ migrated | `channel_heads/models/mars_inference.py` |
| Mars CNN patches (4) | ✅ migrated | `channel_heads/rasterization/mars_patches.py` |
| Mars CNN embeddings (5) | ✅ migrated | `channel_heads/models/embeddings.py` |
| Mars combined inference (6C) | ✅ migrated | `channel_heads/models/mars_combined.py` |
| Earth training | ✅ package-resident | `channel_heads/training/{cnn,xgboost,datasets}.py`; `channel_heads/cli/train_*` (run via `python -m channel_heads train-*`) |
| Regime calibration | ✅ package-resident | `channel_heads/training/regime.py`; `channel_heads/cli/*_regime.py` (run via `python -m channel_heads *-regime`) |

The full pipeline is now package-resident. `channel_heads/cli/` is the CLI
package: a dispatcher plus one thin module per command. Implementations live in
the library subpackages; `channel_heads/pipelines/` invokes the CLI command
modules in-process (plain imports, no `runpy`).

See also: [pipeline.md](pipeline.md) · [modeling.md](modeling.md) ·
[data_management.md](data_management.md) · [notebooks.md](notebooks.md).
