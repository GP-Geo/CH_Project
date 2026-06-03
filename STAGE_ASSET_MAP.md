# STAGE_ASSET_MAP.md

Maps each pipeline stage (see [`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md))
to the scripts, notebooks, package modules, and data artifacts that serve it.

Status key:
- ✅ **complete** — production-quality code + notebook + artifacts exist
- 🔶 **partial** — code exists but notebook is exploratory-only, or artifacts are stale
- ⚠️ **stale** — artifacts present but built on pre-rewrite data; valid for testing, need regeneration before final results
- ❌ **gap** — no dedicated notebook or package support yet

_Last updated: 2026-06-04_

---

## Stage 0 — Project setup and assumptions ✅

| Asset | Location |
|---|---|
| Canonical paths | `channel_heads/io/paths.py` |
| Regime presets | `channel_heads/regimes.py` |
| Unit contracts | `channel_heads/units.py` |
| Docs | `CLAUDE.md`, `docs/architecture.md`, `docs/PIPELINE_DESIGN.md` |

---

## Stage 1 — Earth source-data exploration 🔶

| Asset | Location |
|---|---|
| **QA notebook** | `notebooks/analysis/01_earth_source_data_qa.ipynb` ← DEM coverage, CRS, z-range |
| Archived | `notebooks/archive/01_single_basin_test.ipynb`, `02_multi_basin.ipynb`, `03/04_all_basins.ipynb` (superseded exploratory) |
| Basin metadata | `channel_heads/basin_config.py` |
| DEM paths | `channel_heads/io/paths.py::EARTH_BASINS` |
| Data | `data/cropped_DEMs/` (17 DEMs, RAW_KEEP) |

**Gap closed:** `01_earth_source_data_qa.ipynb` added.

---

## Stage 2 — Earth interactive network exploration 🔶

| Asset | Location |
|---|---|
| **Explorer notebook** | `notebooks/analysis/02_earth_network_explorer.ipynb` ← per-basin threshold/pruning preview |
| Calibration notebooks | `notebooks/diagnostics/dd_threshold_calibration.ipynb`, `earth_network_pruning_experiments.ipynb` |
| Package | `channel_heads/dd_calibration.py`, `channel_heads/pruning.py` |

**Gap closed:** `02_earth_network_explorer.ipynb` added.

---

## Stage 3 — Mars interactive network exploration 🔶

| Asset | Location |
|---|---|
| **Explorer notebook** | `notebooks/mars/00_mars_network_explorer.ipynb` ← Mars networks on MOLA hillshade |
| Analysis notebooks | `notebooks/mars/02_first_meet_pairs.ipynb`, `03_pair_features.ipynb` |
| Package | `channel_heads/mars/topology.py`, `channel_heads/dd_calibration.py` |
| Data | `data/final_valleys/` (RAW_KEEP), `data/Mars/MOLA_Hillshade_Robinson_128ppd.tif` |

**Gap closed:** `00_mars_network_explorer.ipynb` added.

---

## Stage 4 — Earth-Mars regime calibration ✅

| Asset | Location |
|---|---|
| **Calibration notebook** | `notebooks/regime/00_calibration_overview.ipynb` |
| Frozen presets | `channel_heads/regimes.py` (`regA`, `regB`, `regC`) |
| Rationale doc | `docs/REGIME_SELECTION.md` |
| Calibration scripts | `scripts/diagnostics/calibrate_stream_threshold_by_mars_dd.py`, `diag_regB_threshold.py` |
| Data | `data/results/drainage_density_calibration/`, `data/results/experiments/th*_test/` |

---

## Stage 5 — Final Earth network generation and QA ✅

| Asset | Location |
|---|---|
| **QA gate notebook** | `notebooks/analysis/05_earth_network_qa.ipynb` |
| Script | `scripts/cli/build_earth_features_regime.py` |
| Package | `channel_heads/training/regime.py`, `channel_heads/features/earth_enrichment.py` |
| QA report | `data/results/stage5_earth_network_qa_report.csv` (0 hard flags) |

---

## Stage 6 — Earth pair and label generation ✅

| Asset | Location |
|---|---|
| **Pair QA notebook** | `notebooks/training/00_pair_sample_qa.ipynb` ← touching/non-touching visual samples |
| Package | `channel_heads/pairing/earth.py`, `channel_heads/training/labeling.py` |
| Data | `data/results/master_dataset_reg{A,B,C}.csv` |

**Gap closed:** `00_pair_sample_qa.ipynb` added.

---

## Stage 7 — Earth model-input construction ⚠️

| Asset | Location |
|---|---|
| Script | `scripts/cli/build_cnn_patches_regime.py` |
| Package | `channel_heads/rasterization/earth_patches.py`, `earth_batch.py` |
| Notebooks | `notebooks/training/03_feature_engineering.ipynb`, `notebooks/diagnostics/rasterization_diagnostics.ipynb` |
| Data | `data/results/_rasters_reg{A,B,C}/` — **archived** to `data/_stage7_archive_*/` |
| Manifests | `data/results/raster_manifest_reg{A,B,C}.csv` — **archived**, need restoration |

**Status:** Rasters and manifests archived. Regenerate (or restore from archive)
when final regime is chosen. All downstream model artifacts are stale but usable
for testing. See `AGENT_STATE.md` for regeneration commands.

---

## Stage 8 — Model training ⚠️

| Asset | Location |
|---|---|
| Scripts | `scripts/cli/train_cnn_regime.py`, `train_combined_xgb_regime.py`, `train_cnn_baseline.py`, `train_combined_xgb_phase6b.py` |
| Package | `channel_heads/training/cnn.py`, `channel_heads/training/xgboost.py`, `channel_heads/training/datasets.py` |
| Notebooks | `notebooks/training/02_train_classifier.ipynb`, `04_cnn_embeddings.ipynb`, `05_cnn_quick_eval.ipynb` |
| Models | `models/cnn_outlet_reg{A,B,C}.pt`, `models/xgb_geom_plus_cnn_emb_reg{A,B,C}.json` ⚠️ stale |
| Thresholds | `models/optimal_threshold_*.txt`, `models/feature_columns_*.txt` |

---

## Stage 9 — Earth model validation and tuning ⚠️

| Asset | Location |
|---|---|
| Script | `scripts/cli/eval_lobo_cv.py`, `scripts/cli/retune_threshold_regime.py` |
| Package | `channel_heads/eval/lobo.py` |
| Notebooks | `notebooks/diagnostics/lobo_cv.ipynb`, `notebooks/regime/02_threshold_retune.ipynb` |
| Metrics | `models/lobo_cv_metrics.csv`, `models/ALL_MODELS_METRICS.csv` ⚠️ stale |

---

## Stage 10 — Final Mars model-input generation ⚠️

| Asset | Location |
|---|---|
| CLI | `scripts/cli/run_mars_pipeline.py --stage topology\|pairs\|features\|patches\|embeddings` |
| Package | `channel_heads/mars/topology.py`, `mars/pairs.py`, `channel_heads/features/mars_features.py`, `channel_heads/rasterization/mars_patches.py`, `channel_heads/models/embeddings.py` |
| Notebooks | `notebooks/mars/02_first_meet_pairs.ipynb`, `03_pair_features.ipynb` |
| Data | `data/Mars/model_inputs/mars_pair_features_5feat*.parquet`, `mars_cnn_patch_index.parquet` ⚠️ stale |

---

## Stage 11 — Mars inference ⚠️

| Asset | Location |
|---|---|
| CLI | `scripts/cli/run_mars_pipeline.py --stage combined`, `scripts/cli/run_mars_combined_regime.py` |
| Package | `channel_heads/models/mars_combined.py`, `channel_heads/models/mars_inference.py` |
| Notebook | `notebooks/regime/01_mars_inference.ipynb` |
| Predictions | `data/Mars/model_outputs/mars_combined_reg{A,B,C}_predictions.*` ⚠️ stale |

---

## Stage 12 — Mars threshold and prediction analysis ✅

| Asset | Location |
|---|---|
| **Threshold notebook** | `notebooks/mars/05_mars_threshold_sensitivity.ipynb` ← sweep threshold, compare regimes |
| Supporting | `notebooks/mars/04_xgb_inference_5feat.ipynb`, `notebooks/regime/02_threshold_retune.ipynb` |
| Data | `data/Mars/model_outputs/mars_combined_reg{A,B,C}_predictions.parquet` |

---

## Stage 13 — Scientific interpretation ✅

| Asset | Location |
|---|---|
| **Interpretation notebook** | `notebooks/interpretation/00_scientific_summary.ipynb` ← coupling rates, geography, terrain |

---

## Stage 14 — Figures, poster, and reporting 🔶

| Asset | Location |
|---|---|
| Notebooks | `notebooks/presentation/mars_contact_sheets.ipynb`, `per_outlet_touching_pairs.ipynb`, `result_figures.ipynb` |
| Scripts | `scripts/cli/generate_poster_figures.py`, `scripts/cli/make_result_figures.py` |
| Rendering | `scripts/rendering/render_mars_combined_contact_sheets_vector.py`, `render_mars_outlet_touching_pairs.py` |
| Data | `data/exports/*.pdf`, `data/results/figures_models/` ⚠️ stale |

---

## Summary table

| Stage | Status | Notes |
|---|---|---|
| 0 | ✅ | Paths, units, regimes canonical |
| 1 | 🔶 | `01_earth_source_data_qa.ipynb` added |
| 2 | 🔶 | `02_earth_network_explorer.ipynb` added |
| 3 | 🔶 | `00_mars_network_explorer.ipynb` added |
| 4 | ✅ | Regimes frozen, rationale doc written |
| 5 | ✅ | QA gate passed, 0 hard flags |
| 6 | ✅ | `00_pair_sample_qa.ipynb` added |
| 7 | ⚠️ | Rasters archived; regenerate when regime finalised |
| 8 | ⚠️ | Stale models present; re-run after Stage 7 |
| 9 | ⚠️ | Stale LOBO metrics; re-run after Stage 8 |
| 10 | ⚠️ | Stale Mars inputs; re-run after Stage 8 |
| 11 | ⚠️ | Stale predictions; re-run after Stage 10 |
| 12 | ✅ | Threshold sensitivity notebook complete |
| 13 | ✅ | Scientific summary notebook complete |
| 14 | 🔶 | Notebooks present; figures need refresh after Stage 11 |
