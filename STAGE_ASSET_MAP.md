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

**Foundation-verified 2026-06-04:** all 17 `EXAMPLE_DEMS` resolve; `CROPPED_DEMS_DIR`, `FINAL_VALLEYS_DIR`, `MARS_DIR`, `MARS_HILLSHADE`, `MARS_VALLEYS` all exist; all 12 regime parameters (regA/B/C) exactly match `docs/REGIME_SELECTION.md`; `units.py` confirmed sole source of unit conversions. S1 (ΔL unit assumption) resolved — arc-degrees confirmed. Minor: `basin_config.py` has 18 entries (includes `piedepalo` with no DEM on disk); `EXAMPLE_DEMS` correctly has 17.

| Asset | Location |
|---|---|
| Canonical paths | `channel_heads/io/paths.py` |
| Regime presets | `channel_heads/regimes.py` |
| Unit contracts | `channel_heads/units.py` |
| Docs | `CLAUDE.md`, `docs/architecture.md`, `docs/PIPELINE_DESIGN.md` |

---

## Stage 1 — Earth source-data exploration 🔶

**Foundation-verified 2026-06-04:** all 17 DEMs load cleanly; all EPSG:4326, cell 0.000833°, z-ranges geomorphically plausible, 0 QA issues. Toano `z_max` in `basin_config.py` (2914 m) is 87 m above DEM max (2827 m) — expected (DEM crop smaller than paper extent; `z_th=1710 m` unaffected).

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

**Foundation-verified 2026-06-04:** all 17 basins run through full regB extraction + Strahler pruning; no empty networks; survivor ratios 0.14–0.51 (low ratios for Taiwan/Sierra Madre expected for large basins). `apply_strategy()` API confirmed correct.

| Asset | Location |
|---|---|
| **Explorer notebook** | `notebooks/analysis/02_earth_network_explorer.ipynb` ← per-basin threshold/pruning preview |
| Calibration notebooks | `notebooks/diagnostics/dd_threshold_calibration.ipynb`, `earth_network_pruning_experiments.ipynb` |
| Package | `channel_heads/dd_calibration.py`, `channel_heads/pruning.py` |

**Gap closed:** `02_earth_network_explorer.ipynb` added.

---

## Stage 3 — Mars interactive network exploration 🔶

**Foundation-verified 2026-06-04:** Mars topology GeoPackage has 391 networks, 7 coherent layers, all CRS metre-units on Mars sphere, coordinate extents within Mars equatorial circumference — no CRS/unit problems. MOLA hillshade (Robinson) and valleys (Equidistant Cylindrical) use different projections by design; topology pipeline reprojects internally.

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
| Script | `channel_heads/cli/build_earth_features_regime.py` |
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

## Stage 7 — Earth model-input construction ✅

| Asset | Location |
|---|---|
| Script | `channel_heads/cli/build_cnn_patches_regime.py` (now supports `--workers` N) |
| Package | `channel_heads/rasterization/earth_patches.py`, `earth_batch.py` (multiprocess per-basin/chunk render, `n_workers`) |
| Notebooks | `notebooks/training/03_feature_engineering.ipynb`, `notebooks/diagnostics/rasterization_diagnostics.ipynb` |
| Data | `data/results/_rasters_reg{A,B,C}/` — on disk, manifests resolve 0-missing (backup retained in `data/_stage7_archive_20260603_231506/`) |
| Manifests | `data/results/raster_manifest_reg{A,B,C}.csv` — valid (regA 23,742 ok / 0 failed; regB 10,612 ok; regC 28,954 ok) |

**Status (RECONCILED 2026-06-04):** All three regime raster sets are on disk in
`data/results/_rasters_reg{A,B,C}/` and their manifests resolve with **0 missing**.
**regA** was freshly regenerated with the current rasterizer (17/17 basins;
23,742 ok / 2,174 invalid / 0 failed — the old interrupted manifest had 7,607
failed). **regB/regC** were restored from `data/_stage7_archive_20260603_231506/`
(rasterizer-compatible per user; regB 10,612 ok, regC 28,954 ok), archive retained
as backup. The rasterizer now supports optional **multiprocess** per-basin/chunk
rendering (`n_workers` / CLI `--workers`); default 1 is bit-identical to the serial
path (verified output-identical on real data; ~2.4× at 4 workers). Stage 8
training can now consume these manifests. See `AGENT_STATE.md`.

---

## Stage 8 — Model training ✅

**Retrained 2026-06-04** on the reconciled Stage-7 rasters (regA/regB/regC CNN +
combined geom+CNN-emb XGBoost); new metrics reproduce the prior numbers within
noise. Frozen `cnn_outlet_final.pt` / `xgb_touching_classifier.json` untouched.
See `AGENT_STAGE_8_9_RETRAIN.md`.

| Asset | Location |
|---|---|
| Scripts | `channel_heads/cli/train_cnn_regime.py`, `train_combined_xgb_regime.py`, `train_cnn_baseline.py`, `train_combined_xgb_phase6b.py` |
| Orchestrator | `scripts/run_regime_pipeline.sh <regA\|regB\|regC> [full\|retrain]` (regC + retrain-only mode) |
| Package | `channel_heads/training/cnn.py`, `channel_heads/training/xgboost.py`, `channel_heads/training/datasets.py` |
| Notebooks | `notebooks/training/02_train_classifier.ipynb`, `04_cnn_embeddings.ipynb`, `05_cnn_quick_eval.ipynb` |
| Models | `models/cnn_outlet_reg{A,B,C}.pt`, `models/xgb_geom_plus_cnn_emb_reg{A,B,C}.json` ✅ retrained 2026-06-04 |
| Thresholds | `models/optimal_threshold_*.txt`, `models/feature_columns_*.txt` |

---

## Stage 9 — Earth model validation and tuning ✅

**Refreshed 2026-06-04.** LOBO CV + thresholds + `ALL_MODELS_METRICS.csv` rebuilt
on the retrained models; operating thresholds kept precision-oriented
(max-precision@recall≥0.5), F1-optimal recorded as alternative. Fixed a
`parents[1]` project-root bug in `eval_lobo_cv.py` / `retune_threshold_regime.py`
(exposed when the refactor moved them into `scripts/cli/`).

| Asset | Location |
|---|---|
| Script | `channel_heads/cli/eval_lobo_cv.py`, `channel_heads/cli/retune_threshold_regime.py` |
| Package | `channel_heads/eval/lobo.py` |
| Notebooks | `notebooks/diagnostics/lobo_cv.ipynb`, `notebooks/regime/02_threshold_retune.ipynb` |
| Metrics | `models/lobo_cv_metrics.csv`, `models/ALL_MODELS_METRICS.csv` ✅ refreshed 2026-06-04 |

---

## Stage 10 — Final Mars model-input generation ✅

**Regenerated 2026-06-04** with the current rasterizer: 5-class patches (3,682 ok /
103 invalid), `mars_cnn_patch_index.parquet`, embeddings + `tabular_plus_cnn` via
the frozen `cnn_outlet_final.pt`. Old artifacts archived in
`data/_mars_stage10_archive_20260604_040046/`. See `AGENT_STAGE_10_14_MARS.md`.


| Asset | Location |
|---|---|
| CLI | `channel_heads/cli/run_mars_pipeline.py --stage topology\|pairs\|features\|patches\|embeddings` |
| Package | `channel_heads/mars/topology.py`, `mars/pairs.py`, `channel_heads/features/mars_features.py`, `channel_heads/rasterization/mars_patches.py`, `channel_heads/models/embeddings.py` |
| Notebooks | `notebooks/mars/02_first_meet_pairs.ipynb`, `03_pair_features.ipynb` |
| Data | `data/Mars/model_inputs/mars_pair_features_5feat*.parquet`, `mars_cnn_patch_index.parquet` ⚠️ stale |

---

## Stage 11 — Mars inference ✅

**Re-run 2026-06-04** on the new patches: regime combined (regA/B/C) + baseline
combined. Coupling rates 53.0% / 64.1% / 38.6% (regA/B/C), within < 1.6 pp of the
stale run; regime ordering preserved. See `AGENT_STAGE_10_14_MARS.md`.


| Asset | Location |
|---|---|
| CLI | `channel_heads/cli/run_mars_pipeline.py --stage combined`, `channel_heads/cli/run_mars_combined_regime.py` |
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

## Stage 14 — Figures, poster, and reporting ✅

**Refreshed 2026-06-04** on the new predictions: 18 figures (vector contact sheets,
per-outlet drawings, network map, ROC, variant scatters, threshold-sensitivity
sweep) via the headless render scripts. Fixed a `parents[1]` root bug in
`make_result_figures.py`. See `AGENT_STAGE_10_14_MARS.md`.


| Asset | Location |
|---|---|
| Notebooks | `notebooks/presentation/mars_contact_sheets.ipynb`, `per_outlet_touching_pairs.ipynb`, `result_figures.ipynb` |
| Scripts | `channel_heads/cli/generate_poster_figures.py`, `channel_heads/cli/make_result_figures.py` |
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
| 7 | ✅ | Reconciled 2026-06-04: regA regenerated, regB/regC restored; manifests resolve 0-missing |
| 8 | ✅ | Regime CNN/XGBoost **retrained 2026-06-04** on reconciled rasters; metrics ≈ prior within noise |
| 9 | ✅ | LOBO + thresholds + ALL_MODELS_METRICS **refreshed 2026-06-04**; path bug fixed |
| 10 | ✅ | Mars patches/embeddings **regenerated 2026-06-04** (current rasterizer); old archived |
| 11 | ✅ | Mars inference **re-run 2026-06-04** (baseline + regA/B/C); rates ≈ stale within 1.6 pp |
| 12 | ✅ | Threshold sensitivity **refreshed** (`mars_threshold_sensitivity.csv/.png`) |
| 13 | ✅ | Interpretation **refreshed** (`mars_regime_interpretation.csv`; 64.7% cross-regime consensus) |
| 14 | ✅ | 18 figures **refreshed 2026-06-04**; `make_result_figures.py` root bug fixed |
