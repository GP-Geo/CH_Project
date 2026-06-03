# STAGE_ASSET_MAP.md

Maps each `PIPELINE_DESIGN.md` stage (0–14) to the existing scripts,
notebooks, package modules, and data artifacts that serve it.

Status column key:
- ✅ **covered** — production-quality code + notebook exists
- 🔶 **partial** — code exists but no clean notebook, or notebook is exploratory-only
- ❌ **gap** — stage has no dedicated notebook or package support yet

_Last updated: 2026-06-03_

---

## Stage 0 — Project setup and assumptions

| Asset type | Asset |
|-----------|-------|
| Config / contracts | `channel_heads/io/paths.py` (all canonical paths) |
| Regime presets | `channel_heads/regimes.py` (`Regime`, `regA`, `regB`, `regC`) |
| Unit contracts | `channel_heads/units.py` |
| Docs | `CLAUDE.md`, `AGENT_RULES.md`, `docs/architecture.md`, `PIPELINE_DESIGN.md` |

**Status: ✅** — paths, units, and regime objects are canonical.

---

## Stage 1 — Earth source-data exploration

| Asset type | Asset |
|-----------|-------|
| Notebooks | `notebooks/analysis/01_single_basin_test.ipynb` — single basin |
| | `notebooks/analysis/02_multi_basin.ipynb` — multi-basin comparison |
| | `notebooks/analysis/03_all_basins.ipynb`, `04_all_basins_full.ipynb` — all basins |
| Package | `channel_heads/basin_config.py` — basin metadata |
| | `channel_heads/pruning.py` — network pruning |
| | `channel_heads/io/paths.py` — `EARTH_BASINS`, DEM paths |
| Data | `data/cropped_DEMs/` (17 DEMs, `RAW_KEEP`) |

**Status: 🔶** — analysis notebooks exist but are pipeline-oriented, not
source-data QA oriented. No notebook explicitly checks DEM coverage,
projection consistency, or basin boundary validity.

**Gap:** A dedicated `notebooks/analysis/00_earth_source_data_qa.ipynb` that
verifies each basin DEM exists, loads, has valid CRS, and has reasonable
value range would close this stage.

---

## Stage 2 — Earth interactive network exploration

| Asset type | Asset |
|-----------|-------|
| Notebooks | `notebooks/diagnostics/dd_threshold_calibration.ipynb` — DD vs threshold |
| | `notebooks/diagnostics/earth_network_pruning_experiments.ipynb` — pruning sweep |
| | `notebooks/archive/experiment_250th.ipynb`, `350th`, `500th` — archived threshold tests |
| Package | `channel_heads/dd_calibration.py` — `evaluate_basin_metrics`, `summarize_threshold_metrics`, `choose_best_threshold` |
| | `channel_heads/coupling_analysis.py` — network coupling |
| Data outputs | `data/results/experiments/th145_baseline/`, `th250_test/`, `th350_test/`, `th500_test/` (`REPORT`) |

**Status: 🔶** — notebooks exist for individual diagnostics, but there is no
single interactive notebook where a user can pick a basin, set a threshold,
and see the extracted network with immediate visual feedback.

**Gap:** `notebooks/analysis/02_earth_network_explorer.ipynb` — interactive
per-basin threshold + pruning preview calling `dd_calibration` functions.

---

## Stage 3 — Mars interactive network exploration

| Asset type | Asset |
|-----------|-------|
| Notebooks | `notebooks/mars/02_first_meet_pairs.ipynb` — pair topology |
| | `notebooks/mars/03_pair_features.ipynb` — geometric features |
| | `notebooks/mars/dd_hull_mars_vs_earth_complexity.ipynb` — Mars DD / complexity |
| Package | `channel_heads/mars/topology.py` — valley vectors → GeoPackage |
| | `channel_heads/mars/pairs.py` — first-meet pairs |
| | `channel_heads/dd_calibration.py` — `mars_network_geometry`, `mars_network_strahler`, `mars_network_table` |
| Data | `data/final_valleys/` Mars valley vectors (`RAW_KEEP`) |
| | `data/Mars/MOLA_Hillshade_Robinson_128ppd.tif` (`RAW_KEEP`) |
| | `data/Mars/topology/mars_vn_topology_model_ready.gpkg` (`CAN_REGENERATE`) |

**Status: 🔶** — package functions and per-stage notebooks exist; no
single unified "browse Mars networks" notebook with MOLA overlay and
network-summary panel.

**Gap:** `notebooks/mars/00_mars_network_explorer.ipynb` — browse Mars
networks on MOLA hillshade, show Strahler distribution and DD metrics per
network.

---

## Stage 4 — Earth-Mars regime calibration

| Asset type | Asset |
|-----------|-------|
| **Main notebook** | `notebooks/regime/00_calibration_overview.ipynb` ← primary entry |
| Supporting notebooks | `notebooks/diagnostics/dd_threshold_calibration.ipynb` |
| | `notebooks/diagnostics/earth_network_pruning_experiments.ipynb` |
| Scripts | `scripts/diagnostics/calibrate_stream_threshold_by_mars_dd.py` |
| | `scripts/diagnostics/diag_regB_threshold.py` |
| Package | `channel_heads/regimes.py` — frozen `regA`, `regB`, `regC` presets |
| | `channel_heads/dd_calibration.py` — full calibration function suite |
| Data outputs | `data/results/drainage_density_calibration/` (threshold sweeps, pruning maps, complexity calibration) (`REPORT`) |
| | `data/results/experiments/earth_network_pruning/` (`REPORT`) |
| | `data/results/experiments/th*_test/` threshold sweeps (`REPORT`) |

**Status: 🔶** — the calibration was completed historically; `regA/B/C`
presets exist in `channel_heads/regimes.py`; calibration data and figures
exist under `data/results/`. However, the **regime-selection rationale is
not written down in a single frozen notebook** — `00_calibration_overview.ipynb`
exists but may not be a clean read-through from inputs to decision.

**Gaps:**
1. The `notebooks/regime/00_calibration_overview.ipynb` should be audited
   and updated to be the canonical, self-contained record of the regime choice
   (reads package functions, outputs the selected regime and its rationale).
2. A short `docs/REGIME_SELECTION.md` noting the frozen regime parameters and
   the scientific rationale would close this stage definitively.

---

## Stage 5 — Final Earth network generation and QA

| Asset type | Asset |
|-----------|-------|
| Scripts | `scripts/build_earth_features_regime.py` (→ `training.regime.build_regime_feature_dataset`) |
| Package | `channel_heads/training/regime.py` — `build_regime_feature_dataset`, `resolve_regime_basins` |
| | `channel_heads/features/earth_enrichment.py` — per-basin feature CSV generation |
| | `channel_heads/pruning.py` — pruning implementation |
| Data outputs | `data/results/<basin>/` per-basin CSVs (`CAN_REGENERATE`) |
| | `data/results/master_dataset_reg{A,B,C}.csv` tabular datasets (`CAN_REGENERATE`) |
| | `data/results/build_earth_features_reg{A,B,C}_stats.csv` per-basin stats |
| QA notebooks | `notebooks/diagnostics/stream_crossing_qa.ipynb` — stream-crossing QA |
| | `notebooks/training/01_prepare_dataset.ipynb` — dataset prep |

**Status: ❌** — network generation runs via the script, but there is no
dedicated QA notebook that inspects outlier basins, checks aggressively
pruned basins, or visualizes where the regime damaged network structure.

**This is the most important Stage 4/5 gap** (see Stage 4/5 planning note
in `STAGE_45_PLANNING.md`).

---

## Stage 6 — Earth pair and label generation

| Asset type | Asset |
|-----------|-------|
| Package | `channel_heads/pairing/earth.py` — `first_meet_pairs_for_outlet` |
| | `channel_heads/features/earth_enrichment.py` — `add_geometric_features_to_csv`, `default_stream_loader` |
| | `channel_heads/training/labeling.py` — `generate_labeled_dataset`, `filter_hard_negatives` |
| Notebooks | `notebooks/training/01_prepare_dataset.ipynb` |
| Data outputs | `data/results/<basin>/` per-basin pair CSVs (`CAN_REGENERATE`) |
| | `data/results/master_dataset_reg{A,B,C}.csv` — labeled pair datasets |

**Status: ✅** — code is complete and package-resident. No visual
touching/non-touching pair sample notebook exists, but the functionality
is solid.

**Minor gap:** A visual QA sample notebook (e.g., `notebooks/training/00_pair_sample_qa.ipynb`)
showing touching and non-touching pair examples would support scientific
interpretation.

---

## Stage 7 — Earth model-input construction

| Asset type | Asset |
|-----------|-------|
| Package | `channel_heads/features/earth_geometry.py` — geometric features |
| | `channel_heads/features/earth_paths.py` — path traversal |
| | `channel_heads/features/asymmetry.py` — lengthwise asymmetry |
| | `channel_heads/rasterization/earth_patches.py` — single-patch rasterization |
| | `channel_heads/rasterization/earth_batch.py` — batch precompute |
| Scripts | `scripts/build_cnn_patches_regime.py` (→ `training.regime.build_regime_patch_dataset`) |
| Notebooks | `notebooks/training/03_feature_engineering.ipynb` |
| | `notebooks/diagnostics/rasterization_diagnostics.ipynb` |
| Data outputs | `data/results/_rasters_reg{A,B,C}/` (currently `STALE_AFTER_RASTER_FIX`) |
| | `data/results/raster_manifest_reg{A,B,C}.csv` (stale) |

**Status: ✅** — code is complete. Rasters are stale (pre-rewrite) but
regeneration path is clear.

---

## Stage 8 — Model training

| Asset type | Asset |
|-----------|-------|
| Scripts | `scripts/train_cnn_baseline.py` — baseline CNN |
| | `scripts/train_cnn_regime.py` — regime CNN |
| | `scripts/train_cnn_multiseed.py` — multi-seed CNN |
| | `scripts/train_combined_xgb_phase6b.py` — baseline combined XGB |
| | `scripts/train_combined_xgb_regime.py` — regime combined XGB |
| Package | `channel_heads/training/cnn.py` — `train_cnn` |
| | `channel_heads/training/xgboost.py` — `train_combined_variant` |
| | `channel_heads/training/datasets.py` — manifest filtering, CV split |
| | `channel_heads/models/` — architecture, device, XGBoost inference |
| Notebooks | `notebooks/training/02_train_classifier.ipynb` |
| | `notebooks/training/04_cnn_embeddings.ipynb` |
| | `notebooks/training/00_full_pipeline.ipynb` |
| Data outputs | `models/cnn_outlet_final.pt`, `models/cnn_outlet_reg{A,B,C}.pt` (`STALE_AFTER_RASTER_FIX`) |
| | `models/xgb_geom_plus_cnn_*.json` variants (stale) |
| | `models/xgb_geom_only.json` (`CAN_REGENERATE`, raster-independent) |

**Status: ✅** — training code is complete and package-resident.

---

## Stage 9 — Earth model validation and tuning

| Asset type | Asset |
|-----------|-------|
| Scripts | `scripts/eval_lobo_cv.py` (→ `eval.lobo`) |
| | `scripts/retune_threshold_regime.py` (calls `channel_heads.eval`) |
| Package | `channel_heads/eval/lobo.py` — LOBO CV report |
| Notebooks | `notebooks/diagnostics/lobo_cv.ipynb` |
| | `notebooks/diagnostics/regB_threshold.ipynb` |
| | `notebooks/regime/02_threshold_retune.ipynb` |
| | `notebooks/training/05_cnn_quick_eval.ipynb` |
| Data outputs | `models/lobo_cv_metrics.csv` |
| | `models/optimal_threshold_*.txt` |
| | `models/ALL_MODELS_METRICS.csv` |

**Status: ✅** — validation and threshold tuning covered.

---

## Stage 10 — Final Mars model-input generation

| Asset type | Asset |
|-----------|-------|
| CLI | `scripts/cli/run_mars_pipeline.py` — `--stage topology|pairs|features|patches|embeddings` |
| Package | `channel_heads/mars/topology.py`, `mars/pairs.py` |
| | `channel_heads/features/mars_features.py` |
| | `channel_heads/rasterization/mars_patches.py` |
| | `channel_heads/models/embeddings.py` |
| | `channel_heads/pipelines/mars.py` — `run_full_mars_pipeline()` |
| Notebooks | `notebooks/mars/02_first_meet_pairs.ipynb` |
| | `notebooks/mars/03_pair_features.ipynb` |
| Data outputs | `data/Mars/topology/*.gpkg` (`CAN_REGENERATE`) |
| | `data/Mars/model_inputs/mars_pair_features_5feat*.parquet` (`CAN_REGENERATE`) |
| | `data/Mars/model_inputs/cnn_patches_5class/` (`STALE_AFTER_RASTER_FIX`) |
| | `data/Mars/model_inputs/mars_cnn_patch_index.parquet` (stale) |

**Status: ✅** — fully package-resident; Mars pipeline is the most complete part.

---

## Stage 11 — Mars inference

| Asset type | Asset |
|-----------|-------|
| CLI | `scripts/cli/run_mars_pipeline.py --stage combined` |
| | `scripts/run_mars_combined_regime.py` — regime-specific inference |
| Package | `channel_heads/models/mars_combined.py` |
| | `channel_heads/models/mars_inference.py` |
| | `channel_heads/models/regime.py` — regime embedding attach |
| | `channel_heads/pipelines/mars.py` — `run_mars_combined_inference()` |
| Notebooks | `notebooks/regime/01_mars_inference.ipynb` |
| Data outputs | `data/Mars/model_outputs/mars_combined_*_predictions.*` (`CAN_REGENERATE`) |
| | `data/Mars/model_outputs/mars_xgb_predictions_5feat.*` (`CAN_REGENERATE`) |

**Status: ✅** — fully covered.

---

## Stage 12 — Mars threshold and prediction analysis

| Asset type | Asset |
|-----------|-------|
| Notebooks | `notebooks/regime/01_mars_inference.ipynb` — includes basic threshold look |
| | `notebooks/regime/02_threshold_retune.ipynb` — Earth-side retune |
| | `notebooks/mars/04_xgb_inference_5feat.ipynb` — 5-feat XGB predictions |
| Data outputs | `data/Mars/model_outputs/figures_combined/` (`REPORT`) |
| | `data/Mars/model_outputs/mars_model_comparison_summary.csv` |
| | `data/Mars/model_outputs/mars_predictions_by_network_combined.csv` |

**Status: 🔶** — some threshold analysis is embedded in existing notebooks,
but there is no dedicated notebook that systematically varies the Mars
operating threshold and shows the sensitivity of the touching/non-touching
prediction.

**Gap:** `notebooks/mars/05_mars_threshold_sensitivity.ipynb` — load
Mars prediction probabilities, sweep threshold, show touching fraction,
network-level statistics, and high-confidence pair distribution.

---

## Stage 13 — Scientific interpretation

| Asset type | Asset |
|-----------|-------|
| Notebooks | _(none)_ |
| Docs | _(none)_ |

**Status: ❌** — no dedicated notebook or document yet.

**Gap:** `notebooks/interpretation/00_scientific_summary.ipynb` — translate
Mars predictions into geomorphological meaning (e.g., coupling rate by
network, geographic distribution of touching pairs, comparison with Mars
terrain context).

---

## Stage 14 — Figures, poster, and reporting

| Asset type | Asset |
|-----------|-------|
| Notebooks | `notebooks/presentation/mars_contact_sheets.ipynb` |
| | `notebooks/presentation/per_outlet_touching_pairs.ipynb` |
| | `notebooks/presentation/result_figures.ipynb` |
| | `notebooks/presentation/simple_mars_earth_dd_presentation.ipynb` |
| Scripts | `scripts/make_result_figures.py` |
| | `scripts/cli/generate_poster_figures.py` |
| Data outputs | `data/exports/*.pdf` (`REPORT`) |
| | `data/results/figures_models/` (`REPORT`) |
| | `data/Mars/model_inputs/filtering_qa_removed_pairs/` (`REPORT`) |

**Status: 🔶** — presentation notebooks exist; figures are `STALE_AFTER_RASTER_FIX`
for any CNN-derived content. Poster figures generator exists.

---

## Gap summary

| Stage | Status | Primary gap |
|-------|--------|-------------|
| 0 | ✅ | — |
| 1 | 🔶 | No source-data QA notebook |
| 2 | 🔶 | No interactive network explorer |
| 3 | 🔶 | No unified Mars network browser |
| 4 | 🔶 | `00_calibration_overview.ipynb` not verified as clean/self-contained; no frozen rationale doc |
| **5** | **❌** | **No Earth network QA notebook** (outlier basins, aggressive pruning, network damage) |
| 6 | ✅ | Minor: no visual pair-sample QA notebook |
| 7 | ✅ | Rasters stale; regeneration path clear |
| 8 | ✅ | Models stale; retrain path clear |
| 9 | ✅ | — |
| 10 | ✅ | — |
| 11 | ✅ | — |
| 12 | 🔶 | No threshold-sensitivity notebook |
| 13 | ❌ | No interpretation notebook |
| 14 | 🔶 | Figures stale; generation path clear |

**Priority order for new work:**

1. **Stage 5 QA notebook** — gate before retraining; no retrain makes sense
   without confirming the regime-generated networks are scientifically sound.
2. **Stage 4 calibration audit** — confirm `00_calibration_overview.ipynb` is
   self-contained and the regime choice is frozen and documented.
3. **Stage 12 threshold sensitivity** — needed before scientific interpretation.
4. **Stage 13 interpretation** — synthesizes all outputs into a scientific story.
5. Stages 1–3 (exploratory notebooks) — lower urgency; useful for onboarding
   and reproducibility but not blocking any current work.
