# Notebooks

Every notebook calls `channel_heads.*` — no duplicated logic. All root-path
resolution uses `pathlib`; no `sys.path` hacks needed.

There are two tiers:

1. **`notebooks/pipeline/00–14` — the canonical deep dive.** One enumerated,
   self-contained notebook per stage of [`docs/PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md)
   / [`STAGE_ASSET_MAP.md`](../STAGE_ASSET_MAP.md). **Start here.**
2. **Themed folders (`analysis/`, `mars/`, `training/`, `diagnostics/`,
   `presentation/`, `interpretation/`)** — kept as supporting / historical
   material, no longer the canonical reference. (`regime/` has been **archived**
   under `notebooks/archive/regime/`; its calibration + regA–regE selection now
   live in Stage-4 `pipeline/04_earth_mars_regime_calibration`.)

## Pipeline notebooks (`notebooks/pipeline/` — canonical, one per stage)

Each stage notebook is the authoritative, executable treatment of its stage:
scientific narrative + analysis/visualization over the canonical package and
on-disk artifacts. Heavy rebuilds run via the `channel-heads` CLI (each notebook
lists its command). See [`notebooks/pipeline/README.md`](../notebooks/pipeline/README.md);
regenerate the set with `notebooks/pipeline/_build_pipeline_notebooks.py`.

| Stage | Notebook | Builds with (CLI) |
|---|---|---|
| 0 | `00_project_setup_and_assumptions` | — (paths/regimes/units) |
| 1 | `01_earth_source_data_exploration` | — (RAW_KEEP DEMs) |
| 2 | `02_earth_interactive_network_exploration` | — |
| 3 | `03_mars_interactive_network_exploration` | `run-mars-pipeline --stage topology` |
| 4 | `04_earth_mars_regime_calibration` | — (calibration + regA–regE selection → `selected_regimes_AE.csv`) |
| 5 | `05_final_earth_network_generation_and_qa` | `build-earth-features --regime <r>` |
| 6 | `06_earth_pair_and_label_generation` | (pairing/labeling) |
| 7 | `07_earth_model_input_construction` | `build-cnn-patches --regime <r>` |
| 8 | `08_model_training` | `train-cnn-regime`, `train-combined-xgb-regime` |
| 9 | `09_earth_model_validation_and_tuning` | `eval-lobo-cv`, `retune-threshold-regime` |
| 10 | `10_final_mars_model_input_generation` | `run-mars-pipeline --stage patches/embeddings` |
| 11 | `11_mars_inference` | `run-mars-combined-regime --regime <r>` |
| 12 | `12_mars_threshold_and_prediction_analysis` | (threshold sweep) |
| 13 | `13_scientific_interpretation` | (interpretation) |
| 14 | `14_figures_poster_and_reporting` | `make-result-figures`, `generate-poster-figures` |

## Themed notebooks (supporting references)

### `notebooks/analysis/` — Earth data QA and exploration
| Notebook | Stage | Type | Purpose |
|---|---|---|---|
| `01_earth_source_data_qa` | 1 | QA gate | Verify all 17 DEMs exist, load, have valid CRS and z-range |
| `02_earth_network_explorer` | 2 | exploration | Per-basin threshold/pruning preview |
| `05_earth_network_qa` | 5 | QA gate | Formal Stage 5 gate — zero hard flags required before Stage 7 |

### `notebooks/mars/` — Mars pipeline
| Notebook | Stage | Type | Purpose |
|---|---|---|---|
| `00_mars_network_explorer` | 3 | exploration | Mars networks on MOLA hillshade, Strahler/DD summary |
| `02_first_meet_pairs` | 10 | pipeline | First-meet pair extraction (`channel_heads.mars.pairs`) |
| `03_pair_features` | 10 | pipeline | 5-feature table build (`features`) |
| `04_xgb_inference_5feat` | 11 | pipeline | Tabular XGBoost Mars inference (`models.xgboost`) |
| `05_mars_threshold_sensitivity` | 12 | analysis | Threshold sweep, touching fraction, regime comparison |
| `dd_hull_mars_vs_earth_complexity` | 4 | decision | Earth vs Mars DD / complexity calibration |

### `notebooks/archive/regime/` — regime calibration (archived)
Superseded by Stage-4 `pipeline/04_earth_mars_regime_calibration.ipynb`, which now
covers both the Earth↔Mars calibration **and** the regA–regE selection
(`selected_regimes_AE.csv`). Kept for provenance only: `00_calibration_overview`,
`01_mars_inference`, `02_threshold_retune`, `03_optimize_regime_candidates`.

### `notebooks/training/` — Earth model training
| Notebook | Stage | Type | Purpose |
|---|---|---|---|
| `00_pair_sample_qa` | 6 | QA | Visual touching/non-touching pair inspection |
| `01_prepare_dataset` | 6 | pipeline | Dataset preparation |
| `02_train_classifier` | 8 | pipeline | Train XGBoost classifier |
| `03_feature_engineering` | 7 | pipeline | Geometric features + patch preview |
| `04_cnn_embeddings` | 8 | pipeline | CNN training and embedding extraction |
| `05_cnn_quick_eval` | 9 | QA | CNN quick evaluation |

### `notebooks/diagnostics/` — QA and investigation
| Notebook | Stage | Type | Purpose |
|---|---|---|---|
| `dd_threshold_calibration` | 2/4 | decision | Stream-threshold vs Mars DD calibration |
| `earth_network_pruning_experiments` | 2 | exploration | Pruning strategy experiments |
| `lobo_cv` | 9 | QA | Leave-one-basin-out CV (`eval.lobo`) |
| `regB_threshold` | 9 | QA | regB threshold diagnostic |
| `rasterization_diagnostics` | 7 | QA | Patch rasterization sanity checks |
| `stream_crossing_qa` | 6 | QA | Stream-crossing filter QA |

### `notebooks/presentation/` — figures and reporting
| Notebook | Stage | Type | Purpose |
|---|---|---|---|
| `result_figures` | 14 | presentation | Model-comparison / result figures |
| `mars_contact_sheets` | 14 | presentation | Mars contact sheets (vector polylines) |
| `per_outlet_touching_pairs` | 14 | presentation | Per-outlet touching-pair figures |
| `simple_mars_earth_dd_presentation` | 14 | presentation | Earth vs Mars DD presentation |

### `notebooks/interpretation/` — scientific interpretation
| Notebook | Stage | Type | Purpose |
|---|---|---|---|
| `00_scientific_summary` | 13 | interpretation | Coupling rates, geographic distribution, scientific findings |

### `notebooks/archive/` — superseded (do not maintain)
Historical threshold experiments, old exploratory analysis notebooks
(`01_single_basin_test`, `02_multi_basin`, `03/04_all_basins`, `00_full_pipeline`),
and one-off experiment templates. Keep for provenance; do not import from.
See `notebooks/archive/README.md` for the full inventory.

## Rules

- Every pipeline notebook opens with a stage-card markdown cell (stage, prev/next, purpose, inputs, outputs, gate).
- Notebooks call `channel_heads.*` only — no inline copies of package logic.
- Write outputs to `data/results/` or `data/Mars/model_outputs/` only via canonical path constants from `channel_heads.io.paths`.
- Archive (move to `notebooks/archive/`) rather than delete superseded notebooks.
