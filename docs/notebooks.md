# Notebooks

Notebooks are organised by pipeline stage (see [`docs/PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md)).
Each calls `channel_heads.*` — no duplicated logic. All root-path resolution
uses `pathlib`; no `sys.path` hacks needed.

## Pipeline notebooks (numbered by stage)

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

### `notebooks/regime/` — regime calibration
| Notebook | Stage | Type | Purpose |
|---|---|---|---|
| `00_calibration_overview` | 4 | decision | Regime/pruning calibration record (frozen parameters) |
| `01_mars_inference` | 11 | pipeline | Regime Mars inference with CNN embeddings |
| `02_threshold_retune` | 9/12 | decision | Re-tune a regime operating threshold |

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
