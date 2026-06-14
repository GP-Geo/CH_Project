# Data Status

Classification of everything under `data/` and `models/` (both **gitignored**) so
a clean rebuild knows what to keep, what to regenerate, and what is stale.

> See [PIPELINE_RERUN.md](PIPELINE_RERUN.md) for the commands that regenerate the
> `CAN_REGENERATE` / `STALE_*` artifacts, and [PROJECT_STRUCTURE.md §4](PROJECT_STRUCTURE.md)
> for the raw inventory.
>
> Last classified: 2026-06-02. **Update 2026-06-04 (RECONCILED):** the Stage-7
> raster chain is restored. `data/results/_rasters_reg{A,B,C}/` are on disk and
> their `raster_manifest_reg*.csv` resolve with **0 missing** (regA freshly
> regenerated with the current rasterizer: 23,742 ok / 0 failed; regB/regC
> restored from `data/_stage7_archive_20260603_231506/`, which is retained as
> backup). These manifests are now valid inputs for Stage 8.
>
> **Update 2026-06-04 (RETRAINED):** Stage 8–9 ran on these reconciled rasters —
> the regime CNN/XGBoost chains (`cnn_outlet_reg*.pt`,
> `xgb_geom_plus_cnn_emb_reg*.json`, `master_dataset_reg*_with_emb.csv`) flipped
> `STALE_AFTER_RASTER_FIX` → `CAN_REGENERATE`. Baseline (`*_final`, `v4_cnn_full`)
> stays as-is (frozen, not retrained).
>
> **Update 2026-06-04 (MARS REFRESHED):** Stage 10–14 re-ran on the retrained
> models — Mars 5-class patches regenerated with the current rasterizer, embeddings
> /`tabular_plus_cnn` + all combined predictions (baseline + regA/B/C) refreshed,
> and 18 figures rebuilt. The Mars patch/embedding/prediction chain flipped
> `STALE_AFTER_RASTER_FIX` → `CAN_REGENERATE`; old artifacts archived in
> `data/_mars_stage10_archive_20260604_040046/`.

## Legend

| Tag | Meaning |
|-----|---------|
| `RAW_KEEP` | Primary input. Never regenerate, never delete. |
| `CAN_REGENERATE` | Deterministically rebuildable from `RAW_KEEP` + code via the rerun plan. |
| `STALE_AFTER_RASTER_FIX` | Built **before** the direct-final-grid rasterizer rewrite; a raster-dependent artifact that should be regenerated before being trusted. |
| `REPORT` | Rendered figures / summaries; cheap to regenerate, safe to discard. |
| `LEGACY` | Superseded duplicate; archive, do not read in code. |
| `BACKUP` | Pre-rebuild snapshot; keep off-repo, do not commit. |

## Classification

| Path | Tag | Notes |
|------|-----|-------|
| `data/raw/` (SRTM) | `RAW_KEEP` | Original DEM source. |
| `data/cropped_DEMs/*.tif` (17 Earth DEMs) | `RAW_KEEP` | Per-basin Earth inputs. |
| `data/final_valleys/` | `RAW_KEEP` | Mars valley-network vectors (input). |
| `data/Mars/` DEM + MOLA hillshade | `RAW_KEEP` | Mars inputs. |
| `data/Mars/topology/*.gpkg` | `CAN_REGENERATE` | From `channel_heads/cli/run_mars_pipeline.py --stage topology` -> `--stage pairs`. |
| `data/Mars/model_inputs/mars_pair_features_5feat*.parquet` | `CAN_REGENERATE` | Tabular features (`channel_heads/cli/run_mars_pipeline.py --stage features`). Raster-independent. |
| `data/Mars/model_inputs/cnn_patches_5class/` | `CAN_REGENERATE` | 5-class patches — **regenerated 2026-06-04** (current rasterizer; 3,682 ok / 103 invalid). Rebuild via `run_mars_pipeline.py --stage patches`. |
| `data/Mars/model_inputs/mars_cnn_patch_index.parquet`, `*_tabular_plus_cnn.parquet`, `mars_cnn_embeddings.parquet` | `CAN_REGENERATE` | **Regenerated 2026-06-04** from the new patches (`--stage patches`/`embeddings`). |
| `data/Mars/model_outputs/` predictions + figures | `CAN_REGENERATE` | Inference outputs; **refreshed 2026-06-04** on the new patches/embeddings (baseline + regA/B/C + 18 figures). |
| `data/results/<basin>/` per-basin dirs | `CAN_REGENERATE` | Earth pipeline outputs. |
| `data/results/<basin>/rasters/`, `data/results/_rasters_reg{A,B,C}/` | `STALE_AFTER_RASTER_FIX` | Pre-rewrite rasters. |
| `data/results/master_dataset_v2.csv`, 5-feature tables, geom-only XGBoost inputs | `CAN_REGENERATE` | Tabular-only — **unaffected** by the raster fix. |
| `data/results/master_dataset_reg{A,B,C}_with_emb.csv` | `CAN_REGENERATE` | Regime CNN embeddings — **regenerated 2026-06-04** (Stage 8 retrain). |
| `data/results/master_dataset_v4_cnn_full.csv` | `STALE_AFTER_RASTER_FIX` | Baseline CNN embeddings — frozen baseline, **not** retrained this wave. |
| `data/results/raster_manifest*.csv` | `STALE_AFTER_RASTER_FIX` | Index of pre-rewrite rasters. |
| `models/xgb_*_geom_only*.json`, tabular-only models + threshold/feature-col files | `CAN_REGENERATE` | Geometry-only — unaffected by the raster fix. |
| `models/cnn_outlet_reg{A,B,C}.pt`, `models/xgb_geom_plus_cnn_emb_reg{A,B,C}.json` (+ thresholds/feature-cols/metrics) | `CAN_REGENERATE` | Regime CNN / CNN-derived — **retrained 2026-06-04** on reconciled rasters (Stage 8–9). |
| `models/cnn_outlet_final.pt`, `models/xgb_geom_plus_cnn_{emb,logit}.json` (baseline) | `STALE_AFTER_RASTER_FIX` | Frozen/baseline CNN-derived — preserved as-is, regenerate only via an explicit baseline rebuild. |
| `data/exports/*.pdf`, `data/results/figures_*`, `data/Mars/model_outputs/figures*` | `REPORT` | Regenerate from `presentation/` notebooks or render scripts. |
| `data/outputs/` | `LEGACY` | Duplicate of `data/results/` (not read by `channel_heads.io.paths`). Archive. |
| `data/archive/` | `LEGACY` | Archive holding area (gitignored as of 2026-06-02). |
| `data/_rebuild_backup_20260531/` (~97 MB) | `BACKUP` | Pre-rebuild snapshot (gitignored). Keep off-repo. |

| `data/results/experiments/th145_baseline/`, `th250_test/`, `th350_test/`, `th500_test/` | `REPORT` | Stage 4 threshold-sweep outputs (per-basin CSVs + summary figures). Regenerated by `notebooks/archive/experiment_*.ipynb` or equivalent calibration notebook. |
| `data/results/experiments/earth_network_pruning/` | `REPORT` | Stage 4 pruning-strategy sweep (CSV ranking + comparison figures). Regenerated from `notebooks/diagnostics/earth_network_pruning_experiments.ipynb`. |
| `data/results/drainage_density_calibration/` (all subdirs) | `REPORT` | Stage 4 Earth-Mars DD and complexity calibration outputs: threshold sweeps, pruning maps, Strahler comparisons, visual regime grids. Regenerated from `notebooks/diagnostics/dd_threshold_calibration.ipynb` and `notebooks/pipeline/04_earth_mars_regime_calibration.ipynb`. Safe to discard; not read by production code. |
| `data/final_valleys/*.sr.lock` files (12 files) | safe to delete | GIS process-lock artifacts (GILAD session). Zero scientific content; left over from a closed QGIS/ArcGIS session. No code reads these. |

## Key invariant — the rasterizer rewrite

`channel_heads/rasterization/` (formerly `rasterizer.py`) does **direct
final-grid** rasterization. Any artifact in the raster → CNN → CNN-embedding → combined-model
chain built before that change is `STALE_AFTER_RASTER_FIX`. **Tabular-only**
artifacts (geometry features, geom-only XGBoost, the 5-feature Mars tables) are
unaffected and stay `CAN_REGENERATE`.

The production artifacts the project preserves as-is
(`models/xgb_touching_classifier.json`, `models/cnn_outlet_final.pt`) are
**not** to be overwritten by a rebuild unless explicitly intended.
