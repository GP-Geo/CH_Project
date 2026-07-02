# Data Status

Classification of everything under `data/` (gitignored) and `models/`
(**tracked in git as of 2026-07-02** — see `models/MANIFEST.md`) so a clean
rebuild knows what to keep, what to regenerate, and what is stale.

> See [PIPELINE_RERUN.md](PIPELINE_RERUN.md) for the commands that regenerate the
> `CAN_REGENERATE` / `STALE_*` artifacts, and [PROJECT_STRUCTURE.md §4](PROJECT_STRUCTURE.md)
> for the raw inventory.
>
> Last classified: 2026-07-02. **Update 2026-06-04 (RECONCILED):** the Stage-7
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
>
> **Update 2026-06-13 (REGIME REDEFINITION):** the three regimes were redefined
> to the data-driven top-3 presets (T = 0.20/0.25/0.15 km², trim-only; see
> [REGIME_SELECTION.md](REGIME_SELECTION.md)) and the full regime chain was
> retrained on them: `master_dataset_reg*`, `cnn_outlet_reg{A,B,C}.pt`,
> `xgb_geom_plus_cnn_emb_reg{A,B,C}.json`, `optimal_threshold_*_reg*.txt`
> (regA 0.769133 / regB 0.773238 / regC 0.810635) and the Mars
> `mars_combined_reg{A,B,C}_predictions.*` all refreshed on 2026-06-13.
>
> **Update 2026-06-20 (TRUE LOBO):** honest cross-basin validation ran via
> `channel-heads lobo-validate` (`geom_only` + `precomputed_emb` modes) into
> `data/results/lobo/`; the `per_fold_cnn` mode has **not** been run yet. See
> [ROADMAP_AND_RISKS.md](ROADMAP_AND_RISKS.md).

## Legend

| Tag | Meaning |
|-----|---------|
| `RAW_KEEP` | Primary input. Never regenerate, never delete. |
| `CAN_REGENERATE` | Deterministically rebuildable from `RAW_KEEP` + code via the rerun plan. |
| `STALE_AFTER_RASTER_FIX` | Built **before** the direct-final-grid rasterizer rewrite; a raster-dependent artifact that should be regenerated before being trusted. |
| `REPORT` | Rendered figures / summaries; cheap to regenerate, safe to discard. |
| `LEGACY` | Superseded duplicate; archive, do not read in code. |
| `BACKUP` | Pre-rebuild snapshot; keep off-repo, do not commit. |

## Provenance of raw inputs

Where every `RAW_KEEP` asset came from, and the terms attached to it. The
repo's MIT `LICENSE` covers **code only** — data carries its own terms.

| Asset | Source & terms |
|-------|----------------|
| `data/cropped_DEMs/*.tif` (17 Earth DEMs) + `data/raw/output_srtm.tif` | Crops of public-domain **SRTM GL3**, obtained via OpenTopography ([doi:10.5069/G9445JDF](https://doi.org/10.5069/G9445JDF)); the basins are the 18 elongated mountain ranges of Goren & Shelef (2024), Table A1. The uncropped per-basin originals live in the owner's local `GuyPinkasLiran/DEMsG&S24/` (not in git). Attribute NASA SRTM / OpenTopography. |
| `data/Mars/` MOLA DEM + hillshade | **NASA MOLA** (public domain, NASA PDS). `Mars_DEM_reprojected.tif` (~11 GB) is a *derived* equirectangular reprojection whose exact recipe is undocumented — treat it as a frozen input (owner to add the reprojection parameters if recalled). |
| `data/final_valleys/` valley-network vectors | Alemanno, Orofino & Mancarella (2018), *Global map of Martian fluvial systems*, Earth and Space Science 5, 560–577 ([doi:10.1029/2018EA000362](https://doi.org/10.1029/2018EA000362)). Verify the supplement's redistribution terms before republishing the derived GPKGs. |

## Classification

| Path | Tag | Notes |
|------|-----|-------|
| `data/raw/` (SRTM) | `RAW_KEEP` | Original DEM source (`output_srtm.tif` is git-tracked). |
| `data/cropped_DEMs/*.tif` (17 Earth DEMs) | `RAW_KEEP` | Per-basin Earth inputs (`Inyo_strm_crop.tif` is git-tracked as the quickstart example). |
| `data/final_valleys/` | `RAW_KEEP` | Mars valley-network vectors (input; Alemanno et al. 2018). |
| `data/Mars/` DEM + MOLA hillshade | `RAW_KEEP` | Mars inputs. |
| `data/Mars/topology/*.gpkg` | `CAN_REGENERATE` | From `channel-heads run-mars-pipeline --stage topology` → `--stage pairs`. |
| `data/Mars/model_inputs/mars_pair_features_5feat*.parquet` | `CAN_REGENERATE` | Tabular features (`channel-heads run-mars-pipeline --stage features`). Raster-independent. |
| `data/Mars/model_inputs/cnn_patches_5class/` | `CAN_REGENERATE` | 5-class patches — **regenerated 2026-06-04** (current rasterizer; 3,682 ok / 103 invalid). Rebuild via `run-mars-pipeline --stage patches`. |
| `data/Mars/model_inputs/mars_cnn_patch_index.parquet`, `*_tabular_plus_cnn.parquet`, `mars_cnn_embeddings.parquet` | `CAN_REGENERATE` | **Regenerated 2026-06-04** from the new patches (`--stage patches`/`embeddings`). |
| `data/Mars/model_outputs/` predictions + figures | `CAN_REGENERATE` | Inference outputs; **refreshed 2026-06-13** on the redefined-regime models (regA/B/C predictions + figures). |
| `data/results/<basin>/` per-basin dirs | `CAN_REGENERATE` | Earth pipeline outputs. |
| `data/results/<basin>/rasters/` | `STALE_AFTER_RASTER_FIX` | Pre-rewrite per-basin rasters (superseded by `_rasters_reg*`). |
| `data/results/_rasters_reg{A,B,C}/` | `CAN_REGENERATE` | **Reconciled 2026-06-04** (regA regenerated, regB/C restored); inputs to the 2026-06-13 retrain. |
| `data/results/master_dataset_v2.csv`, 5-feature tables, geom-only XGBoost inputs | `CAN_REGENERATE` | Tabular-only — **unaffected** by the raster fix. |
| `data/results/master_dataset_reg{A,B,C}_with_emb.csv` | `CAN_REGENERATE` | Regime CNN embeddings — **regenerated 2026-06-13** (redefined-regime retrain). |
| `data/results/master_dataset_v4_cnn_full.csv` | `STALE_AFTER_RASTER_FIX` | Baseline CNN embeddings — frozen baseline, **not** retrained this wave. |
| `data/results/raster_manifest*.csv` | `CAN_REGENERATE` | Regime manifests resolve with 0 missing (2026-06-04) and fed the Stage-8 retrain; the non-regime `raster_manifest.csv` still indexes pre-rewrite rasters. |
| `data/results/lobo/{regA,regB,regC}/{geom_only,precomputed_emb}/` | `CAN_REGENERATE` | True-LOBO outputs (metrics, per-fold CSVs, leakage audits), 2026-06-20. Rebuild via `channel-heads lobo-validate --regime <r> --mode <m>`. `per_fold_cnn/` does not exist yet (never run). |
| `data/results/poster_figures/` (~14 MB) | `REPORT` | Poster panel PNG/SVGs + `poster_figure_manifest.{csv,md}` written by `notebooks/presentation/14_poster_figure_inventory.ipynb`. |
| `data/results/final_figures/` | `REPORT` | Output dir of `channel-heads generate-poster-figures` (currently empty). |
| `models/xgb_*_geom_only*.json`, tabular-only models + threshold/feature-col files | `CAN_REGENERATE` | Geometry-only — unaffected by the raster fix. |
| `models/cnn_outlet_reg{A,B,C}.pt`, `models/xgb_geom_plus_cnn_emb_reg{A,B,C}.json` (+ thresholds/feature-cols/metrics) | `CAN_REGENERATE` | Regime CNN / CNN-derived — **retrained 2026-06-13** on the redefined presets. |
| `models/cnn_outlet_final.pt`, `models/xgb_geom_plus_cnn_{emb,logit}.json` (baseline) | `STALE_AFTER_RASTER_FIX` | Frozen/baseline CNN-derived — preserved as-is, regenerate only via an explicit baseline rebuild. |
| `models/ALL_MODELS_METRICS.csv` | `REPORT` | **Predates the 2026-06-13 retrain** — where it disagrees with `optimal_threshold_*.txt`, the txt files are authoritative. |
| `models/lobo_cv_metrics.csv` | `REPORT` | *Within-basin* CV summary written by `channel-heads eval-lobo-cv` (last regenerated 2026-07-02). Not a cross-basin estimate — see `data/results/lobo/` for true LOBO. |
| `data/exports/*.pdf`, `data/results/figures_*`, `data/Mars/model_outputs/figures*` | `REPORT` | Regenerate from `presentation/` notebooks or render commands. |
| `data/outputs/` | `LEGACY` | Duplicate of `data/results/` (not read by `channel_heads.io.paths`). Archive. |
| `data/archive/` | `LEGACY` | Archive holding area (gitignored as of 2026-06-02). |
| `data/_rebuild_backup_20260531/` (~97 MB) | `BACKUP` | Pre-rebuild snapshot (gitignored). Keep off-repo. |
| `data/_stage7_archive_20260603_231506/` (~1.8 GB) | `BACKUP` | Pre-reconciliation Stage-7 raster snapshot; source of the regB/C restore. Keep off-repo. |
| `data/_mars_stage10_archive_20260604_040046/` (~64 MB) | `BACKUP` | Pre-refresh Mars Stage-10 artifacts. Keep off-repo. |
| `data/results/experiments/th145_baseline/`, `th250_test/`, `th350_test/`, `th500_test/` | `REPORT` | Stage 4 threshold-sweep outputs (per-basin CSVs + summary figures). Regenerated by `notebooks/archive/experiment_*.ipynb` or equivalent calibration notebook. |
| `data/results/experiments/earth_network_pruning/` | `REPORT` | Stage 4 pruning-strategy sweep (CSV ranking + comparison figures). Regenerated from `notebooks/diagnostics/earth_network_pruning_experiments.ipynb`. |
| `data/results/drainage_density_calibration/` (all subdirs) | `REPORT` | Stage 4 Earth-Mars DD and complexity calibration outputs: threshold sweeps, pruning maps, Strahler comparisons, visual regime grids. Regenerated from `notebooks/diagnostics/dd_threshold_calibration.ipynb` and `notebooks/pipeline/04_earth_mars_regime_calibration.ipynb`. Safe to discard; not read by production code. |
| `data/final_valleys/*.sr.lock` files (12 files) | safe to delete | GIS process-lock artifacts (GILAD session). Zero scientific content; left over from a closed QGIS/ArcGIS session. No code reads these. |

## Off-machine copy

The full `data/` tree (~18 GB) and the owner's `GuyPinkasLiran/DEMsG&S24/`
source DEMs exist only on the owner's machine. **Owner to copy them to
institutional storage / an external drive and record the location here**
(see the checklist in [../HANDOFF.md](../HANDOFF.md)).

## Key invariant — the rasterizer rewrite

`channel_heads/rasterization/` (formerly `rasterizer.py`) does **direct
final-grid** rasterization. Any artifact in the raster → CNN → CNN-embedding → combined-model
chain built before that change is `STALE_AFTER_RASTER_FIX`. **Tabular-only**
artifacts (geometry features, geom-only XGBoost, the 5-feature Mars tables) are
unaffected and stay `CAN_REGENERATE`.

The production artifacts the project preserves as-is
(`models/xgb_touching_classifier.json`, `models/cnn_outlet_final.pt`) are
**not** to be overwritten by a rebuild unless explicitly intended.
