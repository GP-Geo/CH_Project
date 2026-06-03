# Project Structure

Repo map, scripts inventory, and data inventory — consolidated from the former
`PROJECT_STRUCTURE.md`, `scripts/README.md`, and `data/README.md`.

> Descriptive, not prescriptive. Proposed-but-unapplied moves are marked. See
> [ROADMAP_AND_RISKS.md](ROADMAP_AND_RISKS.md) for the refactor plan and
> [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for the package API.
>
> Last consolidated: 2026-06-01.

---

## 1. Top-level layout

```
channel-heads/
├── channel_heads/      # Python package (importable library code)
├── tests/              # pytest suite (maps 1:1 to package modules)
├── scripts/            # headless CLI scripts (pipeline, training, QA, rendering)
├── notebooks/          # foldered by role: training / analysis / mars / regime / diagnostics / presentation / archive (§3a)
├── data/               # inputs + generated outputs (gitignored; §4)
├── models/             # trained model artifacts (gitignored)
├── docs/               # ← consolidated documentation (this folder)
├── env/                # conda environment spec
├── .github/            # CI workflows
├── README.md           # slim user quickstart
└── CLAUDE.md           # thin pointer to docs/
```

### Doc index
| Doc | Scope |
|-----|-------|
| `docs/DEVELOPER_GUIDE.md` | Package API, ML pipeline, testing, conventions. |
| `docs/PROJECT_STRUCTURE.md` | This file — repo/scripts/data inventory. |
| `docs/ROADMAP_AND_RISKS.md` | Open work, refactor plan, scientific risk register. |
| `docs/MARS_PIPELINE.md` | Cross-planet pipeline design contract + phase history. |
| `docs/DATA_STATUS.md` | Per-path classification of `data/`+`models/` (keep / regenerate / stale). |
| `docs/PIPELINE_RERUN.md` | End-to-end regeneration order + guardrails. |
| `notebooks/archive/optimization_review.md` | Notebook-local audit notes (archived with the experiment sweeps). |
| `data/archive/README.md`, `data/outputs/README.LEGACY.md` | In-situ data markers (kept in place). |

---

## 2. `channel_heads/` package

| Module | Role |
|--------|------|
| `coupling_analysis.py` | `CouplingAnalyzer` (coupling detection, mask cache, stream-crossing gate). |
| `first_meet_pairs_for_outlet.py` | Compatibility shim for the Earth first-meet adapter in `pairing/earth.py`. |
| `geometric_analysis.py` | Asymmetry + geometric features + labeling + CSV enrichment. |
| `rasterizer.py` | 5-class 128×128 patch rasterization (direct final-grid); shared `bresenham_line`. |
| `cnn_features.py`, `cnn_model.py`, `cnn_training.py` | CNN dataset + `OutletCNN` + embedding extraction + shared training loop/defaults. |
| `dd_calibration.py` | Drainage-density / threshold calibration. |
| `pruning.py` | Strahler-strip + order-gap pruning. |
| `units.py` | Unit conversions (single source of truth). |
| `regimes.py` | `Regime` dataclass + `REGIMES` presets (regA/B/C). |
| `pairing/` | First-meet pairing package: graph-agnostic core (`dag.py`), Earth/TopoToolbox adapter (`earth.py`), and Mars-graph helpers (`mars_graph.py`). |
| `features/` | Dimensionless feature math: `geometry.py` (angle/azimuth/proximity), `paths.py` (direction/sampling). |
| `inference/` | XGBoost glue: `xgb.py` (load/verify/predict), `device.py` (`pick_device`), `regime.py` (regime-CNN embedding attach). |
| `eval/` | `metrics.py` (F1-opt / max-precision threshold, classification metrics), `splitting.py` (`outlet_group_holdout`, `leave_one_group_out_oof`). |
| `viz/` | Earth DEM/basin plotting (`earth.py`) plus vector figures: `contact_sheet.py`, `curves.py` (ROC), `per_outlet.py`, `stream_crossing.py`. |
| `basin_config.py`, `io/paths.py`, `config.py`, `logging_config.py`, `cli.py`, `stream_utils.py` | Basin params / canonical paths / legacy path shim / logging / CLI / helpers. |

`io.paths.RESULTS_DIR == config.RESULTS_DIR == OUTPUTS_DIR == data/results` (canonical output dir).
Public API is re-exported from each subpackage's `__init__.py`.

---

## 3. `scripts/` inventory

Partially organized: `rendering/` and `diagnostics/` are subfolders. The
maintained Mars CLI is `scripts/cli/run_mars_pipeline.py`; old root-level Mars
wrappers are archived under `scripts/_archive/`. Earth/regime training scripts
remain root-level compatibility entry points.
Categories: CLI = maintained command entry point · MARS = Mars cross-planet ·
REGIME = regime calibration · TRAIN = Earth training · RENDER = visualization ·
QA = diagnostics · MAINT = maintenance.

| Script | Category | Phase/Step | Purpose |
|--------|----------|-----------|---------|
| `cli/run_mars_pipeline.py` | CLI | 1-6C | Maintained Mars stage/all runner over `channel_heads.pipelines`. |
| `_archive/extract_mars_outlet_candidates.py` | MARS | pre-1 | Archived outlet-candidate prototype; superseded by package topology logic. |
| `_archive/build_mars_network_topology.py` | MARS | 1 | Archived root wrapper; use `cli/run_mars_pipeline.py --stage topology`. |
| `_archive/extract_mars_first_meet_pairs.py` | MARS | 2B | Archived root wrapper; use `cli/run_mars_pipeline.py --stage pairs`. |
| `_archive/build_mars_pair_features_5feat.py` | MARS | 3A | Archived root wrapper; use `cli/run_mars_pipeline.py --stage features`. |
| `_archive/run_mars_xgb_inference_5feat.py` | MARS | 3B | Archived root wrapper; use `cli/run_mars_pipeline.py --stage xgb`. |
| `_archive/build_mars_cnn_patches_5class.py` | MARS | 4 | Archived root wrapper; use `cli/run_mars_pipeline.py --stage patches`. |
| `_archive/extract_mars_cnn_embeddings.py` | MARS | 5 | Archived root wrapper; use `cli/run_mars_pipeline.py --stage embeddings`. |
| `_archive/run_mars_combined_xgb_inference.py` | MARS | 6C | Archived root wrapper; use `cli/run_mars_pipeline.py --stage combined`. |
| `train_combined_xgb_phase6b.py` | TRAIN | 6B | Train+persist 3 Earth XGBoost variants. |
| `build_earth_features_regime.py` | REGIME | 2 | Per-basin Earth features under a regime; consumes `channel_heads.regimes`. |
| `build_cnn_patches_regime.py` | REGIME | 3 | Regime CNN patches; consumes `channel_heads.regimes`. |
| `train_cnn_regime.py` | REGIME | 4 | Train per-regime `OutletCNN`. |
| `train_combined_xgb_regime.py` | REGIME | 5 | Train per-regime geom+emb XGBoost. |
| `run_mars_combined_regime.py` | REGIME | 6 | Mars inference under a regime. |
| `retune_threshold_regime.py` | REGIME | aux | Re-tune a regime threshold to F1-optimal. |
| `run_regime_pipeline.sh` | REGIME | orchestrator | Runs regime Steps 2→6. **Refs 5 scripts by path.** |
| `rendering/render_mars_combined_contact_sheets_vector.py` | RENDER | — | Phase 6C contact sheets (vector). |
| `rendering/render_mars_high_conf_emb_contact_sheet.py` | RENDER | — | Top-20 emb-probability pairs. |
| `rendering/render_mars_outlet_touching_pairs.py` | RENDER | — | Per-outlet touching-pair figures. |
| `diagnostics/qa_mars_stream_crossing_filter.py` | QA | — | QA of stream-crossing-dropped pairs. |
| `diagnostics/diag_regB_threshold.py` | QA | — | regB threshold diagnostic. |
| `diagnostics/calibrate_stream_threshold_by_mars_dd.py` | QA | — | Earth Dd calibration across thresholds. |
| `clean-cache.sh`, `setup-hooks.sh` | MAINT | — | Cache cleanup; install pre-push hook. |

### Run order — Mars cross-planet (Phases 1–6C)
```
scripts/cli/run_mars_pipeline.py --stage all

# Per-stage equivalents:
topology -> pairs -> features -> xgb -> patches -> embeddings -> combined
```
### Run order — regime calibration
```
scripts/run_regime_pipeline.sh regA   # = build_earth_features_regime → build_cnn_patches_regime
scripts/run_regime_pipeline.sh regB   #   → train_cnn_regime → train_combined_xgb_regime → run_mars_combined_regime
```

### Known path couplings (verify before moving)
1. `run_regime_pipeline.sh` invokes 5 regime scripts as `scripts/<name>.py`.
2. Most non-wrapper scripts compute `PROJECT_ROOT = Path(__file__).resolve().parents[1]` — moving one level deeper needs `parents[2]` (done for the moved render/diagnostics scripts).
3. `clean-cache.sh`/`setup-hooks.sh` use `cd "$(dirname $0)/.."`; `setup-hooks.sh` generates a hook hardcoding `./scripts/clean-cache.sh`.

### Migration status
- ✅ Applied: Mars Phases 1-6C are package-resident; root Mars scripts are wrappers.
- ✅ Applied: `rendering/` (3), `diagnostics/` (3) — `parents[1]`→`[2]` fixed; no inbound refs. Regime presets and CNN training helpers now live in `channel_heads/`, removing the former sibling-import coupling.
- ⏳ Deferred (path-coupled): `regime/`, `training/`, `maintenance/`.

---

## 3a. `notebooks/` homes

Foldered by role (Phase 6). Every notebook lives at `notebooks/<role>/` (depth 2),
which the in-notebook root-resolution cells assume — keep new notebooks at that depth.

| Home | Contents |
|------|----------|
| `training/` | Earth training pipeline `00_full_pipeline` → `05_cnn_quick_eval`; referenced by `scripts/cli/train_*`, `build_*`. |
| `analysis/` | Earth basin analysis `01_single_basin_test` → `04_all_basins_full`. |
| `mars/` | Mars cross-planet exploration (`dd_hull_mars_vs_earth_complexity`). |
| `regime/` | Regime calibration: `00_calibration_overview`, `01_mars_inference` (→ `channel_heads.inference`). |
| `diagnostics/` | QA / investigative (`rasterization_diagnostics`, `earth_network_pruning_experiments` — source of `channel_heads/pruning.py`). |
| `presentation/` | Presentation / figure generation (`simple_mars_earth_dd_presentation`, result figures, contact sheets). |
| `archive/` | Superseded one-offs (`experiment_*`, `optimization_review.md`). See `notebooks/archive/README.md`. |

### Script ↔ notebook primary-interface map (Phase 6)

Each B-class script is now a thin batch wrapper; the notebook is the primary,
documented interface and calls `channel_heads.*` only. Notebooks execute
read-only (no output regeneration) and are guarded when model artifacts are
absent.

| Notebook (primary) | Wrapper script | Package modules called |
|--------------------|----------------|------------------------|
| `mars/02_first_meet_pairs` | `cli/run_mars_pipeline.py --stage pairs` | `pairing` |
| `mars/03_pair_features` | `cli/run_mars_pipeline.py --stage features` | `features` |
| `mars/04_xgb_inference_5feat` | `cli/run_mars_pipeline.py --stage xgb` | `inference` |
| `regime/01_mars_inference` | `run_mars_combined_regime.py` | `inference`, `inference.regime` |
| `regime/02_threshold_retune` | `retune_threshold_regime.py` | `eval`, `inference` |
| `diagnostics/lobo_cv` | `eval_lobo_cv.py` | `eval` |
| `diagnostics/regB_threshold` | `diagnostics/diag_regB_threshold.py` | `eval`, `inference` |
| `diagnostics/stream_crossing_qa` | `diagnostics/qa_mars_stream_crossing_filter.py` | `pairing`, `viz` |
| `diagnostics/dd_threshold_calibration` | `diagnostics/calibrate_stream_threshold_by_mars_dd.py` | `dd_calibration` |
| `presentation/mars_contact_sheets` | `rendering/render_mars_*_contact_sheet*.py` | `viz` |
| `presentation/per_outlet_touching_pairs` | `rendering/render_mars_outlet_touching_pairs.py` | `viz` |
| `presentation/result_figures` | `make_result_figures.py` | `viz`, `eval`, `inference` |

**A-class (kept CLI-only — heavy compute / model-producing / orchestration):**
`scripts/cli/run_mars_pipeline.py`, `build_earth_features_regime`,
`build_cnn_patches_regime`, `train_cnn_*`, `train_combined_xgb_*`,
`run_*_pipeline.sh`, `run_full_rebuild.sh`, `clean-cache.sh`,
`setup-hooks.sh`.

**C-class (deletion candidate):** `exp_calibration_standardize.py` (dropped per-basin
standardization experiment; no inbound refs).

---

## 4. `data/` and `models/` inventory

All of `data/` and `models/` is **gitignored**. Canonical output dir is
`data/results/`. The "current vs stale" classification lives in
`docs/DATA_STATUS.md` (Phase 7).

| Path | Contents | Status |
|------|----------|--------|
| `data/raw/` | Original SRTM. | input (keep) |
| `data/cropped_DEMs/` | 17 Earth DEMs (`*_strm_crop.tif`). | input (keep) |
| `data/Mars/` | Mars DEM + MOLA hillshade + topology/model_inputs/model_outputs. | active |
| `data/final_valleys/` | Mars valley-network vectors. | input (Mars) |
| `data/results/` | **Canonical** Earth/regime outputs; per-basin dirs, `master_dataset_v2.csv`, `raster_manifest*.csv`, regime rasters. | active |
| `data/outputs/` | Legacy duplicate (not read by `config.py`). | LEGACY (archive candidate) |
| `data/exports/` | Rendered PDFs. | reports |
| `data/archive/` | Archive policy + holding area (empty). | policy |
| `data/_rebuild_backup_20260531/` | Pre-rebuild backup of old `models/`, Mars outputs, derived datasets (~97 MB). | backup (do not commit) |

### Rasterization-fix note
`channel_heads/rasterizer.py` was rewritten to direct final-grid rasterization.
Raster patches / CNN embeddings / models built before that change are
regeneration candidates: `data/results/<basin>/rasters/`,
`data/results/_rasters_reg{A,B,C}/`, `data/Mars/model_inputs/cnn_patches_5class/`,
the `cnn_outlet_*.pt` models, and CNN-derived master datasets. Tabular-only
artifacts (`master_dataset_v2.csv`, 5-feature Mars tables, geom-only XGBoost)
are unaffected. The full regeneration plan is Phase 8 (see ROADMAP).

### Housekeeping (not actioned)
- `final_valleys/*.sr.lock` — stale ESRI lock files (junk; safe to delete).
- `*.DS_Store` — removed by `scripts/clean-cache.sh`.
- `data/archive/` and `data/_rebuild_backup_20260531/` are gitignored; keep the backup off-repo.
