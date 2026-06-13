# Project Structure

Repo map, scripts inventory, and data inventory — consolidated from the former
`PROJECT_STRUCTURE.md`, `scripts/README.md`, and `data/README.md`.

> Descriptive, not prescriptive. Proposed-but-unapplied moves are marked. See
> [ROADMAP_AND_RISKS.md](ROADMAP_AND_RISKS.md) for the refactor plan and
> [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for the package API.
>
> Last consolidated: 2026-06-13 (package-first CLI; modules in subpackages).

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
| `features/` | Dimensionless feature math: `asymmetry.py` (ΔL), `geometry.py` (angle/azimuth/proximity), `paths.py` (direction/sampling), `earth_enrichment.py` (CSV enrichment), `mars_features.py`. |
| `pairing/` | First-meet pairing: graph-agnostic core (`dag.py`), Earth/TopoToolbox adapter (`earth.py`, exports `first_meet_pairs_for_outlet`), Mars-graph helpers (`mars_graph.py`), hard-negative `filtering.py`. |
| `rasterization/` | 5-class 128×128 patch rasterization (`patches.py`, direct final-grid), `drawing.py` (shared `bresenham_line`), `manifest.py`, class `schema.py`. |
| `models/` | XGBoost load/verify/predict (`xgboost.py`), `thresholds.py`, `comparison.py`, CNN (`cnn.py` `OutletCNN`, `cnn_features.py`, `embeddings.py`), `device.py` (`pick_device`), Mars `mars_inference.py`/`mars_combined.py`, `regime.py` (regime-CNN embedding attach). |
| `training/` | Earth training: `cnn.py` (loop/defaults), `xgboost.py` (variants), `datasets.py`, `labeling.py`, `regime.py` (regime dataset builders). |
| `eval/` | `metrics.py` (F1-opt / max-precision threshold, classification metrics), `splitting.py` / `lobo.py` (`outlet_group_holdout`, `leave_one_group_out_oof`, LOBO CV), `diagnostics.py`. |
| `io/` | `paths.py` (canonical paths), `tables.py` (parquet/CSV), `geopackage.py`, `cleanup.py` (generated-data manifest). |
| `pipelines/` | Readable top layer — one function per stage: `earth.py`, `mars.py`, `poster.py`. |
| `viz/` | Earth DEM/basin plotting (`earth.py`) plus vector figures: `contact_sheet.py`, `curves.py` (ROC), `per_outlet.py`, `stream_crossing.py`, `calibration.py`. |
| `cli/` | CLI package: subcommand dispatcher (`__init__.py`) + one module per command; run via `python -m channel_heads <command>`. |
| `dd_calibration.py` | Drainage-density / threshold calibration. |
| `pruning.py` | Strahler-strip + order-gap pruning. |
| `units.py` | Unit conversions (single source of truth). |
| `regimes.py` | `Regime` dataclass + `REGIMES` presets (regA/B/C). |
| `basin_config.py`, `logging_config.py`, `stream_utils.py` | Basin params / logging / `outlet_node_ids_from_streampoi`. |

`io.paths.RESULTS_DIR == OUTPUTS_DIR == data/results` (canonical output dir).
Public API is re-exported from each subpackage's `__init__.py`.

---

## 3. `scripts/` inventory

The command-line surface now lives **in the package**: `channel_heads/cli/` (run
via `python -m channel_heads <command>` or the `channel-heads` console script;
`python -m channel_heads --help` lists all commands). `scripts/` holds only
**shell orchestrators, headless diagnostics, rendering helpers, and archive** —
no Python CLI entry points and no pipeline/model implementation.

Categories: SHELL = orchestration runner · QA = diagnostics · RENDER =
visualization · MAINT = maintenance · ARCHIVE = retained, not maintained.

| Script | Category | Purpose |
|--------|----------|---------|
| `run_regime_pipeline.sh` | SHELL | Regime Steps 2→6; invokes the package CLI (`channel-heads build-earth-features → … → run-mars-combined-regime`) by command name. |
| `run_full_rebuild.sh` | SHELL | Baseline + regime rebuild batch runner over the package CLI. |
| `clean-cache.sh` | MAINT | Cache cleanup; used by the generated pre-push hook. |
| `setup-hooks.sh` | MAINT | Installs a pre-push hook that calls `./scripts/clean-cache.sh`. |
| `diagnostics/calibrate_stream_threshold_by_mars_dd.py` | QA | Earth Dd calibration across thresholds (`channel_heads.dd_calibration`). |
| `diagnostics/diag_regB_threshold.py` | QA | regB threshold diagnostic (`channel_heads.eval` + model loaders). |
| `diagnostics/qa_mars_stream_crossing_filter.py` | QA | QA of stream-crossing-dropped pairs (`channel_heads.pairing` + `viz`). |
| `rendering/render_mars_combined_contact_sheets_vector.py` | RENDER | Phase 6C contact sheets, vector (`channel_heads.viz`). |
| `rendering/render_mars_high_conf_emb_contact_sheet.py` | RENDER | Top-20 emb-probability pairs. |
| `rendering/render_mars_outlet_touching_pairs.py` | RENDER | Per-outlet touching-pair figures. |
| `_archive/*` | ARCHIVE | Superseded Mars wrappers + one-off experiments; use the package CLI / `channel_heads.pipelines` instead. |

The former `scripts/cli/*` wrappers and root-level Mars/Earth/regime training
scripts are now `channel_heads/cli/` commands (e.g. `run-mars-pipeline`,
`train-cnn-regime`, `train-combined-xgb-phase6b`, `eval-lobo-cv`,
`retune-threshold-regime`, `make-result-figures`, `generate-poster-figures`).

### Run order — Mars cross-planet (Phases 1–6C)
```
python -m channel_heads run-mars-pipeline --stage all

# Per-stage equivalents:
topology -> pairs -> features -> xgb -> patches -> embeddings -> combined
```
### Run order — regime calibration
```
scripts/run_regime_pipeline.sh regA   # = build-earth-features → build-cnn-patches
scripts/run_regime_pipeline.sh regB   #   → train-cnn-regime → train-combined-xgb-regime → run-mars-combined-regime
```

### Known path couplings (verify before moving)
1. `run_regime_pipeline.sh` / `run_full_rebuild.sh` invoke the package CLI by
   command name (`python -m channel_heads <command>`); they break only if a
   command is renamed/removed, not if a file moves.
2. The remaining `scripts/{diagnostics,rendering}/*.py` compute `PROJECT_ROOT =
   Path(__file__).resolve().parents[2]` (depth-2 under `scripts/`).
3. `clean-cache.sh`/`setup-hooks.sh` use `cd "$(dirname $0)/.."`; `setup-hooks.sh` generates a hook hardcoding `./scripts/clean-cache.sh`.

### Migration status
- ✅ Complete: the CLI lives in `channel_heads/cli/`; all Mars/Earth/regime
  pipeline + training logic is package-resident. `scripts/` retains only shell
  orchestrators, diagnostics, rendering, and archive.

---

## 3a. `notebooks/` homes

Foldered by role (Phase 6). Every notebook lives at `notebooks/<role>/` (depth 2),
which the in-notebook root-resolution cells assume — keep new notebooks at that depth.

| Home | Contents |
|------|----------|
| `training/` | Earth training pipeline `00_pair_sample_qa` → `05_cnn_quick_eval`; referenced by `channel_heads/cli/train_*`, `build_*`. |
| `analysis/` | Earth basin QA/exploration: `01_earth_source_data_qa`, `02_earth_network_explorer`, `05_earth_network_qa`. |
| `mars/` | Mars cross-planet exploration (`dd_hull_mars_vs_earth_complexity`). |
| `regime/` | Regime calibration: `00_calibration_overview`, `01_mars_inference` (→ `channel_heads.models`). |
| `diagnostics/` | QA / investigative (`rasterization_diagnostics`, `earth_network_pruning_experiments` — source of `channel_heads/pruning.py`). |
| `presentation/` | Presentation / figure generation (`simple_mars_earth_dd_presentation`, result figures, contact sheets). |
| `archive/` | Superseded one-offs (`experiment_*`, `optimization_review.md`). See `notebooks/archive/README.md`. |

### Notebook ↔ command primary-interface map

The notebook is the primary, documented interface and calls `channel_heads.*`
only; the matching CLI command (`python -m channel_heads <command>`) or
diagnostics/rendering script is the batch counterpart. Notebooks execute
read-only (no output regeneration) and are guarded when model artifacts are
absent.

| Notebook (primary) | CLI command / script | Package modules called |
|--------------------|----------------------|------------------------|
| `mars/02_first_meet_pairs` | `run-mars-pipeline --stage pairs` | `pairing` |
| `mars/03_pair_features` | `run-mars-pipeline --stage features` | `features` |
| `mars/04_xgb_inference_5feat` | `run-mars-pipeline --stage xgb` | `models` |
| `regime/01_mars_inference` | `run-mars-combined-regime` | `models`, `models.regime` |
| `regime/02_threshold_retune` | `retune-threshold-regime` | `eval`, `models` |
| `diagnostics/lobo_cv` | `eval-lobo-cv` | `eval` |
| `diagnostics/regB_threshold` | `scripts/diagnostics/diag_regB_threshold.py` | `eval`, `models` |
| `diagnostics/stream_crossing_qa` | `scripts/diagnostics/qa_mars_stream_crossing_filter.py` | `pairing`, `viz` |
| `diagnostics/dd_threshold_calibration` | `scripts/diagnostics/calibrate_stream_threshold_by_mars_dd.py` | `dd_calibration` |
| `presentation/mars_contact_sheets` | `scripts/rendering/render_mars_*_contact_sheet*.py` | `viz` |
| `presentation/per_outlet_touching_pairs` | `scripts/rendering/render_mars_outlet_touching_pairs.py` | `viz` |
| `presentation/result_figures` | `make-result-figures` | `viz`, `eval`, `models` |

**CLI-only (heavy compute / model-producing / orchestration):** `run-mars-pipeline`,
`build-earth-features`, `build-cnn-patches`, `train-cnn-*`, `train-combined-xgb-*`,
`run_regime_pipeline.sh`, `run_full_rebuild.sh`, `clean-cache.sh`,
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
| `data/outputs/` | Legacy duplicate (not read by `channel_heads.io.paths`). | LEGACY (archive candidate) |
| `data/exports/` | Rendered PDFs. | reports |
| `data/archive/` | Archive policy + holding area (empty). | policy |
| `data/_rebuild_backup_20260531/` | Pre-rebuild backup of old `models/`, Mars outputs, derived datasets (~97 MB). | backup (do not commit) |

### Rasterization-fix note
`channel_heads/rasterization/` (formerly `rasterizer.py`) does direct final-grid rasterization.
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
