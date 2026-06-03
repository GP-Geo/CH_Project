# AGENT_RUN_LOG.md — refactor run log

Append one entry per completed slice (newest at top). Keep entries short.

---

## 2026-06-04 — Correction: Stage 7 in progress, not complete

- **Task:** Correct erroneous docs from previous entry that claimed Stages 7–11
  were complete. Stage 7 regA raster generation is currently running.
- **Correction:** The model artifacts in `models/` dated June 1–2 are from the
  pre-rewrite stale run, not a fresh regeneration. Stage 7B is in progress:
  `build_cnn_patches_regime.py --regime regA` running, 12/17 basins done.
  regB and regC not yet started. Stages 8–11 all pending Stage 7 completion.
- **Updated:** `STAGE_ASSET_MAP.md` (Stage 7 🔶 in progress; Stages 8–14 pending),
  `AGENT_STATE.md` (Stage 7 in progress, next steps).

---

## 2026-06-04 — Verify Stages 5–11 complete (INCORRECT — see correction above)

- **Branch:** `refactor/package-first-architecture`
- **Task:** Verify pipeline stages 5–11 executed and update docs.
- **Findings:**
  - Stage 5 QA: `stage5_earth_network_qa_report.csv` exists, 17 basins, 0 hard flags, 5 soft warnings.
  - Stage 7B: `_rasters_regA` present; regB/C rasters may have been cleaned after training. Training artifacts confirm all rasters were generated.
  - Stage 8: `cnn_outlet_reg{A,B,C}.pt`, `xgb_geom_plus_cnn_emb_reg{A,B,C}.json`, `optimal_threshold_*_reg{A,B,C}.txt` all present (dated June 1–2). Metrics: regA ROC-AUC=0.886, regB=0.871, regC=0.882. RegC LOBO pooled AUC (0.710) anomalously low vs fold mean (0.908) — flagged for Stage 9 review.
  - Stage 9: `lobo_cv_metrics.csv`, `ALL_MODELS_METRICS.csv` present.
  - Stage 10: `master_dataset_reg{A,B,C}_with_emb.csv` with 4-d CNN embeddings.
  - Stage 11: `mars_combined_reg{A,B,C}_predictions.*` all present.
- **Actions taken:**
  1. `STAGE_ASSET_MAP.md`: Stages 5/7/8/9/10/11 all marked ✅; metrics recorded; gap summary and priority list updated.
  2. `AGENT_STATE.md`: recorded stages 7B–11 complete with key artifacts and metrics; next task = Stage 12.
- **Not touched:** `data/`, `models/`, notebooks, package code.
- **Next step:** Stage 12 — `notebooks/mars/05_mars_threshold_sensitivity.ipynb`.

---

## 2026-06-03 — Stage 7A archive stale Earth regime raster artifacts

- **Branch:** `refactor/package-first-architecture`
- **Base commit:** `d3bd085` (Stage 7 regeneration plan)
- **Task:** Archive only stale generated Earth regime raster artifacts before
  regeneration. Do not delete data, regenerate patches, run training, touch
  models, or touch notebooks.
- **Actions taken:**
  1. Created timestamped archive folder
     `data/_stage7_archive_20260603_231506/`.
  2. Moved stale generated artifacts with structure preserved relative to
     `data/`:
     - `data/results/_rasters_regA/` ->
       `data/_stage7_archive_20260603_231506/results/_rasters_regA/`
     - `data/results/_rasters_regB/` ->
       `data/_stage7_archive_20260603_231506/results/_rasters_regB/`
     - `data/results/_rasters_regC/` ->
       `data/_stage7_archive_20260603_231506/results/_rasters_regC/`
     - `data/results/raster_manifest_regA.csv` ->
       `data/_stage7_archive_20260603_231506/results/raster_manifest_regA.csv`
     - `data/results/raster_manifest_regB.csv` ->
       `data/_stage7_archive_20260603_231506/results/raster_manifest_regB.csv`
     - `data/results/raster_manifest_regC.csv` ->
       `data/_stage7_archive_20260603_231506/results/raster_manifest_regC.csv`
  3. Added `.gitignore` entry `data/_stage7_archive_*/` so the generated archive
     is not accidentally committed.
- **Verification:** all six archived paths exist; all six original paths are
  absent; `data/results/master_dataset_regA_with_emb.csv`,
  `data/results/master_dataset_regB_with_emb.csv`, and
  `data/results/master_dataset_regC_with_emb.csv` still exist. Archive contains
  73,272 raster files and is about 1.4 GB.
- **Missing expected paths:** none.
- **Not touched:** raw/source/manual data, `models/`, notebooks, package code,
  regenerated rasters/patches, embeddings, predictions, training outputs.
- **Validation:** `git diff --check` clean.
- **Next step:** Stage 7B regeneration only; do not run Stage 8 training until
  regenerated manifests and patches validate.

---

## 2026-06-03 — Stage 7 Earth model-input regeneration plan

- **Branch:** `refactor/package-first-architecture`
- **Base commit:** `1341bfe` (Stage 5 Earth network QA gate)
- **Task:** Report-only Stage 7 regeneration plan after the Stage 5 Earth
  network QA pass. No regeneration, source edits, notebook edits, data changes,
  model changes, deletes, archives, or real pipeline runs.
- **Actions taken:**
  1. Created `AGENT_STAGE7_REGENERATION_PLAN.md` with purpose, readiness
     status, exact Stage 7 inputs, outputs to regenerate, stale/generated output
     classification, canonical `scripts/cli/` command sequence to run later,
     risk controls, validation plan, decision gate, and recommended next action.
  2. Updated `AGENT_STATE.md` to record that Stage 5 passed and Stage 7 now has
     a report-only regeneration plan.
- **Readiness recorded:** Stage 4 rationale exists; Stage 5 QA report exists at
  `data/results/stage5_earth_network_qa_report.csv`; hard flags = 0; soft
  warnings = 6 and non-blocking.
- **Noted input mismatch:** request named `docs/STAGE_ASSET_MAP.md`, but the
  file present in this checkout is `STAGE_ASSET_MAP.md` at repo root. The plan
  uses the root file and marks the path mismatch as documentation cleanup.
- **Not touched:** `data/`, root `models/`, notebooks, generated outputs,
  package code, scripts, rasters, embeddings, predictions, training outputs.
- **Validation:** `git diff --check` clean.
- **Next step:** Stage 7 execution slice: archive stale generated Earth
  model-input artifacts or run the planned regeneration commands, then validate
  manifests/patches before Stage 8 training.

---

## 2026-06-03 — Stage 5 Earth network QA gate

- **Branch:** `refactor/package-first-architecture`
- **Base commit:** `5a98517` (Stage 4 — regime selection rationale)
- **Task:** Create the formal Stage 5 QA gate notebook and update STAGE_ASSET_MAP.md.
- **Actions taken:**
  1. Created `notebooks/analysis/05_earth_network_qa.ipynb` — formal QA gate
     connected to PIPELINE_DESIGN.md. Opens with pipeline stage card.
     Cells: config → imports/paths → load data → cross-regime summary →
     flag analysis (hard + soft) → feature distributions → QC flags →
     per-basin detail for flagged basins → DEM thumbnails (optional) →
     decision gate (raises AssertionError on hard fail, writes QA CSV on pass).
  2. Updated `STAGE_ASSET_MAP.md`: Stage 5 ❌ → 🔶 with gate criteria listed;
     priority order updated to reflect Stages 4 and 5 complete.
  3. Updated `AGENT_STATE.md` to reflect Stage 5 complete and next actions.
- **Not touched:** `data/`, trained models, existing notebooks.
- **Validation:** no tests changed; full pytest still 569 passed.
- **Next step:** Run `notebooks/analysis/05_earth_network_qa.ipynb` and confirm
  PASS, then proceed to Stage 12 (Mars threshold sensitivity) or Stage 7 (rasters).

---

## 2026-06-03 — Agent doc update post-cleanup

- **Branch:** `refactor/package-first-architecture`
- **Base commit:** `77a80c1` (full shim deletion + scripts CLI move)
- **Task:** Update `AGENT_STATE.md` and `AGENT_BACKLOG.md` to reflect the
  completed package-first refactor.
- **Actions taken:**
  1. `AGENT_STATE.md`: updated latest commit, completed-migrations list, canonical
     ownership table (removed shim column entries for deleted modules), replaced
     "Known shims (keep working)" with accurate list of the two remaining
     compatibility surfaces, removed stale "transitional" entries, updated
     next-task section.
  2. `AGENT_BACKLOG.md`: marked Slice 10 (notebook rebuild) as DONE; added
     "Full shim deletion + scripts CLI move" entry as DONE; added Stage 4 and
     Stage 5 as the next bounded slices.
- **Validation:** no code changed.
- **Next step:** Stage 4 — audit `notebooks/regime/00_calibration_overview.ipynb`
  and write `docs/REGIME_SELECTION.md`.

---

## 2026-06-03 — Stage 4/5 planning + housekeeping

- **Branch:** `refactor/package-first-architecture`
- **Base commit:** `caaec26` (data cleanup dry-run)
- **Task:** Four housekeeping items following the dry-run report.
- **Actions taken:**
  1. Restored `models/xgb_touching_classifier.json` from
     `data/_rebuild_backup_20260531/models/` — this is the production model
     expected by `channel_heads/io/paths.py:228` (`XGB_PRODUCTION`).
  2. Updated `docs/DATA_STATUS.md` — added `REPORT` entries for
     `data/results/experiments/` (threshold sweeps + pruning sweep) and
     `data/results/drainage_density_calibration/` (DD/complexity calibration),
     and a "safe to delete" entry for the 12 `.sr.lock` files in `data/final_valleys/`.
  3. Created `STAGE_ASSET_MAP.md` — maps all 15 `PIPELINE_DESIGN.md` stages
     to existing scripts, notebooks, package modules, and data artifacts.
     Key finding: Stage 5 (Earth network QA) is the only fully missing stage
     (❌); Stages 4, 12, 13, 14 are partial (🔶).
  4. Created `STAGE_45_PLANNING.md` — concrete implementation plan for Stage
     4 (calibration audit + rationale doc) and Stage 5 (new QA notebook).
- **Not touched:** `data/`, any rasters, generated outputs, trained artifacts
  (except restoring the missing production model from the project's own backup).
- **Validation:** none required (doc + model-restore only). The model was
  verified to load as valid JSON with the expected XGBoost learner structure.
- **Next step:** Start Stage 4/5 work per `STAGE_45_PLANNING.md`: audit
  `00_calibration_overview.ipynb`, write `docs/REGIME_SELECTION.md`, then
  write `notebooks/analysis/05_earth_network_qa.ipynb`.

## 2026-06-03 — Data cleanup dry-run (report-only)

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `fd789dc`
- **Task:** Produce a report-only inventory of stale, legacy, and unclassified
  data against `docs/DATA_STATUS.md`. No data deleted, moved, or mutated.
- **Report file:** `AGENT_DATA_CLEANUP_DRYRUN.md`
- **Key findings:**
  - `data/outputs/` (179 MB) is `LEGACY` — main near-term safe-delete candidate
    once `data/results/` content is confirmed to supersede it.
  - Regime rasters `_rasters_reg{A,B,C}` (~1.43 GB, ~73k files) and per-basin
    rasters (~5,923 files) are `STALE_AFTER_RASTER_FIX` — delete only after
    retrain decision.
  - `data/Mars/model_inputs/cnn_patches_5class/` (74 MB) is
    `STALE_AFTER_RASTER_FIX`.
  - `models/xgb_touching_classifier.json` is **missing** from `models/` but
    present in the backup. `channel_heads/io/paths.py:228` registers this as
    `XGB_PRODUCTION`. Needs restore or reference update.
  - 12 stale `.sr.lock` GIS lock files in `data/final_valleys/` — safe to
    delete (no scientific content).
  - `data/results/experiments/` and `data/results/drainage_density_calibration/`
    are not classified in `DATA_STATUS.md` — need entries.
  - `data/_rebuild_backup_20260531/` (94 MB) is `BACKUP` — keep off-repo;
    no action until full rebuild is verified.
- **Not touched:** `data/`, root `/models/`, any code or generated artifact.
- **Validation:** no pytest required (doc-only slice).
- **Next step:** Restore `models/xgb_touching_classifier.json` from backup or
  update the CLAUDE.md/paths.py reference; update `DATA_STATUS.md` for the
  unclassified items; then move to `PIPELINE_DESIGN.md` stage-to-asset mapping.

## 2026-06-03 — Archive Mars root wrappers

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `fd789dc`
- **Task:** Execute the archive-readiness report for the low-risk Mars root
  wrappers only, while preserving behavior and leaving Earth/regime/training
  root scripts in place.
- **Archived:** `scripts/build_mars_network_topology.py`,
  `scripts/extract_mars_first_meet_pairs.py`,
  `scripts/build_mars_pair_features_5feat.py`,
  `scripts/run_mars_xgb_inference_5feat.py`,
  `scripts/build_mars_cnn_patches_5class.py`,
  `scripts/extract_mars_cnn_embeddings.py`, and
  `scripts/run_mars_combined_xgb_inference.py` moved to `scripts/_archive/`.
- **Command surface updated:** `scripts/run_full_rebuild.sh` now calls
  `scripts/cli/run_mars_pipeline.py --stage embeddings` and `--stage combined`.
  `docs/PIPELINE_RERUN.md`, `docs/PROJECT_STRUCTURE.md`,
  `docs/DATA_STATUS.md`, `scripts/README.md`, and
  `AGENT_SCRIPT_ARCHIVE_READINESS.md` now point at the maintained Mars CLI or
  package pipeline functions.
- **Left in place:** Earth/regime/training/diagnostic root scripts,
  `scripts/run_mars_combined_regime.py`, `scripts/run_regime_pipeline.sh`, and
  maintenance shell scripts.
- **Not touched:** `data/`, root `/models/`, notebooks, generated outputs,
  trained artifacts, DEMs, shapefiles, GeoPackages, parquet/csv outputs, and
  figures.
- **Validation:** stale-reference `rg` showed no live root-path references
  outside historical audit/run-log entries and archived files. Maintained Mars
  CLI import smoke passed. Focused pytest
  (`tests/test_pipelines.py tests/test_inference.py tests/test_mars_patches.py`)
  -> 40 passed, 6 warnings. Full pytest -> 583 passed, 7 warnings. `git diff
  --check` clean. Targeted ruff on the changed Python rendering script passed.
- **Next step:** A later bounded slice can add dedicated package-backed CLIs for
  Earth/regime/training scripts before considering any further archive moves.

## 2026-06-03 — Scripts CLI organization cleanup

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `5eb525a`
- **Task:** Audit and organize `scripts/` as a CLI/diagnostics/archive layer
  after package-first extraction, without changing behavior or running real
  pipelines.
- **Decision:** No scripts were moved. Reference checks showed root script paths
  are still used by `docs/PIPELINE_RERUN.md`, `docs/PROJECT_STRUCTURE.md`,
  `scripts/run_full_rebuild.sh`, `scripts/run_regime_pipeline.sh`, script
  docstrings, and handoff/audit history. Moving root wrappers would break
  documented commands unless many references and orchestrators were updated in
  the same slice.
- **Docs updated:** `scripts/README.md` now classifies all scripts as
  maintained CLI, wrapper over package API, diagnostics/rendering utility,
  shell/orchestration entry point, or archive. It also records canonical
  package alternatives and a conservative move policy.
- **Scripts moved:** none.
- **Scripts archived:** none. Existing archive entries remain
  `_archive/extract_mars_outlet_candidates.py` and
  `_archive/old_experiments/exp_calibration_standardize.py`.
- **Not touched:** `data/`, root `/models/`, notebooks, generated outputs,
  trained artifacts, DEMs, shapefiles, GeoPackages, parquet/csv outputs,
  figures, and script behavior.
- **Validation:** stale-reference `rg` audit completed before deciding not to
  move files. Full pytest -> 583 passed, 7 warnings. `git diff --check` clean.
  No Python scripts were changed or moved, so import-smoke and targeted ruff
  were not applicable.
- **Next step:** Data cleanup dry-run remains the next backlog item; it must be
  report-only.

## 2026-06-03 — Regime Earth feature-builder wrapper

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `2798251`
- **Task:** Convert only `scripts/build_earth_features_regime.py` into a
  wrapper around package regime helpers; leave patch builder, training/eval
  scripts, pipeline scripts, real data, generated outputs, and artifacts
  untouched.
- **Script converted:** `scripts/build_earth_features_regime.py` now keeps the
  CLI/log setup and delegates to
  `channel_heads.training.regime.build_regime_feature_dataset`.
- **Package helpers added:** `channel_heads.training.regime` now owns
  `regime_feature_paths()`, `regime_basin_feature_cache_path()`,
  `regime_prefilter_distance()`, `assemble_regime_master_dataset()`, and
  `build_regime_feature_dataset()`. The existing package `process_basin()` and
  `resolve_regime_basins()` remain the default implementation path.
- **Tests added:** `tests/test_training_regime.py` now covers output path
  naming, cache load vs. force recompute behavior, no-master behavior,
  hard-negative filter parameter forwarding, stratified subsample parameter
  forwarding and largest-basin adjustment, prefilter distance formula, empty
  basin resolution error behavior, and script import/default-argument smoke.
  All tests use temp paths and mocked processors; no real data writes.
- **Preserved:** regime presets, DEM discovery / basin resolution,
  threshold km²-to-cells conversion, DEM z-threshold masking, `CONNECTIVITY=8`,
  pruning order, `min_basin_px=500`, `max_outlets=40` and `0 -> None`,
  per-outlet prefilter distance
  `max(regime.min_prefilter_px, 2 * sqrt(outlet_basin_px))`,
  `coupling_n_workers`, analyzer calls, hard-negative parameters
  (`max_L_ratio=3.0`, `max_dist_ratio=5.0`), stratified negative subsampling
  (`target_ratio=3.0`, seed `42`, basin-proportional targets, largest-basin
  adjustment, final sort), cache path/load policy, stats CSV path, master CSV
  path, and script logging surface.
- **Validation:** import smoke for `scripts/build_earth_features_regime.py`
  passed. Focused pytest (`tests/test_training_regime.py
  tests/test_geometric_analysis.py`) -> 129 passed. Full pytest -> 583 passed,
  7 warnings. Targeted ruff on changed source/test files passed. `git diff
  --check` clean.
- **Next step:** No remaining Earth/regime training/eval/patch/feature script
  conversion is pending. Future work should be explicit and bounded, e.g. data
  cleanup dry-run or notebook import updates only if requested.

## 2026-06-03 — Regime CNN patch-builder wrapper

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `3c4ceb8`
- **Task:** Convert only `scripts/build_cnn_patches_regime.py` to use the
  package regime/rasterization helpers; leave
  `scripts/build_earth_features_regime.py`, patch/feature data, and pipeline
  scripts untouched.
- **Script converted:** `scripts/build_cnn_patches_regime.py` is now a CLI
  wrapper around `channel_heads.training.regime.build_regime_patch_dataset`.
- **Package helper added:** `channel_heads.training.regime` now owns
  `regime_patch_paths()` and `build_regime_patch_dataset()`, which derive the
  regime master/output/manifest paths, create the regime raster output root,
  call canonical `rasterization.earth_batch.precompute_raster_dataset` with the
  package regime stream loader, write the manifest, and emit the same
  raster-status / zero-basin logging.
- **Tests added:** `tests/test_training_regime.py` now covers regime patch
  output root, manifest path, default `target_size=128`, `threshold=0`
  forwarding, script import/default smoke, and mocked stream-loader
  threshold-to-cells / z-mask / pruning behavior without real data writes.
- **Left unchanged:** `scripts/build_earth_features_regime.py`,
  `scripts/run_mars_combined_regime.py`, `scripts/run_regime_pipeline.sh`,
  `data/`, root `/models/`, notebooks, generated outputs, DEMs, and trained
  artifacts.
- **Preserved:** CLI arguments/defaults, output root
  `RESULTS_DIR / f"_rasters_{regime.name}"`, manifest path
  `RESULTS_DIR / f"raster_manifest_{regime.name}.csv"`, target size default,
  regime threshold km²-to-cells conversion, DEM z-threshold masking, pruning
  order (`pre_remove_max_order` then `order_gap_to_prune`), and
  `precompute_raster_dataset(..., threshold=0)`.
- **Validation:** import smoke for `scripts/build_cnn_patches_regime.py`
  passed. Focused pytest (`tests/test_training_regime.py tests/test_rasterizer.py
  tests/test_mars_patches.py`) -> 61 passed. Full pytest -> 575 passed,
  7 warnings. Targeted ruff on changed source/test files passed. `git diff
  --check` clean.
- **Next step:** Convert `scripts/build_earth_features_regime.py` as its own
  bounded slice; keep real data, trained artifacts, and pipeline scripts
  untouched unless explicitly requested.

## 2026-06-03 — Combined-XGBoost training script wrappers

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `7294024`
- **Task:** Convert only the combined-XGBoost training scripts to call the
  package training helpers; leave Earth/regime feature and patch builders
  untouched.
- **Scripts converted:**
  - `scripts/train_combined_xgb_phase6b.py` — uses
    `training.xgboost.extract_emb_and_logit_strict`,
    `train_combined_variant`, and the feature/threshold file writers. Keeps the
    Phase 6B dataset rebuild, GroupShuffleSplit, variant loop, artifact names,
    metrics CSV, and logging surface in the script.
  - `scripts/train_combined_xgb_regime.py` — uses
    `training.xgboost.extract_emb_strict`, `train_combined_variant`, and the
    feature/threshold file writers. Keeps regime CLI, manifest/model path
    checks, dataset write, GroupShuffleSplit, artifact names, metrics CSV, and
    logging surface in the script.
- **Left unchanged:** `scripts/build_earth_features_regime.py`,
  `scripts/build_cnn_patches_regime.py`, `scripts/run_mars_combined_regime.py`,
  and `scripts/run_regime_pipeline.sh`.
- **Preserved:** CLI/defaults, artifact/model/dataset/metrics/feature-column/
  threshold paths, feature order, XGBoost hyperparameters, strict CNN loading,
  threshold policy (`max_precision_at_recall>=0.50`, fallback 0.5), metrics
  schema/order, and script log messages.
- **Validation:** import smoke for both changed scripts passed. Focused pytest
  (`tests/test_training_xgboost.py tests/test_training_datasets.py
  tests/test_cnn_consolidation.py`) -> 58 passed, 1 warning. Full pytest ->
  571 passed, 7 warnings. `git diff --check` clean. Targeted ruff on the
  changed scripts passed.
- **Next step:** Convert the remaining Earth/regime feature and patch builders
  in a separate slice only if behavior can be kept pinned; do not touch
  generated data or trained artifacts.

## 2026-06-03 — Low-risk Earth/regime script wrappers

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `442d04d`
- **Task:** Convert only the low-risk Earth/regime scripts to call the package
  foundations added in `442d04d`; leave feature/patch builders and pipeline
  scripts untouched.
- **Scripts converted:**
  - `scripts/train_cnn_baseline.py` — uses
    `training.datasets.load_valid_raster_manifest`, `cv_pool`, and
    `deterministic_val_split`.
  - `scripts/train_cnn_regime.py` — same helper conversion for regime manifests
    and Taiwan holdout validation split.
  - `scripts/train_cnn_multiseed.py` — uses package manifest/CV helpers and the
    per-seed deterministic split helper while preserving torch/numpy seeding and
    best-seed selection.
  - `scripts/eval_lobo_cv.py` — wraps `channel_heads.eval.lobo` for dataset
    paths and LOBO report; retains script-level constants and the `lobo` alias.
- **Left unchanged:** `scripts/build_earth_features_regime.py`,
  `scripts/build_cnn_patches_regime.py`, `scripts/run_mars_combined_regime.py`,
  `scripts/run_regime_pipeline.sh`, `scripts/train_combined_xgb_phase6b.py`,
  and `scripts/train_combined_xgb_regime.py`.
- **Preserved:** CLI arguments/defaults, CNN training defaults, Taiwan holdout
  behavior, validation split formula/index order, manifest filtering,
  artifact/model/history paths and filenames, multi-seed seed loop/best-model
  selection, LOBO dataset map, fold-AUC skip behavior, LOBO output schema/path,
  and logging/printed output.
- **Validation:** import smoke for all four changed scripts passed. Focused
  pytest (`tests/test_training_datasets.py tests/test_training_regime.py
  tests/test_eval_lobo.py tests/test_cnn_consolidation.py`) -> 55 passed,
  1 warning. Full pytest -> 571 passed, 7 warnings. `git diff --check` clean.
  Targeted ruff on changed scripts -> clean.
- **Next step:** Convert the remaining transitional scripts in a separate
  slice, starting with the combined-XGB trainers or the regime patch/feature
  builders only if behavior can be pinned without changing scientific output.

## 2026-06-03 — Earth/regime package foundations recovery

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `ad089d8`
- **Task:** Recover the interrupted Earth/regime training extraction to a safe,
  commit-ready state. Keep the self-contained package foundations and tests;
  revert incomplete script edits.
- **Files created:**
  - `channel_heads/training/datasets.py` — raster-manifest filtering, Taiwan CV
    pool, deterministic validation split, and frozen Earth combined feature
    constants/order.
  - `channel_heads/training/xgboost.py` — strict CNN embedding/logit extraction,
    frozen combined-XGBoost config, scale-pos-weight helper, PR-threshold policy,
    combined-variant metrics routine, and feature/threshold file writers.
  - `channel_heads/training/regime.py` — regime feature-build helpers,
    stratified negative subsampling, DEM-to-basin resolution, and regime stream
    loader factory.
  - `channel_heads/eval/lobo.py` — LOBO geom+CNN-embedding XGBoost diagnostic
    dataset map and report helper.
  - Focused tests for each new module.
- **Files reverted before validation:** `scripts/build_earth_features_regime.py`,
  `scripts/eval_lobo_cv.py`, `scripts/train_cnn_baseline.py`,
  `scripts/train_cnn_multiseed.py`, `scripts/train_cnn_regime.py`,
  `scripts/train_combined_xgb_phase6b.py`, and
  `scripts/train_combined_xgb_regime.py`.
- **Not changed:** no script repointing/wrapper conversion in this recovery
  slice; scripts remain transitional. No data, root `/models`, notebooks,
  generated outputs, DEMs/shapefiles/GeoPackages, or trained artifacts touched.
- **Validation:** current Python focused tests -> 19 passed, 2 skipped
  (`xgboost` unavailable there); current Python full suite -> 536 passed,
  19 skipped, 1 warning. `ch-heads` focused tests -> 37 passed, 1 warning.
  `ch-heads` full suite -> 571 passed, 7 warnings. Import smoke passed for
  `channel_heads.training.datasets`, `channel_heads.training.xgboost`,
  `channel_heads.training.regime`, and `channel_heads.eval.lobo`.
  `git diff --check` clean. Ruff unavailable (`python -m ruff` no module,
  `ruff` not on PATH).
- **Next step:** Repoint/convert the transitional Earth/regime scripts as a
  separate slice using the package foundations above; stop if any CLI,
  threshold, feature order, artifact path, schema, class count, or numeric
  behavior would change.

## 2026-06-03 — Regime inference: move into models/regime.py (inference/regime.py → shim)

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `3a84cdc`
- **Task:** Make `channel_heads.models.regime` the canonical home for the
  regime-specific Mars embedding-attach / inference helpers; reduce
  `channel_heads/inference/regime.py` to a compatibility shim. Behavior-preserving.
- **Files created:**
  - `channel_heads/models/regime.py` — `extract_regime_embeddings`,
    `attach_regime_embeddings`, `DEFAULT_BATCH_SIZE` (moved verbatim). Imports
    `DEFAULT_EMBEDDING_DIM` / `OutletCNN` / `OutletPairDataset` from the canonical
    `channel_heads.models.cnn` (was the `cnn_model` shim). Strict
    `load_state_dict(strict=True)`, eval/no-augment forward pass, patch-index
    `patch_status == "ok"` filtering, absolute/project-relative patch path
    resolution, missing-patch dropping, `emb_0..emb_N` overwrite, finite checks,
    and the returned schema (drops `patch_path_abs`) are unchanged.
- **Files updated:**
  - `channel_heads/inference/regime.py` — reduced to a pure re-export shim
    (`extract_regime_embeddings`, `attach_regime_embeddings`,
    `DEFAULT_BATCH_SIZE`, `DEFAULT_EMBEDDING_DIM`).
  - `channel_heads/models/__init__.py` — added `regime` to the torch-optional
    block, re-exported `attach_regime_embeddings` / `extract_regime_embeddings`,
    and updated the docstring + `__all__`.
  - `scripts/run_mars_combined_regime.py` — repointed the import to
    `channel_heads.models.regime` (clearly safe; ruff reordered the import
    block); updated the inline comment.
  - `tests/test_inference_regime.py` — behavior tests now monkeypatch the
    canonical `models.regime.extract_regime_embeddings`; added
    `TestRegimeShimIdentity` (old path is new path; `models` package exposes the
    canonical objects) and a torch-guarded `TestStrictStateDictLoad` (valid
    state dict → finite (n, dim) matrix; mismatched embedding-head state dict →
    `RuntimeError` under `strict=True`), using synthetic `.npy` patches + temp
    paths only.
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not changed:** strict-vs-lenient CNN load split (the lenient
  `models.cnn_features.extract_embeddings` is untouched and NOT merged), Mars/
  regime scientific behavior, feature order, thresholds, model/patch paths,
  output schema, Earth/regime training scripts, `train_*` scripts, `rasterizer.py`,
  `geometric_analysis.py`, `data/`, root `/models/`, notebooks, generated
  outputs, trained artifacts.
- **Validation:** `tests/test_inference_regime.py` → **6 passed**;
  `tests/test_mars_combined.py tests/test_mars_embeddings.py` → **10 passed**;
  full pytest → **534 passed, 7 warnings** (was 530; +4 regime tests). Import
  smoke confirms shim objects are identical to canonical and the script imports
  cleanly. `git diff --check` clean. `ruff check` clean on all touched files
  (the script import block was auto-reordered by `ruff --fix`).
- **Risks:** Low. Implementation moved byte-for-byte; only the import source for
  the CNN classes (shim → canonical, identical objects) and the logger name
  (`__name__`, cosmetic) changed. The four divergent forward-pass extractors
  were deliberately NOT merged.
- **Next step:** Per `AGENT_AUDIT_EARTH_REGIME.md`, the remaining regime/training
  consolidation (`training/regime.py`, `training/xgboost.py`, `eval/lobo.py`,
  etc.) is still future work behind behavior-pinning tests; or the backlog data
  cleanup dry-run (report-only).

## 2026-06-03 — Raster R3: move Earth raster batch precompute

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `c086259`
- **Task:** Move Earth batch raster precompute into the rasterization package
  after R2, preserving output paths, filenames, debug patch behavior,
  status/error strings, QA semantics, and DataFrame columns exactly.
- **Files created:**
  - `channel_heads/rasterization/earth_batch.py` — canonical home for
    `precompute_raster_dataset`, with an internal injectable helper used by the
    legacy shim.
- **Files updated:**
  - `channel_heads/rasterizer.py` — now delegates `precompute_raster_dataset`
    to `rasterization.earth_batch` while passing the legacy module globals
    through, preserving old import and monkeypatch behavior.
  - `channel_heads/rasterization/patches.py` and
    `channel_heads/rasterization/__init__.py` — re-export the canonical package
    batch precompute function and expose the new `earth_batch` module.
  - `tests/test_rasterizer.py` — updates the public-surface assertion for the
    package canonical batch function while keeping legacy wrapper behavior
    covered by existing batch QA tests.
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not changed:** single-patch raster pixels, QA flag definitions, Mars patch
  rendering behavior, regime patch scripts, notebooks, `data/`, root
  `/models/`, trained artifacts, generated outputs.
- **Validation:** import checks passed for `channel_heads.rasterizer`,
  `channel_heads.rasterization`, legacy `rasterize_outlet_pair` /
  `precompute_raster_dataset`, and package `rasterize_outlet_pair` /
  `precompute_raster_dataset`. Targeted pytest (`tests/test_rasterizer.py
  tests/test_mars_patches.py tests/test_cnn_model.py
  tests/test_cnn_consolidation.py`) → **101 passed, 1 warning**. Full pytest →
  **513 passed, 17 skipped, 1 warning**. `git diff --check` clean. `ruff` was
  requested but unavailable (`python -m ruff` reported no installed module and
  no `ruff` binary was on `PATH`).
- **Risks:** Low. The implementation moved structurally; loader signature,
  basin config lookup, output directory/filename, debug saves, `raster_path`
  gating, status/error strings, and output columns are preserved. Package
  callers now get the canonical function; legacy `rasterizer` callers get a
  wrapper to keep historical monkeypatch behavior.
- **Next step:** Rasterization extraction is complete for the audited
  `rasterizer.py` implementation. Recommended next bounded task is the backlog
  data cleanup dry-run report, or an explicit `inference/regime.py`
  consolidation slice from `AGENT_AUDIT_EARTH_REGIME.md`.

## 2026-06-03 — Raster R2: move Earth patch rasterization into package

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `b70b817`
- **Task:** Move only the Earth single-patch rasterization implementation into
  the rasterization package. Leave Earth batch precompute for R3.
- **Files created:**
  - `channel_heads/rasterization/earth_patches.py` — canonical home for
    `bresenham_line`, `_project_to_target_grid`, `_draw_path_on_target_grid`,
    `_draw_edges_on_target_grid`, `_component_count`, `raster_quality_flags`,
    `_get_rc`, `_compute_rotation_angle`, `_rotate_coordinates`, and
    `rasterize_outlet_pair` (moved behavior-preserving).
- **Files updated:**
  - `channel_heads/rasterizer.py` — now re-exports the moved single-patch
    symbols and keeps `precompute_raster_dataset` in place. The pinned private
    `_trace_full_path` compatibility name remains available from this module.
  - `channel_heads/rasterization/patches.py` and
    `channel_heads/rasterization/__init__.py` — re-export the canonical
    single-patch implementation while continuing to expose batch precompute.
  - `channel_heads/rasterization/drawing.py` — imports `bresenham_line` from the
    canonical Earth patch module.
  - `tests/test_rasterizer.py` — pins old/new/package import identity for the
    moved single-patch helpers.
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not changed:** `precompute_raster_dataset` behavior/location, Mars patch
  rendering behavior, regime patch scripts, notebooks, `data/`, root
  `/models/`, trained artifacts, generated outputs.
- **Validation:** import checks passed for `channel_heads.rasterizer`,
  `channel_heads.rasterization`, `channel_heads.rasterization.patches`,
  `channel_heads.rasterization.schema`, `channel_heads.rasterization.earth_patches`,
  legacy `rasterize_outlet_pair` / `precompute_raster_dataset`, and package
  `rasterize_outlet_pair`. Targeted pytest (`tests/test_rasterizer.py
  tests/test_mars_patches.py tests/test_cnn_model.py
  tests/test_cnn_consolidation.py`) → **101 passed, 1 warning**. Full pytest →
  **513 passed, 17 skipped, 1 warning**. `git diff --check` clean. `ruff` was
  requested but unavailable (`python -m ruff` reported no installed module and
  no `ruff` binary was on `PATH`).
- **Risks:** Low. The single-patch implementation moved without changing the
  direct-final-grid projection, dtype, target size, padding, draw order, branch
  protection, confluence overwrite, QA flag semantics, or public imports.
- **Next step:** Raster R3 — move `precompute_raster_dataset` into the
  rasterization package while preserving output paths, filenames, status/error
  strings, debug patch behavior, and DataFrame columns exactly.

## 2026-06-03 — Raster R1: extract shared raster schema constants

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `5f4741e`
- **Task:** First rasterization cleanup slice. Make shared 5-class raster
  schema constants package-resident without moving Earth rasterization logic.
- **Files created:**
  - `channel_heads/rasterization/schema.py` — canonical `BACKGROUND`,
    `BRANCH_A`, `BRANCH_B`, `OTHER_STREAMS`, `CONFLUENCE_MARKER`,
    `NUM_CLASSES`, `CLASS_LABELS`, and `PATCH_FLAG_COLUMNS`.
- **Files updated:**
  - `channel_heads/rasterizer.py` — imports class constants from schema but
    still owns Earth `bresenham_line`, direct-final-grid drawing,
    `raster_quality_flags`, `rasterize_outlet_pair`, and
    `precompute_raster_dataset`.
  - `channel_heads/rasterization/patches.py` and
    `channel_heads/rasterization/__init__.py` — re-export schema constants while
    continuing to re-export the Earth implementation from `rasterizer.py`.
  - `channel_heads/rasterization/manifest.py` — imports `PATCH_FLAG_COLUMNS`
    from schema.
  - `channel_heads/rasterization/mars_patches.py`,
    `channel_heads/models/mars_combined.py`, `channel_heads/models/cnn.py`,
    `channel_heads/training/cnn.py`, and `channel_heads/__init__.py` —
    constant-only imports now prefer schema where safe.
  - `tests/test_rasterizer.py` — pins schema / old-path / package-surface
    constant and label/flag-column compatibility.
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not changed:** Earth single-patch rasterization implementation,
  `precompute_raster_dataset`, Mars patch rendering behavior, regime patch
  scripts, notebooks, `data/`, root `/models/`, trained artifacts, generated
  outputs.
- **Validation:** import checks passed for `channel_heads.rasterizer`,
  `channel_heads.rasterization`, `channel_heads.rasterization.patches`,
  `channel_heads.rasterization.schema`, legacy `rasterize_outlet_pair` /
  `precompute_raster_dataset`, and package `rasterize_outlet_pair`. Targeted
  pytest (`tests/test_rasterizer.py tests/test_mars_patches.py
  tests/test_cnn_model.py tests/test_cnn_consolidation.py`) → **101 passed,
  1 warning**. Full pytest → **513 passed, 17 skipped, 1 warning**.
  `git diff --check` clean. `ruff` was requested but unavailable
  (`python -m ruff` reported no installed module and no `ruff` binary was on
  `PATH`).
- **Risks:** Low. Constants/labels/flag-column ownership changed only; integer
  values, class count, QA flag names, patch dtype/size, draw order, and output
  behavior are unchanged.
- **Next step:** Raster R2 — move Earth single-patch rasterization helpers and
  `rasterize_outlet_pair` into `channel_heads/rasterization/earth_patches.py`;
  do not move `precompute_raster_dataset` until R3.

## 2026-06-03 — Slice 14: repoint internal imports off the geometric_analysis shim

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `ca8220d`
- **Task:** Low-risk cleanup. Now that `geometric_analysis.py` is a pure
  re-export shim, repoint internal package code to import from the canonical
  modules. Keep the shim intact (no exports removed).
- **Files updated:**
  - `channel_heads/__init__.py` — the single `from .geometric_analysis import
    (...)` block was replaced with direct imports from the canonical modules:
    `features.asymmetry` (`LengthwiseAsymmetryAnalyzer`, `PairAsymmetryResult`,
    `compute_asymmetry_statistics`, `compute_delta_L`,
    `merge_coupling_and_asymmetry`), `features.earth_enrichment`
    (`add_geometric_features_to_csv`), `features.earth_geometry`
    (`GEOM_FEATURE_COLS`, `GeometricFeaturesAnalyzer`, `PairGeometricResult`,
    `merge_geometric_features`), `training.labeling` (`filter_hard_negatives`,
    `generate_labeled_dataset`), and `units` (`compute_meters_per_degree`,
    `compute_pixel_size_meters`). Public API and `__all__` unchanged.
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not changed:**
  - `channel_heads/geometric_analysis.py` — left as the compatibility shim;
    still re-exports every symbol (nothing removed).
  - `scripts/build_earth_features_regime.py` — it imports via the top-level
    `channel_heads` public API (`from channel_heads import ...`), **not** from
    `geometric_analysis` directly, so it needed no change; it now resolves
    transitively to the canonical modules.
  - Tests — the existing `Test*Extraction` parity classes already assert both
    canonical-module identity and top-level / shim identity, so they were kept
    as-is (shim tests preserved per the slice brief).
  - rasterizer implementation, notebooks, `data/`, root `/models/`, trained
    artifacts, generated outputs.
- **Validation:** import smoke — top-level `channel_heads.X`, the canonical
  `features.*` / `training.labeling` / `units` objects, and the
  `geometric_analysis` shim re-exports are all the *same* objects. Targeted
  pytest (`test_geometric_analysis.py` + `test_rasterizer.py`) → **154 passed,
  1 warning**; full pytest → **513 passed, 17 skipped, 1 warning**.
  `git diff --check` and `git diff --cached --check` clean. `ruff` was requested
  but unavailable in the recovery environment (`python -m ruff` reported no
  installed module and no `ruff` binary was on `PATH`).
- **Risks:** Very low. Import-source change only; objects are identical across
  paths and `__all__`/public API are unchanged. The shim remains fully
  functional for notebooks/scripts/user code.
- **Next step:** Optional — backlog data cleanup dry-run (report-only), or the
  `inference/regime.py` consolidation per `AGENT_AUDIT_EARTH_REGIME.md`.

## 2026-06-03 — Slice 13: extract Earth enrichment helpers (geometric_analysis → shim)

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `8089216`
- **Task:** Move the CSV-enrichment workflow and default Earth stream loader out
  of `geometric_analysis.py` into `channel_heads/features/earth_enrichment.py`,
  completing the `geometric_analysis.py` split (it becomes a pure re-export
  shim). Fifth/final slice.
- **Files created:**
  - `channel_heads/features/earth_enrichment.py` — `default_stream_loader`,
    `_build_pairs_at_confluence`, `_build_asymmetry_df`, `_add_missing_stream_qc`,
    `add_geometric_features_to_csv`, `_add_geometric_features_cli`, and the
    `StreamLoaderFunc` alias (moved verbatim). Imports `GEOM_FEATURE_COLS` /
    `GeometricFeaturesAnalyzer` from `features.earth_geometry`, `resolve_dem_path`
    from `config`, `_normalize_pair` from `pairing.earth`. Per-module
    `get_logger(__name__)`; has its own `__main__` guard too.
- **Files updated:**
  - `channel_heads/geometric_analysis.py` — **rewritten as a pure re-export
    shim.** It now only imports/re-exports the asymmetry, earth_geometry,
    earth_paths, geometry, training.labeling, earth_enrichment, and units
    symbols (underscored helpers via redundant-alias / `# noqa: F401`), keeps the
    historical type aliases, the `__all__` public list (with `GEOM_FEATURE_COLS`
    added), and the `if __name__ == "__main__": _add_geometric_features_cli()`
    CLI. All prior imports specific to the moved code (`argparse`, `logging`,
    `Path`, `Any`, `numpy`, `pandas`, `resolve_dem_path`, `_normalize_pair`,
    `get_logger`/`logger`, `Callable`/`StreamLoaderFunc` definition) were removed.
  - `tests/test_geometric_analysis.py` — added `TestEnrichmentExtraction`
    (legacy/canonical identity for the public + private enrichment symbols, the
    CLI, the `StreamLoaderFunc` alias, and the top-level
    `add_geometric_features_to_csv`).
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not touched:** scripts (`build_earth_features_regime.py`,
  `build_cnn_patches_regime.py` still reference the helpers via the shim /
  docstrings), notebooks, `data/`, root `/models/`, generated outputs, trained
  artifacts.
- **Validation:** targeted pytest → **154 passed, 1 warning**; full pytest →
  **530 passed, 7 warnings**. `ruff check` clean on `features/earth_enrichment.py`
  and `geometric_analysis.py` (ruff `--fix` organized the re-export blocks).
  `git diff --check` clean. CLI smoke: `python -m channel_heads.geometric_analysis
  --help` prints usage (the runpy double-import RuntimeWarning is the standard,
  pre-existing `-m`-on-imported-module notice). Regime-script import surface
  verified to resolve to the canonical objects.
- **Risks:** Low. Enrichment moved verbatim — CSV schema (overlap_px drop,
  head/L swapping, missing basin/lat=36.0/z_th=0.0 defaults, `missing_stream`
  flags, default threshold 300, write-only-when-`output_csv`) unchanged. Only
  the logger name differs (no test depends on it).
- **Next step:** `geometric_analysis.py` split complete. Optional follow-ups:
  thin the shim / repoint `__init__` + regime script to canonical modules; the
  backlog's data cleanup dry-run; or the `inference/regime.py` consolidation.

## 2026-06-03 — Slice 12: extract labeling and hard-negative filters

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `ce54016`
- **Task:** Move the labeled-dataset assembly and hard-negative filtering out of
  `geometric_analysis.py` into `channel_heads/training/labeling.py`, preserving
  behavior exactly. Fourth slice of the `geometric_analysis.py` split.
- **Files created:**
  - `channel_heads/training/labeling.py` — `generate_labeled_dataset`,
    `filter_hard_negatives`, and the private stream-crossing helpers
    `_line_crosses_stream` / `_build_stream_mask` (moved verbatim). Imports
    `GEOM_FEATURE_COLS` from `features.earth_geometry` and `line_pixels` from
    `stream_utils`. No logging in these functions.
- **Files updated:**
  - `channel_heads/geometric_analysis.py` — removed the moved definitions;
    imports the four symbols from `channel_heads.training.labeling` and
    re-exports them (privates via `# noqa: F401`). Dropped the now-unused
    `import numpy.typing as npt` and `from .stream_utils import line_pixels`.
  - `tests/test_geometric_analysis.py` — added `TestLabelingExtraction`
    (public symbols identical across `geometric_analysis` / `training.labeling`
    / top-level `channel_heads`; private helpers re-exported from the canonical
    module).
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not touched:** CSV enrichment, scripts (`build_earth_features_regime.py`
  still imports `generate_labeled_dataset` / `filter_hard_negatives` via the
  geometric_analysis re-export), notebooks, `data/`, root `/models/`, generated
  outputs, trained artifacts.
- **Validation:** targeted pytest → **152 passed, 1 warning**; full pytest →
  **528 passed, 7 warnings**. `ruff check` clean on `training/labeling.py` and
  `geometric_analysis.py` (ruff `--fix` organized labeling's import block).
  `git diff --check` clean.
- **Risks:** Low. Functions moved verbatim — per-group recursion, NaN-keep
  semantics, positive-median L / distance thresholds, optional stream-crossing
  filter (negatives only), conservative keep-on-error, and final sort keys all
  unchanged.
- **Next step:** Slice 13 — extract CSV enrichment / Earth stream loading into
  `channel_heads/features/earth_enrichment.py`; `geometric_analysis.py` becomes
  a pure re-export shim.

## 2026-06-03 — Slice 11: extract Earth geometry analyzer

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `19dcdcc`
- **Task:** Move the Earth geometric-feature analyzer out of
  `geometric_analysis.py` into `channel_heads/features/earth_geometry.py`,
  preserving behavior exactly. Third slice of the `geometric_analysis.py` split.
- **Files created:**
  - `channel_heads/features/earth_geometry.py` — `GEOM_FEATURE_COLS`,
    `DEFAULT_DIRECTION_SAMPLE_DISTANCE_M`, `PairGeometricResult`,
    `GeometricFeaturesAnalyzer`, `merge_geometric_features` (moved verbatim).
    Imports path helpers from `features.earth_paths`, pure math from
    `features.geometry`, `_build_parents_from_stream` / `_normalize_pair` from
    `pairing.earth`, and `compute_pixel_size_meters` from `units`. Per-module
    `get_logger(__name__)`.
- **Files updated:**
  - `channel_heads/geometric_analysis.py` — removed the moved definitions and
    the now-redundant `dataclass` import; imports the five symbols from
    `features.earth_geometry` and re-exports them. The path-helper / geometry
    imports it keeps for the historical import surface are now re-export-only
    (redundant-alias `X as X` / `# noqa: F401`); `GEOM_FEATURE_COLS` and
    `GeometricFeaturesAnalyzer` are still used internally by the labeling /
    enrichment helpers that remain. `_build_parents_from_stream` import dropped
    (only the analyzer used it; tests import it from `pairing.earth` directly).
  - `tests/test_geometric_analysis.py` — repointed the skip-warning test to
    `patch("channel_heads.features.earth_geometry.logger")` (the analyzer's new
    home), and added `TestEarthGeometryExtraction` (old/new/top-level identity,
    `GEOM_FEATURE_COLS` order, `DEFAULT_DIRECTION_SAMPLE_DISTANCE_M` re-export).
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not touched:** labeling / hard-negative filtering, CSV enrichment,
  stream-crossing helpers, scripts, notebooks, `data/`, root `/models/`,
  generated outputs, trained artifacts.
- **Validation:** targeted pytest → **150 passed, 1 warning**; full pytest →
  **526 passed, 7 warnings**. `ruff check` clean on `features/earth_geometry.py`
  and `geometric_analysis.py` (ruff `--fix` organized the re-export import
  block). `git diff --check` clean.
- **Risks:** Low. Analyzer moved verbatim — feature-column order, x=col / y=-row
  convention, branch-parent Strahler logic, proximity profile, QC flag strings,
  and `evaluate_pairs_for_outlet` skip-warning all unchanged. Only the logger
  name differs (test repointed accordingly).
- **Next step:** Slice 12 — extract labeling / hard-negative filters into
  `channel_heads/training/labeling.py`.

## 2026-06-03 — Slice 10: extract asymmetry helpers

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `c566f72`
- **Task:** Move the Earth lengthwise-asymmetry logic out of
  `geometric_analysis.py` into `channel_heads/features/asymmetry.py`, preserving
  behavior exactly (merge-and-consolidate). Second slice of the
  `geometric_analysis.py` split.
- **Files created:**
  - `channel_heads/features/asymmetry.py` — `PairAsymmetryResult`,
    `compute_delta_L`, `LengthwiseAsymmetryAnalyzer`,
    `compute_asymmetry_statistics`, `merge_coupling_and_asymmetry` (moved
    verbatim). Imports `_detect_cellsize` from `features.earth_paths` and
    `compute_meters_per_degree` from `units`; logger is per-module
    `get_logger(__name__)` (warnings emitted identically; no test patches the
    asymmetry logger).
- **Files updated:**
  - `channel_heads/geometric_analysis.py` — removed the moved definitions; now
    imports the five symbols from `channel_heads.features.asymmetry` and
    re-exports them. `compute_meters_per_degree` kept as a re-export (in
    `__all__`). No change to the geometry analyzer, labeling, or enrichment.
  - `tests/test_geometric_analysis.py` — added `TestAsymmetryExtraction`
    pinning old (`geometric_analysis`) / new (`features.asymmetry`) / top-level
    (`channel_heads`) import identity for all five moved symbols.
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md`.
- **Not touched:** geometry analyzer, labeling / hard-negative filtering, CSV
  enrichment, scripts, notebooks, `data/`, root `/models/`, generated outputs,
  trained artifacts.
- **Validation:** targeted pytest → **147 passed, 1 warning**; full pytest →
  **523 passed, 7 warnings**. `ruff check` clean on
  `features/asymmetry.py` and `geometric_analysis.py`. `git diff --check` clean.
- **Risks:** Low. Logic moved verbatim; S1 upstream-distance conversion policy,
  negative-length warn/clamp, head-order normalization with L swapping, NaN-safe
  statistics, and merge keys all unchanged. Only the logger name differs
  (cosmetic; not asserted by any test).
- **Next step:** Slice 11 — extract the Earth geometry analyzer into
  `channel_heads/features/earth_geometry.py`.

## 2026-06-03 — Slice 9: extract Earth path helpers

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `8200a9b`
- **Task:** Move the Earth/TopoToolbox path helpers out of
  `channel_heads/geometric_analysis.py` into a canonical feature submodule,
  preserving behavior exactly (merge-and-consolidate, not a rewrite). First
  implementation slice of the `geometric_analysis.py` split (audit step 2).
- **Files created:**
  - `channel_heads/features/earth_paths.py` — canonical home for
    `_build_children_from_parents`, `_trace_path_downstream`,
    `_compute_direction_vector`, `_trace_full_path`, `_sample_path_coords`,
    `_detect_cellsize` (moved verbatim), plus the small private deps
    `_euclidean_2d` / `_normalize_vector` they require and the `EPSILON` /
    `MIN_EDGES_FOR_DIRECTION` constants. Self-contained (no
    `geometric_analysis` import) to avoid an import cycle; owns the
    `NodeId` / `ParentsList` / `ChildrenDict` / `Coord2D` type aliases it uses.
- **Files updated:**
  - `channel_heads/geometric_analysis.py` — removed the moved definitions and
    the duplicate `EPSILON` / `MIN_EDGES_FOR_DIRECTION` constants; now imports
    all of them from `channel_heads.features.earth_paths` and re-exports for
    backward compatibility (two re-export-only names marked `# noqa: F401`).
    Dropped the now-unused `import math` and `from collections import
    defaultdict`. `DEFAULT_DIRECTION_SAMPLE_DISTANCE_M`, type aliases,
    analyzers, asymmetry, labeling, hard-negative filtering, and CSV enrichment
    are unchanged.
  - `channel_heads/rasterizer.py` — repointed `from .geometric_analysis import
    _trace_full_path` to `from .features.earth_paths import _trace_full_path`
    (its `_build_children_from_parents` still comes from `pairing.earth`, a
    separate variant left untouched).
  - `tests/test_geometric_analysis.py` — added `TestEarthPathsExtraction`
    proving old/new import-path identity for all moved helpers, that
    `EPSILON` / `MIN_EDGES_FOR_DIRECTION` re-export from the canonical module
    (values 1e-10 / 3), and that `rasterizer._trace_full_path` is the canonical
    object.
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md` — Slice 9 recorded;
    next recommended task set to extracting Earth asymmetry.
- **Not touched:** asymmetry logic, `GeometricFeaturesAnalyzer`, labeling /
  hard-negative filtering, CSV enrichment, `pairing.earth`'s separate
  `_build_children_from_parents`, scripts, notebooks, `data/`, root `/models/`,
  generated outputs, trained artifacts.
- **Validation:** targeted pytest
  (`tests/test_geometric_analysis.py tests/test_rasterizer.py`) → **146 passed,
  1 warning** (was 143; +3 extraction tests). Full pytest → **522 passed, 7
  warnings**. `ruff check` clean on `features/earth_paths.py` and
  `geometric_analysis.py`; pre-existing I001 import-sort findings in
  `rasterizer.py` (1) and `tests/test_geometric_analysis.py` (3) were verified
  to exist identically on HEAD and were left as-is per CLAUDE.md (repo-wide ruff
  has known pre-existing errors; run targeted). `git diff --check` clean.
- **Risks:** Low. Helpers moved byte-for-byte; greedy child choice, unreachable
  → `[]`, sample-fraction exclusion of the confluence endpoint, weighted
  direction QC flags, cellsize detection, and the x=col / y=-row convention are
  unchanged. Old import paths and re-exported constants resolve to the same
  objects (pinned). No circular import (earth_paths is self-contained; `features`
  package was already imported by `geometric_analysis`).
- **Next step:** Extract Earth asymmetry (`PairAsymmetryResult`,
  `compute_delta_L`, `LengthwiseAsymmetryAnalyzer`, stats, asymmetry merge) into
  `channel_heads/features/asymmetry.py`, preserving the S1 unit policy.

## 2026-06-02 — Slice 8: behavior-pinning tests for geometric analysis and rasterizer

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `24820f8`
- **Task:** Test-only Slice 8 checkpoint before future extraction of
  `channel_heads/geometric_analysis.py` and `channel_heads/rasterizer.py`. No
  implementation code changed.
- **Files updated:**
  - `tests/test_geometric_analysis.py` — added behavior-pinning coverage for
    `GEOM_FEATURE_COLS` order, `compute_delta_L`, head normalization and length
    swapping, `_trace_full_path`, `_sample_path_coords`, hard-negative filtering
    including grouping and stream-crossing behavior, labeled-dataset assembly,
    and CSV enrichment edge behavior.
  - `tests/test_rasterizer.py` — added behavior-pinning coverage for exact
    5-class constants, import identity across `rasterizer` /
    `rasterization.patches` / `rasterization`, confluence marker overwrite,
    small-target direct-final-grid connectivity, and `precompute_raster_dataset`
    manifest/status/error behavior.
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md` — recorded Slice 8
    completion and set Slice 9 as the next recommended task.
- **Not touched:** implementation source, scripts, notebooks, `data/`, root
  `/models/`, generated outputs, trained artifacts, DEMs, shapefiles,
  GeoPackages, CSV/parquet outputs, figures.
- **Validation:** targeted pytest
  (`python -m pytest tests/test_geometric_analysis.py tests/test_rasterizer.py`)
  passed: **143 passed, 1 warning**. Full pytest passed:
  **502 passed, 17 skipped, 1 warning**. `ruff` was requested but unavailable
  (`python -m ruff` reported no installed module and no `ruff` binary was on
  `PATH`). `git diff --check` clean.
- **Next step:** Slice 9 — data cleanup dry-run report only; do not mutate data
  or model artifacts.

## 2026-06-02 — Slices 6/7: geometric analysis and rasterizer ownership audit

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `b35320f`
- **Task:** Combined read-only audit for Slice 6 (`geometric_analysis.py`) and
  Slice 7 (`rasterizer.py` / `channel_heads/rasterization/`). No
  implementation code changed.
- **Files created:**
  - `AGENT_AUDIT_GEOMETRIC_ANALYSIS.md` — reference map, function/class
    classification, duplication map, recommended canonical ownership,
    behavior-preservation list, proposed slices, risks, and required tests.
  - `AGENT_AUDIT_RASTERIZER.md` — reference map, module/script classification,
    duplication map, recommended rasterization ownership, frozen patch-contract
    behavior list, proposed slices, risks, and required tests.
- **Files updated:**
  - `AGENT_STATE.md` — recorded Slice 6/7 completion and set the next
    recommended task to a behavior-pinning checkpoint before further moves.
  - `AGENT_BACKLOG.md` — marked Slices 6/7 done and inserted Slice 8 for
    behavior-pinning tests/checkpoint before data cleanup.
  - `AGENT_RUN_LOG.md` — this entry.
- **Key findings:**
  - `geometric_analysis.py` is not disposable. It still owns Earth/TopoToolbox
    path traversal, lengthwise asymmetry, Earth geometric feature generation,
    labeled-dataset assembly, hard-negative filtering, default stream loading,
    and CSV enrichment. Pure feature math, Mars path helpers, Mars feature
    generation, units, and pairing helpers already have package owners.
  - `rasterizer.py` still owns the frozen Earth 5-class patch contract, direct
    final-grid drawing, QA flags, and Earth batch precompute. Mars Phase 4 is
    already package-resident in `rasterization/mars_patches.py`, while
    `rasterization/patches.py` is currently only a re-export surface.
  - Recommended next step is behavior-pinning tests before moving any of this
    logic. High-risk contracts include feature order, S1 unit behavior, QC flag
    strings, hard-negative semantics, raster class values, direct-final-grid
    connectivity, manifest schemas, regime node-ID consistency, and shim
    identity.
- **Not touched:** implementation source, scripts, notebooks, `data/`, root
  `/models/`, generated outputs, DEMs, shapefiles, GeoPackages, CSV/parquet
  outputs, figures.
- **Validation:** no pytest run (markdown/handoff audit only, per user
  instructions). `git diff --check` clean.
- **Next step:** Slice 8 — behavior-pinning checkpoint/tests before any
  ownership moves.

## 2026-06-02 — Slice 5: Earth/regime training ownership audit

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `cda8281`
- **Task:** Slice 5 — read-only audit of the remaining Earth/regime training
  scripts and package modules, including `channel_heads/inference/regime.py`.
  No implementation code changed.
- **Files created:**
  - `AGENT_AUDIT_EARTH_REGIME.md` — reference map, script classification,
    duplication map, recommended canonical ownership, proposed implementation
    slices, risks, and required tests.
- **Files updated:**
  - `AGENT_STATE.md` — recorded Slice 5 audit completion, current
    `inference/regime.py` recommendation, and Slice 6 as next task.
  - `AGENT_BACKLOG.md` — marked Slice 5 done.
  - `AGENT_RUN_LOG.md` — this entry.
- **Key findings:**
  - Earth/regime scripts are not disposable. Real logic remains in regime Earth
    feature generation, regime patch generation, baseline/regime combined-XGB
    training, Mars regime inference output writing, threshold retuning, and
    LOBO-CV diagnostics.
  - Existing package owners already cover regime presets, generic CNN
    architecture/dataset/training loop, generic lenient CNN embeddings,
    XGBoost inference helpers, grouped split/threshold primitives, and strict
    regime embedding attachment.
  - Recommended future homes: `training/datasets.py`, expanded
    `training/cnn.py`, `training/xgboost.py`, `training/regime.py`,
    `models/regime.py`, and `eval/lobo.py`. Scripts should become thin wrappers
    only after tests pin feature order, threshold policies, strict/lenient CNN
    state loading, embedding overwrite, artifact paths, LOBO behavior, and Mars
    regime output schemas.
- **Not touched:** implementation source, scripts, notebooks, `data/`, root
  `/models/`, generated outputs, DEMs, shapefiles, GeoPackages, CSV/parquet
  outputs, figures, `geometric_analysis.py`, `rasterizer.py`.
- **Validation:** no pytest run (markdown/handoff audit only, per slice
  instructions). `git diff --check` clean.
- **Next step:** Slice 6 — read-only `geometric_analysis.py` audit.

## 2026-06-02 — Slice 4: repoint scripts to canonical imports

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `36d8f3b`
- **Task:** Slice 4 — cleanup script imports after the package-first
  consolidation. Repoint thin scripts from compatibility shims to canonical
  package modules where safe, and remove script-local `pick_device()` copies.
  Behavior-preserving: no CLI changes, no threshold/model-path/hyperparameter
  changes, no forward-pass changes, no archive moves.
- **Files updated:**
  - `scripts/build_cnn_patches_regime.py`,
    `scripts/build_earth_features_regime.py`,
    `scripts/diagnostics/calibrate_stream_threshold_by_mars_dd.py`,
    `scripts/diagnostics/diag_regB_threshold.py`,
    `scripts/retune_threshold_regime.py`,
    `scripts/run_mars_combined_regime.py`,
    `scripts/train_cnn_baseline.py`, `scripts/train_cnn_multiseed.py`,
    `scripts/train_cnn_regime.py`, `scripts/train_combined_xgb_phase6b.py`,
    `scripts/train_combined_xgb_regime.py` — imports now target canonical
    modules (`io.paths`, `models.cnn`, `models.device`, `models.xgboost`,
    `training.cnn`) instead of shims where safe.
  - `scripts/train_combined_xgb_phase6b.py`,
    `scripts/train_combined_xgb_regime.py` — local `pick_device()` copies removed;
    both scripts use `channel_heads.models.device.pick_device` (same `mps` >
    `cuda` > `cpu` order).
  - `scripts/diagnostics/calibrate_stream_threshold_by_mars_dd.py`,
    `scripts/diagnostics/diag_regB_threshold.py` — tiny ruff cleanup in touched
    files only (unused imports removed; two semicolon-combined statements split).
  - `AGENT_STATE.md`, `AGENT_BACKLOG.md`, `AGENT_RUN_LOG.md` — handoff state
    updated for Slice 4 completion and Slice 5 as next task.
- **Not touched (per slice scope):** `channel_heads/` source, `data/`, root
  `/models/`, notebooks, generated outputs, trained artifacts. No scripts were
  archived because none were clearly dead in this pass.
- **Validation:** smoke-imported all 11 changed scripts with
  `conda run -n ch-heads python` (pass; only Matplotlib temp-cache warning due
  unwritable `~/.matplotlib`). Full `pytest`: **503 passed, 7 warnings**.
  `ruff check` clean on all changed scripts. `git diff --check` clean.
- **Risks:** Low. Import-only/script-local helper dedup; strict state-dict load
  in combined trainers, lenient/strict extractor split, CLI arguments, output
  paths, thresholds, feature order, and model artifacts unchanged.
- **Next step:** Slice 5 — read-only Earth/regime training audit, including
  `channel_heads/inference/regime.py`.

## 2026-06-02 — Slice 3d: move CNN training core into training package

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `8ecbd9e refactor(models): move CNN embedding helpers into models`
- **Task:** Slice 3d — move the CNN training core out of
  `channel_heads/cnn_training.py` into a new `channel_heads/training/` package;
  reduce `cnn_training.py` to a shim. Separates model definitions from training
  code. Behavior-preserving (no rewrite). Completes Slice 3.
- **Files created:**
  - `channel_heads/training/__init__.py` — training-layer package (minimal,
    torch-free; does not eager-import the torch submodule).
  - `channel_heads/training/cnn.py` — real `train_cnn` loop + `DEFAULT_*` /
    `HOLDOUT_BASIN` / `RANDOM_STATE` (moved verbatim). Logger name kept as
    `"channel_heads.cnn_training"` (logging behavior unchanged). Imports
    `OutletCNN`/`OutletPairDataset` from `channel_heads.models.cnn` and
    re-exports `pick_device` from `channel_heads.models.device`.
- **Files updated:**
  - `channel_heads/cnn_training.py` — reduced to a pure re-export shim
    (`train_cnn`, `pick_device`, six `DEFAULT_*`, `HOLDOUT_BASIN`,
    `RANDOM_STATE`).
  - `channel_heads/models/cnn.py` — repointed the lazy `__getattr__` + the
    `TYPE_CHECKING` block to source training symbols from
    `channel_heads.training.cnn` (lazy still required: `training.cnn` imports
    `models.cnn`, so an eager re-export would cycle). Docstring/comments updated.
  - `channel_heads/models/__init__.py` — docstring refreshed (cnn / cnn_features
    / embeddings / training.cnn relationships).
  - `tests/test_cnn_consolidation.py` — added `TestTrainingCoreConsolidated`
    (old/new `train_cnn` identity, module location, all `DEFAULT_*` +
    `HOLDOUT_BASIN` + `RANDOM_STATE` unchanged and identical across paths, and
    the `models.cnn` lazy re-export resolving to `training.cnn`).
- **Not touched (per slice scope):** training scripts (`scripts/train_cnn_*.py`,
  `scripts/train_combined_xgb_*.py`), `inference/regime.py`, `mars_combined.py`,
  `rasterizer.py`, `geometric_analysis.py`, trained artifacts, `data/`, root
  `/models/`, notebooks.
- **Validation:** import checks under 4 cold entry points (channel_heads,
  cnn_training shim, training.cnn, models.cnn lazy) — all resolve, no cycle,
  old `is` new for `train_cnn`/`pick_device`; defaults (60, 1e-3, 1e-4, 64, 0.3,
  12), `HOLDOUT_BASIN="taiwan"`, `RANDOM_STATE=42` identical across paths;
  `models.cnn.train_cnn is training.cnn.train_cnn`; all `pick_device` paths
  unified onto `models.device`. The three unmodified `scripts/train_cnn_*`
  trainers import cleanly via the shim. Targeted (`test_cnn_model.py`,
  `test_cnn_consolidation.py`): 51 passed. Full `pytest`: **503 passed, 7
  warnings** (was 497; +6 training-core tests). `ruff` clean on all new/changed
  files. `git diff --check` clean.
- **Risks:** Low. `train_cnn` moved byte-for-byte (loop, loss/optimizer,
  early-stopping, checkpoint, seeding, logger name all unchanged). Only
  structural change is the package relocation + repointed lazy re-export, both
  verified cycle-free.
- **Next step:** Slice 4 — scripts cleanup / archive (repoint thin scripts to
  canonical imports; archive dead scripts).

## 2026-06-02 — Slice 3c: move CNN embedding helpers into models

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `6b2e31b refactor(models): deduplicate CNN training device selection`
- **Task:** Slice 3c — move the generic/Earth CNN embedding helpers from
  `channel_heads/cnn_features.py` into the canonical models layer; reduce
  `cnn_features.py` to a shim. Behavior-preserving (no rewrite).
- **Files created:** `channel_heads/models/cnn_features.py` — real
  `extract_embeddings`, `merge_cnn_features`, `CNN_FEATURE_COLS` (moved verbatim;
  imports `OutletCNN`/`OutletPairDataset`/`DEFAULT_EMBEDDING_DIM` from
  `channel_heads.models.cnn`). Keeps the **lenient default `load_state_dict`**
  (no `strict=`, no missing/unexpected check) — deliberately distinct from the
  strict extractors in `inference/regime.py` / `models/mars_combined.py`.
- **Files updated:**
  - `channel_heads/cnn_features.py` — reduced to a pure re-export shim.
  - `channel_heads/models/embeddings.py` — repointed the embedding import to
    `channel_heads.models.cnn_features` (+ docstring); import block re-sorted by
    ruff. Mars Phase-5 orchestration logic unchanged.
  - `channel_heads/__init__.py` — repointed the CNN-embedding import to
    `from .models.cnn_features import ...` (one isort-ordered line pair).
  - `tests/test_cnn_consolidation.py` — added `TestCNNFeaturesConsolidated`
    (old/new path identity, `CNN_FEATURE_COLS` unchanged, canonical module,
    and a synthetic `extract_embeddings` smoke proving the lenient load path).
- **Not touched (per slice scope):** `inference/regime.py`, `mars_combined.py`
  forward-pass logic, `cnn_training.py`, `rasterizer.py`, `geometric_analysis.py`,
  trained artifacts, `data/`, root `/models/`, notebooks, scripts.
- **Validation:** import checks under multiple entry points (channel_heads,
  cnn_features shim, models.cnn_features, models.embeddings) — all resolve, no
  cycle, old `is` new for all three symbols and from the top-level package.
  Targeted (`test_mars_embeddings.py`, `test_cnn_consolidation.py`,
  `test_cnn_model.py`): 49 passed; with `test_cnn_features.py` added: 60 passed.
  Full `pytest`: **497 passed, 7 warnings** (was 492; +5 consolidation tests).
  `ruff` clean on the new/changed model files; the `__init__.py` I001 is
  pre-existing (verified on HEAD), test-file E402s match the existing
  `importorskip` pattern. `git diff --check` clean.
- **Risks:** Low. Embedding implementation moved byte-for-byte; lenient-load and
  DataFrame schema preserved (the existing `test_cnn_features.py` exercises the
  full extract path through the shim). The strict/lenient split is intentionally
  kept — no forward-pass extractors were merged.
- **Next step:** Slice 3d — move the CNN training core (`train_cnn` + defaults)
  into a new `channel_heads/training/` package; `cnn_training.py` → shim.

## 2026-06-02 — Slice 3b: deduplicate CNN training device selection

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `a1b5b29 refactor(models): move CNN architecture into models`
- **Task:** Slice 3b — remove the duplicate `pick_device` in
  `channel_heads/cnn_training.py`; use the canonical
  `channel_heads.models.device.pick_device`.
- **Files updated:**
  - `channel_heads/cnn_training.py` — deleted the local `pick_device` function;
    added `from channel_heads.models.device import pick_device` (re-export, so
    `from channel_heads.cnn_training import pick_device` still works). No change
    to the training loop, hyperparameters, early stopping, dataset, or
    architecture.
  - `tests/test_cnn_consolidation.py` — added `TestPickDeviceDeduplicated`
    asserting `cnn_training.pick_device is models.device.pick_device` and the
    lazy `models.cnn.pick_device is models.device.pick_device`.
- **Not touched (per slice scope):** training loop / hyperparameters / early
  stopping / dataset / architecture; `scripts/` (the
  `train_combined_xgb_*.py` inline `pick_device` copies remain — Slice 4);
  notebooks, `data/`, root `/models/`.
- **Validation:** import checks — `cnn_training.pick_device`,
  `models.cnn.pick_device`, and `models.device.pick_device` are all the *same*
  object (`is`), returns `mps` here. Targeted (`test_cnn_model.py`,
  `test_cnn_consolidation.py`, `test_inference.py`): 59 passed. Full `pytest`:
  **492 passed, 7 warnings** (was 490; +2 dedup tests). `ruff` clean on
  `cnn_training.py` (test-file E402s are the pre-existing `importorskip`
  pattern). `git diff --check` clean.
- **Risks:** Very low — pure de-duplication onto an identical canonical
  implementation (`mps` > `cuda` > `cpu`); no platform device-selection change.
- **Next step:** Slice 3c — move Earth embedding features (`cnn_features.py`)
  into the models layer with a shim.

## 2026-06-02 — Slice 3a: CNN architecture move into models/cnn.py

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `3b51fb0 docs(agents): record CNN ownership audit`
- **Task:** Slice 3a — promote `channel_heads/models/cnn.py` to the canonical
  home of the CNN architecture/dataset; reduce `channel_heads/cnn_model.py` to a
  shim. Behavior-preserving (no rewrite).
- **Files updated:**
  - `channel_heads/models/cnn.py` — now holds the real `OutletCNN`,
    `OutletPairDataset`, `encode_raster_onehot`, `DEFAULT_EMBEDDING_DIM`,
    `DEFAULT_TARGET_SIZE` (architecture copied verbatim; `NUM_CLASSES` imported
    from `channel_heads.rasterizer`). Training symbols (`train_cnn`,
    `pick_device`, `DEFAULT_*`, `HOLDOUT_BASIN`) re-exported **lazily** via module
    `__getattr__` (+ a `TYPE_CHECKING` import so linters see them) to avoid the
    `cnn_training → cnn_model(shim) → models.cnn` import cycle.
  - `channel_heads/cnn_model.py` — reduced to a pure re-export shim of the five
    architecture symbols + `NUM_CLASSES` (preserves its historical namespace).
  - `channel_heads/__init__.py` — one-line repoint of the CNN architecture import
    to `from .models.cnn import ...` (prefers canonical path; cnn_features stays
    on the shim as it is out of scope).
- **Files created:** `tests/test_cnn_consolidation.py` — proves old/new paths are
  the *same* objects, instantiation from both paths, **state-dict keys pinned**
  (artifact contract), forward/embed shapes, dataset behavior, lazy training
  re-export, and the shim re-exporting `NUM_CLASSES`.
- **Not touched (per slice scope):** `cnn_features.py`, `cnn_training.py`,
  `models/embeddings.py`, `inference/regime.py`, `mars_combined.py`,
  `rasterizer.py`, trained artifacts, `data/`, notebooks.
- **Validation:** import smoke checks under 6 entry points (channel_heads,
  cnn_model shim, cnn_training, models.cnn, cnn_features, models.embeddings) —
  all resolve, no cycle, old `is` new. Targeted tests (consolidation + cnn_model
  + cnn_features + mars_embeddings + mars_combined + inference_regime): 61
  passed. Full `pytest`: **490 passed, 7 warnings** (was 471 before; +19
  consolidation tests). `ruff` clean on `models/cnn.py` and `cnn_model.py`; the
  `__init__.py` I001 and the test E402s are pre-existing patterns (verified on
  HEAD / matching `test_cnn_model.py`'s `importorskip`). `git diff --check` clean.
- **Risks:** Low. Architecture moved byte-for-byte (state-dict keys/dims/forward
  unchanged → `strict=True` artifact load preserved). The only structural
  novelty is the lazy training re-export, required and verified to break the
  import cycle without changing the curated surface.
- **Next step:** Slice 3b — dedup `pick_device` in `cnn_training.py` onto
  `channel_heads/models/device.py`.

## 2026-06-02 — Slice 2: CNN audit (audit-only)

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `2cf2cac refactor(models): move torch device selection into models`
- **Task:** Slice 2 — read-only ownership audit of `cnn_model.py`,
  `cnn_features.py`, `cnn_training.py`, `models/cnn.py`, `models/embeddings.py`,
  `models/device.py`. Produce a merge-and-consolidate plan. No source edits.
- **Files created:** `AGENT_AUDIT_CNN.md` — full audit (reference map,
  duplication map, canonical-ownership recommendation, models/ vs future
  training/ split, shim plan, risks, proposed sub-slices 3a–3d).
- **Files updated:** `AGENT_STATE.md` (model-layer status + next task),
  this log.
- **Key findings:**
  - `cnn_model.py` = architecture SoT; `cnn_features.py` = Earth embedding path
    (lenient load); `cnn_training.py` = training core + a 3rd duplicate
    `pick_device`; `models/cnn.py` = thin re-export (good canonical name);
    `models/embeddings.py` = real Mars Phase-5 orchestration (not redundant).
  - `pick_device` triplicated (cnn_training + 2 scripts) besides canonical.
  - **Four divergent CNN forward-pass extractors** (lenient vs strict load;
    embed vs logit vs both) across `cnn_features`, `inference/regime`,
    `mars_combined`, and 2 scripts — **must not be blindly merged.**
  - Recommended: architecture → `models/cnn.py` (promote to real); Earth
    embeddings → models layer; training → future `training/cnn.py`; flat
    modules → shims. Forward-pass unification deferred to a separate
    test-guarded slice (touches out-of-scope files).
- **Tests run:** none (no code change), per slice spec.
- **Risks:** none to source/data (doc-only). Frozen `cnn_outlet_final.pt`
  architecture lock + lenient/strict load divergence flagged as the main
  hazards for Slice 3.
- **Next step:** Slice 3a — move CNN architecture into `models/cnn.py` with a
  shim (see `AGENT_AUDIT_CNN.md` §6).

## 2026-06-02 — Slice 1: device consolidation

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `90fd694 refactor(models): move XGBoost inference implementation into models`
- **Task:** Slice 1 — make `channel_heads/models/device.py` the canonical home of
  `pick_device()`; reduce `channel_heads/inference/device.py` to a shim.
- **Files created:** `channel_heads/models/device.py` (canonical `pick_device`).
- **Files updated:**
  - `channel_heads/inference/device.py` — reduced to re-export shim.
  - `channel_heads/inference/__init__.py` — docstring (still re-exports via shim).
  - `channel_heads/models/__init__.py` — expose `device` submodule + `pick_device`.
  - `channel_heads/models/embeddings.py`, `channel_heads/models/mars_combined.py`
    — repointed `pick_device` import to `channel_heads.models.device`.
  - `tests/test_inference.py` — added canonical-location/shim-identity test.
- **Tests run:** targeted `tests/test_inference.py` + `tests/test_mars_combined.py`
  (25 passed); full `pytest` → **471 passed, 7 warnings**. `ruff` clean on touched
  files (import-sort autofix applied to `models/__init__.py` + `embeddings.py`).
  `git diff --check` clean.
- **Risks:** Low — pure move + re-export, no behavior change. Verified no circular
  import (`models.device` imports torch lazily; importing `channel_heads.inference`
  first still resolves). Old imports (`channel_heads.inference.device`,
  `from channel_heads.inference import pick_device`) preserved. A separate
  `pick_device` in `cnn_training.py` (via `models/cnn.py`) was intentionally left
  untouched (CNN module, out of scope) — flagged for the CNN audit.
- **Next step:** Slice 2 — CNN audit (audit-only).

## 2026-06-02 — Agent handoff workflow setup

- **Branch:** `refactor/package-first-architecture`
- **Base commit before this entry:** `90fd694 refactor(models): move XGBoost inference implementation into models`
- **Task:** Create project agent handoff files so Claude Code or Codex can
  resume the package-first refactor without chat history.
- **Files created:**
  - `AGENT_RULES.md` — binding rules for refactor agents.
  - `AGENT_STATE.md` — current branch, canonical ownership, shims, transitional
    areas, next task.
  - `AGENT_BACKLOG.md` — slice-based backlog (device, CNN audit/consolidation,
    scripts cleanup, regime audit, geometric_analysis audit, rasterizer audit,
    data dry-run, notebooks).
  - `AGENT_RUN_LOG.md` — this log.
  - `AGENTS.md` — concise Codex instructions.
- **Files updated:**
  - `CLAUDE.md` — added an "Agent refactor workflow" pointer to the handoff files.
- **Tests run:** none required — markdown-only change. `git diff --check` clean.
- **Risks:** none to source/data; documentation only. State claims were verified
  against the repo (shims for `inference/xgb.py`, `config.py`, `plotting_utils.py`,
  `first_meet_pairs_for_outlet.py`; `inference/device.py` and `inference/regime.py`
  still present and transitional).
- **Next step:** Slice 1 — device consolidation (`inference/device.py` →
  `models/device.py` with a shim).
