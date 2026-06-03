# AGENT_BACKLOG.md — slice-based refactor backlog

Each entry is one bounded slice. Do one at a time, top to bottom unless a
dependency dictates otherwise. After completing a slice, update
`AGENT_STATE.md` and append to `AGENT_RUN_LOG.md`. Follow `AGENT_RULES.md`.

Standard tests:
- Targeted: `conda run -n ch-heads python -m pytest <files>`
- Full: `conda run -n ch-heads python -m pytest`

---

## Slice 1 — Device consolidation (DONE)

> Completed — see `AGENT_RUN_LOG.md`. `pick_device` is canonical in
> `channel_heads/models/device.py`; `channel_heads/inference/device.py` is a shim.


- **Goal:** Make `channel_heads/models/device.py` the canonical home of
  `pick_device()`; reduce `inference/device.py` to a compatibility shim.
- **Allowed files:** `channel_heads/models/device.py` (new),
  `channel_heads/models/__init__.py`, `channel_heads/inference/device.py`,
  `channel_heads/inference/__init__.py`, `channel_heads/models/mars_combined.py`,
  `channel_heads/models/embeddings.py`, `tests/test_inference.py` (and/or a new
  device test).
- **Forbidden files:** `inference/regime.py`, `inference/xgb.py`, any CNN
  module, `data/`, root `/models/`, notebooks, generated outputs.
- **Tests to run:** `tests/test_inference.py`, `tests/test_mars_combined.py`,
  then full pytest.
- **Suggested commit:** `refactor(models): move torch device selection into models`
- **Stop condition:** Stop if moving the import changes device selection on any
  platform, or if a circular import appears between `models` and `inference`.

## Slice 2 — CNN audit (audit-only) (DONE)

> Completed — see `AGENT_AUDIT_CNN.md` and `AGENT_RUN_LOG.md`. Canonical homes
> recommended (architecture → `models/cnn.py`; embeddings → models layer;
> training → future `training/`); refined sub-slices 3a–3d defined.

- **Goal:** Read-only comparison of `cnn_model.py`, `cnn_features.py`,
  `cnn_training.py` against `channel_heads/models/` (cnn, embeddings). Produce
  an ownership recommendation (which is canonical, what becomes a shim).
- **Allowed files:** none modified except appending findings to
  `AGENT_RUN_LOG.md` / a short audit note.
- **Forbidden files:** all source edits, `data/`, root `/models/`, notebooks.
- **Tests to run:** none (no code change) — optionally `import` checks.
- **Suggested commit:** `docs(agents): record CNN ownership audit`
- **Stop condition:** Do not change any CNN code in this slice. Stop after
  writing the recommendation.

## Slice 3 — CNN model consolidation (DONE)

> Refined by the Slice 2 audit into sub-slices 3a–3d (see `AGENT_AUDIT_CNN.md`
> §6), all complete — see `AGENT_RUN_LOG.md`:
> **3a** architecture → `models/cnn.py` (`cnn_model.py` → shim);
> **3b** `pick_device` dedup in `cnn_training.py` → `models/device.py`;
> **3c** Earth embeddings → `models/cnn_features.py` (`cnn_features.py` → shim);
> **3d** training core → `training/cnn.py` (`cnn_training.py` → shim).
> The four divergent forward-pass extractors were deliberately NOT merged.

- **Goal:** Move the canonical CNN implementation into `channel_heads/models/`
  per the audit; reduce the legacy module(s) to shims. 5-class patches MUST
  remain 5-class to match the Earth-trained CNN.
- **Allowed files:** the CNN modules named by the audit + `models/` targets +
  affected tests.
- **Forbidden files:** trained CNN artifacts in root `/models/`, `data/`,
  notebooks, anything outside the audit scope.
- **Tests to run:** CNN/embedding tests, Mars Phase 4/5 tests, then full pytest.
- **Suggested commit:** `refactor(models): consolidate CNN implementation into models`
- **Stop condition:** Stop if class count, patch geometry, or embedding output
  changes in any way.

## Slice 4 — Scripts cleanup / archive (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Thin scripts were repointed to canonical
> package imports where safe, and the inline `pick_device()` copies in the
> combined-XGB trainers now use `channel_heads.models.device.pick_device`. No
> scripts were archived because none were clearly dead in this pass.

- **Goal:** Repoint thin scripts to canonical package imports where safe; move
  clearly dead scripts to an `_archive/` location (do not delete). Keep scripts
  runnable.
- **Allowed files:** files under `scripts/` only.
- **Forbidden files:** `channel_heads/` source, `data/`, root `/models/`,
  notebooks, generated outputs.
- **Tests to run:** full pytest (scripts are mostly import-checked); smoke-import
  changed scripts.
- **Suggested commit:** `refactor(scripts): repoint to canonical imports; archive dead scripts`
- **Stop condition:** Stop if a script's runtime behavior or CLI surface would
  change. Archiving ≠ deleting.

## Slice 5 — Earth / regime training audit (DONE)

> Completed — see `AGENT_AUDIT_EARTH_REGIME.md` and `AGENT_RUN_LOG.md`. Audit
> only: no implementation code changed. Key recommendation is to keep
> Earth/regime scripts as wrappers later, after moving real logic into
> `training/datasets.py`, `training/cnn.py`, `training/xgboost.py`,
> `training/regime.py`, `models/regime.py`, and `eval/lobo.py` with tests
> protecting feature order, thresholds, strict/lenient CNN loading, artifact
> paths, LOBO behavior, and Mars regime embedding overwrite semantics.

- **Goal:** Read-only audit of the Earth/regime training + `inference/regime.py`
  to plan a future consolidation without changing regime behavior.
- **Allowed files:** audit note + `AGENT_RUN_LOG.md` only.
- **Forbidden files:** all regime/training source edits, `data/`, root
  `/models/`, notebooks.
- **Tests to run:** none.
- **Suggested commit:** `docs(agents): record Earth/regime training audit`
- **Stop condition:** Do not modify regime behavior. Stop after the note.

## Slice 6 — `geometric_analysis.py` audit (audit-only) (DONE)

> Completed — see `AGENT_AUDIT_GEOMETRIC_ANALYSIS.md` and
> `AGENT_RUN_LOG.md`. Audit only: no implementation code changed. Key
> recommendation is to split `geometric_analysis.py` into
> `features/earth_paths.py`, `features/asymmetry.py`,
> `features/earth_geometry.py`, `training/labeling.py`, and
> `features/earth_enrichment.py`, with `geometric_analysis.py` kept as a shim
> after tests pin unit behavior, path traversal, feature/QC column order,
> hard-negative semantics, CSV enrichment behavior, and private helper imports.

- **Goal:** Read-only review of `geometric_analysis.py` ownership/boundaries;
  recommend a target package location and any shim plan.
- **Allowed files:** audit note + `AGENT_RUN_LOG.md` only.
- **Forbidden files:** `geometric_analysis.py` (no edits), `data/`, root
  `/models/`, notebooks.
- **Tests to run:** none.
- **Suggested commit:** `docs(agents): record geometric_analysis audit`
- **Stop condition:** Audit only. No source change.

## Slice 7 — `rasterizer.py` audit (audit-only) (DONE)

> Completed — see `AGENT_AUDIT_RASTERIZER.md` and `AGENT_RUN_LOG.md`. Audit
> only: no implementation code changed. Key recommendation is to move shared
> raster constants/schema and Earth patch rasterization/precompute into
> `channel_heads/rasterization/`, keep Mars Phase 4 in
> `rasterization/mars_patches.py`, move regime patch orchestration into future
> `training/regime.py`, and leave `rasterizer.py` as a shim after behavior tests
> pin the frozen 5-class CNN patch contract.

- **Goal:** Read-only review of `rasterizer.py` vs `channel_heads/rasterization/`;
  recommend canonical ownership and shim plan.
- **Allowed files:** audit note + `AGENT_RUN_LOG.md` only.
- **Forbidden files:** `rasterizer.py` (no edits), `data/`, root `/models/`,
  notebooks.
- **Tests to run:** none.
- **Suggested commit:** `docs(agents): record rasterizer audit`
- **Stop condition:** Audit only. No source change.

## Slice 8 — Behavior-pinning checkpoint before package moves (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Added behavior-pinning tests for
> `channel_heads/geometric_analysis.py` and `channel_heads/rasterizer.py`
> without touching implementation code. The new coverage pins feature-column
> order, asymmetry math, path traversal/sampling, hard-negative semantics,
> labeled dataset assembly, CSV enrichment edge behavior, 5-class raster
> constants, import identity, direct-final-grid behavior, confluence overwrite,
> and precompute manifest/status behavior.

- **Goal:** Add or produce a concrete test plan for focused behavior-pinning
  tests before moving Earth/regime, geometric-analysis, or rasterization logic.
  This should protect the contracts identified in
  `AGENT_AUDIT_EARTH_REGIME.md`, `AGENT_AUDIT_GEOMETRIC_ANALYSIS.md`, and
  `AGENT_AUDIT_RASTERIZER.md`.
- **Allowed files:** tests and handoff docs if tests are added; otherwise a
  checkpoint note only. Keep implementation code unchanged unless the user
  explicitly requests moving into a refactor slice.
- **Forbidden files:** implementation refactors, `data/`, root `/models/`,
  notebooks, generated outputs.
- **Tests to run:** targeted tests added/updated in this checkpoint; no full
  pytest required unless code changes beyond tests.
- **Suggested commit:** `test(agents): pin Earth regime feature and raster contracts`
- **Stop condition:** Do not move package ownership yet. Stop after tests or the
  test plan/checkpoint are in place.

## Slice 9 — Extract Earth path helpers (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Moved the Earth/TopoToolbox path helpers
> (`_build_children_from_parents`, `_trace_path_downstream`,
> `_compute_direction_vector`, `_trace_full_path`, `_sample_path_coords`,
> `_detect_cellsize`) plus the small private deps (`_euclidean_2d`,
> `_normalize_vector`) and the `EPSILON` / `MIN_EDGES_FOR_DIRECTION` constants
> into `channel_heads/features/earth_paths.py`. `geometric_analysis.py`
> re-exports them; `rasterizer.py` repointed its `_trace_full_path` import to the
> canonical module. Asymmetry, the geometry analyzer, labeling, hard-negative
> filtering, and CSV enrichment were NOT moved.
>
> Note: this slice was issued by the user under the label "Slice 9", replacing
> the original Slice 9 (data cleanup dry-run), which is retained below as a later
> task.

- **Goal:** Extract Earth/TopoToolbox path helpers from
  `geometric_analysis.py` into a canonical `features/earth_paths.py` while
  preserving behavior exactly (merge-and-consolidate).
- **Allowed files:** `channel_heads/features/earth_paths.py` (new),
  `channel_heads/geometric_analysis.py`, `channel_heads/rasterizer.py` (safe
  import repoint only), `tests/test_geometric_analysis.py`, handoff docs.
- **Forbidden files:** asymmetry/analyzer/labeling/enrichment moves, `data/`,
  root `/models/`, notebooks, generated outputs.
- **Tests to run:** `tests/test_geometric_analysis.py tests/test_rasterizer.py`,
  then full pytest.
- **Suggested commit:** `refactor(features): extract Earth path helpers`
- **Stop condition:** Stop if any path-tracing branch choice, QC flag, or
  coordinate convention would change.

## Slice 10 — Extract asymmetry helpers (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Moved `PairAsymmetryResult`,
> `compute_delta_L`, `LengthwiseAsymmetryAnalyzer`,
> `compute_asymmetry_statistics`, and `merge_coupling_and_asymmetry` into
> `channel_heads/features/asymmetry.py`; `geometric_analysis.py` re-exports
> them. S1 upstream-distance unit policy unchanged.
> Commit: `refactor(features): extract asymmetry helpers`.

## Slice 11 — Extract Earth geometry analyzer (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Moved `GEOM_FEATURE_COLS`,
> `DEFAULT_DIRECTION_SAMPLE_DISTANCE_M`, `PairGeometricResult`,
> `GeometricFeaturesAnalyzer`, and `merge_geometric_features` into
> `channel_heads/features/earth_geometry.py`; `geometric_analysis.py`
> re-exports. Feature-column order, y-axis convention, Strahler behavior, QC
> flags, and skip-warning logging unchanged.
> Commit: `refactor(features): extract Earth geometry analyzer`.

## Slice 12 — Extract labeling / hard-negative filters (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Moved `generate_labeled_dataset`,
> `filter_hard_negatives`, `_line_crosses_stream`, `_build_stream_mask` into
> `channel_heads/training/labeling.py`; `geometric_analysis.py` re-exports.
> Per-group recursion, NaN-keep semantics, thresholds, stream-crossing filter,
> and sort keys unchanged.
> Commit: `refactor(training): extract labeling and hard-negative filters`.

## Slice 13 — Extract Earth enrichment helpers (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Moved `default_stream_loader`,
> `_build_pairs_at_confluence`, `_build_asymmetry_df`, `_add_missing_stream_qc`,
> `add_geometric_features_to_csv`, `_add_geometric_features_cli` (and the
> `StreamLoaderFunc` alias) into `channel_heads/features/earth_enrichment.py`.
> `geometric_analysis.py` is now a **pure re-export shim** and still serves the
> `python -m channel_heads.geometric_analysis` CLI. CSV schema unchanged.
> Commit: `refactor(features): extract Earth enrichment helpers`.
>
> The `geometric_analysis.py` split (Slices 9–13) is complete.

## Slice 14 — Repoint internal imports off the geometric_analysis shim (DONE)

> Completed — see `AGENT_RUN_LOG.md`. `channel_heads/__init__.py` now imports
> the asymmetry / geometry / enrichment / labeling / unit symbols directly from
> the canonical modules instead of from the `geometric_analysis` shim. The shim
> is unchanged (no exports removed); `scripts/build_earth_features_regime.py`
> imports via the top-level `channel_heads` API and needed no change. Public
> API, `__all__`, and all object identities are unchanged.
> Commit: `refactor(features): repoint internal imports from geometric analysis shim`.

## Raster Slice R1 — Extract shared raster schema/constants (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Added
> `channel_heads/rasterization/schema.py` as the canonical home for the frozen
> 5-class patch constants, `CLASS_LABELS`, and `PATCH_FLAG_COLUMNS`. Repointed
> constant-only internal imports to schema where safe. `rasterizer.py`,
> `rasterization.patches`, and `rasterization` still expose compatibility values.
> No Earth rasterization or batch-precompute implementation moved.
> Commit: `refactor(rasterization): extract shared raster schema constants`.

## Raster Slice R2 — Move Earth single-patch rasterization implementation (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Moved the Earth single-patch
> implementation and required helpers into
> `channel_heads/rasterization/earth_patches.py`. `channel_heads/rasterizer.py`
> re-exports the moved names and still owns only Earth batch precompute in this
> area. `rasterization.patches` remains the curated public re-export surface.
> Commit: `refactor(rasterization): move Earth patch rasterization into package`.

- **Goal:** Move the Earth single-patch rasterization implementation into
  `channel_heads/rasterization/earth_patches.py`, while preserving behavior
  exactly and keeping old import paths working.
- **Allowed files:** `channel_heads/rasterization/earth_patches.py` (new),
  `channel_heads/rasterization/patches.py`,
  `channel_heads/rasterization/__init__.py`, `channel_heads/rasterizer.py`,
  `channel_heads/rasterization/drawing.py` only if an import repoint is clearly
  safe, tests and handoff docs.
- **Move only:** `bresenham_line`, `_project_to_target_grid`,
  `_draw_path_on_target_grid`, `_draw_edges_on_target_grid`, `_component_count`,
  `raster_quality_flags`, `_get_rc`, `_compute_rotation_angle`,
  `_rotate_coordinates`, and `rasterize_outlet_pair`.
- **Forbidden:** moving `precompute_raster_dataset`, changing Mars patch logic,
  changing regime scripts, changing class values/count/dtype/target size,
  padding, rotation, draw order, branch protection, confluence overwrite, QA
  semantics, output paths, data, root `/models/`, notebooks, generated outputs.
- **Tests to run:** import checks, targeted raster/CNN/Mars tests, full pytest,
  `git diff --check`, ruff only if available.
- **Suggested commit:** `refactor(rasterization): move Earth patch rasterization into package`
- **Stop condition:** Stop if any raster pixels, QA flags, import identity, or
  Mars patch behavior would change.

## Raster Slice R3 — Move Earth batch precompute (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Moved Earth batch precompute into
> `channel_heads/rasterization/earth_batch.py`. `channel_heads/rasterizer.py`
> now delegates through a compatibility wrapper so old imports and legacy
> monkeypatch behavior continue to work. `rasterization.patches` and
> `rasterization` re-export the canonical package function.
> Commit: `refactor(rasterization): move Earth raster batch precompute`.

- **Goal:** Move `precompute_raster_dataset` into the rasterization package
  after R2, while preserving loader signature, output paths, filenames, debug
  patch behavior, status/error strings, and output columns exactly.
- **Allowed files:** `channel_heads/rasterization/earth_patches.py` or a new
  `channel_heads/rasterization/earth_batch.py`,
  `channel_heads/rasterization/patches.py`,
  `channel_heads/rasterization/__init__.py`, `channel_heads/rasterizer.py`,
  tests and handoff docs.
- **Forbidden:** changing regime patch script behavior, touching data/root
  `/models`/notebooks/generated outputs, changing manifest schema, output paths,
  status/error strings, QA flag semantics, or real artifacts.
- **Tests to run:** import checks, targeted raster/CNN/Mars tests, full pytest,
  `git diff --check`, ruff only if available.
- **Suggested commit:** `refactor(rasterization): move Earth raster batch precompute`
- **Stop condition:** Stop if batch precompute output columns/status/error
  behavior would change.

## Slice 15 — Move regime inference helpers into models/regime.py (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Moved `extract_regime_embeddings`,
> `attach_regime_embeddings`, and `DEFAULT_BATCH_SIZE` into
> `channel_heads/models/regime.py`; `channel_heads/inference/regime.py` is now a
> pure re-export shim. Strict `load_state_dict(strict=True)`, `patch_status ==
> "ok"` filtering, missing-patch dropping, abs/relative patch path resolution,
> `emb_*` overwrite, finite checks, and the returned schema are unchanged.
> `scripts/run_mars_combined_regime.py` imports from the canonical module; the
> lenient `models.cnn_features.extract_embeddings` was NOT merged.
> Commit: `refactor(models): move regime inference helpers into models`.

## Recovery Slice — Earth/regime package foundations (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Added package foundations and tests for
> `channel_heads/training/datasets.py`, `channel_heads/training/xgboost.py`,
> `channel_heads/training/regime.py`, and `channel_heads/eval/lobo.py`.
> Incomplete edits to Earth/regime scripts were reverted before validation.
> Scripts remain transitional; no wrapper conversion was forced in this recovery
> slice.
> Commit: `refactor(training): add Earth regime package foundations`.

## Slice 16a — Repoint low-risk Earth/regime scripts to package foundations (DONE)

> Completed — see `AGENT_RUN_LOG.md`. `scripts/train_cnn_baseline.py`,
> `scripts/train_cnn_regime.py`, `scripts/train_cnn_multiseed.py`, and
> `scripts/eval_lobo_cv.py` now wrap the package foundations added in
> `442d04d`. Manifest filtering, Taiwan holdout/CV-pool selection,
> deterministic validation splits, multi-seed split seeding, LOBO dataset map,
> fold-AUC skip behavior, output schemas, output paths, CLI/defaults, and
> logging/printing surfaces are preserved.
> Commit: `refactor(scripts): wrap low-risk Earth regime training scripts`.

## Slice 16b — Repoint combined-XGBoost scripts to package foundations (DONE)

> Completed — see `AGENT_RUN_LOG.md`. `scripts/train_combined_xgb_phase6b.py`
> and `scripts/train_combined_xgb_regime.py` now wrap
> `channel_heads.training.xgboost` for strict CNN extraction, frozen XGBoost
> config, PR-threshold tuning, metrics schema, and feature/threshold writers.
> Feature order, hyperparameters, strict loading, threshold fallback policy,
> artifact paths, output files, and script logging are preserved.
> Commit: `refactor(scripts): wrap combined XGBoost training scripts`.

## Slice 16c — Repoint regime CNN patch builder (DONE)

> Completed — see `AGENT_RUN_LOG.md`. `scripts/build_cnn_patches_regime.py`
> now wraps `channel_heads.training.regime.build_regime_patch_dataset`, which
> uses the package regime stream loader and canonical Earth batch rasterization.
> CLI/defaults, output root, manifest path, target size, threshold-to-cells
> conversion, DEM z-threshold masking, pruning order, and
> `precompute_raster_dataset(..., threshold=0)` behavior are preserved.
> Commit: `refactor(scripts): wrap regime CNN patch builder`.

## Slice 16d — Repoint remaining Earth/regime feature builder (DONE)

> Completed — see `AGENT_RUN_LOG.md`. `scripts/build_earth_features_regime.py`
> now wraps `channel_heads.training.regime.build_regime_feature_dataset`.
> Basin resolution, per-basin cache handling, stats/master output naming,
> hard-negative filtering, stratified subsampling, max-outlets handling, and
> script CLI/defaults are preserved. No real data or generated outputs were
> touched.
> Commit: `refactor(scripts): wrap regime Earth feature builder`.

## Earth/regime script conversion checkpoint

> All Earth/regime training/eval/patch/feature scripts covered by Slices
> 16a-16d are now wrappers around package foundations. `run_mars_combined_regime.py`
> and `run_regime_pipeline.sh` were intentionally out of scope and remain
> pipeline entry points rather than conversion targets.

## Scripts / CLI organization cleanup (DONE)

> Completed — see `AGENT_RUN_LOG.md`. Audited the full `scripts/` tree and
> updated `scripts/README.md` with a conservative classification: maintained
> CLI, wrapper over package API, diagnostics/rendering utility,
> shell/orchestration entry point, and archive. No scripts were moved because
> root paths are still referenced by docs and shell orchestrators. No scripts
> were deleted or archived in this phase.
> Commit: `chore(scripts): organize CLI and wrapper scripts`.

## Slice 9 (original) — Data cleanup dry-run (DONE)

> Completed — see `AGENT_DATA_CLEANUP_DRYRUN.md` and `AGENT_RUN_LOG.md`.
> Report-only. No data was deleted or moved. Key findings: `data/outputs/`
> (179 MB, LEGACY) is the main safe-delete candidate; ~1.4 GB of regime
> rasters + ~74 MB Mars CNN patches are STALE_AFTER_RASTER_FIX; production
> model `models/xgb_touching_classifier.json` is missing from `models/` but
> present in the backup. `DATA_STATUS.md` needs entries for
> `data/results/experiments/`, `data/results/drainage_density_calibration/`,
> and the `.sr.lock` files in `data/final_valleys/`.

- **Goal:** Produce a **dry-run only** report of candidate stale/generated data
  per `docs/DATA_STATUS.md`. No deletion, no moves.
- **Allowed files:** a report note only.
- **Forbidden files:** `data/`, root `/models/`, any data mutation.
- **Tests to run:** none.
- **Suggested commit:** `docs(agents): data cleanup dry-run report`
- **Stop condition:** Never delete or move data in this slice. Report only.

## Slice 10 — Notebook rebuild (later, explicit request only)

- **Goal:** Update notebooks to canonical imports once shims are stable.
- **Allowed files:** `notebooks/` — only when the user explicitly requests it.
- **Forbidden files:** everything else unless specified.
- **Tests to run:** notebook execution / nbmake if configured.
- **Suggested commit:** `refactor(notebooks): move to canonical package imports`
- **Stop condition:** Do not touch notebooks unless explicitly requested.
