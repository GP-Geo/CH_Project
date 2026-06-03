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

## Slice 9 (original) — Data cleanup dry-run (later)

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
