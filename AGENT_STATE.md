# AGENT_STATE.md — current refactor state

Snapshot for resuming the package-first refactor without chat history.
Update this file after every completed slice.

_Last updated: 2026-06-03 (Stage 7A archive preparation)_

## Git

- **Current branch:** `refactor/package-first-architecture`
- **Latest stable commit:** `77a80c1` — package-first cleanup: deleted all 11
  shim modules from `channel_heads/`, moved all 11 Python scripts from `scripts/`
  root to `scripts/cli/`, updated all 25 notebooks to canonical imports, removed
  `pipelines/_delegate.py`, rewrote `inference/__init__.py` to import from
  canonical modules. 569 tests pass.
- **Working tree:** clean.
- Slices are committed directly to this branch (not a per-slice branch). Do not
  merge into `main`; do not push.

## Completed package-first migrations

- **Mars pipeline is fully package-resident** (Phases 1–6C). Mars scripts are
  thin wrappers; the pipeline no longer delegates via `_delegate.py`.
  - Phase 1 topology → `channel_heads`
  - Phase 2 pair extraction → `channel_heads`
  - Phase 3A Mars feature generation → `channel_heads/features/mars_features.py`
  - Phase 3B tabular Mars XGBoost inference → `channel_heads/models/mars_inference.py`
  - Phase 4 Mars CNN patch generation → `channel_heads/rasterization/`
  - Phase 5 CNN embeddings → `channel_heads/models/embeddings.py`
  - Phase 6C combined Mars XGBoost inference → `channel_heads/models/mars_combined.py`
- **Pairing:** Earth/TopoToolbox first-meet logic → `channel_heads/pairing/earth.py`.
- **Visualization:** Earth plotting → `channel_heads/viz/earth.py`.
- **Paths/config:** canonical path ownership → `channel_heads/io/paths.py`.
- **XGBoost inference:** implementation → `channel_heads/models/xgboost.py`.
- **Scripts cleanup / organization:** Mars root wrappers for Phases 1-6C are
  archived under `scripts/_archive/`; use `scripts/cli/run_mars_pipeline.py`
  for Mars stage/all runs. All remaining Python scripts moved from `scripts/`
  root to `scripts/cli/`; `scripts/` root now contains only `.sh` files.
- **Full shim deletion (commit 77a80c1):** All 11 shim modules deleted from
  `channel_heads/`. All internal package code, notebooks, and scripts now import
  directly from canonical modules. `inference/__init__.py` rewritten to import
  from `channel_heads.models.{device,xgboost,regime}`. `pipelines/_delegate.py`
  deleted; `pipelines/earth.py` and `pipelines/poster.py` use inline `runpy`
  to invoke `scripts/cli/` scripts. All 25 notebooks updated to canonical imports
  via a batch rewrite script.
- **Notebook canonical imports (commit 77a80c1):** All 25 notebooks updated in
  the same cleanup pass — no shim import strings remain in any notebook.
- **Earth/regime training audit:** ownership plan recorded in
  `AGENT_AUDIT_EARTH_REGIME.md` (audit-only; no implementation moved).
- **Earth/regime package foundations:** reusable helpers now live in
  `channel_heads/training/datasets.py`, `channel_heads/training/xgboost.py`,
  `channel_heads/training/regime.py`, and `channel_heads/eval/lobo.py`, with
  behavior-pinning tests. Incomplete script edits from the interrupted
  extraction were reverted; script repointing/wrapper conversion is deferred.
- **Low-risk Earth/regime script wrappers:** `scripts/train_cnn_baseline.py`,
  `scripts/train_cnn_regime.py`, `scripts/train_cnn_multiseed.py`, and
  `scripts/eval_lobo_cv.py` now call the package foundations for manifest
  filtering, Taiwan CV pool, deterministic validation split, and LOBO reporting.
  CLI/defaults/artifact paths/output paths/log/print behavior are unchanged.
  Feature/patch builders, combined-XGB trainers, and regime pipeline scripts
  were converted in later bounded slices where applicable.
- **Combined-XGBoost script wrappers:** `scripts/train_combined_xgb_phase6b.py`
  and `scripts/train_combined_xgb_regime.py` now call
  `channel_heads.training.xgboost` for strict CNN extraction, frozen XGBoost
  config, threshold tuning, metrics assembly, and feature/threshold file
  writers. CLI/defaults/artifact paths/output paths/metrics files/log behavior,
  feature order, strict loading, and threshold policy are unchanged.
- **Regime CNN patch-builder wrapper:** `scripts/build_cnn_patches_regime.py`
  now calls `channel_heads.training.regime` for regime patch paths,
  regime stream loading, canonical Earth batch rasterization, output-root
  creation, manifest writing, and raster status/basin logging. CLI/defaults,
  output root (`RESULTS_DIR / f"_rasters_{regime.name}"`), manifest path
  (`RESULTS_DIR / f"raster_manifest_{regime.name}.csv"`), `target_size=128`,
  regime threshold-to-cells conversion, DEM z-threshold masking, pruning order,
  and `precompute_raster_dataset(..., threshold=0)` behavior are unchanged.
- **Regime Earth feature-builder wrapper:** `scripts/build_earth_features_regime.py`
  now calls `channel_heads.training.regime.build_regime_feature_dataset` for
  basin resolution, per-basin cache handling, stats writing, optional master
  assembly, hard-negative filtering, and stratified negative subsampling.
  CLI/defaults, cache paths, stats/master paths, regime threshold/pruning
  behavior, `CONNECTIVITY=8`, min basin / max outlet defaults, per-outlet
  prefilter distance, `coupling_n_workers`, hard-negative parameters, and
  subsampling policy are unchanged.
- **Geometric analysis audit:** ownership plan recorded in
  `AGENT_AUDIT_GEOMETRIC_ANALYSIS.md` (audit-only; no implementation moved).
- **Rasterizer audit:** ownership plan recorded in `AGENT_AUDIT_RASTERIZER.md`
  (audit-only; no implementation moved).
- **Behavior-pinning checkpoint:** `tests/test_geometric_analysis.py` and
  `tests/test_rasterizer.py` now pin high-risk contracts before future
  extraction. Implementation source remains untouched.
- **Rasterization schema constants:** shared 5-class patch constants,
  `CLASS_LABELS`, and `PATCH_FLAG_COLUMNS` are canonical in
  `channel_heads/rasterization/schema.py` (Raster R1).
- **Earth patch rasterization:** direct-final-grid Earth single-patch
  implementation lives in `channel_heads/rasterization/earth_patches.py`
  (Raster R2), and Earth batch precompute lives in
  `channel_heads/rasterization/earth_batch.py` (Raster R3).
  `channel_heads/rasterizer.py` is now a compatibility wrapper/re-export
  surface for these rasterization APIs.

## Canonical ownership (current truth)

| Concern | Canonical module | Notes |
|---------|------------------|-------|
| Mars pipeline | `channel_heads.pipelines.mars` + `channel_heads.models.*` | `scripts/cli/` are thin wrappers |
| Earth first-meet pairing | `channel_heads/pairing/earth.py` | shim deleted |
| Earth plotting | `channel_heads/viz/earth.py` | shim deleted |
| Path/config surface | `channel_heads/io/paths.py` | shim deleted |
| XGBoost model load / validate / predict / threshold | `channel_heads/models/xgboost.py` | shim deleted; `inference/__init__.py` re-exports from here |
| Torch device selection (`pick_device`) | `channel_heads/models/device.py` | shim deleted; `inference/__init__.py` re-exports from here |
| Regime embedding helpers (`extract_regime_embeddings`, `attach_regime_embeddings`) | `channel_heads/models/regime.py` | shim deleted; `inference/__init__.py` re-exports from here |
| CNN architecture / dataset / one-hot | `channel_heads/models/cnn.py` | shim deleted |
| Generic/Earth CNN embedding helpers | `channel_heads/models/cnn_features.py` | shim deleted |
| CNN training core (`train_cnn`) | `channel_heads/training/cnn.py` | shim deleted |
| Earth/regime raster-manifest, CV-pool, split, combined feature constants | `channel_heads/training/datasets.py` | `scripts/cli/train_*.py` are wrappers |
| Earth combined-XGBoost training helpers | `channel_heads/training/xgboost.py` | `scripts/cli/train_combined_xgb_*.py` are wrappers |
| Earth/regime feature-build, negative subsampling, regime patch orchestration | `channel_heads/training/regime.py` | `scripts/cli/build_*_regime.py` are wrappers |
| LOBO diagnostic report | `channel_heads/eval/lobo.py` | `scripts/cli/eval_lobo_cv.py` is a thin wrapper |
| Pure feature math | `channel_heads/features/geometry.py` | `geometric_analysis.py` still re-exports for compatibility |
| Earth/TopoToolbox path helpers | `channel_heads/features/earth_paths.py` | `geometric_analysis.py` re-exports |
| Earth lengthwise asymmetry | `channel_heads/features/asymmetry.py` | `geometric_analysis.py` re-exports |
| Earth geometry analyzer (`GEOM_FEATURE_COLS`, `GeometricFeaturesAnalyzer`, …) | `channel_heads/features/earth_geometry.py` | `geometric_analysis.py` re-exports |
| Earth labeling / hard-negative filtering | `channel_heads/training/labeling.py` | `geometric_analysis.py` re-exports |
| Earth CSV enrichment / stream loading | `channel_heads/features/earth_enrichment.py` | `geometric_analysis.py` re-exports; keeps `__main__` CLI |
| `channel_heads/geometric_analysis.py` | pure re-export shim (no implementation) | retained for backward compat |
| Mars projected path helpers | `channel_heads/features/paths.py` | — |
| Mars feature table generation | `channel_heads/features/mars_features.py` | `scripts/cli/run_mars_pipeline.py --stage features` |
| Unit conversions | `channel_heads/units.py` | `geometric_analysis.py` and `dd_calibration.py` re-export selected helpers |
| Shared raster patch schema | `channel_heads/rasterization/schema.py` | `rasterization/patches.py` and `rasterization/__init__.py` re-export for compat |
| Mars CNN patch generation | `channel_heads/rasterization/mars_patches.py` | `scripts/cli/run_mars_pipeline.py --stage patches` |
| Earth 5-class single-patch rasterization | `channel_heads/rasterization/earth_patches.py` | `rasterization/patches.py` and `rasterization/__init__.py` re-export for compat |
| Earth raster batch precompute | `channel_heads/rasterization/earth_batch.py` | `rasterization/patches.py` and `rasterization/__init__.py` re-export; accepts `rasterize_func=` kwarg for testability |

## Model-layer status

- **XGBoost consolidation: DONE.** Real implementation lives in
  `channel_heads/models/xgboost.py`; `channel_heads/inference/xgb.py` is a pure
  re-export shim. Old imports (`channel_heads.inference.xgb`,
  `from channel_heads.inference import ...`) resolve to the same objects.
- **Device consolidation: DONE.** `pick_device()` now lives in
  `channel_heads/models/device.py`; `channel_heads/inference/device.py` is a pure
  re-export shim. `models/mars_combined.py` and `models/embeddings.py` import
  `pick_device` from `channel_heads.models.device`; it is also re-exported from
  `channel_heads.models`. Old imports (`channel_heads.inference.device`,
  `from channel_heads.inference import pick_device`) still resolve to the same
  object. As of Slice 3b the **CNN training module no longer has its own copy** —
  `cnn_training.py` imports `pick_device` from `channel_heads.models.device`, so
  there is now a single canonical implementation across the package.
- **Regime inference consolidation: DONE.** The regime-CNN embedding /
  patch-index merge glue (`extract_regime_embeddings`,
  `attach_regime_embeddings`, `DEFAULT_BATCH_SIZE`) now lives canonically in
  `channel_heads/models/regime.py` (moved verbatim; imports the CNN classes from
  `channel_heads.models.cnn`). `channel_heads/inference/regime.py` is a pure
  re-export shim. Strict `load_state_dict(strict=True)`, `patch_status == "ok"`
  filtering, dropping pairs without a patch, absolute/project-relative patch
  path resolution, `emb_0..emb_N` overwrite, finite-value checks, and the
  returned schema (drops `patch_path_abs`) are all unchanged.
  `scripts/run_mars_combined_regime.py` imports from the canonical module; old
  imports (`channel_heads.inference.regime`) still resolve to the same objects.
  `channel_heads.models` re-exports `attach_regime_embeddings` /
  `extract_regime_embeddings` (torch-optional). Pinned by
  `tests/test_inference_regime.py` (identity + strict-load + drop/override/finite).
  The four divergent forward-pass extractors remain deliberately **not** merged.
- **CNN audit (Slice 2): DONE.** See `AGENT_AUDIT_CNN.md`. Recommended canonical
  homes: architecture/dataset → `channel_heads/models/cnn.py` (promote the
  current re-export to real); Earth embeddings (`cnn_features.py`) → models
  layer; training core (`cnn_training.py`) → future `channel_heads/training/`;
  flat `cnn_*` modules → shims. The four divergent CNN forward-pass extractors
  (lenient vs strict load) are **not** to be merged in the model slice.
- **CNN architecture consolidation (Slice 3a): DONE.** The real `OutletCNN`,
  `OutletPairDataset`, `encode_raster_onehot`, `DEFAULT_EMBEDDING_DIM`,
  `DEFAULT_TARGET_SIZE` now live in `channel_heads/models/cnn.py`;
  `channel_heads/cnn_model.py` is a pure re-export shim (also re-exports
  `NUM_CLASSES` to match its historical namespace). Architecture moved
  byte-for-byte — state-dict keys, dims, constructor defaults, and forward
  shapes unchanged; `strict=True` artifact loading preserved (pinned by
  `tests/test_cnn_consolidation.py`). `models/cnn.py` re-exports the training
  core (`train_cnn`, `pick_device`, `DEFAULT_*`, `HOLDOUT_BASIN`) from
  `cnn_training.py` **lazily** via module `__getattr__` to avoid the
  `cnn_training → cnn_model(shim) → models.cnn` import cycle (with a
  `TYPE_CHECKING` block so linters still see the names). Old imports
  (`channel_heads.cnn_model`, `channel_heads.models.cnn`, top-level
  `channel_heads`) all resolve to the same objects.
- **CNN training `pick_device` dedup (Slice 3b): DONE.** The duplicate
  `pick_device` body in `channel_heads/cnn_training.py` was removed; the module
  now does `from channel_heads.models.device import pick_device` (re-export).
  `from channel_heads.cnn_training import pick_device` and the lazy
  `channel_heads.models.cnn.pick_device` both resolve to the *same* object as
  `channel_heads.models.device.pick_device` (pinned by
  `tests/test_cnn_consolidation.py::TestPickDeviceDeduplicated`). Training loop,
  hyperparameters, early stopping, dataset behavior, and architecture untouched.
  The script-level `pick_device` copies in `scripts/train_combined_xgb_*.py`
  remain (scripts are out of scope until Slice 4).
- **Earth/generic embedding helpers consolidation (Slice 3c): DONE.** The real
  `extract_embeddings`, `merge_cnn_features`, and `CNN_FEATURE_COLS` now live in
  `channel_heads/models/cnn_features.py` (moved verbatim; imports the CNN
  architecture from `channel_heads.models.cnn`). `channel_heads/cnn_features.py`
  is a pure re-export shim. The **lenient default `load_state_dict`** (no
  `strict=`, no missing/unexpected check) and the manifest-keyed DataFrame
  schema are preserved exactly — deliberately distinct from the strict
  extractors in `inference/regime.py` and `models/mars_combined.py` (NOT merged).
  Internal consumers repointed to the canonical module
  (`channel_heads/__init__.py`, `models/embeddings.py`). `models/embeddings.py`
  remains the Mars Phase-5 orchestration surface (only its import line +
  docstring changed). Old imports (`channel_heads.cnn_features`,
  `from channel_heads.cnn_features import extract_embeddings, CNN_FEATURE_COLS`,
  top-level `channel_heads`) all resolve to the same objects (pinned by
  `tests/test_cnn_consolidation.py::TestCNNFeaturesConsolidated`).
- **CNN training core → `training/` package (Slice 3d): DONE.** The real
  `train_cnn` loop + `DEFAULT_*` / `HOLDOUT_BASIN` / `RANDOM_STATE` now live in
  the new `channel_heads/training/cnn.py` (moved verbatim; logger name kept as
  `"channel_heads.cnn_training"` so logging behavior is unchanged; imports the
  CNN classes from `channel_heads.models.cnn` and re-exports `pick_device` from
  `channel_heads.models.device`). `channel_heads/cnn_training.py` is a pure
  re-export shim. `models/cnn.py`'s lazy `__getattr__` (and `TYPE_CHECKING`
  block) now source the training symbols from `channel_heads.training.cnn` — the
  lazy mechanism is still required because `training.cnn` imports `models.cnn`
  (would cycle if eager). Old imports (`channel_heads.cnn_training`,
  `channel_heads.training.cnn`, and `models.cnn`'s lazy surface) all resolve to
  the same objects; the three `scripts/train_cnn_*.py` trainers (unmodified)
  still import cleanly via the shim. Pinned by
  `tests/test_cnn_consolidation.py::TestTrainingCoreConsolidated`. This completes
  the CNN consolidation (Slice 3).
- **Scripts cleanup / archive (Slice 4): DONE.** Thin scripts under `scripts/`
  were repointed from compatibility shims to canonical modules where safe:
  `channel_heads.io.paths`, `channel_heads.models.cnn`,
  `channel_heads.models.device`, `channel_heads.models.xgboost`, and
  `channel_heads.training.cnn`. The inline `pick_device()` copies in
  `scripts/train_combined_xgb_phase6b.py` and
  `scripts/train_combined_xgb_regime.py` were removed in favor of the canonical
  `channel_heads.models.device.pick_device` implementation (same `mps` > `cuda`
  > `cpu` order). No scripts were archived because none were clearly dead in
  this pass. CLI surfaces, model paths, hyperparameters, strict/lenient
  state-dict behavior, and outputs unchanged.
- **Earth/regime training audit (Slice 5): DONE.** See
  `AGENT_AUDIT_EARTH_REGIME.md`. Key findings: Earth/regime scripts are not
  disposable; real logic remains in regime feature generation, regime patch
  generation, baseline/regime combined-XGB training, regime Mars inference,
  threshold retuning, and LOBO-CV diagnostics. Existing package owners already
  cover regime presets, generic CNN architecture/training, generic lenient CNN
  embeddings, XGBoost inference helpers, eval split/threshold primitives, and
  strict regime embedding attachment. Recommended future homes:
  `training/datasets.py`, expanded `training/cnn.py`,
  `training/xgboost.py`, `training/regime.py`, `models/regime.py`, and
  `eval/lobo.py`, with scripts reduced to wrappers only after tests pin
  behavior.
- **Geometric analysis audit (Slice 6): DONE.** See
  `AGENT_AUDIT_GEOMETRIC_ANALYSIS.md`. Key findings:
  `geometric_analysis.py` is not disposable; it owns Earth/TopoToolbox path
  traversal, lengthwise asymmetry, Earth geometric feature generation,
  labeled-dataset assembly, hard-negative filtering, default stream loading,
  and CSV enrichment. Pure math (`features.geometry`), Mars projected path
  helpers (`features.paths`), Mars feature generation (`features.mars_features`),
  units (`units.py`), and pairing helpers (`pairing.earth`) are already package
  owners. Recommended future homes: `features/earth_paths.py`,
  `features/asymmetry.py`, `features/earth_geometry.py`,
  `training/labeling.py`, and `features/earth_enrichment.py`, with
  `geometric_analysis.py` reduced to a shim only after tests pin feature order,
  unit behavior, QC flags, hard-negative semantics, and private helper imports.
- **Rasterizer audit (Slice 7): DONE.** See `AGENT_AUDIT_RASTERIZER.md`. Key
  findings: Mars Phase 4 patch generation is package-resident in
  `rasterization/mars_patches.py`, but Earth patch rasterization and shared
  5-class constants still live in `rasterizer.py`.
  `rasterization/patches.py` is currently a curated re-export, not the real
  implementation. Recommended future homes: a shared rasterization schema,
  promoted `rasterization/patches.py` or `earth_patches.py` for Earth
  rasterization/precompute, `rasterizer.py` as a shim, and regime patch
  orchestration under future `training/regime.py`.
- **CSV enrichment extraction + shim completion (Slice 13): DONE.**
  `default_stream_loader`, `add_geometric_features_to_csv`, the private helpers
  (`_build_pairs_at_confluence`, `_build_asymmetry_df`, `_add_missing_stream_qc`),
  the `StreamLoaderFunc` alias, and the `_add_geometric_features_cli` CLI now
  live canonically in `channel_heads/features/earth_enrichment.py` (moved
  verbatim; per-module `get_logger(__name__)`). **`geometric_analysis.py` is now
  a pure re-export shim** — it imports/re-exports the asymmetry, geometry, path,
  labeling, enrichment, and unit symbols, keeps the type aliases for
  compatibility, and keeps the `if __name__ == "__main__"` CLI working
  (`python -m channel_heads.geometric_analysis`). The CSV schema (overlap_px
  drop, head/L swapping, missing basin/lat=36.0/z_th=0.0 defaults,
  `missing_stream` flags, default threshold 300, write-only-when-requested) is
  unchanged. `scripts/build_earth_features_regime.py` and the top-level
  `channel_heads` API still resolve through the shim. Pinned by
  `tests/test_geometric_analysis.py::TestEnrichmentExtraction`.
- **Labeling / hard-negative extraction (Slice 12): DONE.**
  `generate_labeled_dataset`, `filter_hard_negatives`, and the private
  stream-crossing helpers (`_line_crosses_stream`, `_build_stream_mask`) now
  live canonically in `channel_heads/training/labeling.py` (moved verbatim;
  imports `GEOM_FEATURE_COLS` from `features.earth_geometry` and `line_pixels`
  from `stream_utils`). `geometric_analysis.py` imports and re-exports them
  (privates via `# noqa: F401`), and dropped its now-unused `numpy.typing` /
  `stream_utils.line_pixels` imports. Per-group recursion, NaN-keep semantics,
  positive-median thresholds, the optional stream-crossing filter (negatives
  only), conservative keep-on-error, and final sort keys are unchanged. Pinned
  by `tests/test_geometric_analysis.py::TestLabelingExtraction`.
- **Earth geometry analyzer extraction (Slice 11): DONE.** `GEOM_FEATURE_COLS`,
  `DEFAULT_DIRECTION_SAMPLE_DISTANCE_M`, `PairGeometricResult`,
  `GeometricFeaturesAnalyzer`, and `merge_geometric_features` now live
  canonically in `channel_heads/features/earth_geometry.py` (moved verbatim;
  per-module `get_logger(__name__)`). `geometric_analysis.py` imports and
  re-exports them (the path-helper / geometry re-exports there are now
  re-export-only, using redundant-alias / `# noqa: F401` form). `GEOM_FEATURE_COLS`
  order, the x=col / y=-row convention, branch-parent Strahler behavior, QC flag
  strings, and skip-warning logging are unchanged. The skip-warning test now
  patches `channel_heads.features.earth_geometry.logger` (the analyzer's new
  module). Pinned by
  `tests/test_geometric_analysis.py::TestEarthGeometryExtraction`.
- **Asymmetry extraction (Slice 10): DONE.** The Earth lengthwise-asymmetry
  symbols (`PairAsymmetryResult`, `compute_delta_L`,
  `LengthwiseAsymmetryAnalyzer`, `compute_asymmetry_statistics`,
  `merge_coupling_and_asymmetry`) now live canonically in
  `channel_heads/features/asymmetry.py` (moved verbatim; logger is per-module
  `get_logger(__name__)`, warnings unchanged). `geometric_analysis.py` imports
  and re-exports them; top-level `channel_heads` and the regime script
  (`scripts/build_earth_features_regime.py`) still resolve via the re-export.
  The S1 upstream-distance unit policy is unchanged. Pinned by
  `tests/test_geometric_analysis.py::TestAsymmetryExtraction`.
- **Earth path helper extraction (Slice 9): DONE.** The Earth/TopoToolbox path
  helpers (`_build_children_from_parents`, `_trace_path_downstream`,
  `_compute_direction_vector`, `_trace_full_path`, `_sample_path_coords`,
  `_detect_cellsize`) plus their small private dependencies (`_euclidean_2d`,
  `_normalize_vector`) and the `EPSILON` / `MIN_EDGES_FOR_DIRECTION` constants
  now live canonically in `channel_heads/features/earth_paths.py` (moved
  verbatim). `geometric_analysis.py` imports them and re-exports them, so old
  imports (`from channel_heads.geometric_analysis import _trace_full_path`, the
  underscored helpers used by tests, `EPSILON`, `MIN_EDGES_FOR_DIRECTION`) all
  resolve to the *same* objects. `rasterizer.py` was repointed to import
  `_trace_full_path` from `channel_heads.features.earth_paths` (its
  `_build_children_from_parents` still comes from `pairing.earth` — a separate
  variant, deliberately untouched). Asymmetry, `GeometricFeaturesAnalyzer`,
  labeling, hard-negative filtering, and CSV enrichment were NOT moved. Pinned
  by `tests/test_geometric_analysis.py::TestEarthPathsExtraction`.
- **Behavior-pinning tests (Slice 8): DONE.** Focused tests now cover
  `geometric_analysis.py` contracts for feature-column order, `compute_delta_L`,
  head normalization / length swapping, `_trace_full_path`, `_sample_path_coords`,
  hard-negative filtering, labeled-dataset assembly, and CSV enrichment edge
  behavior. Rasterizer tests now pin 5-class constants, old/new import identity,
  confluence overwrite behavior, small-target direct-final-grid connectivity,
  and `precompute_raster_dataset` columns/status/error behavior. No
  implementation code was changed.

## Known shims (still active — keep working)

Only two compatibility surfaces remain:

- `channel_heads/geometric_analysis.py` — pure re-export shim for all
  Earth feature / enrichment / labeling / asymmetry / path symbols. No
  implementation. Retained because external callers (notebooks, legacy scripts)
  may still import from it. Safe to delete once all callers confirmed.
- `channel_heads/inference/__init__.py` — re-exports `pick_device`,
  `load_xgb_model`, `load_threshold`, `predict_with_threshold`,
  `verify_feature_matrix`, `verify_model_feature_order`,
  `extract_regime_embeddings`, `attach_regime_embeddings` from canonical
  `channel_heads.models.*`. Retained for any callers that do
  `from channel_heads.inference import ...`.

All other shims listed in earlier snapshots of this file have been **deleted**
(commit `77a80c1`):
  `cnn_model.py`, `cnn_features.py`, `cnn_training.py`, `config.py`,
  `first_meet_pairs_for_outlet.py`, `plotting_utils.py`, `rasterizer.py` (was
  partial shim — also deleted), `inference/xgb.py`, `inference/device.py`,
  `inference/regime.py`, `pipelines/_delegate.py`.

## Remaining compatibility surfaces (not transitional — intentional)

- `channel_heads/geometric_analysis.py` — pure re-export shim. No
  implementation. All consumers repointed internally; external callers still
  work. See §Known shims above for deletion criteria.
- `channel_heads/inference/__init__.py` — thin re-export surface over
  `models.device`, `models.xgboost`, `models.regime`. Not a shim to delete soon.
- `channel_heads/rasterization/patches.py` and `channel_heads/rasterization/__init__.py`
  — curated re-export surfaces over `schema.py`, `earth_patches.py`,
  `earth_batch.py`. No implementation. Maintained as the public rasterization API.

The four divergent forward-pass CNN extractors (lenient vs strict load) in
`models/mars_combined.py`, `models/embeddings.py`, and `models/regime.py`
were deliberately **not** merged — they encode different loading contracts.

## Next recommended task

The package-first refactor is **complete**. No shims remain except the two
intentional compatibility surfaces listed above. Scientific pipeline work is next.

See `STAGE_45_PLANNING.md` for the full plan.

**Completed (Stage 4):**
- `notebooks/regime/00_calibration_overview.ipynb` — fixed import block:
  replaced `from build_earth_features_regime import (DEM_TO_BASIN, REGIMES,
  stratified_subsample_negatives)` with canonical
  `from channel_heads.regimes import REGIMES` +
  `from channel_heads.training.regime import DEM_TO_BASIN, stratified_subsample_negatives`.
  Removed obsolete `scripts/` sys.path insert. Updated 4 run_script paths to
  `scripts/cli/`.
- `docs/REGIME_SELECTION.md` — created; freezes regime parameters, calibration
  rationale, evidence pointers, and downstream artifact map.

**Completed (Stage 5):**
- `notebooks/analysis/05_earth_network_qa.ipynb` — formal QA gate notebook
  created. Opens with pipeline stage card (stage/prev/next/purpose/inputs/
  outputs/decision gate). Hard gate on: zero error basins, ≥5 000 total pairs
  per regime, no NaN in feature columns. Soft warnings: basins <10 pairs,
  touching ratio outside [0.05, 0.95]. Writes `stage5_earth_network_qa_report.csv`
  on PASS. `STAGE_ASSET_MAP.md` updated: Stage 5 ❌ → 🔶.
- `data/results/stage5_earth_network_qa_report.csv` exists and records a PASS:
  hard flags = 0, soft warnings = 6. Non-blocking warnings are yoro low pair
  counts in regA/regB and low regC touching ratios for finisterre, luliang,
  sierramadre, and vallefertil.

**Completed (Stage 7 planning):**
- `AGENT_STAGE7_REGENERATION_PLAN.md` created as a report-only plan for Earth
  model-input regeneration after Stage 4 regime selection and Stage 5 QA pass.
  It lists required inputs, stale/generated outputs, canonical `scripts/cli/`
  commands to run later, risk controls, validation gates, and the decision gate
  before Stage 8/9. No data, models, notebooks, rasters, embeddings,
  predictions, training outputs, or package code were modified.

**Completed (Stage 7A archive preparation):**
- Stale generated Earth regime raster artifacts were moved, not deleted, to
  `data/_stage7_archive_20260603_231506/` with structure preserved relative to
  `data/`:
  - `results/_rasters_regA/`
  - `results/_rasters_regB/`
  - `results/_rasters_regC/`
  - `results/raster_manifest_regA.csv`
  - `results/raster_manifest_regB.csv`
  - `results/raster_manifest_regC.csv`
- Verification passed: archived paths exist, original paths are absent, and
  `data/results/master_dataset_reg{A,B,C}_with_emb.csv` remain in place. Archive
  holds 73,272 raster files and is about 1.4 GB. `data/_stage7_archive_*/` is
  gitignored so the generated archive is not committed.
- No raw/source/manual data, models, notebooks, package code, raster
  regeneration, patch regeneration, or training was touched.

**Next:**
- Run Stage 7B regeneration commands from `AGENT_STAGE7_REGENERATION_PLAN.md`
  when ready:
  `python scripts/cli/build_cnn_patches_regime.py --regime regA|regB|regC -v`.
- Do not proceed to Stage 8 model training/retraining or Stage 9 validation
  until Stage 7 regenerated patches/manifests pass validation.
