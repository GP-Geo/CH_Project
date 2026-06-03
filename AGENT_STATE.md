# AGENT_STATE.md — current refactor state

Snapshot for resuming the package-first refactor without chat history.
Update this file after every completed slice.

_Last updated: 2026-06-03_

## Git

- **Current branch:** `refactor/package-first-architecture`
- **Latest stable commit:** Geometric analysis + rasterizer behavior-pinning
  tests (Slice 8; see `AGENT_RUN_LOG.md`); prior was the Slice 6/7 ownership
  audit.
- **Working tree:** clean at time of writing after the behavior-pinning commit.
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
- **XGBoost inference:** implementation → `channel_heads/models/xgboost.py`
  (this is the most recent slice, commit `90fd694`).
- **Scripts cleanup:** thin scripts now import canonical package modules directly
  where safe; no scripts were archived.
- **Earth/regime training audit:** ownership plan recorded in
  `AGENT_AUDIT_EARTH_REGIME.md` (audit-only; no implementation moved).
- **Geometric analysis audit:** ownership plan recorded in
  `AGENT_AUDIT_GEOMETRIC_ANALYSIS.md` (audit-only; no implementation moved).
- **Rasterizer audit:** ownership plan recorded in `AGENT_AUDIT_RASTERIZER.md`
  (audit-only; no implementation moved).
- **Behavior-pinning checkpoint:** `tests/test_geometric_analysis.py` and
  `tests/test_rasterizer.py` now pin high-risk contracts before future
  extraction. Implementation source remains untouched.

## Canonical ownership (current truth)

| Concern | Canonical module | Shim / legacy path |
|---------|------------------|--------------------|
| Mars pipeline | `channel_heads.pipelines.mars` + `channel_heads.models.*` | Mars scripts are thin wrappers |
| Earth first-meet pairing | `channel_heads/pairing/earth.py` | `channel_heads/first_meet_pairs_for_outlet.py` (shim) |
| Earth plotting | `channel_heads/viz/earth.py` | `channel_heads/plotting_utils.py` (shim) |
| Path/config surface | `channel_heads/io/paths.py` | `channel_heads/config.py` (shim) |
| XGBoost model load / validate / predict / threshold | `channel_heads/models/xgboost.py` | `channel_heads/inference/xgb.py` (shim) |
| Torch device selection (`pick_device`) | `channel_heads/models/device.py` | `channel_heads/inference/device.py` (shim) |
| CNN architecture / dataset / one-hot (`OutletCNN`, `OutletPairDataset`, `encode_raster_onehot`, `DEFAULT_EMBEDDING_DIM`, `DEFAULT_TARGET_SIZE`) | `channel_heads/models/cnn.py` | `channel_heads/cnn_model.py` (shim) |
| Generic/Earth CNN embedding helpers (`extract_embeddings`, `merge_cnn_features`, `CNN_FEATURE_COLS`) | `channel_heads/models/cnn_features.py` | `channel_heads/cnn_features.py` (shim) |
| CNN training core (`train_cnn`, `DEFAULT_*`, `HOLDOUT_BASIN`, `RANDOM_STATE`) | `channel_heads/training/cnn.py` | `channel_heads/cnn_training.py` (shim) |
| Pure feature math | `channel_heads/features/geometry.py` | `channel_heads/geometric_analysis.py` aliases for compatibility |
| Earth/TopoToolbox path helpers (`_build_children_from_parents`, `_trace_path_downstream`, `_compute_direction_vector`, `_trace_full_path`, `_sample_path_coords`, `_detect_cellsize`, `_euclidean_2d`, `_normalize_vector`, `EPSILON`, `MIN_EDGES_FOR_DIRECTION`) | `channel_heads/features/earth_paths.py` | `channel_heads/geometric_analysis.py` re-exports for compatibility |
| Earth lengthwise asymmetry (`PairAsymmetryResult`, `compute_delta_L`, `LengthwiseAsymmetryAnalyzer`, `compute_asymmetry_statistics`, `merge_coupling_and_asymmetry`) | `channel_heads/features/asymmetry.py` | `channel_heads/geometric_analysis.py` re-exports for compatibility |
| Earth geometry analyzer (`GEOM_FEATURE_COLS`, `DEFAULT_DIRECTION_SAMPLE_DISTANCE_M`, `PairGeometricResult`, `GeometricFeaturesAnalyzer`, `merge_geometric_features`) | `channel_heads/features/earth_geometry.py` | `channel_heads/geometric_analysis.py` re-exports for compatibility |
| Mars projected path helpers | `channel_heads/features/paths.py` | none |
| Mars feature table generation | `channel_heads/features/mars_features.py` | Mars script wrappers |
| Unit conversions | `channel_heads/units.py` | `geometric_analysis.py` and `dd_calibration.py` re-export selected helpers |
| Mars CNN patch generation | `channel_heads/rasterization/mars_patches.py` | `scripts/build_mars_cnn_patches_5class.py` wrapper |
| Earth 5-class patch rasterization | `channel_heads/rasterizer.py` for now | `channel_heads/rasterization/patches.py` re-export surface |

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
- **`inference/regime.py`: TRANSITIONAL.** Holds regime-CNN embedding /
  patch-index merge glue. Leave untouched until the Earth/regime training
  audit. Do not change regime behavior now.
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

## Known shims (keep working)

- `channel_heads/inference/xgb.py` → `channel_heads/models/xgboost.py`
- `channel_heads/inference/device.py` → `channel_heads/models/device.py`
- `channel_heads/first_meet_pairs_for_outlet.py` → `channel_heads/pairing/earth.py`
- `channel_heads/plotting_utils.py` → `channel_heads/viz/earth.py`
- `channel_heads/config.py` → `channel_heads/io/paths.py`
- `channel_heads/cnn_model.py` → `channel_heads/models/cnn.py`
- `channel_heads/cnn_features.py` → `channel_heads/models/cnn_features.py`
- `channel_heads/cnn_training.py` → `channel_heads/training/cnn.py`

## Known transitional / not-yet-audited areas

- `channel_heads/inference/regime.py` — audited in Slice 5. Keep untouched for
  now; future recommendation is to move it to `channel_heads/models/regime.py`
  and leave `inference/regime.py` as a shim after tests pin strict load,
  patch-index filtering, embedding overwrite, and finite checks.
- CNN modules: **all three flat `cnn_*` modules are now shims** —
  `cnn_model.py` → `models/cnn.py` (3a), `cnn_features.py` →
  `models/cnn_features.py` (3c), `cnn_training.py` → `training/cnn.py` (3d);
  `pick_device` deduped onto `models/device.py` (3b). Slice 3 (CNN consolidation)
  is complete. The four divergent forward-pass extractors in `inference/regime.py`,
  `models/mars_combined.py`, and `scripts/train_combined_xgb_*.py` were
  deliberately **not** merged (see `AGENT_AUDIT_CNN.md` §3b/§5).
- `geometric_analysis.py` — audited in Slice 6, behavior-pinned in Slice 8.
  Earth path helpers extracted to `features/earth_paths.py` (Slice 9); asymmetry
  to `features/asymmetry.py` (Slice 10); the Earth geometry analyzer to
  `features/earth_geometry.py` (Slice 11). Still owns the stream-crossing helpers
  (`_line_crosses_stream`, `_build_stream_mask`), labeling / hard-negative
  filtering (`generate_labeled_dataset`, `filter_hard_negatives`), and CSV
  enrichment (`default_stream_loader`, `add_geometric_features_to_csv`, helpers,
  CLI). Remaining recommendation is to split labeling into
  `training/labeling.py` (Slice 12) and enrichment into
  `features/earth_enrichment.py` (Slice 13), then leave `geometric_analysis.py`
  as a pure re-export shim.
- `rasterizer.py` — audited in Slice 7 and behavior-pinned in Slice 8. Keep
  untouched for now; future
  recommendation is to move shared constants/schema and Earth rasterization into
  `channel_heads/rasterization/`, then leave `rasterizer.py` as a shim.
- `scripts/` — cleanup pass complete as of Slice 4. Scripts may still import
  transitional modules directly when that is the canonical current surface
  (notably `channel_heads.inference.regime`); no dead scripts were archived.

## Next recommended task

**Slice 12 — extract labeling / hard-negative filters** (`generate_labeled_dataset`,
`filter_hard_negatives`, `_line_crosses_stream`, `_build_stream_mask`) into
`channel_heads/training/labeling.py`, with `geometric_analysis.py` re-exporting
for compatibility. Preserve per-group recursion, NaN-keep semantics, the
positive-median thresholds, the optional stream-crossing filter (negatives
only), conservative keep-on-error, and the final sort keys.

The backlog's standalone **data cleanup dry-run (report-only)** item remains
open as a later task; do not delete, move, or mutate any data/model/generated
artifact when it is picked up.

> Note: the user-issued geometric_analysis split slices are numbered 9–13 in the
> prompts; the backlog's original Slice 9 was the data cleanup dry-run, retained
> as a separate later task.
