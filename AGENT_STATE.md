# AGENT_STATE.md — current refactor state

Snapshot for resuming the package-first refactor without chat history.
Update this file after every completed slice.

_Last updated: 2026-06-02_

## Git

- **Current branch:** `refactor/package-first-architecture`
- **Latest stable commit:** CNN embedding helpers into models (Slice 3c; see
  `AGENT_RUN_LOG.md`); prior was `6b2e31b refactor(models): deduplicate CNN training device selection`.
- **Working tree:** clean at time of writing.
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

## Known shims (keep working)

- `channel_heads/inference/xgb.py` → `channel_heads/models/xgboost.py`
- `channel_heads/inference/device.py` → `channel_heads/models/device.py`
- `channel_heads/first_meet_pairs_for_outlet.py` → `channel_heads/pairing/earth.py`
- `channel_heads/plotting_utils.py` → `channel_heads/viz/earth.py`
- `channel_heads/config.py` → `channel_heads/io/paths.py`
- `channel_heads/cnn_model.py` → `channel_heads/models/cnn.py`
- `channel_heads/cnn_features.py` → `channel_heads/models/cnn_features.py`

## Known transitional / not-yet-audited areas

- `channel_heads/inference/regime.py` — transitional; defer to Earth/regime audit.
- CNN modules: `cnn_model.py` → shim to `models/cnn.py` (Slice 3a) and
  `cnn_features.py` → shim to `models/cnn_features.py` (Slice 3c) are both done.
  `cnn_training.py` no longer duplicates `pick_device` (Slice 3b done) but is
  still the real home of the training loop/defaults — the training-core move to a
  future `channel_heads/training/` package is the remaining sub-slice (3d). See
  `AGENT_AUDIT_CNN.md` §6.
- `geometric_analysis.py`, `rasterizer.py` — audit-only, no refactor yet.
- `scripts/` — several still import `from channel_heads.inference import ...`
  (acceptable for thin scripts); cleanup/archive pass pending.

## Next recommended task

**Slice 3d — CNN training core → future `channel_heads/training/` package:**
move `train_cnn` + the `DEFAULT_*` / `HOLDOUT_BASIN` / `RANDOM_STATE`
hyperparameters out of `channel_heads/cnn_training.py` into a new
`channel_heads/training/cnn.py`; reduce `cnn_training.py` to a shim;
`models/cnn.py`'s lazy `__getattr__` re-export should source the training
symbols from the new home. Keep scripts importing `channel_heads.cnn_training`
(shim) — repointing them is Slice 4. Preserve every default value. Tests:
`tests/test_cnn_model.py`, full pytest. Stop if any default changes. See
`AGENT_AUDIT_CNN.md` §6; do not merge the four forward-pass extractors.
