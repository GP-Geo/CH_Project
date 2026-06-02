# AGENT_STATE.md — current refactor state

Snapshot for resuming the package-first refactor without chat history.
Update this file after every completed slice.

_Last updated: 2026-06-02_

## Git

- **Current branch:** `refactor/package-first-architecture`
- **Latest stable commit:** CNN architecture move (Slice 3a; see
  `AGENT_RUN_LOG.md`); prior was `3b51fb0 docs(agents): record CNN ownership audit`.
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
  object. Note: a **separate** `pick_device` copy in `cnn_training.py`
  (re-exported via `models/cnn.py`) was intentionally left untouched — it is a
  CNN module, out of scope here, and is a candidate for the CNN slices.
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
  `channel_heads`) all resolve to the same objects. `cnn_training.py` itself was
  NOT touched (still defines its own `pick_device` — dedup deferred to Slice 3b).

## Known shims (keep working)

- `channel_heads/inference/xgb.py` → `channel_heads/models/xgboost.py`
- `channel_heads/inference/device.py` → `channel_heads/models/device.py`
- `channel_heads/first_meet_pairs_for_outlet.py` → `channel_heads/pairing/earth.py`
- `channel_heads/plotting_utils.py` → `channel_heads/viz/earth.py`
- `channel_heads/config.py` → `channel_heads/io/paths.py`
- `channel_heads/cnn_model.py` → `channel_heads/models/cnn.py`

## Known transitional / not-yet-audited areas

- `channel_heads/inference/regime.py` — transitional; defer to Earth/regime audit.
- CNN modules: `cnn_model.py` is now a shim → `models/cnn.py` (Slice 3a done).
  `cnn_features.py` (Earth embeddings) and `cnn_training.py` (training core +
  duplicate `pick_device`) are still real and not yet consolidated — see Slices
  3b/3c/3d in `AGENT_AUDIT_CNN.md` §6.
- `geometric_analysis.py`, `rasterizer.py` — audit-only, no refactor yet.
- `scripts/` — several still import `from channel_heads.inference import ...`
  (acceptable for thin scripts); cleanup/archive pass pending.

## Next recommended task

**Slice 3b — `pick_device` dedup in `cnn_training.py`:** replace the local
`pick_device` copy in `channel_heads/cnn_training.py` with
`from channel_heads.models.device import pick_device` (keep the name exported so
`models/cnn.py`'s lazy re-export and the trainer scripts still resolve it).
Tests: `tests/test_cnn_model.py` (asserts `pick_device()` returns a valid
device), full pytest. Stop if device selection changes on any platform. See
`AGENT_AUDIT_CNN.md` §6 for the remaining 3c/3d sub-slices and the "do not merge
the four forward-pass extractors" rule.
