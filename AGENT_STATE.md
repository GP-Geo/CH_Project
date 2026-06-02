# AGENT_STATE.md — current refactor state

Snapshot for resuming the package-first refactor without chat history.
Update this file after every completed slice.

_Last updated: 2026-06-02_

## Git

- **Current branch:** `refactor/package-first-architecture`
- **Latest stable commit:** device consolidation (see `AGENT_RUN_LOG.md` for hash);
  prior was `90fd694 refactor(models): move XGBoost inference implementation into models`.
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
  (lenient vs strict load) are **not** to be merged in the model slice. No CNN
  code changed yet.

## Known shims (keep working)

- `channel_heads/inference/xgb.py` → `channel_heads/models/xgboost.py`
- `channel_heads/inference/device.py` → `channel_heads/models/device.py`
- `channel_heads/first_meet_pairs_for_outlet.py` → `channel_heads/pairing/earth.py`
- `channel_heads/plotting_utils.py` → `channel_heads/viz/earth.py`
- `channel_heads/config.py` → `channel_heads/io/paths.py`

## Known transitional / not-yet-audited areas

- `channel_heads/inference/regime.py` — transitional; defer to Earth/regime audit.
- CNN modules (`cnn_model.py`, `cnn_features.py`, `cnn_training.py`) — not yet
  audited against `models/`. Audit-only first.
- `geometric_analysis.py`, `rasterizer.py` — audit-only, no refactor yet.
- `scripts/` — several still import `from channel_heads.inference import ...`
  (acceptable for thin scripts); cleanup/archive pass pending.

## Next recommended task

**Slice 3a — CNN architecture consolidation:** move `OutletCNN`,
`OutletPairDataset`, `encode_raster_onehot`, `DEFAULT_EMBEDDING_DIM`,
`DEFAULT_TARGET_SIZE` from `channel_heads/cnn_model.py` into
`channel_heads/models/cnn.py` (promote to real); reduce `cnn_model.py` to a
shim. Byte-for-byte architecture move only — `cnn_outlet_final.pt` is loaded
`strict=True`. See `AGENT_AUDIT_CNN.md` §6 for the full 3a–3d sub-slice plan and
the explicit "do not merge the four forward-pass extractors" rule.
