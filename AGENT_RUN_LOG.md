# AGENT_RUN_LOG.md — refactor run log

Append one entry per completed slice (newest at top). Keep entries short.

---

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
