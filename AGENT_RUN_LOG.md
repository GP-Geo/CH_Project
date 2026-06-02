# AGENT_RUN_LOG.md — refactor run log

Append one entry per completed slice (newest at top). Keep entries short.

---

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
