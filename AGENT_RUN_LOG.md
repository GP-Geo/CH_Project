# AGENT_RUN_LOG.md — refactor run log

Append one entry per completed slice (newest at top). Keep entries short.

---

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
