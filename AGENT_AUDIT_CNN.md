# AGENT_AUDIT_CNN.md — Slice 2: CNN / model / training ownership audit

_Audit-only. No implementation code was changed. Date: 2026-06-02._

Scope audited (read-only):

- `channel_heads/cnn_model.py`
- `channel_heads/cnn_features.py`
- `channel_heads/cnn_training.py`
- `channel_heads/models/cnn.py`
- `channel_heads/models/embeddings.py`
- `channel_heads/models/device.py`

Goal: produce a **merge-and-consolidate** plan for CNN / model / training
ownership. Newer modules are **not** assumed better; useful old logic is
preserved.

---

## 1. What each module actually is

| Module | Kind today | Real content |
|--------|-----------|--------------|
| `channel_heads/cnn_model.py` | **REAL (architecture SoT)** | `OutletCNN` (`nn.Module`), `OutletPairDataset`, `encode_raster_onehot`, `DEFAULT_EMBEDDING_DIM=4`, `DEFAULT_TARGET_SIZE=128`. Imports `NUM_CLASSES` from `.rasterizer`. The trained artifact `cnn_outlet_final.pt` is keyed to this exact architecture. |
| `channel_heads/cnn_features.py` | **REAL (Earth embedding path)** | `extract_embeddings` (**lenient** load: no `strict=`, no missing/unexpected check; returns a manifest-keyed DataFrame), `merge_cnn_features` (merge on `outlet/confluence/head_1/head_2`), `CNN_FEATURE_COLS`. |
| `channel_heads/cnn_training.py` | **REAL (training core)** | `train_cnn` loop + hyperparameter defaults (`DEFAULT_EPOCHS=60`, `LR=1e-3`, …, `HOLDOUT_BASIN="taiwan"`, `RANDOM_STATE=42`) **+ a duplicate `pick_device`**. |
| `channel_heads/models/cnn.py` | **SHIM (curated surface)** | Pure re-export of `cnn_model` + `cnn_training` symbols. No logic. |
| `channel_heads/models/embeddings.py` | **REAL (Mars Phase-5 orchestration)** | Manifest prep, embedding-table assembly (incl. skipped rows), validation, merge-with-tabular, figures, export. **Calls** `cnn_features.extract_embeddings`; it is *not* a duplicate of it. |
| `channel_heads/models/device.py` | **REAL (canonical `pick_device`)** | Consolidated in Slice 1; lazy torch import. |

Key point: `models/cnn.py` is already a thin re-export named the way the backlog
expects the canonical CNN home to be named. `models/embeddings.py` is a genuine
higher layer (Mars pipeline glue), **not** redundant with `cnn_features.py`.

---

## 2. Reference map (who imports what)

`cnn_model.py` (architecture + dataset) — widely depended on:

- `channel_heads/__init__.py` (public API)
- `channel_heads/cnn_features.py`, `channel_heads/cnn_training.py`
- `channel_heads/models/cnn.py`, `channel_heads/models/embeddings.py`
- `channel_heads/inference/regime.py` (lazy-ish, top-level import)
- `channel_heads/models/mars_combined.py` (lazy, inside `extract_logits`)
- scripts: `train_cnn_baseline.py`, `train_cnn_multiseed.py`,
  `train_cnn_regime.py`, `train_combined_xgb_phase6b.py`,
  `train_combined_xgb_regime.py`
- tests: `test_cnn_model.py`, `test_cnn_features.py`

`cnn_features.py`:

- `channel_heads/__init__.py`
- `channel_heads/models/embeddings.py` (Mars Phase 5)
- tests: `test_cnn_features.py`

`cnn_training.py`:

- `channel_heads/models/cnn.py`
- scripts: `train_cnn_baseline.py`, `train_cnn_multiseed.py`,
  `train_cnn_regime.py`
- tests: `test_cnn_model.py`

`models/cnn.py`:

- `channel_heads/models/__init__.py` only.

`models/embeddings.py`:

- `channel_heads/models/__init__.py`
- `channel_heads/pipelines/mars.py`
- tests: `test_mars_embeddings.py`

Dependency direction today: `models/cnn.py` and `models/embeddings.py` depend
**upward** on the flat `cnn_*` modules. Consolidation must flip this (flat
modules become shims that import from `models/`) without breaking any of the
import sites above.

---

## 3. Duplication map

### 3a. `pick_device` — triplicated (besides the canonical one)

| Location | Status |
|----------|--------|
| `channel_heads/models/device.py` | **canonical** (Slice 1) |
| `channel_heads/inference/device.py` | already a shim → canonical |
| `channel_heads/cnn_training.py:37` | **duplicate** (eager, module-level torch) |
| `scripts/train_combined_xgb_phase6b.py:127` | **duplicate** |
| `scripts/train_combined_xgb_regime.py:85` | **duplicate** |

All three duplicates are byte-identical logic (`mps` > `cuda` > `cpu`).
Safe to collapse onto `models/device.pick_device`.

### 3b. "load CNN state-dict + forward pass" — four divergent copies

This is the **scientifically sensitive** duplication. Four functions implement
the same shape (build `OutletCNN`, load state-dict, eval, no-aug
`OutletPairDataset`, batched forward) but differ in ways that matter:

| Function | Load mode | Output | Returns |
|----------|-----------|--------|---------|
| `cnn_features.extract_embeddings` | **lenient** (`load_state_dict` w/o `strict=`) | `embed()` | DataFrame (+`emb_*` cols) |
| `inference/regime.extract_regime_embeddings` | **strict** (raises on mismatch) | `embed()` | `np.ndarray` |
| `models/mars_combined.extract_logits` | **strict** | `forward()` logit | `np.ndarray` |
| `scripts/train_combined_xgb_phase6b.extract_emb_and_logit` | **strict** | `embed()` **and** logit | `(ndarray, ndarray)` |
| `scripts/train_combined_xgb_regime.py` (inline) | **strict** | embed (+logit) | tuple |

`models/embeddings.load_cnn_model_for_embeddings` is a fifth strict loader, but
it only *loads* (the forward pass is delegated to `extract_embeddings`).

The lenient-vs-strict difference is deliberate (regime.py's docstring calls this
out). **These must not be blindly merged** — collapsing them onto one helper
could silently change a lenient load to strict (or vice-versa) and alter which
inputs are accepted. They are owned by their respective pipelines and several
sit in out-of-scope files (`inference/regime.py`, `mars_combined.py`, scripts).

### 3c. `emb_*` column-name lists

`CNN_FEATURE_COLS` (in `cnn_features.py`) is the source of truth, but
`scripts/train_combined_xgb_*` and `models/embeddings.py` each recompute
`[f"emb_{i}" for i in range(DEFAULT_EMBEDDING_DIM)]`. Minor; `models/embeddings`
already asserts the two agree at runtime.

---

## 4. Recommended canonical ownership

Follow the established package-first pattern (real impl in `models/`, flat
module reduced to a shim), **split by concern**:

| Concern | Recommended canonical home | Flat module becomes |
|---------|---------------------------|---------------------|
| CNN architecture + dataset + dims (`OutletCNN`, `OutletPairDataset`, `encode_raster_onehot`, `DEFAULT_EMBEDDING_DIM`, `DEFAULT_TARGET_SIZE`) | `channel_heads/models/cnn.py` (promote from re-export to REAL) | `channel_heads/cnn_model.py` → shim |
| Earth embedding extraction/merge (`extract_embeddings`, `merge_cnn_features`, `CNN_FEATURE_COLS`) | `channel_heads/models/` (e.g. `models/cnn_features.py`, or fold into `models/embeddings.py`) | `channel_heads/cnn_features.py` → shim |
| Training loop + hyperparameters (`train_cnn`, `DEFAULT_EPOCHS`, …, `HOLDOUT_BASIN`, `RANDOM_STATE`) | **future** `channel_heads/training/cnn.py` | `channel_heads/cnn_training.py` → shim |
| `pick_device` (CNN/training) | `channel_heads/models/device.py` (already canonical) | drop duplicate in `cnn_training.py`; import canonical |

### What should move to `models/`

- Architecture/dataset from `cnn_model.py` → `models/cnn.py` (it is already the
  curated name; make it the real owner).
- Earth embedding features from `cnn_features.py` → a model-layer module.

### What should move to a future `training/` package

- The entire `train_cnn` loop + the `DEFAULT_*` / `HOLDOUT_BASIN` /
  `RANDOM_STATE` hyperparameters from `cnn_training.py` → `training/cnn.py`.
  Training is orthogonal to inference/model definition; it is only used by the
  `scripts/train_cnn_*` trainers. This is the natural home once a `training/`
  package is introduced. Until then, keep `cnn_training.py` real but dedup its
  `pick_device`.

### What should remain a shim

- `channel_heads/cnn_model.py`, `channel_heads/cnn_features.py`,
  `channel_heads/cnn_training.py` — reduce each to a pure re-export once their
  real content moves. Notebooks/scripts/user code import these flat paths; they
  must keep resolving.
- `channel_heads/models/cnn.py` keeps re-exporting training symbols
  (`train_cnn`, `DEFAULT_*`, `pick_device`) for its curated surface even after
  the training core relocates — it just sources them from the new homes.
- `channel_heads/__init__.py` public API stays unchanged (it imports from the
  flat modules, which will resolve through shims).

### What should NOT be consolidated now (kept as-is, flagged)

- The four divergent forward-pass extractors (§3b). Leave each with its pipeline.
  A shared `_run_cnn(..., strict: bool, want_logit: bool)` helper is *possible*
  but is a separate, test-guarded slice — and two of the copies live in
  out-of-scope files (`inference/regime.py`, `mars_combined.py`).

---

## 5. Risks

1. **Frozen artifact / architecture lock.** `cnn_outlet_final.pt` is loaded
   `strict=True` in most paths. Relocating `OutletCNN` must be a byte-for-byte
   move — no rename of attributes/layers, no dim changes. Any drift breaks every
   strict load. 5-class input (`NUM_CLASSES`) must stay 5-class.
2. **Lenient vs strict load divergence (§3b).** The biggest trap. Do not unify
   `extract_embeddings` (lenient) with the strict extractors during a "cleanup".
3. **Circular imports.** `models/cnn.py` → `models/device.py` is fine. But
   `cnn_features` ↔ `models/embeddings` (embeddings imports cnn_features today)
   must not become circular when cnn_features moves into `models/`. Sequence the
   moves so the shim points one direction only.
4. **`channel_heads/__init__.py` torch-optional guard.** CNN imports sit behind
   `try/except ImportError` (`_HAS_TORCH`). Shims must preserve import-without-
   torch behavior so the top-level package still imports on a torch-less env.
5. **Scripts are out of scope for Slice 3** but import the flat modules; their
   inline `pick_device` / extractor copies are addressed in Slice 4 (scripts),
   not here. Don't repoint scripts in the model slice.
6. **`inference/regime.py` and `mars_combined.py` are out of scope** (regime is
   transitional; mars_combined was just consolidated). Their CNN-loading copies
   stay until the regime/Earth-training audit (Slice 5).

---

## 6. Proposed implementation slices (refines backlog Slice 3)

Ordered low-risk → higher-risk; each is independently testable and commits green.

- **3a — CNN architecture → `models/cnn.py`.** Move `OutletCNN`,
  `OutletPairDataset`, `encode_raster_onehot`, `DEFAULT_EMBEDDING_DIM`,
  `DEFAULT_TARGET_SIZE` into `models/cnn.py` (real). Reduce `cnn_model.py` to a
  shim re-exporting from `models/cnn.py`. `models/cnn.py` keeps re-exporting the
  training symbols (now sourced from `cnn_training.py` until 3d). Tests:
  `test_cnn_model.py`, `test_cnn_features.py`, `test_mars_embeddings.py`,
  `test_inference_regime.py`, then full pytest.
  *Stop if* any layer/attr name or dim changes, or a strict load fails.

- **3b — `pick_device` dedup in `cnn_training.py`.** Replace the local copy with
  `from channel_heads.models.device import pick_device` (keep the name exported
  so `models/cnn.py` and the trainers still get it). Tests: `test_cnn_model.py`
  (asserts `pick_device()` returns a valid device), full pytest.
  *Stop if* device selection changes on any platform.

- **3c — Earth embedding features → models layer.** Move `extract_embeddings`,
  `merge_cnn_features`, `CNN_FEATURE_COLS` into `models/` (new
  `models/cnn_features.py` or fold into `models/embeddings.py`). Reduce
  `cnn_features.py` to a shim. Re-point `models/embeddings.py` and
  `channel_heads/__init__.py` to the canonical source (keeping shim imports
  valid). Tests: `test_cnn_features.py`, `test_mars_embeddings.py`, full pytest.
  *Stop if* the lenient-load behavior or DataFrame schema changes.

- **3d (future, needs `training/` package) — training core → `training/cnn.py`.**
  Move `train_cnn` + `DEFAULT_*` + `HOLDOUT_BASIN` + `RANDOM_STATE` into a new
  `channel_heads/training/cnn.py`. Reduce `cnn_training.py` to a shim;
  `models/cnn.py` re-exports training symbols from the new home. Scripts keep
  importing `channel_heads.cnn_training` (shim) — repointing them is Slice 4.
  Tests: `test_cnn_model.py`, full pytest. *Stop if* any default value changes.

- **(Deferred, separate risk-guarded slice) — forward-pass extractor unification.**
  Only if explicitly requested: introduce one parametrized loader/forward helper
  with `strict` and output-mode flags, with tests pinning lenient-vs-strict
  behavior, then migrate the four call sites. Touches out-of-scope files; do not
  bundle into Slice 3.

---

## 7. Bottom line

- **Canonical CNN model** → `channel_heads/models/cnn.py` (promote to real).
- **Canonical Earth embeddings** → models layer; `cnn_features.py` → shim.
- **Training** → future `channel_heads/training/cnn.py`; `cnn_training.py` →
  shim (and drop its `pick_device` duplicate now-ish via 3b).
- **`models/embeddings.py`** is correct as-is (Mars orchestration) — keep.
- **Do not** merge the four strict/lenient forward-pass extractors in the model
  slice; that is a separate, test-guarded effort touching out-of-scope files.
</content>
</invoke>
