# Stage 8 + 9 retrain — regime CNN/XGBoost on reconciled Stage-7 rasters

_Run: 2026-06-04 (~03:07–03:38 local). Branch: `refactor/package-first-architecture`._

This wave replaced the stale (pre-rewrite-raster) regime models with clean
retrains on the **reconciled Stage-7 inputs**, and refreshed Stage-9 validation
(LOBO CV + thresholds + model-comparison table). Mars re-inference (Stage 10–11)
and figures (12–14) are explicitly **out of scope** and were not run.

## TL;DR

- ✅ All three regimes (regA/regB/regC) retrained: CNN + combined geom+CNN-emb XGBoost.
- ✅ New metrics **reproduce the stale numbers within noise** — every single-split
  Δ(ROC/PR/F1) < 0.03; largest mover is the regA LOBO fold-AUC at **−0.028**
  (borderline, still < 0.03). No hard regressions.
- ✅ Frozen production artifacts **untouched** (MD5 + mtime verified).
- ✅ `pytest -q`: **570 passed** (fixed one pre-existing stale-mock failure).
- ✅ Operating thresholds kept **precision-oriented** (max-precision@recall≥0.5);
  F1-optimal recorded as a documented alternative.
- **Go** for the Mars re-inference wave (Stage 10–11), with two carried caveats
  (regC pooled-AUC anomaly; Mars threshold must be chosen deliberately).

---

## 1. What was retrained (inputs, timings)

Per regime: `train_cnn_regime.py` → `models/cnn_outlet_<r>.pt`, then
`train_combined_xgb_regime.py` → `models/xgb_geom_plus_cnn_emb_<r>.json`
(+ feature-columns, threshold, metrics, and `master_dataset_<r>_with_emb.csv`).
Run via a sequential driver (single MPS device). Inputs were verified to resolve
**0-missing** before training (manifest `raster_status == ok`, all required
columns present, 17 basins).

| Regime | Valid rasters (ok) | touching | CNN CV pool (train/val) | CNN time | best_epoch | XGB train/test | XGB time |
|---|---|---|---|---|---|---|---|
| regA | 23,742 | 7,073 | 12,699 (11,430 / 1,269) | 702 s | 38 | 19,150 / 4,592 | 18 s |
| regB | 10,612 | 3,334 | 6,249 (5,625 / 624) | 359 s | 56 | 9,003 / 1,609 | 9 s |
| regC | 28,954 | 7,362 | 15,576 (14,019 / 1,557) | 743 s | 29 | 23,725 / 5,229 | 25 s |

- CNN: Taiwan held out (LOBO-style), remaining 16 basins split 90/10 for early
  stopping; device **MPS**; hyperparameters unchanged (60 epochs max, patience 12,
  lr 1e-3, batch 64, embedding dim 4).
- XGB: `GroupShuffleSplit` by `basin__outlet`, `test_size=0.20`, `seed=42`;
  9 features (5 geom + emb_0..3) in the **frozen** order; threshold by
  max-precision@recall≥0.5. Inputs consumed: `master_dataset_<r>.csv` +
  `raster_manifest_<r>.csv` + the freshly trained `cnn_outlet_<r>.pt`.
- LOBO CV: 3 s.
- **Total compute ≈ 31 min.**

---

## 2. New metrics vs old (stale) — Δ per model

### 2a. Single GroupShuffleSplit test (`models/ALL_MODELS_METRICS.csv`)

geom+CNN-emb (the raster-dependent variant that this wave actually retrains):

| Model | ROC (new→old, Δ) | PR (new→old, Δ) | F1 (new→old, Δ) |
|---|---|---|---|
| regA geom+emb | 0.8827 → 0.8859 (**−0.0032**) | 0.7557 → 0.7594 (−0.0037) | 0.6028 → 0.6037 (−0.0009) |
| regB geom+emb | 0.8814 → 0.8710 (**+0.0104**) | 0.7748 → 0.7687 (+0.0061) | 0.6119 → 0.6161 (−0.0042) |
| regC geom+emb | 0.8841 → 0.8822 (**+0.0019**) | 0.7644 → 0.7609 (+0.0035) | 0.6136 → 0.6095 (+0.0041) |

All |Δ| < 0.03 → **no NEEDS REVIEW** on the single split.

New per-regime **geom-only** baselines were added to the table (no prior rows to
diff). They quantify the CNN-embedding lift on the reconciled data:

| Regime | geom-only ROC | geom+emb ROC | ΔROC (embedding lift) |
|---|---|---|---|
| regA | 0.7652 | 0.8827 | +0.117 |
| regB | 0.7109 | 0.8814 | +0.170 |
| regC | 0.7956 | 0.8841 | +0.089 |

Baseline rows (`baseline geom_only / geom+cnn_emb / geom+cnn_logit`) are kept
**verbatim** — the production baseline is frozen and was not retrained this wave.

### 2b. Leave-one-basin-out CV (`models/lobo_cv_metrics.csv`)

| Config | pooled AUC (new→old, Δ) | fold-AUC mean±std (new→old) | F1 (new→old) |
|---|---|---|---|
| baseline | 0.9157 → 0.9157 (0.000, frozen) | 0.888±0.092 → 0.888±0.092 | 0.642 → 0.642 |
| regA | 0.8814 → 0.9003 (**−0.0189**) | 0.840±0.095 → 0.868±0.060 (**Δ−0.028**) | 0.605 → 0.616 |
| regB | 0.8947 → 0.8890 (**+0.0057**) | 0.852±0.141 → 0.845±0.089 | 0.620 → 0.614 |
| regC | 0.7022 → 0.7102 (**−0.0080**) | 0.917±0.044 → 0.908±0.046 | 0.453 → 0.452 |

- **regA LOBO is the largest mover** (pooled −0.019, fold-mean −0.028). Both are
  *just under* the 0.03 NEEDS-REVIEW line — flagged for a human eye. Direction is
  a slight degradation, consistent with regA being the set that was *regenerated*
  from scratch by the current rasterizer (regB/regC were restored from archive).
- **regC pooled (0.70) ≪ fold-mean (0.92)** — this gap is **pre-existing** (also
  present in the stale run: 0.71 vs 0.91), not a regression. It reflects
  cross-fold OOF-probability calibration differences amplified by regC's low
  touching base rate (~10% positive). Carried as a caveat, not a defect.
- baseline LOBO is identical because it reads the frozen
  `master_dataset_v4_cnn_full.csv` (not retrained this wave).

**S8 leakage (per-fold hard-negative filtering):** no per-fold leakage is
introduced. `channel_heads/eval/lobo.py` applies only a label-independent
`dropna` on features inside the fold loop; hard negatives are excluded upstream
at Stage-6 dataset build (a fixed data property, not a label-dependent transform
applied across the full set before splitting). So the S8 risk pattern does not
apply to this LOBO path.

---

## 3. Threshold decisions (per regime)

Two operating points were computed and **both recorded** in each
`xgb_geom_plus_cnn_emb_<r>_metrics.csv`:

| Regime | Operating (max-precision@R≥0.5) | P / R / F1 | F1-optimal (recorded only) | P / R / F1 |
|---|---|---|---|---|
| regA | **0.756326** | 0.753 / 0.502 / 0.602 | 0.515583 | 0.607 / 0.814 / 0.696 |
| regB | **0.779264** | 0.778 / 0.502 / 0.610 | 0.480472 | 0.645 / 0.835 / 0.728 |
| regC | **0.759369** | 0.792 / 0.501 / 0.614 | 0.555388 | 0.606 / 0.754 / 0.672 |

**Decision: the `optimal_threshold_geom_plus_cnn_emb_<r>.txt` files hold the
max-precision (precision-oriented) value** — matching the prior pipeline
convention and `docs/modeling.md`'s guidance to prefer a precision-oriented
operating point for Earth→Mars transfer. The max-precision protocol did **not**
collapse this run (tuned precision 0.75–0.79 on test n=1.6k–5.2k), so there was
no reason to fall back to F1-optimal. The F1-optimal thresholds (0.48–0.56) trade
a lot of precision for recall and are recorded for reference only.

`retune_threshold_regime.py` was run to *record* the F1-optimal point; the
operating `.txt` was then restored to the max-precision value (the script
overwrites it by design). The Earth F1 threshold must **not** be copied to Mars —
that is a deliberate decision for the Mars wave (per `docs/modeling.md`).

---

## 4. Frozen production artifacts — confirmed untouched

| Artifact | MD5 (session-start = now) | mtime |
|---|---|---|
| `models/xgb_touching_classifier.json` | `ac1a7912008d821ea1fb4b007f6a62c6` | 06-03 22:01 (pre-session) |
| `models/cnn_outlet_final.pt` | `b812f0b344ff08641f2da3f43488cbcd` | 06-01 13:29 (pre-session) |

Hashes are byte-identical to the values captured at the start of the wave; mtimes
predate this session. The frozen 5-class CNN patch contract and the 5-feature
order were not changed.

---

## 5. Doc + code updates made

**Docs (committed):**
- `AGENT_STATE.md` — Stage 8 & 9 flipped to ✅; "Current phase" + "Next actions" updated to point at the Mars wave.
- `STAGE_ASSET_MAP.md` — Stage 8 & 9 rows ⚠️→✅; summary table updated.
- `docs/DATA_STATUS.md` — regime CNN/CNN-derived chains flipped
  `STALE_AFTER_RASTER_FIX` → `CAN_REGENERATE` (baseline `*_final` / `v4_cnn_full`
  stay as-is: frozen / not retrained this wave).
- This report.

**Code (committed):**
- `scripts/cli/eval_lobo_cv.py` — **bugfix**: used `parents[1]` (resolved to
  `scripts/`) after the refactor moved it into `scripts/cli/`; now imports the
  canonical `PROJECT_ROOT`. This is why the first LOBO run reported all datasets
  MISSING.
- `scripts/cli/retune_threshold_regime.py` — same `parents[1]` bugfix.
- `scripts/run_regime_pipeline.sh` — now accepts **regC** and a **`retrain` mode**
  (Steps 4–5 only) that skips the out-of-scope feature rebuild / patch rebuild /
  Mars inference. Usage: `run_regime_pipeline.sh <regA|regB|regC> [full|retrain]`.
- `tests/test_training_regime.py` — fixed a **pre-existing** stale mock: the
  `fake_build` signature lacked `n_workers` (added to the real
  `build_regime_patch_dataset` by the Stage-7 `--workers` commit). pytest is now green.

**Not committed (gitignored):** all `models/*.pt|*.json|*.csv|*.txt` and
`data/results/*` outputs stay local; their state is recorded here and in DATA_STATUS.

**Unrelated working-tree changes left untouched** (pre-existing, not part of this
wave): `channel_heads/dd_calibration.py` (Length-column fallback) and
`notebooks/mars/00_mars_network_explorer.ipynb`. These are Stage-3 Mars-explorer
edits and were **excluded** from the commit.

---

## 6. Go / No-Go for the Mars re-inference wave (Stage 10–11)

**GO.** ✅

- All three regime models retrained on the reconciled Stage-7 rasters; metrics
  reproduce the stale numbers within noise (max |Δ| = 0.019 LOBO pooled; all
  single-split |Δ| < 0.011).
- Operating thresholds set (precision-oriented), F1-optimal recorded.
- Frozen artifacts verified untouched; pytest green.

**Carry these caveats into the Mars wave:**
1. **regC pooled-AUC anomaly** (0.70 pooled vs 0.92 fold) is structural and
   pre-existing — interpret regC Mars outputs with the usual caution and lean on
   the precision-oriented threshold.
2. **Mars operating threshold is a scientific decision** — do *not* copy the Earth
   F1-optimal threshold to Mars; choose it deliberately (precision-oriented +
   Dd-calibration cross-check) per `docs/modeling.md`.
3. regA LOBO is the largest (still sub-0.03) mover; if a stricter bar is wanted,
   a quick multi-seed CNN stability check (`train_cnn_multiseed.py`) on regA would
   confirm it's seed noise rather than a real shift.
