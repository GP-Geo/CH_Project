# Stage 10–14 — Mars re-inference + analysis + figures on retrained regime models

_Run: 2026-06-04 (~04:00–04:09 local). Branch: `refactor/package-first-architecture`._

Follows `AGENT_STAGE_8_9_RETRAIN.md`. With the regime models retrained on the
reconciled Stage-7 rasters, this wave refreshed the **Mars** side end-to-end:
re-generated the stale Mars CNN patches with the current rasterizer, re-ran
inference (baseline + regA/B/C), and refreshed the Stage 12–14 analysis + figures.

## TL;DR

- ✅ **Stage 10** — Mars 5-class patches regenerated with the current
  (direct-final-grid) rasterizer; embeddings + `tabular_plus_cnn` refreshed via the
  **frozen** `cnn_outlet_final.pt`. 3,785 patches (3,682 ok / 103 invalid) — same
  pair coverage as before.
- ✅ **Stage 11** — regime combined inference (regA/B/C) + baseline combined,
  all on the new patches. Coupling rates moved < 1.6 pp vs the stale run and the
  regime ordering (regB > regA > regC) is preserved.
- ✅ **Stage 12–13** — threshold-sensitivity sweep + interpretation written
  (`mars_threshold_sensitivity.csv/.png`, `mars_regime_interpretation.csv`).
- ✅ **Stage 14** — 18 figures refreshed (contact sheets, per-outlet vector
  drawings, network map, ROC, variant scatters, sensitivity curve).
- ✅ Frozen production artifacts untouched (MD5 verified before/after); `pytest -q`
  **570 passed**; old Mars artifacts archived (reversible).

---

## 1. Why Stage 10 had to re-run patches (consistency)

The regime CNNs were retrained on **new-rasterizer** Earth patches (Stage 7/8).
The Mars patches on disk were dated **05-27 (old rasterizer)** and flagged
`STALE_AFTER_RASTER_FIX` in `DATA_STATUS`. Feeding old-rasterizer Mars patches to
the new CNNs would be a train/inference rasterization mismatch. `mars_patches.py`
is frozen to the **same 5-class direct-final-grid contract** as the Earth
rasterizer, so regenerating produces Mars patches consistent with the retrained
models. The "keep Mars as-is" decision refers to Mars *geometry* (topology/pairs/
features are shared across regimes), **not** to surviving the rasterizer rewrite —
so the shared patches are regenerated **once** and reused across all regimes.

**Stage 10 steps (all read-only on frozen models):**
1. `run_mars_pipeline.py --stage patches` → 3,785 patches (3,682 ok / 103 invalid)
   + `mars_cnn_patch_index.parquet`. All `.npy` rewritten 04:01.
2. `run_mars_pipeline.py --stage embeddings` → `mars_cnn_embeddings.parquet` +
   `mars_model_input_tabular_plus_cnn.parquet` via **frozen** `cnn_outlet_final.pt`.

---

## 2. Stage 11 — Mars inference (touching rates vs stale)

Per-regime: `run_mars_combined_regime.py --regime <r>` (recomputes regime
embeddings from the new patches with `cnn_outlet_<r>.pt`, applies
`xgb_geom_plus_cnn_emb_<r>.json` at its operating threshold). Baseline combined:
`run_mars_pipeline.py --stage combined`.

| Regime | operating thr | new touching | old touching | Δ | new high-conf (≥0.8) |
|---|---|---|---|---|---|
| regA | 0.7563 | 53.0% | 53.7% | −0.7 pp | 45.7% |
| regB | 0.7793 | 64.1% | 63.9% | +0.2 pp | 61.7% |
| regC | 0.7594 | 38.6% | 37.1% | +1.5 pp | 34.3% |

All 3,682 pairs / 391 networks; probs in [0,1], 0 NaN. Coupling rates reproduce
the stale run within < 1.6 pp — the expected consistency from regenerating patches
with the new rasterizer + retrained models. Regime ordering preserved
(regB sparsest network → most coupling; regC intermediate → least).

---

## 3. Stage 12 — threshold sensitivity

`mars_threshold_sensitivity.csv` + `.png`: touching fraction vs decision threshold
(0.30–0.95) per regime, with the operating (max-precision@recall≥0.5) point marked.

**Operating-threshold decision (documented, not silently changed):** the Mars
operating point per regime stays the **precision-oriented** max-precision threshold
carried from Stage 9 — *not* the Earth F1-optimal threshold (per `docs/modeling.md`,
which warns against copying the Earth F1 point to label-free Mars). The sweep shows
sensitivity is moderate near the operating point (regime curves are smooth, no
cliff), so the precision-oriented choice is stable. A future, more permissive
operating point can be read directly off the sweep CSV if desired — that remains a
deliberate scientific choice, surfaced here rather than baked in.

---

## 4. Stage 13 — interpretation

`mars_regime_interpretation.csv`:

| Regime | touching frac | networks w/ ≥1 touch | frac networks w/ touch | mean per-network touch frac |
|---|---|---|---|---|
| regA | 0.530 | 382 / 391 | 0.977 | 0.571 |
| regB | 0.641 | 383 / 391 | 0.980 | 0.682 |
| regC | 0.386 | 372 / 391 | 0.951 | 0.443 |

**Cross-regime agreement** (same call across all three regimes at their operating
thresholds, n=3,682 pairs): **all-touching 33.9%**, all-non-touching 30.8% →
**64.7% consensus**; the remaining 35.3% are regime-sensitive. This is the
project's calibration-uncertainty quantification — no single regime is "correct".

---

## 5. Stage 14 — figures (18 refreshed)

All under `data/Mars/model_outputs/figures_combined/` unless noted:
- `contact_sheet_high_conf_both.png`, `contact_sheet_disagreement.png` — **vector
  polyline** contact sheets (per the project's preferred Mars figure style).
- `per_outlet/mars_outlet_touching_pairs_net*.png` (×10) — vector per-outlet drawings.
- `mars_networks_channels.png`, `roc_curves.png` (`data/results/figures_models/`).
- `scatter_emb_vs_logit.png`, `scatter_tabular_vs_emb.png`,
  `probability_histograms.png`, `touching_pct_by_variant.png` (variant comparison).
- `mars_threshold_sensitivity.png` (Stage 12 sweep).
- `generate_poster_figures.py` → `data/results/final_figures/`.

---

## 6. Frozen artifacts + bug fix + archive

- **Frozen untouched:** `cnn_outlet_final.pt` (`b812f0b3…`) and
  `xgb_touching_classifier.json` (`ac1a7912…`) — MD5 identical before/after the
  whole wave; used read-only for baseline embeddings/inference.
- **Bug fixed:** `scripts/cli/make_result_figures.py` used `parents[1]` (resolved
  to `scripts/data/...`) — same refactor artifact fixed in Stage 9 for
  `eval_lobo_cv.py` / `retune_threshold_regime.py`. Now `parents[2]`. `pytest`
  570 passed; ruff clean.
- **Archived (reversible, per "never delete data"):**
  `data/_mars_stage10_archive_20260604_040046/` holds the old patch index,
  embeddings, `tabular_plus_cnn`, and all old combined predictions.

---

## 7. Status

The pipeline is now **end-to-end current on the reconciled rasters** (Stages 0–14).
Mars predictions, analysis, and figures all derive from the retrained regime models
on new-rasterizer patches. Remaining judgement call for any final publication: the
Mars operating threshold (precision-oriented now; revisit deliberately via the
sensitivity CSV if a different precision/recall balance is wanted).
