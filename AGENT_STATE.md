# AGENT_STATE.md — current project state

_Last updated: 2026-06-04_

## Branch

- **Active branch:** `refactor/package-first-architecture`
- Do not push or merge into `main`.

## Refactor status: COMPLETE

The package-first refactor is done (commit `77a80c1`). All core logic lives in
`channel_heads/`; `scripts/cli/` contains thin wrappers only. Two intentional
compatibility surfaces remain:

- `channel_heads/geometric_analysis.py` — pure re-export shim
- `channel_heads/inference/__init__.py` — thin re-export over `models.*`

## Pipeline status

See [`STAGE_ASSET_MAP.md`](STAGE_ASSET_MAP.md) for per-stage coverage.

Key facts:
- **Stages 0–3:** ✅ foundation-verified 2026-06-04. S1 (critical ΔL unit risk) resolved — `s.upstream_distance()` confirmed to return arc-degrees (CalnAlpine max=0.2404°); `LengthwiseAsymmetryAnalyzer` correctly applies `compute_meters_per_degree()` from `units.py`. Minor: `basin_config.py` lists 18 basins (includes `piedepalo`) but no `piedepalo` DEM exists on disk; `EXAMPLE_DEMS` in `paths.py` correctly has 17 entries. Does not affect training.
- **Stages 0–6:** ✅ complete; Earth feature datasets in `data/results/`.
- **Stage 7:** ✅ **RECONCILED 2026-06-04.** All three regime raster sets are on
  disk in `data/results/_rasters_reg{A,B,C}/` and their manifests resolve with
  **0 missing**:
  - **regA** — freshly regenerated with the current rasterizer (17/17 basins,
    Taiwan included): manifest `raster_status` = 23,742 ok / 2,174 invalid / **0
    failed** (the old interrupted manifest had 7,607 failed). Verified 0 missing.
  - **regB / regC** — restored from `data/_stage7_archive_20260603_231506/` (user
    confirmed rasterizer-compatible). Manifests resolve 0 missing (regB 10,612 ok;
    regC 28,954 ok).
  - The `data/_stage7_archive_20260603_231506/` copies are retained as backup.
  - Rasterizer now supports optional **multiprocess** per-basin/chunk rendering
    (`n_workers` / CLI `--workers`); default 1 is bit-identical to the prior
    serial path. Verified output-identical on real data (regA `inyo`, and
    finisterre 900-pair byte-for-byte), with ~2.4× speedup at 4 workers
    (process-based to sidestep the GIL; threads gave no gain).
- **Stages 8–9:** ✅ **retrained + validated 2026-06-04** on the reconciled
  Stage-7 rasters. All three regime CNNs (`cnn_outlet_reg{A,B,C}.pt`) and combined
  geom+CNN-emb XGBoosts (`xgb_geom_plus_cnn_emb_reg{A,B,C}.json`) were retrained;
  LOBO CV + thresholds + `ALL_MODELS_METRICS.csv` refreshed. New metrics reproduce
  the old stale numbers within noise (all single-split Δ < 0.03; largest mover is
  regA LOBO fold-AUC −0.028, still sub-threshold). Frozen production artifacts
  verified untouched. See `AGENT_STAGE_8_9_RETRAIN.md`.
- **Stages 10–11:** ⚠️ **stale — Mars re-inference pending.** Mars predictions in
  `data/Mars/model_outputs/` still come from the pre-retrain regime models; refresh
  them next (`run_mars_combined_regime.py`) now that Stages 8–9 are clean. Usable
  for structural testing only until re-run.
- **Stages 12–14:** 🔶 threshold sensitivity, interpretation, and figures
  notebooks in `notebooks/mars/`, `notebooks/interpretation/`, and
  `notebooks/presentation/` — runnable on current (stale) predictions; refresh
  after the retrain.

## Current phase (2026-06-04)

**Stages 0–9 complete; Mars re-inference (10–11) is the next wave.**
Foundation (0–3) verified with S1 resolved, regimes frozen (4), Earth networks +
pairs + labels done (5–6), regime rasters reconciled (7), and the regime
**models + validation are now retrained on the clean rasters (8–9)** — see
`AGENT_STAGE_8_9_RETRAIN.md`. Only the downstream Mars chain (10–11) and the
figures (12–14) still reflect the pre-retrain models.

## Next actions (Stage 10 → 11 Mars re-inference)

Stage 7 rasters/manifests and the Stage 8–9 regime models are clean — **do not**
regenerate them unless a regime parameter changes. The patch builder supports
`--workers N`; `scripts/run_regime_pipeline.sh <regA|regB|regC> [full|retrain]`
now has a `retrain`-only mode (Steps 4–5) and accepts regC.

1. Rerun Mars inference per regime: `run_mars_combined_regime.py --regime reg{A,B,C}`
   (regenerates Mars 5-class patches → embeddings via the **frozen**
   `cnn_outlet_final.pt` → combined predictions). Choose the Mars operating
   threshold deliberately (precision-oriented + Dd-calibration) — do **not** copy
   the Earth F1 threshold.
2. Re-run Stage 12–14 notebooks (threshold sensitivity, interpretation, figures)
   on the refreshed predictions.
