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
- **Stages 8–11:** ⚠️ **stale — retrain pending.** Model artifacts and Mars
  predictions exist in `models/` and `data/Mars/model_outputs/`, but the regime
  CNN/XGBoost variants were trained on the *old* (pre-rewrite / interrupted)
  rasters. Now that Stage 7 is reconciled, these are the next thing to refresh.
  Usable for structural testing only until retrained.
- **Stages 12–14:** 🔶 threshold sensitivity, interpretation, and figures
  notebooks in `notebooks/mars/`, `notebooks/interpretation/`, and
  `notebooks/presentation/` — runnable on current (stale) predictions; refresh
  after the retrain.

## Current phase (2026-06-04)

**Foundation verified + Stage 7 reconciled; ready for a clean Stage 8 retrain.**
Stages 0–7 are complete/verified: foundation (0–3) checked with S1 resolved,
regimes frozen (4), Earth networks + pairs + labels done (5–6), and all three
regime raster sets + manifests are clean (7, resolve 0-missing). The regime
*models* (8) and everything downstream (9–11) are still trained on pre-rewrite
data, so the next bounded step is to retrain on the reconciled rasters.

## Next actions (Stage 8 → 11 retrain)

Stage 7 rasters/manifests are already reconciled — **do not** regenerate them
unless a regime parameter changes. The patch builder now supports
`--workers N` (multiprocess) if you ever do regenerate.

1. Retrain per regime: `train_cnn_regime.py` then `train_combined_xgb_regime.py`
   for regA / regB / regC (or `scripts/run_regime_pipeline.sh regA|regB`).
2. Validate: `eval_lobo_cv.py` + `retune_threshold_regime.py`; refresh
   `models/ALL_MODELS_METRICS.csv`.
3. Rerun Mars inference per regime: `run_mars_combined_regime.py`.
4. Re-run Stage 12–14 notebooks (threshold sensitivity, interpretation, figures)
   on the refreshed predictions.
