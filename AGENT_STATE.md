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
- **Stages 8–11:** ✅ model artifacts and Mars predictions exist in `models/` and
  `data/Mars/model_outputs/` — built on pre-rewrite data, adequate for structural
  testing and scientific review.
- **Stages 12–14:** 🔶 threshold sensitivity, interpretation, and figures
  notebooks in `notebooks/mars/`, `notebooks/interpretation/`, and
  `notebooks/presentation/`.

## Next actions when regime is finalised

1. Restore or regenerate regime rasters:
   ```bash
   python scripts/cli/build_cnn_patches_regime.py --regime regA -v
   python scripts/cli/build_cnn_patches_regime.py --regime regB -v
   python scripts/cli/build_cnn_patches_regime.py --regime regC -v
   ```
2. Retrain: `train_cnn_regime.py` and `train_combined_xgb_regime.py` for each regime.
3. Rerun Mars pipeline: `run_mars_pipeline.py --stage all`.
4. Rerun Mars inference: `run_mars_combined_regime.py`.
