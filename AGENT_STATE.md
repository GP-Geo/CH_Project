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
- **Stages 0–6:** ✅ complete; Earth feature datasets in `data/results/`.
- **Stage 7:** ⚠️ **NEEDS REVIEW — inconsistent/interrupted raster state.**
  The live manifests `data/results/raster_manifest_reg{A,B,C}.csv` (17 basins each)
  reference `data/results/_rasters_reg{A,B,C}/` paths, but those raster directories
  are **not present on disk** in `data/results/` — they were archived to
  `data/_stage7_archive_20260603_231506/`. Do **not** trust these manifests for
  downstream training: most `raster_path` entries resolve to MISSING files.
  - **regA is mid-rebuild (interrupted, 2026-06-04).** A regA raster regen was
    started and stopped; the output is fragmented:
    `data/results/_rasters_regA/` = 5 basins (toano→yoro);
    `…/_stage7_archive_…/_rasters_regA_partial_rerun/` = 12 basins (calnalpine→taiwan).
    The clean **pre-rewrite** full 17-basin set is archived at
    `…/_stage7_archive_…/_rasters_regA/` (May 31, STALE — pre-rasterizer-rewrite).
  - Before any regime is finalised, **reconcile this state**: regenerate rasters
    + manifests cleanly with the current rasterizer (preferred), rather than
    relying on the fragmented/archived sets.
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
