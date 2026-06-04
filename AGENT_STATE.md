# AGENT_STATE.md — current project state

_Last updated: 2026-06-04_

## Branch

- **Active branch:** `refactor/package-first-architecture`
- Do not push or merge into `main`.

## Refactor status: COMPLETE

The package-first refactor is done. All logic lives in `channel_heads/`,
**including the CLI**: the former `scripts/cli/*` wrappers are now the
`channel_heads/cli/` package (a dispatcher + one module per command), run via
`python -m channel_heads <command>` or the `channel-heads` console script.

The previous compatibility shims have been **removed** (2026-06-04): the
`channel_heads/geometric_analysis.py` and `channel_heads/inference/` re-export
shims are gone — import from the canonical `channel_heads.features.*` /
`channel_heads.models.*` modules.

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
- **Stages 10–11:** ✅ **Mars re-inference done 2026-06-04.** Mars 5-class patches
  regenerated with the current rasterizer (3,682 ok / 103 invalid), embeddings +
  `tabular_plus_cnn` refreshed via the frozen `cnn_outlet_final.pt`, and combined
  inference re-run (baseline + regA/B/C). Coupling rates moved < 1.6 pp vs stale;
  regime ordering preserved. See `AGENT_STAGE_10_14_MARS.md`.
- **Stages 12–14:** ✅ **refreshed 2026-06-04.** Threshold-sensitivity sweep +
  interpretation written (`mars_threshold_sensitivity.csv/.png`,
  `mars_regime_interpretation.csv`); 18 figures regenerated (vector contact sheets,
  per-outlet drawings, network map, ROC, variant scatters). Mars operating
  threshold kept precision-oriented (documented, not the Earth F1 point).

## Current phase (2026-06-04)

**Pipeline end-to-end current on the reconciled rasters (Stages 0–14).**
Foundation (0–3) verified with S1 resolved, regimes frozen (4), Earth networks +
pairs + labels done (5–6), regime rasters reconciled (7), regime models +
validation retrained (8–9, `AGENT_STAGE_8_9_RETRAIN.md`), and the **Mars chain +
analysis + figures are refreshed on the retrained models (10–14,
`AGENT_STAGE_10_14_MARS.md`)**. No stage is now built on pre-rewrite data.

## Next actions (final review / publication)

The compute pipeline is complete and self-consistent. Remaining items are
judgement / polish, not rebuilds:

1. **Mars operating threshold** — currently precision-oriented (max-precision@
   recall≥0.5) per regime. If a different precision/recall balance is wanted for
   the final result, read it off `data/Mars/model_outputs/mars_threshold_sensitivity.csv`
   and document the choice (a deliberate scientific decision — not the Earth F1 point).
2. **Optional robustness:** regA multi-seed CNN check (`train_cnn_multiseed.py`)
   if the Stage-9 regA LOBO fold-AUC mover (−0.028, sub-threshold) needs tightening.
3. Final figure/poster polish via `notebooks/presentation/` if presentation-ready
   styling is needed beyond the headless renders.
