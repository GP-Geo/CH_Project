# Regime Selection — Stage 4 calibration rationale

This document freezes the parameter choices for the three Earth network
complexity regimes (regA, regB, regC) that are upstream of all regime-trained
models. **Do not change these parameters without a full retrain of every
downstream artifact** (CNN, XGBoost, thresholds, Mars inference outputs).

The canonical evidence is in
`notebooks/pipeline/04_earth_mars_regime_calibration.ipynb` and the calibration data
under `data/results/drainage_density_calibration/`.

---

## The calibration problem

Earth training networks are extracted with TopoToolbox using a fixed area
threshold (cells above which a pixel is considered a channel). The threshold
controls network *density*: a low threshold produces fine, dense networks with
many short segments; a high threshold yields sparse, trunk-dominated networks.

Mars valley networks are already-mapped polylines — their density reflects the
geomorphic signal that survived preservation. Applying an Earth-trained model
to Mars only makes sense if the Earth training networks have **comparable
structural complexity** to the Mars networks being scored.

The **drainage-density convex-hull metric** (Dd-hull) is the calibration
signal: for each basin, compute the total stream length divided by the convex
hull area of the channel network. Matching the Earth per-basin Dd-hull
distribution to the Mars Dd-hull distribution selects which Earth network
complexity is most analogous to the Mars networks.

---

## Design space

Each regime bundles three parameters:

| Parameter | Controls |
|-----------|----------|
| `threshold_km2` | Minimum contributing area (km²) to initiate a channel — sets base network density |
| `pre_remove_max_order` | Drop all Strahler-order segments ≤ this value before coupling analysis — removes the finest-order tips |
| `order_gap_to_prune` | Remove a segment if its order is ≥ this many orders below the local trunk — prunes dangling spur segments |

The sweep tested thresholds from 0.01 km² to 0.25 km² and pruning levels from
full network to Strahler ≥ 5, measuring Dd-hull against Mars reference geometry
(`data/results/drainage_density_calibration/complexity_calibration/dd_master_sweep_complexity.csv`).

---

## Frozen regime presets

Canonical code: `channel_heads/regimes.py`, `REGIMES` dict.

These are the **top-3 data-driven regimes** by `scientific_score` (lower =
better) from the eligibility-constrained candidate sweep in
`notebooks/pipeline/04_earth_mars_regime_calibration.ipynb` (§3), written to
`selected_regimes_AE.csv`. All three are plain `trim` (drop 1st order only) with
**no order-gap delta pruning**.

| Regime | `threshold_km2` | `pre_remove_max_order` | `order_gap_to_prune` | `scientific_score` | Network character |
|--------|-----------------|------------------------|----------------------|--------------------|-------------------|
| **regA** | 0.20 km² | ≤ 1 (drop 1st order only) | 0 (none) | 3.986 (best) | Intermediate density |
| **regB** | 0.25 km² | ≤ 1 (drop 1st order only) | 0 (none) | 4.089 | Sparsest base network |
| **regC** | 0.15 km² | ≤ 1 (drop 1st order only) | 0 (none) | 4.168 | Densest base network |

All three regimes use `coupling_n_workers=4` and `min_prefilter_px=30.0`.

The three regimes are the best-scoring, **scientifically defensible** candidates
that survived the hard filters (`length_retained ≥ 0.30`, `median Strahler ≥ 2`,
low-order pruning only) and span the practical threshold neighbourhood
(0.15–0.25 km²) whose Earth Dd-hull medians sit closest to Mars. Running all
three and comparing Mars inference outputs quantifies calibration uncertainty —
no single regime is declared "correct".

> **Prior presets (superseded).** These replace the earlier hand-frozen presets
> — regA (T=0.05, pre_remove=2, order_gap=4), regB (T=0.25, pre_remove=1,
> order_gap=4), regC (T=0.10, pre_remove=1, order_gap=4). The `*_reg{A,B,C}`
> model chain was **retrained on the new presets on 2026-06-13** (see
> "Downstream artifacts keyed to regime" below), so on-disk artifacts are valid.

---

## Selection evidence

The calibration sweep is visualised in several figure sets under
`data/results/drainage_density_calibration/`:

| Subdirectory | Key output |
|--------------|------------|
| `complexity_calibration/` | Per-threshold/pruning Dd-hull vs Mars; regime grid comparison |
| `strahler_trim/` | Effect of Strahler trimming on Dd-hull; Earth vs Mars Strahler distributions |
| `full/`, `practical_range/` | Full-sweep and practical-range threshold scans |
| `simple_presentation/` | Summary figures for the paper |

The figures `fig_regime_grid_dd_hull.png` and `fig_mars_vs_earth_at_matched_complexity.png`
(under `complexity_calibration/`) are the primary evidence that these three
parameter combinations produce Earth networks whose Dd-hull distributions
overlap with Mars.

---

## Threshold-range robustness (checked 2026-06-14)

The frozen choices were stress-tested by **extending the threshold cap from
0.5 km² to 1.5 km²** and re-scoring every variant (the recompute
`regime_optimization/dd_sweep_highthr_to1p5.csv` reproduces the dense-low sweep
to `diff = 0` at the 0.5 km² overlap). **The three regimes survive unchanged:**

- **`trim` degrades monotonically with threshold** (score 0.5 → 4.64, 0.75 →
  5.36, 1.0 → 6.00) and becomes **ineligible by ~1.5 km²** (median Strahler < 2).
  Trimming the 1st-order fringe of an already-coarse network shrinks the convex
  hull faster than the length, so Dd-hull *rises* (1.37 → 1.88 → 2.13), moving
  away from Mars (0.224). No better `trim`/`prune` regime exists above 0.5 km².
- **Delta-3 / delta-4 are inert** at ≥ ~0.15 km²: identical Dd-hull and Strahler,
  3rd–4th-decimal length change only — the order-gap rule never fires on shallow
  coarse networks (this confirms §3a of notebook 04 empirically at high
  thresholds).
- A `full` (no-pruning) candidate at ~1.0 km² scores 4.079 (≈ regB) but matches
  Mars by **threshold alone, not pruning** — a different simplification
  philosophy, still far denser than Mars in absolute Dd. Not adopted; the three
  `trim` regimes stand.

See [`regimes_summary.md`](regimes_summary.md#threshold-range-robustness-checked-2026-06-14)
for the at-a-glance version.

---

## What is frozen

These items are **upstream of all trained models** and must not change without
triggering a full retrain:

1. `threshold_km2` for all three regimes
2. `pre_remove_max_order` for all three regimes
3. `order_gap_to_prune` for all three regimes
4. The `min_prefilter_px = 30.0` floor (matches the production `2 * sqrt(300)` effective floor)
5. The 5-class CNN patch contract (BACKGROUND / BRANCH_A / BRANCH_B / OTHER_STREAMS / CONFLUENCE_MARKER)

The `coupling_n_workers` value affects only runtime performance, not results.

---

## Downstream artifacts keyed to regime

> **Retrained 2026-06-13 on the redefined presets** (commit `e5a1083`): the
> full chain (Stage 5 build → CNN → XGBoost → threshold → Mars inference) was
> regenerated on the data-driven regimes above, so the on-disk regime
> artifacts and Mars predictions are **valid**. New operating thresholds:
> regA **0.769133**, regB **0.773238**, regC **0.810635**
> (`models/optimal_threshold_geom_plus_cnn_emb_reg{A,B,C}.txt` are
> authoritative). If a preset is ever changed again, regenerate in that same
> order — see `docs/PIPELINE_RERUN.md`.

| Artifact | Location | Keyed to |
|----------|----------|----------|
| Regime feature dataset | `data/results/master_dataset_reg{A,B,C}.csv` | all three |
| Per-basin stats | `data/results/build_earth_features_reg{A,B,C}_stats.csv` | all three |
| Regime CNN | `models/cnn_outlet_reg{A,B,C}.pt` | per regime |
| Regime XGBoost | `models/xgb_geom_plus_cnn_emb_reg{A,B,C}.json` | per regime |
| Regime threshold | `models/optimal_threshold_geom_plus_cnn_emb_reg{A,B,C}.txt` | per regime |
| Regime rasters | `data/results/_rasters_reg{A,B,C}/` | per regime (reconciled 2026-06-04; inputs to the 2026-06-13 retrain) |
| Mars regime predictions | `data/Mars/model_outputs/mars_combined_reg{A,B,C}_predictions.parquet` | per regime |

See `docs/DATA_STATUS.md` for current status of each artifact.

---

## Related docs

- `docs/regimes_summary.md` — at-a-glance quick reference for the three regimes
- `docs/modeling.md` — model variants and the Mars threshold issue
- `docs/PIPELINE_RERUN.md` — regeneration order (regimes are Stage 4–5)
- `notebooks/pipeline/04_earth_mars_regime_calibration.ipynb` — runnable calibration evidence
- `channel_heads/regimes.py` — canonical code for `Regime` dataclass and `REGIMES` dict
