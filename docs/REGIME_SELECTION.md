# Regime Selection — Stage 4 calibration rationale

This document freezes the parameter choices for the three Earth network
complexity regimes (regA, regB, regC) that are upstream of all regime-trained
models. **Do not change these parameters without a full retrain of every
downstream artifact** (CNN, XGBoost, thresholds, Mars inference outputs).

The canonical evidence is in
`notebooks/regime/00_calibration_overview.ipynb` and the calibration data
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

| Regime | `threshold_km2` | `pre_remove_max_order` | `order_gap_to_prune` | Network character |
|--------|-----------------|------------------------|----------------------|-------------------|
| **regA** | 0.05 km² | ≤ 2 (drop 1st+2nd order) | ≥ 4 | Dense base network, aggressive tip removal |
| **regB** | 0.25 km² | ≤ 1 (drop 1st order only) | ≥ 4 | Sparse base network, minimal pruning |
| **regC** | 0.10 km² | ≤ 1 (drop 1st order only) | ≥ 4 | Intermediate density, minimal pruning |

All three regimes use `coupling_n_workers=4` and `min_prefilter_px=30.0`.

The three regimes **bracket the plausible range** of Earth network complexities
that match Mars: regB is the sparsest (closest to Mars Dd-hull medians in the
bulk of the sweep), regA is the densest, and regC sits between them. Running
all three and comparing Mars inference outputs is the project's approach to
quantifying calibration uncertainty — no single regime is declared "correct".

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

| Artifact | Location | Keyed to |
|----------|----------|----------|
| Regime feature dataset | `data/results/master_dataset_reg{A,B,C}.csv` | all three |
| Per-basin stats | `data/results/build_earth_features_reg{A,B,C}_stats.csv` | all three |
| Regime CNN | `models/cnn_outlet_reg{A,B,C}.pt` | per regime |
| Regime XGBoost | `models/xgb_geom_plus_cnn_emb_reg{A,B,C}.json` | per regime |
| Regime threshold | `models/optimal_threshold_geom_plus_cnn_emb_reg{A,B,C}.txt` | per regime |
| Regime rasters | `data/results/_rasters_reg{A,B,C}/` | per regime (STALE — pre-rewrite) |
| Mars regime predictions | `data/Mars/model_outputs/mars_combined_reg{A,B,C}_predictions.parquet` | per regime |

See `docs/DATA_STATUS.md` for current status of each artifact.

---

## Related docs

- `docs/modeling.md` — model variants and the Mars threshold issue
- `docs/PIPELINE_RERUN.md` — regeneration order (regimes are Stage 4–5)
- `notebooks/regime/00_calibration_overview.ipynb` — runnable calibration evidence
- `channel_heads/regimes.py` — canonical code for `Regime` dataclass and `REGIMES` dict
