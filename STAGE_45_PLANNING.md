# Stage 4/5 Planning

Planning note for the next active development: Stage 4 (regime calibration
cleanup) and Stage 5 (Earth network QA before retraining).

_Written: 2026-06-03. See `STAGE_ASSET_MAP.md` for full pipeline coverage._

---

## Why these two stages first

All downstream work — raster regeneration, CNN retraining, Mars inference
update — depends on having verified, frozen Earth networks. The current
situation:

- Regime presets (`regA/B/C`) exist in `channel_heads/regimes.py`.
- Feature datasets (`master_dataset_reg{A,B,C}.csv`) exist but were built
  on pre-rewrite rasters.
- The calibration decision was made historically but is not captured in a
  clean, self-contained notebook.
- There is **no notebook that checks whether the regime-generated Earth
  networks are scientifically sound** before model training.

Stage 4 closes the documentation gap. Stage 5 creates the QA gate.

---

## Stage 4 — What needs to happen

### Current state

`notebooks/regime/00_calibration_overview.ipynb` exists. The calibration
data lives under `data/results/drainage_density_calibration/` (threshold
sweeps, pruning maps, complexity calibration, DD hull comparisons — 39 MB
of figures and CSVs). `channel_heads/regimes.py` has the frozen `Regime`
objects with threshold, pruning, and complexity parameters.

The gap is not in the code — it is in **whether the notebook is still
runnable** (it predates the package-first refactor, so imports may point
at old module paths) and **whether the regime choice is documented as a
decision**, not just as parameters.

### Work items

1. **Audit `notebooks/regime/00_calibration_overview.ipynb`**
   - Check that all imports resolve via canonical package paths
     (e.g., `channel_heads.dd_calibration`, `channel_heads.regimes`,
     `channel_heads.io.paths`)
   - Verify it produces the key calibration figures from the frozen data
   - If imports are stale, update them (Slice 10 notebook update)

2. **Write `docs/REGIME_SELECTION.md`** (or add a section to `docs/modeling.md`)
   - State the chosen regimes (`regA/B/C`) with their parameters
   - Record the selection rationale (why these thresholds, why this pruning,
     what Earth-Mars metric guided the choice)
   - Mark this as frozen: "these parameters are upstream of all trained models
     and must not change without a full retrain"

### Package support already exists

`channel_heads/regimes.py` has all three `Regime` objects.
`channel_heads/dd_calibration.py` has `evaluate_basin_metrics`,
`summarize_threshold_metrics`, `choose_best_threshold`, `mars_network_strahler`,
`mars_network_table`, `mars_network_geometry`. No new package code needed for
Stage 4 — only notebook + documentation work.

---

## Stage 5 — What needs to happen

### The gap

There is no notebook that answers: *after building the regime Earth networks,
are the per-basin results scientifically reasonable?*

Specific questions that need inspection:
1. Which basins have very few outlets after regime pruning?
2. Which basins lost the most network length to aggressive pruning?
3. Are there basins where the regime threshold damaged the network structure?
4. For a few representative basins, do the extracted networks look visually
   correct when overlaid on the DEM?
5. Are the touching/non-touching label distributions reasonable per basin?

Currently, `data/results/build_earth_features_reg{A,B,C}_stats.csv` captures
some per-basin statistics, but they are not visualised or reviewed in any notebook.

### New notebook: `notebooks/analysis/05_earth_network_qa.ipynb`

**Inputs:**
- `data/results/master_dataset_reg{A,B,C}.csv` (per-regime labeled pair datasets)
- `data/results/build_earth_features_reg{A,B,C}_stats.csv` (per-basin stats)
- `data/cropped_DEMs/` (for network overlay visualizations)

**Calls (package functions that already exist):**
- `channel_heads.io.paths` — `RESULTS_DIR`, `CROPPED_DEMS_DIR`
- `channel_heads.regimes` — regime objects for threshold / pruning parameters
- `channel_heads.basin_config` — basin metadata (lat, DEM path)
- `channel_heads.training.regime` — `resolve_regime_basins`
- Possibly `channel_heads.dd_calibration.evaluate_basin_metrics` for
  on-the-fly metrics if needed

**What it should show:**
1. Per-basin outlet count, touching/non-touching ratio, network stats table
   across all three regimes
2. Flag basins with fewer than N outlets or extreme touching ratios
3. For 2–3 flagged basins: DEM overlay with extracted network (vector lines),
   colored by pruning tier
4. Distribution plots: delta-L per regime, feature distributions by label

**New package code needed (if any):**
The notebook will largely call existing package functions. The main potential
gap is a helper to load a per-basin regime result CSV and join it with basin
metadata for visualization. This could be a thin `channel_heads/viz/earth_qa.py`
or just inline in the notebook — decide during implementation.

---

## Implementation order

```
1. Audit 00_calibration_overview.ipynb imports (quick, 1–2 hrs)
   → if stale imports found: update notebook (Slice 10 scope)
   → if clean: move to step 2

2. Write docs/REGIME_SELECTION.md  (1 hr)
   → freeze regime rationale in prose

3. Write notebooks/analysis/05_earth_network_qa.ipynb  (half day)
   → start from master_dataset + stats CSVs
   → add DEM overlay for 2–3 flagged basins
   → no new package code unless a clear reusable helper emerges

4. If QA reveals a real problem in the regime-generated data:
   → stop and assess whether to re-run build_earth_features_regime.py
      with adjusted parameters before any raster or CNN work
   → update AGENT_STATE.md with the finding
```

---

## Dependencies on upstream work

- The `master_dataset_reg{A,B,C}.csv` files exist and are `CAN_REGENERATE`.
  The existing CSVs are usable for QA even though the rasters are stale
  (the tabular features are raster-independent).
- No stale-raster regeneration is needed before running the QA notebook
  (it only reads tabular data and DEMs, not raster patches).
- The QA notebook is therefore **safe to write and run right now**.
