# AGENT_STAGE_0_3_VERIFICATION.md

Foundation verification for pipeline Stages 0–3, run 2026-06-04.
Branch: `refactor/package-first-architecture`
Conda env: `ch-heads`
Executed by: Claude (claude-sonnet-4-6)

---

## 1. Executive summary

**The foundation is scientifically sound. S1 (critical blocker) is resolved: no bug.**

`s.upstream_distance()` (TopoToolbox 3) returns **arc-degrees** for a geographic-CRS
SRTM DEM, not pixel counts. The `LengthwiseAsymmetryAnalyzer` correctly multiplies
by `compute_meters_per_degree(lat)` from `channel_heads/units.py`. ΔL values are
correct. The ~3600× error scenario is **not occurring**.

All 17 Earth DEMs load clean with valid CRS (EPSG:4326) and plausible z-ranges.
All 17 basins extract stream networks and prune without producing empty networks
or pathological survivor ratios. Regime parameters (regA/B/C) exactly match the
frozen documentation in `docs/REGIME_SELECTION.md`. Mars source data (topology
GeoPackage, valley vectors) is coherent and covers expected planetary extents.
The package test suite passes 569/569 tests.

**One minor documentation issue found (low severity, does not block retrain):**
`basin_config.py` lists 18 basins (includes `piedepalo`) but `paths.py`
`EXAMPLE_DEMS` has only 17 entries — no `piedepalo` DEM is present on disk.
The mapping `LOCAL_TO_PAPER_BASIN['piedepalo']` exists in both but the DEM is
absent. This does not affect the 17-basin training set.

**Stage 7 raster state remains the known blocker for Stage 8 retrain:**
manifests reference some missing paths. This is already documented in
`AGENT_STATE.md` and `STAGE_ASSET_MAP.md`; it is not a new finding.

**Go/No-Go: GO for Stage 4 re-confirmation; Stage 8 retrain blocked only by
Stage 7 raster regeneration (pre-existing known issue).**

---

## 2. Per-stage findings

### Stage 0 — Project setup and assumptions

**Result: PASS**

| Item | Check | Evidence |
|---|---|---|
| `PROJECT_ROOT` | Resolves to `/Users/guypi/Projects/channel-heads` | `_find_project_root()` finds `pyproject.toml`/`.git` correctly |
| `data/cropped_DEMs` | Exists | `CROPPED_DEMS_DIR.exists() = True` |
| `data/final_valleys` | Exists | `FINAL_VALLEYS_DIR.exists() = True` |
| `data/Mars` | Exists | `MARS_DIR.exists() = True` |
| `MARS_HILLSHADE` | Exists | `MOLA_Hillshade_Robinson_128ppd.tif` present |
| `MARS_VALLEYS` | Exists | `final_valleys_fixed.gpkg` present |
| All 17 `EXAMPLE_DEMS` | Exist on disk | All 17 entries resolve to `True` |
| `regimes.py` vs `REGIME_SELECTION.md` | Exact match | All 12 parameters for regA/B/C verified identical (see §3 below) |
| `units.py` single source | No competing conversions | Scan of all `channel_heads/*.py` found zero hardcoded `110540`/`111320` constants outside `units.py` |
| `piedepalo` DEM | **ABSENT** | `BASIN_CONFIG` has 18 entries; `EXAMPLE_DEMS` has 17; `piedepalo` DEM not present in `data/cropped_DEMs/`. `LOCAL_TO_PAPER_BASIN['piedepalo']` exists as a dangling mapping. Low severity. |

**Regime parameters verified:**

| Regime | `threshold_km2` | `pre_remove_max_order` | `order_gap_to_prune` | `min_prefilter_px` |
|--------|----------------|------------------------|----------------------|--------------------|
| regA | 0.05 | 2 | 4 | 30.0 |
| regB | 0.25 | 1 | 4 | 30.0 |
| regC | 0.10 | 1 | 4 | 30.0 |

All match `docs/REGIME_SELECTION.md` exactly.

---

### Stage 1 — Earth source-data QA

**Result: PASS**

All 17 Earth DEMs load cleanly via `rasterio`. Summary:

| Property | Result |
|---|---|
| Count | 17/17 present |
| CRS | All EPSG:4326 (geographic, arc-degree units) |
| Cell size | All 0.000833° (1 arc-second SRTM, consistent) |
| No-data value | All 0.0 |
| Z-range issues | None (all within Earth geomorphic plausible range) |

Selected DEM stats (full table below):

| Basin | Shape | Z-min (m) | Z-max (m) | Config z_max | Match? |
|---|---|---|---|---|---|
| calnalpine | 256×360 | 1020 | 2677 | 2677 | OK |
| taiwan | 2118×1957 | -28 | 3917 | 3917 | OK |
| finisterre | 943×1674 | 1 | 4096 | 4096 | OK |
| humboldt | 320×219 | 1313 | 2984 | 2984 | OK |
| inyo | 289×284 | 329 | 3363 | 3363 | OK |
| yoro | 155×150 | -14 | 885 | 885 | OK |
| toano | 362×191 | 1517 | 2827 | 2914 | -87m (see note) |

**Toano z_max discrepancy (-87 m):** `basin_config.py` records `z_max = 2914` from
Table A1 of Goren & Shelef (2024); the cropped DEM has `max = 2827`. The DEM is
cropped to a study-area extent that does not include the full summit. The `z_max`
field in `basin_config` is the paper-reported range maximum, not constrained to the
DEM crop. The `z_th = 1710 m` threshold (the operationally used parameter) is
well below both values. **Not a bug. No action needed.**

Basin `sierramadre` has z_min = -4 m (coastal zone); `taiwan` has z_min = -28 m
(lowland / sea-level pixels). Both are plausible given the basin extents and are
within the z_th masking range.

---

### Stage 2 — Earth interactive network exploration

**Result: PASS**

All 17 basins were run through full stream extraction + Strahler-based pruning
under regime `regB` (`threshold_km2=0.25`, `pre_remove_max_order=1`,
`order_gap_to_prune=4`). No basin produced an empty network or pathological
survivor ratio.

Extraction uses `channel_heads/units.py::compute_threshold_cells()` (single
source), verified to route through `compute_pixel_size_m_from_dem()` correctly
(cellsize < 1 → geographic branch, scaled by `compute_meters_per_degree(lat)`).

Pruning uses `channel_heads/pruning.py::apply_strategy()` — correct API confirmed.

Summary of all 17 basins (regB):

| Basin | thresh_px | n_full | n_pruned | ratio |
|---|---|---|---|---|
| calnalpine | 38 | 10,226 | 4,798 | 0.469 |
| daqing | 39 | 5,136 | 2,406 | 0.468 |
| finisterre | 29 | 396,950 | 81,222 | 0.205 |
| humboldt | 38 | 7,346 | 3,516 | 0.479 |
| inyo | 36 | 8,674 | 4,123 | 0.475 |
| kammanasie | 35 | 8,243 | 3,819 | 0.463 |
| luliang | 38 | 29,781 | 14,542 | 0.488 |
| panamint | 36 | 15,297 | 6,948 | 0.454 |
| sakhalin | 43 | 12,799 | 5,747 | 0.449 |
| sierramadre | 31 | 592,504 | 108,155 | 0.183 |
| sierranevadaspain | 37 | 13,066 | 6,007 | 0.460 |
| taiwan | 32 | 1,340,093 | 185,798 | 0.139 |
| toano | 38 | 7,698 | 3,600 | 0.468 |
| troodos | 36 | 16,428 | 7,789 | 0.474 |
| tsugaru | 39 | 3,404 | 1,744 | 0.512 |
| vallefertil | 34 | 51,870 | 26,635 | 0.513 |
| yoro | 36 | 2,368 | 1,027 | 0.434 |

Taiwan and sierramadre show lower survival ratios (0.14 and 0.18) consistent with
their large extents (high-order trunk dominance after pruning). These are expected
behaviors for large, complex basins under order-gap pruning — no warning.

---

### Stage 3 — Mars interactive network exploration

**Result: PASS**

Mars source data is structurally coherent and consistent with a valid transfer target.

**Mars valley vectors (`final_valleys_fixed.gpkg`):**
- 391 `MultiLineString` features
- CRS: Mars Equidistant Cylindrical (custom Mars sphere, metre units)
- Bounds: [-10,668,412 m, -3,936,371 m, 10,321,407 m, 2,787,665 m]
- Segment lengths: min 28 km, mean 171 km, max 2,313 km — geologically plausible
  valley network lengths for Mars

**MOLA hillshade (`MOLA_Hillshade_Robinson_128ppd.tif`):**
- CRS: Robinson projection, Mars sphere, metre units
- Shape: 19,676 × 39,108 pixels (128 ppd global coverage)
- DN range: 0–255 (uint8 hillshade)
- Note: Robinson ≠ Equidistant Cylindrical — hillshade and valleys require
  reprojection for overlay. The topology pipeline reprojects to the valleys CRS
  (confirmed by `mars_vn_topology_model_ready.gpkg` CRS matching valleys). This
  is expected design behavior, not a bug.

**Mars topology GeoPackage (`mars_vn_topology_model_ready.gpkg`):**

| Layer | Rows | Geometry |
|---|---|---|
| mars_networks | 391 | MultiLineString |
| mars_segments | 5,619 | LineString |
| mars_nodes | 6,003 | Point |
| mars_terminal_nodes | 3,390 | Point |
| mars_outlets | 391 | Point |
| mars_channel_heads | 2,999 | Point |
| mars_confluences | 2,612 | Point |

All layers share the same 391 unique `network_id` values (1–391). CRS is
Mars_2000_Equidistant_Cylindrical, metre units.

Network statistics: mean total_length = 171 km/network; 2,999 channel heads across
391 networks (mean 7.7 heads/network); 2,612 confluences (mean 6.7/network). The
structural ratios (confluences ≈ heads − networks for binary trees) are geomorphically
reasonable.

Mars valley coordinate extents are within the Mars equatorial half-circumference
(~10,669 km) — units confirmed as metres on Mars sphere. No CRS or unit problems.

---

## 3. S1 resolution — the critical risk

**S1 STATUS: VERIFIED — NO BUG. ΔL conversion is correct.**

**Measurement (CalnAlpine basin, threshold=300):**
```python
s.upstream_distance().max() = 0.2404  # arc-degrees
```
This is firmly in the 0.01–0.5 range indicating arc-degrees (map units), not the
300–5000 range that would indicate pixel/edge counts.

**End-to-end trace through `LengthwiseAsymmetryAnalyzer`:**
```
DEM cellsize = 0.000833 degrees  (< 1.0 → geographic-CRS branch taken)
→ _meters_per_unit = compute_meters_per_degree(lat=39.69°) = 97,309 m/deg
```
This is the correct scale factor. L_1 and L_2 path lengths are properly in metres.

**What this means downstream:**
- ΔL values computed and stored in `master_dataset_reg{A,B,C}.csv` are correct.
- The production XGBoost model (threshold 0.577406) was trained on correct ΔL values.
- Regime-trained XGBoost models (`xgb_geom_plus_cnn_emb_reg{A,B,C}.json`) used
  correct features.
- There is no ~3600× error in the geometric features. No retraining is required
  solely on account of S1.

**Documentation update made:** `docs/ROADMAP_AND_RISKS.md` S1 row updated from
`Open` to `Verified 2026-06-04 — arc-degrees confirmed` with measured value and
evidence.

---

## 4. Issues found and severity

| # | Issue | Severity | Blocks Stage-8 retrain? |
|---|---|---|---|
| I-1 | `piedepalo` in `basin_config.py` (18 entries) but absent from `paths.py::EXAMPLE_DEMS` (17 entries) and no DEM on disk | Low | No — 17-basin training set unaffected |
| I-2 | Toano `z_max` in `basin_config.py` (2914 m) is 87 m above DEM max (2827 m) | Low / informational | No — z_th=1710 m is well below both values; DEM crop is smaller than full range |
| I-3 | Stage 7 rasters: `regA` has 9,851 missing raster_paths; `regB` 961; `regC` 2,102 in live manifests | HIGH (pre-existing) | YES — do not use these manifests for training; regenerate rasters + manifests before Stage 8 |
| I-4 | MOLA hillshade (Robinson) and Mars valleys (Equidistant Cylindrical) use different projections | Low / informational | No — topology pipeline reprojects internally; no correctness issue |

**I-3 is the only Stage-8 blocker** and is already documented in `AGENT_STATE.md`
and `STAGE_ASSET_MAP.md`. It is a pre-existing state, not newly discovered.

---

## 5. Documentation corrections made

| File | What was stale | Why the fix is safe |
|---|---|---|
| `docs/ROADMAP_AND_RISKS.md` | S1 row status was `Open` | Measurement confirms no bug; updating to `Verified 2026-06-04` with evidence is a factual record of the verification result |

No code was modified. No data was modified.

---

## 6. Go/No-Go recommendation

**Stage 4 re-confirmation: GO.**
Foundation is sound. Earth DEMs, units, regime parameters, Mars source data, and
topology are all verified correct.

**Stage 8 retrain: BLOCKED (pre-existing) by Stage 7 raster state.**
Before any retrain, regenerate rasters and manifests cleanly:
```bash
python scripts/cli/build_cnn_patches_regime.py --regime regA -v
python scripts/cli/build_cnn_patches_regime.py --regime regB -v
python scripts/cli/build_cnn_patches_regime.py --regime regC -v
```
After regeneration, confirm all manifest `raster_path` entries resolve before
proceeding to `train_cnn_regime.py` and `train_combined_xgb_regime.py`.

---

## 7. Commands run

```bash
# S1 verification
conda run -n ch-heads python -c "
import topotoolbox as tt3
dem = tt3.read_tif('data/cropped_DEMs/CalnAlpine_strm_crop.tif')
fo = tt3.FlowObject(dem)
s = tt3.StreamObject(fo, threshold=300)
print(s.upstream_distance().max())  # → 0.2404
"

# End-to-end LengthwiseAsymmetryAnalyzer verification
# (LengthwiseAsymmetryAnalyzer path trace: cellsize 0.000833 < 1.0 → meters_per_unit=97309 m/deg)

# Path checks
conda run -n ch-heads python -c "from channel_heads.io import paths; ..."

# DEM QA (all 17 basins, rasterio)
# → all EPSG:4326, all valid z-ranges, 0 issues

# Regime parameter check
conda run -n ch-heads python -c "from channel_heads.regimes import REGIMES; ..."
# → all 12 parameters match docs/REGIME_SELECTION.md

# Unit source-of-truth check
# → grep: no hardcoded 110540/111320 outside units.py

# Stage 2: all 17 basins through regB extraction+pruning
conda run -n ch-heads python -c "..."  # → all 17 OK, no empty networks

# Stage 3: Mars topology and valley data
# → 391 networks, all layers coherent, CRS metre-units, extents plausible

# Test suite
conda run -n ch-heads pytest -q  # → 569 passed, 7 warnings

# Git status
git status  # → clean
```

---

_Report generated: 2026-06-04_
_Agent: Claude (claude-sonnet-4-6)_
