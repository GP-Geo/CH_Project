# Roadmap & Risks

Consolidated from `PROJECT_REVIEW_AND_IMPROVEMENTS.md` (risk register) and
`improvement.md` (future work), plus the active refactor plan.

> See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) and
> [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for current state.

---

## 1. Scientific / code risks to verify manually

These affect scientific correctness and should be reviewed by a human. **S1 is
the highest priority** and is the motivation for centralizing unit logic in
`channel_heads/units.py`.

| # | File | Issue | Severity | Status |
|---|------|-------|----------|--------|
| **S1** | `features/asymmetry.py` / `units.py` | **`upstream_distance()` unit assumption.** ΔL assumes `s.upstream_distance()` returns map units (arc-degrees for SRTM). If TopoToolbox returns pixel/edge counts, the meters conversion is wrong by ~`cellsize_deg` (≈1/3600), making ΔL ~3600× off. Verify against TopoToolbox docs/source. | **Critical** | **Verified 2026-06-04 — arc-degrees confirmed.** CalnAlpine: `s.upstream_distance().max() = 0.2404` (arc-degrees); `LengthwiseAsymmetryAnalyzer.meters_per_unit = 97309 m/deg` (= `compute_meters_per_degree(39.69°)`, cellsize 0.000833 < 1 branch taken). ΔL conversion is correct. Not a bug. |
| S2 | `units.py` | Geometric mean in `compute_meters_per_degree` biases near-E–W / N–S paths. Acceptable for elongated ranges; document. | Low | Documented |
| S3 | `features/earth_paths.py` | `_trace_path_downstream` greedy child choice could follow wrong branch on unexpected topology. | Low | Manual |
| S4 | `coupling_analysis.py` | Pre-filter distance threshold (`multiplier·√threshold`) is geometrically optimistic for elongated basins — may skip real touching pairs. | Medium | Manual |
| S5 | `coupling_analysis.py` | `contact_px` double-counts diagonal contacts under 8-connectivity (≈2×). `touching` bool is still correct. | Medium | Manual |
| S6 | `features/earth_geometry.py` | `strahler_order_diff` uses the branch parent's order, not the head's. Defensible; document. | Low | Manual |
| S7 | `pairing/dag.py` via `pairing/earth.py` | O(k²) pairs at highly-branched confluences (e.g. Taiwan). Correct but can be slow. | Low | Monitor |
| S8 | `training/labeling.py` | `filter_hard_negatives` on the full dataset before LOBO CV → mild train/test leakage. Call per fold. | Medium | Manual |

**S1 verification recipe:**
```python
import topotoolbox as tt3
dem = tt3.read_tif("data/cropped_DEMs/CalnAlpine_strm_crop.tif")
s = tt3.StreamObject(tt3.FlowObject(dem), threshold=300)
print(s.upstream_distance().max())   # ~0.01–0.5 → arc-degrees; ~300–5000 → pixel counts
```

---

## 2. Active refactor plan (this initiative)

| Phase | Goal | State |
|------|------|-------|
| 0 | Conservative cleanup (move render/diagnostics scripts, remove dead rasterizer helpers, data markers) | ✅ done |
| 1 | Documentation consolidation → `docs/` | ✅ done |
| 2 | `channel_heads/units.py` — single source of truth for units (closes S1) | ✅ done |
| 3 | Regime presets → `channel_heads/regimes.py`; remove `sys.path.insert` coupling | ✅ done |
| 4 | Package extraction (`pairing/`, `features/`, `inference/`, `eval/`, `viz/`); de-duplicate Earth/Mars logic | ✅ done |
| 5 | ~~`channel_heads/calibration.py`~~ — **dropped** (2026-06-01): per-basin standardization was neutral-to-negative, so calibration is deprioritized. Threshold/calibration analysis stays a notebook task, not a core module. | ❌ dropped |
| 6 | Foldered notebook homes (`mars/`, `regime/`, `presentation/`, `diagnostics/`, `training/`, `analysis/`, `archive/`); B-class scripts demoted to thin wrappers, notebooks call `channel_heads.*` only | ✅ done |
| 7 | `docs/DATA_STATUS.md` — classify every output (RAW_KEEP / CAN_REGENERATE / STALE_AFTER_RASTER_FIX / …) | ✅ done |
| 8 | Full clean pipeline rerun plan (regenerate Earth→Mars with chosen params) | ✅ documented (`docs/PIPELINE_RERUN.md`) |

### Duplication clusters dissolved (Phases 2–6, all ✅)
1. **Feature math** → `channel_heads/features/` (`geometry.py`, `paths.py`).
2. **First-meet pairing** → `channel_heads/pairing/` (`dag.py`, `mars_graph.py`).
3. **Rasterization** → `channel_heads/rasterization/` (`earth_patches.bresenham_line` shared; larger polyline draw kept separate by design).
4. **Unit conversions** → `channel_heads/units.py`.
5. **XGBoost inference glue** (loaders / verify / predict / regime embeddings) → `channel_heads/models/`.
6. **Evaluation** (thresholding, metrics, grouped/LOBO splits) → `channel_heads/eval/`.
7. **Vector figures** (contact sheets, ROC, per-outlet, stream-crossing QA) → `channel_heads/viz/`.

Every B-class script now keeps a thin `main()` wrapper that calls these modules;
the matching `notebooks/<home>/` notebook is the primary, documented interface
and calls `channel_heads.*` only (no duplicated cell logic). See
[PROJECT_STRUCTURE.md §3a](PROJECT_STRUCTURE.md) for the notebook ↔ script map.

---

## 3. Architecture / engineering improvements (from review)

- ~~**W1** Split `geometric_analysis.py` (2200-line monolith).~~ **✅ Done** — split into `features/{asymmetry,geometry,earth_geometry,paths,earth_paths}.py`, `training/labeling.py`, `pairing/filtering.py`, and `features/earth_enrichment.py`; the monolith and its re-export shim are removed.
- **W2** Unify the duplicated `_build_children_from_parents` (basin-scoped vs global).
- **E5** Lazy-import `topotoolbox` in `cli/_analyze.py` (try/except with helpful message).
- ~~**E6** Update placeholder GitHub URLs in `pyproject.toml`.~~ **✅ Done** — point to `github.com/GP-Geo/CH_Project`.
- Remove deprecated no-op `use_meters` parameter (next major version).
- Validate `CHANNEL_HEADS_ROOT` path existence in `io.paths`.

## 4. Performance (profiled 2026-02-02)

- **Bounding-box crop in `pair_touching()`** — boolean ops run on full DEM masks though pairs occupy <1% of the grid. Cropping to the union bbox gave **259×** (Finisterre 2.8M px) / **966×** (Taiwan 9.9M px) speedup on that method, ~1.8× overall (Amdahl: `dependencemap()` is the other ~55%). Medium priority, low effort, high impact.
- Optional: multiprocessing across outlets; R-tree spatial index for confluence queries.

## 5. Testing backlog

- ~~Verify S1 units~~ — **Done 2026-06-04** (see §1 S1 row; arc-degrees confirmed).
- Integration test on a small real DEM (CalnAlpine, ~1547 pairs).
- `filter_hard_negatives(s=...)` stream-crossing test; negative-L warning test; vectorized `_sample_path_coords` regression test.
- `contact_px` 4-connectivity test; pre-filter no-false-negative test on elongated basins.
- Data-leakage audit: confirm `filter_hard_negatives` runs per fold in `notebooks/training/01`.

## 6. Other future items

- Data versioning (DVC) for large DEMs/outputs.
- CLI: `--compute-asymmetry`, `--lat`, progress bars.
- Notebook hygiene: jupytext pairing for reviewable diffs; reproducibility (pinned versions, system info).
