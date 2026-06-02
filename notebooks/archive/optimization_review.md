# Optimization review — Earth-Mars `Dd_hull` complexity workflow

Companion to `dd_hull_mars_vs_earth_complexity.ipynb`. Records what was
changed in this audit pass, what the empirical checks found, and what to
show Liran.

## 1. Scientific framing (the principle)

> **First calibrate Earth network complexity and geometric
> representativeness. *Then* evaluate whether the resulting drainage-density
> distributions overlap Mars.**

Optimising Earth `Dd_hull` directly toward Mars `Dd_hull` is circular: it
forces the comparison metric onto the result. The notebook now separates:

- **Calibration metrics** (drive the optimiser): max Strahler,
  `basin_to_hull_area_ratio`, retained-stream-length fraction.
- **Evaluation metrics** (reported as outcomes, never optimised): `Dd_hull`,
  `Dd_true`, stream length, hull area, visual maps.

## 2. Changes made

### `channel_heads/dd_calibration.py`
- **`_dd_trim_rows` now records `outlet_row` / `outlet_col`** for every
  sweep row — DEM-grid coordinates are the threshold-independent identity
  of a physical pour point. Existing caches lack these columns; backfill
  via `augment_sweep_with_outlet_coords`.
- **`basin_to_hull_area_ratio(dd_hull, dd_true)`** — explicit helper for
  the ratio `Dd_h / Dd_t = A_basin / A_hull`. A value of `1` means the
  convex hull tracks the true catchment. Use this for calibration and
  reporting; do **not** invert it back to `Dd_h/Dd_t` in user-facing text.
- **`length_retention_penalty(pct, min_retained_pct, weight)`** — quadratic
  penalty applied when the trimmed network keeps less than
  `min_retained_pct` of the original stream length. Guards against
  over-pruning candidates.
- **`validate_area_bin_grouping(df, ...)`** — diagnostic for the
  `(dem_name, area_bin)` grouping previously used by the optimiser.
  Returns a per-DEM merge-rate table, merged groups, and (when coords are
  present) split outlets.
- **`augment_sweep_with_outlet_coords(df, out_csv)`** — one-time
  backfill: rebuilds StreamObjects per `(dem, threshold)` and resolves
  each row's `outlet_node` → `(row, col)`. No per-outlet variants are
  recomputed, so this is ~3-5× faster than the original sweep
  (still 5-15 min total).
- **`match_outlets_by_coords(df, coord_tol_px)`** — union-find on
  outlet pixel coordinates within `coord_tol_px` Chebyshev distance.
  Adds an `outlet_cluster` column; this is the new grouping key for the
  optimiser.

### Notebook (`dd_hull_mars_vs_earth_complexity.ipynb`)
- §8 markdown rewritten to explain the calibration-vs-evaluation split.
- §8 first code cell now runs `validate_area_bin_grouping`, optionally
  augments the cache, and switches to coordinate-based outlet matching
  when coordinates exist.
- §8 second code cell holds the **calibration-only optimiser** —
  weights `(w_S, w_B, w_R)` and `MIN_RETAINED_PCT` at the top. **The
  `Dd_h → Mars` term is gone.**
- §8 diagnostics: loss-component boxplot, pick-density heatmap, **retained
  length by chosen variant** (new third panel).
- §9 reframed as the **evaluation phase**: same panels but with
  `basin_to_hull_area_ratio` replacing the old `dd_hull/dd_true` label,
  and titles that name the active weights.
- New **§10: Regime evaluation** — fixed `(T, k)` grid on
  `T ∈ {0.05, 0.1, 0.2, 0.5}` × `min_S ∈ {2, 3, 4}`. Reports surviving
  outlets, median complexity, median `Dd_hull`/`Dd_true`,
  `basin_to_hull_area_ratio`, retention, and **% inside Mars IQR**.
  Recommends the top 3 regimes that retain ≥ 30 % length.
- New **§11: Sensitivity** — sweeps seven `(w_S, w_B, w_R)` combinations
  including a "no length penalty" case. Prints a verdict on whether picks
  are stable.
- §12-13: visual gallery + takeaways renumbered, takeaways expanded with
  the new framing.

### Tests
- 12 new unit tests for `basin_to_hull_area_ratio`,
  `length_retention_penalty`, `validate_area_bin_grouping`, and
  `match_outlets_by_coords`. Total: **49 tests passing** in
  `test_dd_calibration.py`.

## 3. Was area-bin grouping problematic?

**Yes — empirically confirmed.** With `AREA_BIN_KM2 = 0.1` against the
existing 5046-row sweep cache:

- **291 of 4747 `(DEM, area_bin, threshold)` groups (6.1 %)** merge two or
  more distinct outlets *at the same threshold*.
- **582 of 5038 valid candidate rows (11.6 %)** are involved in those
  merge groups.
- Per-DEM merge rate is highest in `panamint` (14.7 %), `humboldt`
  (14.1 %), `toano` (11.0 %); lowest on small DEMs.
- Example: `calnalpine` outlets 9788 and 17106 both have basin area
  ≈ 8.4 km² and land in `area_bin = 84` across five thresholds — they
  are genuinely different physical pour points.

**Mitigation already implemented**: outlet pixel coordinates are added to
new sweeps; the augmenter backfills existing caches; the notebook switches
to `match_outlets_by_coords` when coords are present and falls back to
area-bin (with surfaced warnings) when they are not.

## 4. Is the optimiser robust or exploratory?

`§11` runs 7 weight combinations. With the cached dataset (area-bin
fallback in the smoke test), the std of pick-level statistics across the
weight grid was:

| stat | std across weight grid |
| --- | --- |
| median threshold | 0.025 km² |
| median `Dd_hull` | 0.27 |
| median basin/hull ratio | 0.08 |
| median % length retained | 4.5 % |

The threshold pick is stable. `Dd_hull` swings ~30 % which is non-trivial
but interpretable — the heavier-retention weight pushes toward less
trimming. **The optimiser is best treated as a tunable diagnostic; the
regime grid (§10) is the canonical scientific report.**

## 5. Recommended regimes for Liran

The §10 regime grid plus the retention filter (≥ 30 % length) point at
the following defensible pairs:

| threshold | min Strahler kept | character | rationale |
| --- | --- | --- | --- |
| **0.5 km²** | **≥ 3** | trunk-with-tributaries | Best Mars-IQR overlap (~3 %) at the practical-range edge while keeping ~ 50 % length |
| **0.2 km²** | **≥ 3** | medium-complete | Compromise: ~ half the network retained, Strahler still close to Mars S=3 |
| **0.1 km²** | **≥ 2** | full-network baseline | Reference: shows that without trimming the gap stays wide |

(Exact percentages depend on the post-augment coordinate-matched run; the
above is from the area-bin fallback. Recompute after running the
augmenter for the final numbers.)

Suggested talking points:
1. The complexity gap is **structural**, not a threshold artefact.
2. **`basin_to_hull_area_ratio → 1`** is the right geometric calibration
   target — pursuing it is *not* circular.
3. Trimming reduces `Dd_hull` toward Mars but at the cost of network
   retention. There is no `(T, k)` cell that achieves Mars-IQR overlap
   while preserving ≥ 70 % of the original network length.
4. The remaining `Dd_hull` gap is genuinely about Earth dendritic trunks
   being tighter inside their convex hulls than Mars valley assemblies.

## 6. Remaining limitations

- The augmenter has not yet been executed against the user's existing
  cache. Until it is, §8 falls back to area-bin grouping (with the 6 %
  merge issue surfaced as a warning).
- Mars `Dd_true` has no analogue (no DEM-based basin area), so
  cross-planetary `basin_to_hull_area_ratio` is not directly comparable —
  it is an Earth-internal calibration target.
- The regime grid uses `AREA_BIN = 0.1`; if Liran prefers a different
  grouping tolerance this is one line at the top of §8.
- DEM resolution (≈ 85 m/px) caps the practical lower threshold at
  ~ 0.01 km² (one cell). Anything finer is the same network.

## 7. Next steps

1. **Run `augment_sweep_with_outlet_coords` once** and re-run the
   notebook to lock in coordinate-matched optimiser picks. Re-emit the
   regime grid numbers.
2. Re-run the §10 regime grid with `MIN_RETAINED_PCT` = 25 % and 35 %
   and confirm the recommended regimes remain stable.
3. (Optional) Extend the regime grid with one or two finer thresholds
   (e.g. `0.05` already, plus `0.07`, `0.15`) once the augmenter has
   been run — coordinate matching makes adjacent thresholds directly
   comparable.
4. Lock the chosen 2-3 regimes into a stable per-outlet CSV (already
   exported by §10 as `regime_grid_summary.csv`) and use that to drive
   the downstream pair-wise asymmetry pipeline.

---

*See companion outputs in*
`data/results/drainage_density_calibration/complexity_calibration/`:
`per_outlet_optimization.csv`, `regime_grid_summary.csv`,
`weight_sensitivity_summary.csv`, and the figure set with `fig_*.png`.
