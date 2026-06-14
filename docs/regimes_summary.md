# Regimes — summary (quick reference)

At-a-glance summary of the three frozen Earth-complexity regimes used to bracket
Mars and quantify calibration uncertainty. Full rationale lives in
[`REGIME_SELECTION.md`](REGIME_SELECTION.md); runnable evidence is
[`notebooks/pipeline/04_earth_mars_regime_calibration.ipynb`](../notebooks/pipeline/04_earth_mars_regime_calibration.ipynb).

## The three regimes (frozen)

Canonical code: `channel_heads/regimes.py` (`REGIMES`). All three are plain
**`trim`** (drop 1st-order tips only), **no** order-gap (delta) pruning.

| Regime | `threshold_km2` | pruning | `order_gap_to_prune` | `scientific_score` | Network character |
|--------|-----------------|---------|----------------------|--------------------|-------------------|
| **regA** | 0.20 | trim (remove order 1) | 0 (none) | 3.986 (best) | Intermediate density |
| **regB** | 0.25 | trim (remove order 1) | 0 (none) | 4.089 | Sparsest base network |
| **regC** | 0.15 | trim (remove order 1) | 0 (none) | 4.168 | Densest base network |

Shared: `coupling_n_workers=4`, `min_prefilter_px=30.0`.
Mars targets used for scoring: Dd-hull median ≈ **0.224**, Strahler median = **3**.

## Why these three

- They are the **top-3 distinct, eligibility-passing** candidates by
  `scientific_score = 0.80·dd + 0.10·complexity + 0.10·length` (lower = better),
  from the candidate sweep = `threshold` × order-removal (`full`/`trim`/`ge3`) ×
  delta (`none`/`3`/`4`).
- Hard filters: `length_retained ≥ 0.30`, median Strahler ≥ 2, low-order pruning
  only (`pre_remove ≤ 2`).
- They bracket the practical threshold neighbourhood (0.15–0.25 km²) whose Earth
  Dd-hull medians sit closest to Mars. Running all three and reporting the spread
  of Mars results **is** the calibration-uncertainty estimate — no single regime
  is declared "correct".

## Parameter meanings

| Parameter | Controls |
|-----------|----------|
| `threshold_km2` | Min contributing area to initiate a channel — sets base network density |
| `pre_remove_max_order` | Drop all Strahler-order segments ≤ this before analysis (`1` = `trim`) |
| `order_gap_to_prune` | Delta rule: prune a tributary if `trunk_order − branch_order ≥` this (`0` = off) |

## Threshold-range robustness (checked 2026-06-14)

Question raised: *would allowing coarser thresholds (up to 1.5 km²), or applying
delta pruning, produce a better regime?* The sweep was re-run with the cap raised
0.5 → 1.5 km² (recompute `dd_sweep_highthr_to1p5.csv`, validated to `diff = 0` vs
the dense-low sweep at the 0.5 km² overlap). Conclusion: **no — keep the three
regimes as-is.**

- **Trimming gets worse with threshold.** `trim` score degrades monotonically
  (0.5 → 4.64, 0.75 → 5.36, 1.0 → 6.00) and goes **ineligible by ~1.5 km²**
  (median Strahler drops below 2). Removing the 1st-order fringe from an already
  coarse network shrinks the convex hull faster than the length, so Dd-hull
  paradoxically *rises* (1.37 → 1.88 → 2.13), moving **away** from Mars (0.224).
- **Delta-3 / delta-4 are inert at these thresholds.** At ≥ ~0.15 km² (and
  emphatically at 1.0–1.5 km²) they return *identical* Dd-hull and Strahler and
  shave only 3rd–4th-decimal length — the network is too shallow (max Strahler
  ≈ 3–4) for the order-gap rule to ever fire. They cannot change the ranking.
- **One new candidate appeared but was not adopted:** `full` (no pruning) at
  ~1.0 km² scores 4.079 (≈ regB). It matches Mars complexity via a coarse
  threshold *alone*, but it is a different simplification *philosophy* (match by
  threshold, not pruning) and is still far denser than Mars in absolute Dd
  (dd_score 5.10). Adopting it would be a deliberate scientific change, not an
  improvement to the existing `trim` family — so the frozen regimes stand.

Evidence: `data/results/drainage_density_calibration/regime_optimization/dd_sweep_highthr_to1p5.csv`.

## Downstream artifacts keyed to regime

Each regime has its own `*_reg{A,B,C}` CNN, XGBoost, operating threshold, feature
dataset, rasters, and Mars predictions. See
[`REGIME_SELECTION.md`](REGIME_SELECTION.md#downstream-artifacts-keyed-to-regime)
and [`DATA_STATUS.md`](DATA_STATUS.md) for current status, and
[`PIPELINE_RERUN.md`](PIPELINE_RERUN.md) for regeneration order.

## See also

- [`REGIME_SELECTION.md`](REGIME_SELECTION.md) — full Stage-4 calibration rationale (frozen parameters)
- [`notebooks/pipeline/04_earth_mars_regime_calibration.ipynb`](../notebooks/pipeline/04_earth_mars_regime_calibration.ipynb) — runnable selection evidence
- `channel_heads/regimes.py` — canonical `Regime` dataclass and `REGIMES` dict
