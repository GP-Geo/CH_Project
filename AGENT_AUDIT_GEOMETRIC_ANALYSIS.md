# AGENT_AUDIT_GEOMETRIC_ANALYSIS.md - Slice 6 audit

Date: 2026-06-02

Scope: read-only audit of `channel_heads/geometric_analysis.py` and adjacent
feature/unit/training call sites. No implementation code was changed.

## Executive Summary

`channel_heads/geometric_analysis.py` is still a real implementation module, not
a disposable legacy file. It owns Earth/TopoToolbox-specific feature generation,
lengthwise asymmetry, labeled-dataset assembly, hard-negative filtering, stream
loading, and CSV enrichment.

Some lower-level pieces already have package homes:

- Pure scalar/vector feature math lives in `channel_heads/features/geometry.py`.
- Mars projected path helpers live in `channel_heads/features/paths.py`.
- Mars feature-table orchestration lives in
  `channel_heads/features/mars_features.py`.
- Unit conversions live in `channel_heads/units.py`.
- First-meet parent extraction and pair normalization live in
  `channel_heads/pairing/earth.py`.

Future work should split `geometric_analysis.py` into package modules with
`geometric_analysis.py` left as a compatibility shim. The split must be guarded
by behavior-pinning tests because several scientific contracts are encoded in
private helpers and DataFrame column order.

## Reference Map

| File | Current role | Real logic present |
|------|--------------|--------------------|
| `channel_heads/geometric_analysis.py` | Earth asymmetry, Earth geometric features, labeling/filtering, CSV enrichment | Yes. Owns TopoToolbox traversal, upstream-distance conversion policy, feature/QC column assembly, hard-negative filtering, and loader/enrichment behavior. |
| `channel_heads/features/geometry.py` | Pure feature math | Yes, already canonical for angle, azimuth, azimuth difference, proximity profile. |
| `channel_heads/features/paths.py` | Mars projected-metre path helpers | Yes, already canonical for Mars LineString direction and sampling helpers. Earth path helpers remain separate because they depend on node IDs, child maps, and unit conversion. |
| `channel_heads/features/mars_features.py` | Mars Phase 3A five-feature table | Yes, already package-resident. Reuses `features.geometry` and `features.paths`, owns Mars graph/Strahler/filter audit behavior. |
| `channel_heads/features/__init__.py` | Curated feature API | Re-exports pure feature helpers and Mars feature builders. |
| `channel_heads/units.py` | Unit conversion single source of truth | Yes, already canonical. `geometric_analysis.py` re-exports selected functions. |
| `channel_heads/dd_calibration.py` | Drainage-density calibration | Relevant as a unit-helper re-export and DEM-threshold/pixel-size user; not a target for this split. |
| `scripts/build_earth_features_regime.py` | Regime Step 2 Earth feature builder | Real orchestration logic. Uses `LengthwiseAsymmetryAnalyzer`, `GeometricFeaturesAnalyzer`, `generate_labeled_dataset`, and `filter_hard_negatives`. |
| `channel_heads/rasterizer.py` | Earth patch rasterization | Depends on `geometric_analysis._trace_full_path`, which should move before or with rasterizer cleanup. |
| `tests/test_geometric_analysis.py` | Current broad behavior coverage | Strong coverage of helper math, analyzers, hard-negative filtering, stream crossing, projected CRS, y-axis convention, and proximity profile. More pinning needed before extraction. |
| `tests/test_features_geometry.py` | Pure geometry coverage | Confirms pure helpers and `geometric_analysis` aliases are same objects. |
| `tests/test_units.py` | Unit helper identity | Confirms `geometric_analysis` and top-level API re-export the canonical unit functions. |
| `docs/ROADMAP_AND_RISKS.md` | Existing risk register | Calls this split the highest-priority architectural item and records S1/S2/S3/S6/S8 risks. |

## Function / Class Classification

### Already Canonical Elsewhere

- `_angle_between_vectors`, `_compute_azimuth`, `_azimuth_difference`,
  `_compute_proximity_profile` are aliases to `features.geometry`.
- `compute_meters_per_degree` and `compute_pixel_size_meters` are imported from
  `units.py` and re-exported for compatibility.
- `_build_parents_from_stream` and `_normalize_pair` are imported from
  `pairing.earth`.

### Earth Feature Logic That Should Move

- `GEOM_FEATURE_COLS`
- `_trace_path_downstream`
- `_compute_direction_vector`
- `_trace_full_path`
- `_sample_path_coords`
- `_detect_cellsize`
- `PairGeometricResult`
- `GeometricFeaturesAnalyzer`

Recommended future home: `channel_heads/features/earth_geometry.py` plus a
small shared schema module for column constants.

### Earth Asymmetry Logic That Should Move

- `PairAsymmetryResult`
- `compute_delta_L`
- `LengthwiseAsymmetryAnalyzer`
- `compute_asymmetry_statistics`
- `merge_coupling_and_asymmetry`

Recommended future home: `channel_heads/features/asymmetry.py`.

### Training Dataset / Labeling Logic That Should Move

- `generate_labeled_dataset`
- `filter_hard_negatives`
- `_line_crosses_stream`
- `_build_stream_mask`
- `merge_geometric_features`

Recommended future home: `channel_heads/training/labeling.py` or
`channel_heads/training/datasets.py`. Prefer `training/labeling.py` if it stays
Earth-label specific; keep a re-export from `training/datasets.py` only if later
training recipes need a broader dataset API.

### IO / Enrichment Logic That Should Move

- `default_stream_loader`
- `_build_pairs_at_confluence`
- `_build_asymmetry_df`
- `_add_missing_stream_qc`
- `add_geometric_features_to_csv`
- `_add_geometric_features_cli`

Recommended future home: `channel_heads/features/earth_enrichment.py` or
`channel_heads/training/earth_features.py`. Use `io.paths` for DEM/path lookup
once behavior is pinned.

### Keep As Compatibility Surface Later

`channel_heads/geometric_analysis.py` should become a shim that re-exports the
public objects above. Keep private helper re-exports temporarily for tests and
legacy users that currently import `_trace_full_path`, `_sample_path_coords`,
and other underscored helpers.

## Duplication Map

| Concern | Existing package owner | Duplicated / transitional owner | Future direction |
|---------|------------------------|----------------------------------|------------------|
| Pure angle/azimuth/proximity math | `features.geometry` | Aliased in `geometric_analysis.py`; tested for identity | Keep canonical in `features.geometry`; preserve aliases during shim phase. |
| Mars projected path helpers | `features.paths` | Similar Earth helper logic in `geometric_analysis.py` | Keep separate unless a generic path abstraction proves behavior-identical. Earth helpers use node IDs and unit scaling. |
| Unit conversions | `units.py` | Re-exported in `geometric_analysis.py` and `dd_calibration.py` | Keep `units.py` canonical; do not change S1 policy during split. |
| Child-map construction | `pairing.earth`, local `_build_children_from_parents`, rasterizer imports pairing helper too | Multiple local variants with different accepted scopes | Centralize only after tests pin basin-scoped vs global child-map semantics. |
| Earth path tracing | None outside `geometric_analysis.py` | `rasterizer.py` imports `_trace_full_path` | Move Earth path tracing to `features/earth_paths.py` or `features/earth_geometry.py`; repoint rasterizer later. |
| Feature column order | `geometric_analysis.GEOM_FEATURE_COLS`, Mars `MODEL_FEATURES` | Training scripts duplicate five model features | Add schema constants carefully. Earth `GEOM_FEATURE_COLS` includes distance/QC columns; Mars `MODEL_FEATURES` is the five XGB features only. |
| Hard-negative filtering | `geometric_analysis.py` | Regime script calls it; Mars audit documents it as training-only and skipped | Move to training layer; preserve per-group recursion and stream-crossing optional behavior. |
| Stream loading / CSV enrichment | `geometric_analysis.py` | Regime script has separate regime loader/orchestration | Move enrichment/loader separately from regime-specific dataset build. |
| Mars five-feature generation | `features.mars_features` | No remaining script implementation | Leave as package-owned; do not merge Earth analyzers into Mars feature orchestration blindly. |

## Recommended Canonical Ownership

### `channel_heads/features/geometry.py`

Keep as the canonical home for pure coordinate/vector primitives:

- `angle_between_vectors`
- `compute_azimuth`
- `azimuth_difference`
- `compute_proximity_profile`

No Earth/Mars IO or TopoToolbox logic should move here.

### `channel_heads/features/earth_paths.py`

Create this or fold it into `earth_geometry.py` for Earth/TopoToolbox path
helpers:

- `_build_children_from_parents`
- `_trace_path_downstream`
- `_compute_direction_vector`
- `_trace_full_path`
- `_sample_path_coords`
- `_detect_cellsize`

This should be the future import target for `rasterizer.py`, replacing
`from .geometric_analysis import _trace_full_path`.

### `channel_heads/features/asymmetry.py`

Own lengthwise asymmetry:

- `PairAsymmetryResult`
- `compute_delta_L`
- `LengthwiseAsymmetryAnalyzer`
- `compute_asymmetry_statistics`
- `merge_coupling_and_asymmetry`

Keep the current upstream-distance meter conversion exactly until the S1 risk
is resolved separately.

### `channel_heads/features/earth_geometry.py`

Own Earth geometric feature generation:

- `GEOM_FEATURE_COLS` or import it from a small schema module.
- `PairGeometricResult`
- `GeometricFeaturesAnalyzer`
- `merge_geometric_features` if kept in the feature layer.

This module depends on Earth stream graph adapters and unit conversion; it
should not own training label logic.

### `channel_heads/training/labeling.py`

Own label/dataset post-processing:

- `generate_labeled_dataset`
- `filter_hard_negatives`
- `_line_crosses_stream`
- `_build_stream_mask`

This aligns with the Earth/regime audit: hard-negative filtering and labeled
dataset assembly are training-dataset behavior, not Mars inference behavior.

### `channel_heads/features/earth_enrichment.py`

Own CSV enrichment and default Earth stream loading:

- `default_stream_loader`
- `_build_pairs_at_confluence`
- `_build_asymmetry_df`
- `_add_missing_stream_qc`
- `add_geometric_features_to_csv`

This should become the package entry point behind any CLI wrapper that enriches
CSV files with geometry.

### `channel_heads/geometric_analysis.py`

Become a re-export shim after extraction. Keep public imports stable from:

- top-level `channel_heads`
- `channel_heads.geometric_analysis`
- tests and old notebooks that import underscored helpers

Only remove private helper shims after explicit deprecation work.

## Behavior That Must Be Preserved Exactly

- `GEOM_FEATURE_COLS` order:
  `orientation_diff_deg`, `headhead_dist_m`, `headhead_dist_norm`,
  `apex_angle_deg`, `strahler_order_diff`, `proximity_mean_m`,
  `proximity_max_m`, `proximity_profile_norm`, `qc_flags`.
- Five model-feature order remains:
  `orientation_diff_deg`, `headhead_dist_norm`, `apex_angle_deg`,
  `strahler_order_diff`, `proximity_profile_norm`.
- `compute_delta_L = 2 * abs(L_ij - L_ji) / (L_ij + L_ji)`, with zero total
  returning `0.0`.
- `LengthwiseAsymmetryAnalyzer` upstream-distance conversion policy, including
  the current S1 assumption and projected/geographic branch behavior.
- Negative upstream distances warn, are clamped to zero, and still produce rows.
- Head ordering normalization: pairs are sorted by head ID and `L_1` / `L_2`
  are swapped to follow the normalized order.
- Earth coordinate convention: x = column, y = negative row.
- Direction vector calculation: weighted average over downstream path edges;
  include the edge that crosses the sample-distance threshold; QC flags
  `single_edge` and `short_path`.
- `_trace_path_downstream` greedy child choice toward the target on unexpected
  branching.
- `_trace_full_path` returns an empty list when the target is unreachable.
- `_sample_path_coords` samples fractions `0, 1/n, ..., (n-1)/n`, excluding the
  confluence endpoint.
- `strahler_order_diff` uses branch-parent order, not head order.
- `headhead_dist_norm` is `headhead_dist_m / (L_1 + L_2)` when path length is
  positive; otherwise NaN plus `zero_path_length`.
- `qc_flags` strings and comma-join behavior.
- `filter_hard_negatives`:
  - Preserve all positives.
  - If no positives or no negatives, return original rows.
  - Keep negatives with NaN length/distance values.
  - L threshold = positive median `(L_1 + L_2) * max_L_ratio`.
  - Distance threshold = positive median `headhead_dist_m * max_dist_ratio`.
  - Optional group recursion when `group_col` exists.
  - Optional stream-crossing filter only removes negatives.
  - Conservative keep on stream/node lookup errors.
  - Final sort by `outlet`, `confluence`, `head_1`, `head_2`.
- `generate_labeled_dataset` sets `y = touching.astype(int)` and merges on
  `outlet`, `confluence`, `head_1`, `head_2`.
- `add_geometric_features_to_csv`:
  - Drops deprecated `overlap_px`.
  - Normalizes `head_1` / `head_2` and swaps `L_1` / `L_2` if present.
  - Adds missing geom columns before processing.
  - Uses default latitude `36.0` and default `z_th = 0.0` when columns are
    absent.
  - Marks failed basins/outlets with `missing_stream` in `qc_flags`.
  - Default stream threshold remains `300`.
  - Writes CSV only when `output_csv` is provided.

## Proposed Implementation Slices

1. **Behavior-pinning tests first.**
   Add tests for public object identity, GEOM feature order, S1 conversion
   branch behavior, head normalization/L swapping, CSV enrichment edge cases,
   hard-negative filter ordering, and `_trace_full_path` import compatibility.

2. **Extract pure Earth path helpers.**
   Create `features/earth_paths.py` and repoint only tests/imports. Keep
   `geometric_analysis.py` re-exporting private helpers.

3. **Extract asymmetry.**
   Move `PairAsymmetryResult`, `compute_delta_L`,
   `LengthwiseAsymmetryAnalyzer`, stats, and asymmetry merge into
   `features/asymmetry.py`.

4. **Extract Earth geometry analyzer.**
   Move `PairGeometricResult`, `GEOM_FEATURE_COLS`,
   `GeometricFeaturesAnalyzer`, and geometry merge into
   `features/earth_geometry.py`.

5. **Extract labeling and hard-negative filtering.**
   Move dataset labeling/filtering into `training/labeling.py`, then repoint
   `scripts/build_earth_features_regime.py` only after tests prove identical.

6. **Extract CSV enrichment / loader.**
   Move default stream loading and CSV enrichment into
   `features/earth_enrichment.py`, using `io.paths` carefully.

7. **Reduce `geometric_analysis.py` to a shim.**
   Re-export all public symbols and temporarily re-export private helpers used
   by tests/rasterizer/notebooks.

8. **Update docs/tests import surfaces.**
   Update developer docs and tests after source identity is pinned.

## Risks

1. **S1 upstream-distance units.** This is unresolved and high priority. The
   audit recommends preserving behavior, not fixing it during extraction.
2. **Feature-order drift.** XGBoost training/inference depends on exact column
   order and presence.
3. **Private helper users.** `rasterizer.py` imports `_trace_full_path`; tests
   import many underscored helpers. A shim must preserve them initially.
4. **Basin-scoped child maps.** Local and pairing/rasterizer child-map helpers
   have different accepted inputs. Blind unification can change traversal.
5. **QC flag drift.** Flags are behavioral output and must not be renamed or
   reordered without tests.
6. **Training leakage policy.** `filter_hard_negatives` is currently full-dataset
   in some workflows; moving it should not silently change when it is applied.
7. **TopoToolbox import/load side effects.** Keep heavy imports and DEM reads out
   of pure feature modules where possible.

## Tests Needed For Future Moves

| Future move | Tests to add or expand |
|-------------|------------------------|
| `features/earth_paths.py` | Identity/regression tests for `_trace_path_downstream`, `_trace_full_path`, `_sample_path_coords`, direction-vector QC flags, and unreachable-target behavior. |
| `features/asymmetry.py` | Tests for projected vs geographic unit branches, negative-distance warning/clamp, pair normalization with L swapping, empty output schema, and top-level/shim object identity. |
| `features/earth_geometry.py` | Tests for `GEOM_FEATURE_COLS` order, y-axis negation, branch-parent Strahler behavior, path length normalization, QC flag strings, and merge order. |
| `training/labeling.py` | Tests for `generate_labeled_dataset` merge behavior, hard-negative global/per-group behavior, NaN keep semantics, stream-crossing optional filter, and final sort keys. |
| `features/earth_enrichment.py` | Temp-CSV tests for `overlap_px` drop, head/L swapping, missing basin/lat/z_th defaults, `missing_stream` flags, output path writes, and loader failure handling. |
| Shim phase | Tests that old imports from `channel_heads.geometric_analysis` and top-level `channel_heads` resolve to the new canonical objects. |

## Stop Conditions For Future Refactor Slices

Stop and ask before changing:

- Upstream-distance unit policy or S1 assumptions.
- Any feature column order or feature definition.
- Path tracing branch-choice behavior.
- QC flag names/formatting.
- Hard-negative thresholds, NaN keep behavior, group recursion, or stream filter.
- CSV output schema.
- Regime feature-generation behavior in `scripts/build_earth_features_regime.py`.
- Any data, root `models/`, notebooks, generated outputs, DEMs, shapefiles,
  GeoPackages, parquet/CSV outputs, or figures.
