# AGENT_AUDIT_RASTERIZER.md - Slice 7 audit

Date: 2026-06-02

Scope: read-only audit of `channel_heads/rasterizer.py`,
`channel_heads/rasterization/`, Earth/regime patch scripts, Mars patch package
code, and relevant docs/tests. No implementation code was changed.

## Executive Summary

The rasterization layer is partially package-first but not fully consolidated.
Mars Phase 4 patch generation is package-resident in
`channel_heads/rasterization/mars_patches.py`, while Earth patch generation and
the frozen 5-class CNN patch contract still live in `channel_heads/rasterizer.py`.

`channel_heads/rasterization/patches.py` is currently a curated re-export, not a
real implementation module. Future work should promote the Earth rasterization
implementation into `channel_heads/rasterization/patches.py` (or
`earth_patches.py`) and reduce `channel_heads/rasterizer.py` to a shim. Do this
only after behavior-pinning tests, because patch geometry is part of the CNN
model contract and stale artifacts already exist from the rasterizer rewrite.

## Reference Map

| File | Current role | Real logic present |
|------|--------------|--------------------|
| `channel_heads/rasterizer.py` | Earth 5-class rasterization and batch precompute | Yes. Owns class constants, Bresenham, direct final-grid projection/drawing, QA flags, Earth `rasterize_outlet_pair`, and `precompute_raster_dataset`. |
| `channel_heads/rasterization/patches.py` | Curated Earth/Mars patch API | No independent logic. Re-exports from `rasterizer.py` and defines `CLASS_LABELS`. |
| `channel_heads/rasterization/drawing.py` | Mars drawing primitives | Yes. Owns projected-metre to image coordinate conversion, Mars-compatible rotation, polyline drawing, and re-exports `bresenham_line`. |
| `channel_heads/rasterization/manifest.py` | Mars patch-index schema | Yes. Owns manifest columns/status validation and empty QA flags. Currently Mars-specific even though QA flag names match Earth. |
| `channel_heads/rasterization/mars_patches.py` | Mars Phase 4 patch generation | Yes. Package-resident Mars 5-class patch renderer, manifest writer, and QA contact-sheet logic. |
| `channel_heads/rasterization/__init__.py` | Curated rasterization API | Re-exports Earth shim surface, Mars patch generation, drawing, and manifest helpers. |
| `scripts/build_cnn_patches_regime.py` | Regime Step 3 Earth patch builder | Real transitional logic. Owns regime stream loader/pruning and manifest path writes; delegates rasterization to `precompute_raster_dataset`. |
| `scripts/build_mars_cnn_patches_5class.py` | Mars Phase 4 script | Thin wrapper over `channel_heads.pipelines.build_mars_cnn_patches`. |
| `channel_heads/pipelines/mars.py` | Mars package pipeline | Phase 4 calls `channel_heads.rasterization.build_mars_cnn_patches` directly. |
| `tests/test_rasterizer.py` | Earth rasterizer and shared Mars smoke tests | Broad behavior coverage for direct projection, class values, QA flags, batch precompute, invalid/debug paths, and Mars class parity. |
| `tests/test_rasterize_shared.py` | Bresenham shared primitive tests | Covers endpoint inclusion and 8-connected line behavior. |
| `tests/test_mars_patches.py` | Mars patch + manifest tests | Covers Mars shape/dtype/classes, confluence marker, branch protection, alias, and manifest schema validation. |
| `docs/DATA_STATUS.md` / `docs/PIPELINE_RERUN.md` | Artifact status docs | Mark raster-dependent artifacts as stale after the direct-final-grid rewrite. |

## Script / Module Classification

### Keep As Thin CLI Wrappers Later

- `scripts/build_mars_cnn_patches_5class.py` is already thin.
- `scripts/build_cnn_patches_regime.py` should become thin after regime stream
  loader/orchestration moves into package code.

### Keep As Package Implementation

- `channel_heads/rasterization/mars_patches.py` should remain the Mars package
  implementation.
- `channel_heads/rasterization/drawing.py` should remain the Mars drawing
  primitive module unless a proven shared abstraction replaces it.
- `channel_heads/rasterization/manifest.py` should remain manifest/schema logic,
  with possible expansion for Earth manifests later.

### Move Into Package Later

- `channel_heads/rasterizer.py` implementation should move to
  `channel_heads/rasterization/patches.py` or
  `channel_heads/rasterization/earth_patches.py`.
- `channel_heads/rasterizer.py` should become a compatibility shim re-exporting
  constants, helpers, `rasterize_outlet_pair`, `precompute_raster_dataset`, and
  `raster_quality_flags`.

## Duplication Map

| Concern | Existing owner | Duplicated / transitional owner | Future direction |
|---------|----------------|----------------------------------|------------------|
| 5-class constants | `rasterizer.py` | Imported by `mars_patches.py`; `patches.py` re-exports; CNN modules import `NUM_CLASSES` from `rasterizer.py` | Move constants to `rasterization/schema.py` or `patches.py`; keep `rasterizer.py` shim. |
| Bresenham line | `rasterizer.py` | `drawing.py` imports and re-exports it | Keep one implementation. If moved, preserve `channel_heads.rasterizer.bresenham_line` identity/shim. |
| Rotation angle | `rasterizer._compute_rotation_angle` | `drawing.compute_rotation_angle` is a Mars port | Do not merge blindly; add parity tests first because Earth uses row/col nodes and Mars uses projected metres converted to row/col. |
| Coordinate rotation | `rasterizer._rotate_coordinates` | `drawing.rotate_rc` | Same behavior but different naming/surface. Consider shared low-level primitive after tests. |
| Direct final-grid projection | `rasterizer._project_to_target_grid`; local `project_to_target` inside `mars_patches.render_pair_patch` | Similar algorithm in two data models | Keep separate initially; if generalized, tests must compare Earth/Mars simple-Y outputs and connectivity. |
| Branch drawing with protection | `rasterizer._draw_path_on_target_grid`; `drawing.draw_polyline` | Same policy adapted to nodes vs LineStrings | Keep separate until path representation abstraction exists. |
| QA flags | `rasterizer.raster_quality_flags` | Mars imports it; manifest repeats flag column names | Move QA flags and flag columns into a schema/QA module only with identity tests. |
| Earth batch manifest | `precompute_raster_dataset` appends `raster_*` columns to master CSV | Mars uses `manifest.py` patch-index schema | Keep schemas distinct; Earth manifest is master-row based, Mars manifest is patch-index based. |
| Regime stream loading | `scripts/build_cnn_patches_regime.py` | Future `training/regime.py` from Earth/regime audit | Move into regime package workflow, not into low-level rasterization. |

## Recommended Canonical Package Ownership

### `channel_heads/rasterization/schema.py` (new)

Recommended future home for constants shared by Earth, Mars, CNN, and manifest
code:

- `BACKGROUND = 0`
- `BRANCH_A = 1`
- `BRANCH_B = 2`
- `OTHER_STREAMS = 3`
- `CONFLUENCE_MARKER = 4`
- `NUM_CLASSES = 5`
- `CLASS_LABELS`
- Possibly `PATCH_FLAG_COLUMNS`

This avoids `models/cnn.py` importing `NUM_CLASSES` from a future shim.

### `channel_heads/rasterization/patches.py`

Promote this from re-export surface to the canonical Earth patch implementation:

- `bresenham_line`
- `_project_to_target_grid`
- `_draw_path_on_target_grid`
- `_draw_edges_on_target_grid`
- `_component_count`
- `raster_quality_flags`
- `_get_rc`
- `_compute_rotation_angle`
- `_rotate_coordinates`
- `rasterize_outlet_pair`
- `precompute_raster_dataset`

Alternative: create `earth_patches.py` for the real implementation and keep
`patches.py` as the curated public surface. This is cleaner if private helper
exports need to remain temporary.

### `channel_heads/rasterization/drawing.py`

Keep as Mars projected-geometry drawing support:

- `xy_to_rc`
- `linestring_to_rc`
- `compute_rotation_angle`
- `rotate_rc`
- `draw_polyline`

Only unify with Earth drawing after tests prove identical behavior across the
coordinate-model boundary.

### `channel_heads/rasterization/mars_patches.py`

Keep as canonical Mars Phase 4:

- `render_pair_patch`
- `rasterize_mars_pair` alias
- `build_mars_cnn_patches`
- QA contact sheets
- patch manifest writes

### `channel_heads/rasterization/manifest.py`

Keep Mars patch-index schema. Consider adding an Earth manifest helper later
only if it can preserve the existing `precompute_raster_dataset` DataFrame
columns exactly.

### `channel_heads/rasterizer.py`

After extraction, reduce to a pure compatibility shim. Preserve old imports from:

- `channel_heads.rasterizer`
- `channel_heads.rasterization.patches`
- `channel_heads.rasterization`
- `channel_heads.models.cnn` and legacy CNN tests that use `NUM_CLASSES`

## Behavior That Must Be Preserved Exactly

### Shared CNN Patch Contract

- Five classes only:
  - `0 = BACKGROUND`
  - `1 = BRANCH_A`
  - `2 = BRANCH_B`
  - `3 = OTHER_STREAMS`
  - `4 = CONFLUENCE_MARKER`
- `NUM_CLASSES = 5`.
- Patch dtype is `uint8`.
- Default patch size is `128 x 128`.
- No normalization and no augmentation in patch generation.
- Direct final-grid projection must remain; do not return to native draw then
  nearest-neighbor resize.
- Branch A/B mutual protection must remain.
- Confluence marker overwrites branch values.

### Earth `rasterize_outlet_pair`

- Builds parents from stream, collects basin nodes from outlet, then builds a
  basin-scoped children map.
- Uses `_trace_full_path(head, confluence, children)` for each branch.
- Rotation puts the head midpoint above the confluence in raster coordinates.
- Rotation center is the confluence.
- Bounding box is computed over branch A union branch B after rotation.
- Minimum row/column span is `2.0`; padding default is `0.2`.
- Draw order:
  1. all basin edges as `OTHER_STREAMS`
  2. branch A as `BRANCH_A` protecting branch B
  3. branch B as `BRANCH_B` protecting branch A
  4. confluence marker as `CONFLUENCE_MARKER`
- `grid_shape` remains an argument even though current direct-final-grid
  implementation does not use it for clipping.
- Swapping `head_1` and `head_2` swaps branch labels.

### Earth `precompute_raster_dataset`

- Loader signature remains `(basin, lat, z_th, threshold) -> (s, dem) | None`.
- Basin config is loaded with `get_basin_config(basin_name)`.
- Output directory remains `output_dir / basin / "rasters"`.
- File name remains `{outlet}_{head_1}_{head_2}.npy`.
- Always saves the debug patch path when rasterization succeeds.
- `raster_path` is populated only when `branches_connected` is true.
- Invalid patches have `raster_status = "invalid"`, no `raster_path`, and
  `raster_error = "qa_failed:" + comma-joined failed flags`.
- Missing config and missing DEM/stream become `skipped` with existing error
  strings.
- Rasterizer exceptions become `failed` with `"{ExceptionType}: {exc}"`.
- Output columns stay:
  `raster_path`, `raster_debug_path`, `raster_status`, `raster_error`,
  `has_branch_a`, `has_branch_b`, `has_confluence`,
  `branch_a_connected`, `branch_b_connected`, `branches_connected`.

### Regime Patch Script Behavior

- Regime stream loader must rebuild the same pruned stream as
  `scripts/build_earth_features_regime.py` so node IDs resolve.
- Threshold is regime km2 converted to cells through DEM pixel size.
- DEM z-threshold masking stays before flow/stream construction.
- Pruning order stays `pre_remove_max_order` then `order_gap_to_prune`.
- `precompute_raster_dataset(..., threshold=0)` remains acceptable because the
  regime loader ignores the forwarded threshold.
- Output root currently used by code is `RESULTS_DIR / f"_rasters_{regime.name}"`.
- Manifest path stays `RESULTS_DIR / f"raster_manifest_{regime.name}.csv"`.

### Mars Patch Behavior

- `TARGET_SIZE = 128`, `PADDING_FRAC = 0.2`,
  `PATCH_ENCODING = "earth_5class"`, `MOLA_CELL_SIZE_M = 200.0`.
- Mars coordinates convert projected metres to image row/col by
  `row = -y / cell_size_m`, `col = x / cell_size_m`.
- Bounding box is over rotated branch A/B paths, not all streams.
- Other streams draw before branch A/B.
- `rasterize_mars_pair is render_pair_patch`.
- Manifest schema/order from `PATCH_INDEX_COLUMNS` stays fixed.
- `patch_status` values stay `ok`, `invalid`, `failed`, `skipped`.
- `patch_path` is project-root relative when written.
- `patch_shape` format is `"128,128"`, dtype string is `"uint8"`.
- `build_mars_cnn_patches` returns
  `{"manifest", "n_ok", "n_invalid", "n_failed", "paths"}`.
- QA contact-sheet constants and selection policy stay unchanged unless moved
  under separate visualization audit.

## Proposed Implementation Slices

1. **Behavior-pinning tests first.**
   Add tests for old/new import identity, constants, direct projection,
   branch-protection edge cases, Earth manifest exact columns/status strings,
   regime output-root/manifest path decisions, and Mars manifest return shape.

2. **Extract shared schema/constants.**
   Create `rasterization/schema.py`; repoint modules while keeping
   `rasterizer.py` and `rasterization.patches` re-export identities.

3. **Move Earth patch implementation.**
   Promote `rasterization/patches.py` or create `earth_patches.py`; move Earth
   helpers and `rasterize_outlet_pair` with a `rasterizer.py` shim.

4. **Move Earth batch precompute.**
   Move `precompute_raster_dataset` after single-patch behavior is pinned.
   Preserve DataFrame columns and status/error strings.

5. **Resolve `geometric_analysis._trace_full_path` dependency.**
   Repoint Earth rasterization to the future Earth path helper from the Slice 6
   split, or move that helper first.

6. **Package regime patch generation.**
   Move `make_regime_stream_loader` and Step 3 orchestration into the future
   `channel_heads/training/regime.py`; leave
   `scripts/build_cnn_patches_regime.py` as a CLI wrapper.

7. **Optional drawing primitive cleanup.**
   Consider unifying `_compute_rotation_angle` / `rotate_rc` only after parity
   tests across Earth node coordinates and Mars projected geometries.

8. **Docs and artifact-status update.**
   Update docs to name the new canonical module, while preserving the warning
   that raster-dependent artifacts are stale after the direct-final-grid rewrite.

## Risks

1. **CNN artifact compatibility.** Changing class values, class count, target
   size, dtype, or orientation invalidates CNN model behavior.
2. **Direct projection regression.** Returning to draw-then-resize can drop the
   one-pixel confluence marker and disconnect thin branches.
3. **Import cycles.** `models/cnn.py` imports `NUM_CLASSES`; moving constants
   must not create a `models -> rasterization -> models` cycle.
4. **Earth/Mars coordinate models differ.** Mars ports operate on LineStrings in
   projected metres; Earth operates on node IDs and DEM row/col arrays.
5. **Private helper dependency.** `rasterizer.py` imports
   `geometric_analysis._trace_full_path`; the geometric split and raster split
   should be ordered carefully.
6. **Manifest schema drift.** Earth and Mars manifests are similar but not the
   same. Do not force one schema prematurely.
7. **Regime node-ID consistency.** Regime patches must use the same pruned
   stream as regime feature generation.
8. **Generated artifacts.** Tests must use temp paths and must not touch
   `data/`, root `models/`, DEMs, GeoPackages, CSV/parquet outputs, or figures.

## Tests Needed For Future Moves

| Future move | Tests to add or expand |
|-------------|------------------------|
| Schema constants | Old/new import identity for `BACKGROUND`, `BRANCH_A`, `BRANCH_B`, `OTHER_STREAMS`, `CONFLUENCE_MARKER`, `NUM_CLASSES`; CNN imports remain cycle-free. |
| Earth patch move | Existing `tests/test_rasterizer.py` plus object-identity tests for shim exports and direct final-grid connectivity at small target sizes. |
| Earth batch precompute move | Temp-path tests for `ok`, `invalid`, `failed`, `skipped`, debug path writes, no `raster_path` for invalid, exact error strings, and column order. |
| Drawing cleanup | Property/parity tests for rotation/coordinate transforms and branch-protection on simple Y geometries. |
| Regime patch package | Monkeypatch DEM loader/pruning and `precompute_raster_dataset`; assert threshold conversion, pruning arguments, output root, manifest path, target-size default, and loader failure behavior. |
| Mars patch package | Existing Mars tests plus `build_mars_cnn_patches(write=False)` tests with mocked GPKG/parquet reads to assert manifest rows/status counts without writing artifacts. |
| Shim phase | Tests that `channel_heads.rasterizer`, `channel_heads.rasterization.patches`, and `channel_heads.rasterization` all expose the same callable/constant objects. |

## Stop Conditions For Future Refactor Slices

Stop and ask before changing:

- Class values, class count, dtype, target size, padding, rotation, or draw order.
- Direct final-grid projection.
- Branch A/B protection or confluence marker overwrite behavior.
- QA flag names or connectivity definition.
- Earth `precompute_raster_dataset` status/error strings or columns.
- Regime patch output roots/manifest names or pruning/threshold behavior.
- Mars `PATCH_INDEX_COLUMNS`, `PATCH_ENCODING`, or patch path relativity.
- Any data, root `models/`, notebooks, generated outputs, DEMs, shapefiles,
  GeoPackages, parquet/CSV outputs, or figures.
