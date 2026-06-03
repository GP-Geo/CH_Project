# AGENT_ROOT_MODULE_AUDIT.md

Audit-only report for remaining root-level `channel_heads/*.py` modules.

Scope:
- Inspected actual root-level Python modules currently on disk.
- Inspected imports/usages in `channel_heads/`, `tests/`, `scripts/`, `docs/`,
  `README.md`, and `AGENT_*.md`.
- No source code was moved or modified.

Current root-level modules:
- `channel_heads/__init__.py`
- `channel_heads/basin_config.py`
- `channel_heads/cli.py`
- `channel_heads/coupling_analysis.py`
- `channel_heads/dd_calibration.py`
- `channel_heads/logging_config.py`
- `channel_heads/pruning.py`
- `channel_heads/regimes.py`
- `channel_heads/stream_utils.py`
- `channel_heads/units.py`

Older handoff docs still mention previous root shims such as
`geometric_analysis.py`, `rasterizer.py`, `cnn_model.py`, `cnn_features.py`,
`cnn_training.py`, `config.py`, `first_meet_pairs_for_outlet.py`, and
`plotting_utils.py`; those files are not present in the current checkout and
are outside this audit's file list.

## Classification Legend

1. `keep at package root`
2. `move into an existing subpackage`
3. `split into multiple modules`
4. `public API facade`
5. `unclear`

## Proposed Ownership Table

| Module | Current responsibility | Main import/use signal | Classification | Proposed ownership |
|--------|------------------------|------------------------|----------------|--------------------|
| `__init__.py` | Top-level curated package surface, metadata, optional torch exports, subpackage imports. | README and developer docs use `from channel_heads import CouplingAnalyzer, ...`; tests pin top-level re-exports for units/CNN/raster APIs. | 4. public API facade | Keep at root permanently as a curated facade. Avoid moving implementation here; keep imports shallow where possible and preserve optional torch guards. |
| `basin_config.py` | Goren & Shelef basin metadata, local-to-paper name map, `get_basin_config`, `get_z_th`, reference `delta_L`. | Used by `training.regime`, raster batch precompute, diagnostic scripts, README/docs. | 1. keep at package root | Keep at root as project-level scientific metadata/config. A future `channel_heads/config/` package could absorb it, but no existing subpackage is a cleaner owner today. |
| `cli.py` | Installed `ch-analyze` console command for one-DEM coupling analysis. | `pyproject.toml` maps `ch-analyze = "channel_heads.cli:main"`; README/docs document it. | 4. public API facade | Keep at root for the installed entry point. If a CLI subpackage is introduced later, make this a thin facade to `channel_heads.cli.analyze:main`; do not move in a root cleanup slice. |
| `coupling_analysis.py` | `CouplingAnalyzer`, `PairTouchResult`, basin-mask cache, pair-touch/contact computation, prefilter and stream-crossing gate. | Directly imported by tests, `training.regime`, `viz.earth`, `cli.py`; README exposes `CouplingAnalyzer` as a primary API. | 2. move into an existing subpackage | Move implementation to `channel_heads/training/coupling.py` or `channel_heads/training/labeling.py` adjacent to label generation. Leave `channel_heads/coupling_analysis.py` as a re-export shim and keep top-level `channel_heads.CouplingAnalyzer`. |
| `dd_calibration.py` | Large mixed module: Earth threshold sweeps, Mars mapped-network Dd/Strahler metrics, hull geometry, trimming, pruning-variant payloads, sweep cache augmentation, calibration scoring helpers. | Direct tests cover many helpers; `training.regime` imports threshold/cell/trim helpers; diagnostics script imports many public helpers; docs identify this as DD calibration. | 3. split into multiple modules | Split in stages. Move Mars mapped-network table/Strahler helpers to `channel_heads/mars/calibration.py`; move Earth threshold/trim sweep orchestration to `channel_heads/training/calibration.py`; move trim/pruning helpers into `channel_heads/pruning.py` or `channel_heads/training/pruning.py`; keep `dd_calibration.py` as a compatibility facade. |
| `logging_config.py` | Package logger setup, env-driven log level/file handling, `get_logger`, `set_verbose`. | Imported across `mars`, `models`, `features`, `rasterization`, `pipelines`, scripts. | 1. keep at package root | Keep at root as cross-cutting infrastructure. Moving to `io` would be misleading because it is not data IO. |
| `pruning.py` | Strahler graph construction, order-gap pruning, combined regime pruning strategy. | Used by `training.regime`, tested directly, docs describe it as regime pruning. Depends on `dd_calibration.trim_to_min_order`. | 2. move into an existing subpackage | Move implementation to `channel_heads/training/pruning.py` because it is currently part of regime training/data construction. First pull `trim_to_min_order` and `trim_first_order` out of `dd_calibration.py` into this owner to remove the reverse dependency. Keep root `pruning.py` as a shim. |
| `regimes.py` | `Regime` dataclass and `REGIMES` presets for regA/regB/regC. | Used by all regime CLI wrappers and tests; docs refer to it as the canonical preset map. | 1. keep at package root | Keep at root as project-wide named presets. Do not bury in `training/` because the presets are also consumed by Mars inference and scripts. |
| `stream_utils.py` | Small mixed helper module: Bresenham-like `line_pixels` and `outlet_node_ids_from_streampoi`. | `line_pixels` used by coupling/labeling/DD tests; outlet helper used by CLI and regime training. | 3. split into multiple modules | Eventually move `line_pixels` to `channel_heads/rasterization/drawing.py` or a shared geometry helper, and move `outlet_node_ids_from_streampoi` to an Earth stream/pairing helper. Keep root `stream_utils.py` as a shim until all call sites are repointed. |
| `units.py` | Single source of truth for meters/degree, pixel size, threshold cells, stream length, basin area, drainage density. | Tests explicitly pin `units` as canonical and assert `dd_calibration`/top-level re-export identity. Many modules consume it. | 1. keep at package root | Keep at root. It is intentionally cross-cutting and already extracted from older modules. |

## Module Notes

### Keep At Root

`basin_config.py`, `logging_config.py`, `regimes.py`, and `units.py` are
package-level infrastructure or scientific constants used across multiple
subpackages. Moving them now would add indirection without reducing complexity.

### Public Facades

`__init__.py` and `cli.py` should stay as public surfaces. They can delegate to
canonical modules, but their import paths are part of the documented API:
- `from channel_heads import ...`
- `ch-analyze` via `channel_heads.cli:main`

### Move Candidates

`coupling_analysis.py` is implementation-heavy and domain-specific. Its closest
existing owner is `training/` because it produces ground-truth coupling labels
used by Earth/regime training data. A move should preserve:
- `channel_heads.coupling_analysis.CouplingAnalyzer`
- `channel_heads.CouplingAnalyzer`
- DataFrame schema from `evaluate_pairs_for_outlet`
- prefilter and stream-crossing behavior
- cache/thread-safety behavior

`pruning.py` is also implementation-heavy and currently specific to regime
dataset construction. It should move only after `trim_to_min_order` and
`trim_first_order` are separated from `dd_calibration.py`, because the current
dependency direction is backwards: `pruning.py` imports trim helpers from the
larger calibration module.

### Split Candidates

`dd_calibration.py` is the main remaining root-level aggregation point. It has
at least four responsibilities:
- shared DD/hull metric functions and statuses,
- Earth DEM threshold sweeps,
- Mars mapped-network reference geometry and Strahler summaries,
- pruning/trim comparison helpers and cache augmentation utilities.

It should not be moved wholesale. A safe split should preserve
`channel_heads.dd_calibration` as the public facade while extracting smaller
canonical owners.

`stream_utils.py` is small but mixed. Splitting it is lower priority than
`dd_calibration.py`, and should wait until `coupling_analysis.py` and
`training.labeling` call a shared canonical line helper.

## Safe Migration Order

1. Add/confirm import-identity tests for every root facade before moving code.
   Minimum coverage: `channel_heads.__init__`, `channel_heads.coupling_analysis`,
   `channel_heads.dd_calibration`, `channel_heads.pruning`, and
   `channel_heads.stream_utils`.

2. Break the `pruning.py -> dd_calibration.py` dependency.
   Move `trim_to_min_order` and `trim_first_order` into the pruning owner first,
   then make `dd_calibration.py` re-export them. Focused tests:
   `tests/test_pruning.py tests/test_dd_calibration.py tests/test_training_regime.py`.

3. Split `dd_calibration.py` in small pieces.
   Suggested first extraction: Mars mapped-network helpers to
   `channel_heads/mars/calibration.py` because they do not require Earth DEMs.
   Suggested second extraction: Earth threshold sweep helpers to
   `channel_heads/training/calibration.py`. Keep `dd_calibration.py` as a
   re-export facade until all docs/scripts are repointed.

4. Move `coupling_analysis.py` implementation.
   Put `CouplingAnalyzer` and `PairTouchResult` in
   `channel_heads/training/coupling.py` or a similarly named training module.
   Leave `channel_heads/coupling_analysis.py` as a shim. Focused tests:
   `tests/test_coupling_analysis.py tests/test_coupling_parallel.py
   tests/test_training_regime.py tests/test_viz_earth.py`.

5. Split `stream_utils.py` only after the coupling move.
   Move `line_pixels` to the canonical geometry/raster helper used by
   coupling and labeling; move `outlet_node_ids_from_streampoi` to an Earth
   stream/pairing helper. Keep root re-exports. Focused tests:
   `tests/test_stream_utils.py tests/test_dd_calibration.py
   tests/test_coupling_analysis.py tests/test_training_regime.py`.

6. Leave root infrastructure in place.
   Do not move `basin_config.py`, `logging_config.py`, `regimes.py`, `units.py`,
   `__init__.py`, or `cli.py` unless a later design explicitly introduces a
   config/CLI subpackage and updates the public API contract.

## Suggested Future Slices

1. `refactor(pruning): make pruning the canonical Strahler trim owner`
2. `refactor(calibration): split Mars drainage-density calibration helpers`
3. `refactor(calibration): split Earth threshold sweep helpers`
4. `refactor(training): move coupling analyzer behind root shim`
5. `refactor(utils): split stream utility helpers`

## Residual Risks

- `dd_calibration.py` is large and highly tested; broad moves are likely to
  create import cycles unless trim/pruning is separated first.
- `CouplingAnalyzer` is a documented top-level API; any move must preserve old
  imports exactly.
- `stream_utils.line_pixels` affects stream-crossing gates and labels. Treat it
  as scientific behavior, not a cosmetic helper.
- `units.py` contains the open S1 upstream-distance unit assumption. Do not
  change unit behavior during module ownership cleanup.

