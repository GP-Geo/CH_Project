# Developer Guide

Canonical developer documentation for **channel-heads** — paired channel-head
coupling detection on Earth (training) and Mars (inference), after Goren &
Shelef (2024).

> Companion docs: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) (repo/data layout),
> [ROADMAP_AND_RISKS.md](ROADMAP_AND_RISKS.md) (open work + scientific risks),
> [MARS_PIPELINE.md](MARS_PIPELINE.md) (cross-planet pipeline & phase history).

---

## 1. Overview

The project identifies pairs of channel heads that meet at confluences and
determines whether their drainage basins are spatially **coupled** (touching).
It then trains an ML classifier on Earth DEMs and applies it to Martian
valley-network topology.

> Goren, L. & Shelef, E.: *Channel concavity controls planform complexity of
> branching drainage networks*, Earth Surf. Dynam., 12, 1347–1369, 2024.
> https://doi.org/10.5194/esurf-12-1347-2024

## 2. Environment

```bash
conda env create -f env/environment.yml
conda activate ch-heads
pip install -e ".[dev,geo,viz,cnn,ml]"
python -c "from channel_heads import CouplingAnalyzer; print('OK')"
```

Key deps: topotoolbox 0.0.6+, numpy 2.0+, pandas 2.0+, rasterio, geopandas,
scikit-image, xgboost 3.2+, scikit-learn 1.7+, torch (CNN), pytest 8+.

Validation commands used throughout:
```bash
conda run -n ch-heads pytest -q
conda run -n ch-heads ruff check <files>      # targeted; repo-wide has known pre-existing errors
```

## 3. Package architecture (`channel_heads/`)

| Module | Role |
|--------|------|
| `coupling_analysis.py` | `CouplingAnalyzer` — basin coupling detection (parallel-safe, cached masks, stream-crossing gate). |
| `features/` | Dimensionless feature math: `asymmetry.py` (lengthwise ΔL), `geometry.py` / `paths.py` (angles, azimuth, proximity, sampling), `earth_enrichment.py` (CSV enrichment), `mars_features.py`. |
| `pairing/` | First-meet pairing: graph-agnostic core (`dag.py`), Earth/TopoToolbox adapter (`earth.py`, exports `first_meet_pairs_for_outlet`), Mars-graph helpers (`mars_graph.py`), hard-negative `filtering.py`. |
| `rasterization/` | 5-class 128×128 patch generation (`patches.py`, Earth + Mars), drawing primitives (`drawing.py`), `manifest.py`, and the class `schema.py`. Canonical for both planets. |
| `models/` | XGBoost load/verify/predict (`xgboost.py`), `thresholds.py`, model `comparison.py`, CNN (`cnn.py` `OutletCNN` 5→…→4-dim, `cnn_features.py`, `embeddings.py`), `device.py` (`pick_device`), Mars `mars_inference.py` / `mars_combined.py`, regime-CNN embedding attach (`regime.py`). |
| `training/` | Earth training: shared CNN loop (`cnn.py`), XGBoost variants (`xgboost.py`), dataset prep (`datasets.py`), labeling/filtering (`labeling.py`), regime dataset builders (`regime.py`). |
| `eval/` | Threshold tuning + classification metrics (`metrics.py`), grouped splits (`splitting.py`), `diagnostics.py`, **true-LOBO engine + leakage audit** (`lobo.py`), per-fold CNN factory (`lobo_cnn.py`; CLI `lobo-validate`). |
| `io/` | Canonical paths (`paths.py`), parquet/CSV table I/O (`tables.py`), GeoPackage I/O (`geopackage.py`), generated-data cleanup manifest (`cleanup.py`). |
| `pipelines/` | Readable top layer — one function per stage (`earth.py`, `mars.py`, `poster.py`). |
| `viz/` | Earth DEM/basin plotting plus vector figures: contact sheets, ROC curves, per-outlet, stream-crossing QA, calibration, poster/report figures (`poster.py`, incl. conformal Mars hillshade overlays). |
| `cli/` | CLI package: subcommand dispatcher + one module per command (run via `python -m channel_heads <command>` or the `channel-heads` / `ch-analyze` console scripts). |
| `dd_calibration.py` | Drainage-density / threshold calibration helpers. |
| `pruning.py` | Strahler-strip + order-gap network pruning (regime pipeline). |
| `units.py` | **Single source of truth for unit conversions** (added Phase 2 — see §6). |
| `regimes.py` | `Regime` dataclass + `REGIMES` presets (regA/B/C). |
| `basin_config.py`, `logging_config.py`, `stream_utils.py` | Basin params (z-thresholds, example DEMs), logging, `outlet_node_ids_from_streampoi`. |

Public API is re-exported from each subpackage's `__init__.py`.

**Notebooks are the primary interface; the CLI is the headless surface.** All
reusable logic lives in the modules above; each `channel_heads/cli/` command is
a thin `main(argv)` wrapper (`python -m channel_heads <command>`), and the
matching `notebooks/<home>/` notebook calls `channel_heads.*` (no duplicated
cell logic) and runs read-only. See [PROJECT_STRUCTURE.md §3a](PROJECT_STRUCTURE.md)
for the notebook map. New notebooks must live at `notebooks/<role>/` (depth 2)
so the root-resolution cells work, and must not embed "phase" numbering.

## 4. Core API (quick reference)

```python
import numpy as np, topotoolbox as tt3
from channel_heads import (
    CouplingAnalyzer, LengthwiseAsymmetryAnalyzer, GeometricFeaturesAnalyzer,
    first_meet_pairs_for_outlet, outlet_node_ids_from_streampoi,
    merge_coupling_and_asymmetry, merge_geometric_features,
    get_z_th, get_basin_config, EXAMPLE_DEMS, get_output_dir,
)

dem = tt3.read_tif(str(EXAMPLE_DEMS["inyo"]))
dem.z[dem.z < get_z_th("inyo")] = np.nan
fd = tt3.FlowObject(dem)
s  = tt3.StreamObject(fd, threshold=300)

coupling = CouplingAnalyzer(fd, s, dem)
for outlet in outlet_node_ids_from_streampoi(s):
    pairs, _ = first_meet_pairs_for_outlet(s, outlet)
    df = coupling.evaluate_pairs_for_outlet(outlet, pairs)
    coupling.clear_cache()   # between outlets
```

CLI: `ch-analyze dem.tif -o out.csv --threshold 300 --connectivity 8 -v`.

## 5. ML pipeline

**Earth (training):** DEM → flow → stream → first-meet pairs → coupling +
asymmetry + geometric features → labeled dataset → XGBoost.

- Production tabular model: `models/xgb_touching_classifier.json`, **5
  dimensionless features** (`orientation_diff_deg`, `headhead_dist_norm`,
  `apex_angle_deg`, `strahler_order_diff`, `proximity_profile_norm`), decision
  threshold **0.577406**. Dimensionless by design → portable to Mars.
- Earth CNN: `models/cnn_outlet_final.pt` — 128×128 uint8 5-class patches
  (BACKGROUND=0, BRANCH_A=1, BRANCH_B=2, OTHER_STREAMS=3, CONFLUENCE_MARKER=4),
  `embedding_dim=4`, no normalization, augment off at inference.
- Notebooks `training/00`–`05` implement dataset build → train → CNN.

**Mars (inference)** and the **regime-calibration** re-runs are documented in
[MARS_PIPELINE.md](MARS_PIPELINE.md). The CNN is **not** optional — the dual
track (tabular + CNN) is a design contract.

> ⚠ The production `xgb_touching_classifier.json` is **preserved as-is**.
> Research/regime variants live alongside it with explicit suffixes
> (`_geom_only`, `_geom_plus_cnn_emb`, `_reg{A,B,C}`).

## 6. Units (`channel_heads/units.py`)

Single source of truth for all unit handling (added in the Phase 2 refactor):
meters/degree, DEM pixel size (geographic vs projected), km²↔pixel-cells,
stream length (km), basin & hull area (km²), drainage density, and the
documented resolution of the `upstream_distance()` unit question (**risk S1**).
Former call sites (`dd_calibration.py`, `features/asymmetry.py`) route through
`units.py`; behavior is unchanged. See ROADMAP_AND_RISKS.md for S1.

## 7. Testing

```bash
pytest tests/ -v                                  # full suite
pytest tests/ -v --cov=channel_heads              # with coverage
pytest tests/test_coupling_analysis.py -v         # one file
```

Mock objects live in `tests/conftest.py` (`MockGridObject`, `MockFlowObject`,
`MockStreamObject`; fixtures `simple_y_network`, `complex_network`,
`touching_basins_network`). Test files map 1:1 to package modules.

When adding analysis code: create the module under `channel_heads/`, export it
from `__init__.py`, add tests under `tests/`, and update this guide +
PROJECT_STRUCTURE.md.

## 8. Conventions

- PEP 8, type hints (mypy in CI), NumPy-style docstrings, `black`, `ruff`.
- Prefer vectorized numpy/pandas over Python loops.
- Use `channel_heads.io.paths` paths, never hardcode.
- Call `clear_cache()` between outlets; use `evaluate_pairs_for_outlet_parallel`
  for large outlets.
- Never delete data; mark legacy instead (see PROJECT_STRUCTURE.md / archive policy).

## 9. CI

GitHub Actions on push/PR to `main`: pytest (3.11, 3.12), black + ruff,
mypy (informational). See `.github/workflows/tests.yml`.
