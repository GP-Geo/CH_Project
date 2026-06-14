# Channel-Head Coupling — Earth → Mars

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TopoToolbox](https://img.shields.io/badge/TopoToolbox-0.0.6-green.svg)](https://github.com/TopoToolbox/pytopotoolbox)

Automated detection, analysis, and ML-based classification of **coupled channel
heads** in drainage networks derived from DEMs — trained on Earth, applied to
Mars valley networks. Based on Goren & Shelef (2024,
[doi:10.5194/esurf-12-1347-2024](https://doi.org/10.5194/esurf-12-1347-2024)).

## What this does

The pipeline pairs channel heads that meet at confluences ("first-meet" pairs),
detects whether their basins are spatially **coupled** (touching), and predicts
coupling with an **XGBoost** classifier (Earth held-out test AUC ≈ 0.92) augmented
by a **5-class CNN** that reads the local network geometry. Because every feature
is **dimensionless**, the Earth-trained model transfers to Martian valley networks
without retraining (**Earth → Mars transfer learning**).

**Scientific goal.** Quantify how often Martian valley-network channel heads are
coupled, and compare that signature to Earth's. Coupling is a fingerprint of the
erosional/hydrological process that carved a network, so the Earth-vs-Mars
coupling rate is evidence about how Martian valleys formed. Because Earth networks
are far denser than Mars, the comparison is bracketed by **three frozen
"complexity regimes"** (`regA/B/C`, thresholds 0.20 / 0.25 / 0.15 km²) that span
the Earth-network simplifications closest to Mars; reporting the **spread** across
the three regimes *is* the calibration-uncertainty estimate (no single regime is
declared "correct"). See [docs/regimes_summary.md](docs/regimes_summary.md).

The project is **package-first**: all logic lives in `channel_heads/` and runs
through `channel_heads.pipelines`; notebooks and the `channel_heads.cli` package
are thin layers on top. See [docs/architecture.md](docs/architecture.md).

## Install

```bash
conda env create -f env/environment.yml
conda activate ch-heads
pip install -e ".[dev,geo,viz,cnn,ml]"
python -c "from channel_heads import CouplingAnalyzer; print('OK')"
channel-heads --help          # list all pipeline/training/figure commands
```

## Quick start (single basin)

```python
import numpy as np, topotoolbox as tt3
from channel_heads import (
    CouplingAnalyzer, first_meet_pairs_for_outlet, get_z_th, EXAMPLE_DEMS,
)

dem = tt3.read_tif(str(EXAMPLE_DEMS["inyo"]))
dem.z[dem.z < get_z_th("inyo")] = np.nan          # elevation mask
fd = tt3.FlowObject(dem)
s  = tt3.StreamObject(fd, threshold=300)

pairs, heads = first_meet_pairs_for_outlet(s, outlet_id=5)
results = CouplingAnalyzer(fd, s, dem, connectivity=8).evaluate_pairs_for_outlet(5, pairs)
print(results)
```

Batch CLI: `channel-heads analyze data/cropped_DEMs/Inyo_strm_crop.tif -o out.csv --threshold 300 -v`

## Pipeline stages (0 → 14)

The canonical, end-to-end treatment is the enumerated notebook set in
[`notebooks/pipeline/`](notebooks/pipeline/) — **one self-contained notebook per
stage** (start here). Heavy rebuilds run through the `channel-heads` CLI command
listed for each stage. Full design: [docs/PIPELINE_DESIGN.md](docs/PIPELINE_DESIGN.md);
per-stage I/O and dependencies: [docs/pipeline.md](docs/pipeline.md); per-stage
asset coverage: [STAGE_ASSET_MAP.md](STAGE_ASSET_MAP.md).

| Stage | Notebook (`notebooks/pipeline/`) | Rebuild command |
|------:|----------------------------------|-----------------|
| 0 | `00_project_setup_and_assumptions` | — (paths / regimes / units) |
| 1 | `01_earth_source_data_exploration` | — (17 Earth DEMs) |
| 2 | `02_earth_interactive_network_exploration` | — |
| 3 | `03_mars_interactive_network_exploration` | `run-mars-pipeline --stage topology` |
| 4 | `04_earth_mars_regime_calibration` | — (regA–regE selection) |
| 5 | `05_final_earth_network_generation_and_qa` | `build-earth-features --regime <r>` |
| 6 | `06_earth_pair_and_label_generation` | (pairing / labeling) |
| 7 | `07_earth_model_input_construction` | `build-cnn-patches --regime <r>` |
| 8 | `08_model_training` | `train-cnn-regime`, `train-combined-xgb-regime` |
| 9 | `09_earth_model_validation_and_tuning` | `eval-lobo-cv`, `retune-threshold-regime` |
| 10 | `10_final_mars_model_input_generation` | `run-mars-pipeline --stage patches/embeddings` |
| 11 | `11_mars_inference` | `run-mars-combined-regime --regime <r>` |
| 12 | `12_mars_threshold_and_prediction_analysis` | (threshold sweep) |
| 13 | `13_scientific_interpretation` | (interpretation) |
| 14 | `14_figures_poster_and_reporting` | `make-result-figures`, `generate-poster-figures` |

## Reproduce the main workflow

```bash
# 1. Mars cross-planet pipeline, Phases 1–6C (topology → pairs → features →
#    xgb → patches → embeddings → combined). Outputs land under data/Mars/.
channel-heads run-mars-pipeline --stage all
# …or from Python:  from channel_heads import pipelines; pipelines.run_full_mars_pipeline()

# 2. Regime calibration rebuild (Earth features → CNN patches → CNN →
#    combined XGB → Mars inference), per regime:
scripts/run_regime_pipeline.sh regA      # then regB, regC

# 3. Final figures / poster:
channel-heads make-result-figures
channel-heads generate-poster-figures
```

All `data/` and `models/` artifacts are **regenerable**; the full regeneration
order and guardrails are in [docs/PIPELINE_RERUN.md](docs/PIPELINE_RERUN.md).
Validate the package at any time with `conda run -n ch-heads pytest -q`.

## Repository layout

```
channel_heads/      Python package — all pipeline/model/raster/training/viz logic
  cli/              command surface (python -m channel_heads <command>)
  pipelines/        readable top layer, one function per stage (earth/mars/poster)
  features/ pairing/ rasterization/ models/ training/ eval/ io/ viz/ mars/
tests/              pytest suite (598 tests; maps ~1:1 to package modules)
scripts/            shell orchestrators, headless diagnostics, rendering, _archive/
notebooks/          pipeline/ (canonical 00–14) + themed + archive/ (§ below)
docs/               all project documentation (see index below)
data/               inputs + generated outputs   (gitignored)
models/             trained model artifacts        (gitignored)
env/                conda environment spec
```

### Where things live

| You want… | Look in |
|-----------|---------|
| Library code / public API | `channel_heads/` (re-exported from each `__init__.py`) |
| Command-line surface | `channel_heads/cli/` → `channel-heads <command>` |
| **Recommended user-facing notebooks** | [`notebooks/pipeline/00–14`](notebooks/pipeline/) (canonical deep dive) |
| Earth DEMs (17, raw input) | `data/cropped_DEMs/`, `data/raw/` |
| Mars DEM / MOLA hillshade / valley vectors | `data/Mars/`, `data/final_valleys/` |
| Canonical pipeline outputs | `data/results/` (per-basin dirs, master datasets, regime rasters) |
| Mars predictions | `data/Mars/model_outputs/` |
| Rendered figures / reports | `data/results/poster_figures/`, `data/exports/` |
| Trained models | `models/` — production `xgb_touching_classifier.json` (threshold 0.577406) + `cnn_outlet_final.pt`; regime variants `*_reg{A,B,C}` |

Data categories, cleanup policy, and the safe `channel_heads.io.cleanup` tool:
[docs/data_management.md](docs/data_management.md); per-path keep/regenerate
status: [docs/DATA_STATUS.md](docs/DATA_STATUS.md).

## Notebooks

Two tiers (full catalogue: [docs/notebooks.md](docs/notebooks.md)):

1. **`notebooks/pipeline/00–14` — canonical deep dive.** One enumerated,
   self-contained notebook per pipeline stage. **Start here.**
2. **Themed folders** (`analysis/`, `mars/`, `training/`, `diagnostics/`,
   `presentation/`, `interpretation/`) — supporting / historical material that
   backs individual stages.

Every notebook calls `channel_heads.*` only (no duplicated logic) and writes
outputs solely via canonical path constants from `channel_heads.io.paths`.

## Archived material (kept for provenance, not maintained)

Nothing scientific is deleted — superseded work is **archived in place**:

- `notebooks/archive/` — early exploratory / one-off notebooks
  (`00_full_pipeline`, `01–04` basin runs, `experiment_*`). See
  [`notebooks/archive/README.md`](notebooks/archive/README.md).
- `notebooks/archive/regime/` — regime-calibration notebooks
  (`00_calibration_overview`, `01_mars_inference`, `02_threshold_retune`,
  `03_optimize_regime_candidates`), **superseded** by Stage-4
  `pipeline/04_earth_mars_regime_calibration`.
- `scripts/_archive/` — superseded Mars wrappers + one-off experiments; replaced
  by `channel-heads run-mars-pipeline --stage <…>` commands.
- `data/outputs/` — legacy duplicate of `data/results/` (marked, gitignored);
  `data/_rebuild_backup_*`, `data/_stage7_archive_*`, `data/_mars_stage10_archive_*`
  — dated backups kept off-repo.

## Documentation

All project documentation lives in [`docs/`](docs/):

| Doc | Contents |
|-----|----------|
| [docs/architecture.md](docs/architecture.md) | **Start here** — package-first layout + public API |
| [docs/pipeline.md](docs/pipeline.md) · [docs/PIPELINE_DESIGN.md](docs/PIPELINE_DESIGN.md) | Stage-by-stage flow, inputs/outputs, dependency graph |
| [docs/modeling.md](docs/modeling.md) | Model variants + the Mars operating-threshold issue |
| [docs/regimes_summary.md](docs/regimes_summary.md) · [docs/REGIME_SELECTION.md](docs/REGIME_SELECTION.md) | The three frozen regimes + calibration rationale |
| [docs/MARS_PIPELINE.md](docs/MARS_PIPELINE.md) | Mars cross-planet pipeline design + phase history |
| [docs/data_management.md](docs/data_management.md) · [docs/DATA_STATUS.md](docs/DATA_STATUS.md) | Data categories, cleanup, per-path status |
| [docs/PIPELINE_RERUN.md](docs/PIPELINE_RERUN.md) | End-to-end regeneration order + guardrails |
| [docs/notebooks.md](docs/notebooks.md) · [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Notebook catalogue · repo/scripts/data inventory |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) · [docs/ROADMAP_AND_RISKS.md](docs/ROADMAP_AND_RISKS.md) | Package API, testing, conventions · open work + risk register |

Developer hub for contributors: [CLAUDE.md](CLAUDE.md).

## License

MIT — see [LICENSE](LICENSE).
