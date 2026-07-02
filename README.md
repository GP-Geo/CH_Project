# Channel-Head Coupling — Earth → Mars

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TopoToolbox](https://img.shields.io/badge/TopoToolbox-0.0.6-green.svg)](https://github.com/TopoToolbox/pytopotoolbox)

A reusable framework for detecting **coupled channel heads** in drainage
networks derived from DEMs (Goren & Shelef 2024,
[doi:10.5194/esurf-12-1347-2024](https://doi.org/10.5194/esurf-12-1347-2024)):
train the classifiers on Earth DEMs, then transfer them to Mars valley
networks. The pipeline is **data-agnostic** — swap in your own per-basin DEM
GeoTIFFs on the Earth side, or your own target DEM + valley-network vectors on
the Mars side, and the same stages 0–14 apply.

> **Which branch?** Active development (and the handoff state) lives on
> `refactor/package-first-architecture`; the default `main` predates the
> package-first refactor and will be fast-forwarded at the owner's discretion.

## What this does

The pipeline pairs channel heads that meet at confluences ("first-meet" pairs),
detects whether their basins are spatially **coupled** (touching), and predicts
coupling with an **XGBoost** classifier (Earth held-out test AUC ≈ 0.92) augmented
by a **5-class CNN** that reads the local network geometry. Because every feature
is **dimensionless**, the Earth-trained model transfers to Martian valley networks
without retraining (**Earth → Mars transfer learning**).

> **Which AUC?** The ≈ 0.92 figure is the **within-basin** held-out test AUC.
> Honest cross-basin generalization (leave-one-basin-out,
> `channel-heads lobo-validate`) is geometry-only **pooled AUC ≈ 0.77–0.78**
> (per-basin mean ≈ 0.70–0.74); the geom+CNN cross-basin variant
> (`per_fold_cnn`) has not been run yet.

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
```

## Verify your install in 5 minutes

1. **Import + CLI check** (no data needed):
   `python -c "from channel_heads import CouplingAnalyzer; print('OK')"`, then
   `channel-heads --help` to list all pipeline/training/figure commands.
2. **Test suite** (no data needed — fixtures are fully synthetic):
   `conda run -n ch-heads pytest -q --no-cov` (600+ tests, ~20 s).
3. **Quick start below** (~3 s) — runs against the one DEM tracked in the repo
   (`data/cropped_DEMs/Inyo_strm_crop.tif`), so it works on a fresh clone.

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

pairs, heads = first_meet_pairs_for_outlet(s, outlet=5)
results = CouplingAnalyzer(fd, s, dem, connectivity=8).evaluate_pairs_for_outlet(5, pairs)
print(results)
```

An **outlet** is a terminal node of the extracted stream network (where one
drainage tree exits the DEM), and every analysis is scoped to one outlet's tree;
enumerate and browse outlets interactively in
[`notebooks/analysis/02_earth_network_explorer.ipynb`](notebooks/analysis/02_earth_network_explorer.ipynb).
This example uses the tracked Inyo DEM, so it runs straight after `git clone`.

Batch CLI: `channel-heads analyze data/cropped_DEMs/Inyo_strm_crop.tif -o out.csv --threshold 300 -v`

## Data: what you need and where the originals came from

The framework expects two kinds of input — formats, not specific files, so you
can bring your own:

- **Earth (training):** one DEM GeoTIFF per basin / mountain range, placed in
  `data/cropped_DEMs/`.
- **Mars (target):** a DEM GeoTIFF (`data/Mars/`) plus valley-network polyline
  vectors (`data/final_valleys/`).

Provenance of the original datasets used in this project:

| Dataset | Source |
|---------|--------|
| Earth per-basin DEMs | Crops of public-domain **SRTM GL3** obtained via OpenTopography ([doi:10.5069/G9445JDF](https://doi.org/10.5069/G9445JDF)), covering the 18 mountain ranges of Goren & Shelef (2024), Table A1 |
| Mars DEM / hillshade | **NASA MOLA** (public domain); `Mars_DEM_reprojected.tif` is a derived reprojection whose exact recipe is undocumented — treat it as a frozen input |
| Mars valley network | Alemanno, Orofino & Mancarella (2018), *Global map of Martian fluvial systems*, Earth and Space Science 5, 560–577 ([doi:10.1029/2018EA000362](https://doi.org/10.1029/2018EA000362)) |

The full original data (~18 GB) is **not in the repo** — it lives with the
owner (contact via [HANDOFF.md](HANDOFF.md)). One small example DEM
(`data/cropped_DEMs/Inyo_strm_crop.tif`, 161 KB) is tracked so install
verification works on a fresh clone. `models/` **is** tracked in git (~3 MB)
with checksums + provenance in [`models/MANIFEST.md`](models/MANIFEST.md).
Derived `data/` artifacts are regenerable from these inputs
([docs/PIPELINE_RERUN.md](docs/PIPELINE_RERUN.md)); the original inputs are not.

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

**Derived** `data/` artifacts are regenerable (the original inputs are not —
see the data section above); `models/` is tracked in git with checksums in
[`models/MANIFEST.md`](models/MANIFEST.md). The full regeneration order and
guardrails are in [docs/PIPELINE_RERUN.md](docs/PIPELINE_RERUN.md).
Validate the package at any time with `conda run -n ch-heads pytest -q`.

**Wall clock.** The regime pipeline (step 2) retrains the CNNs — expect
**hours per regime on CPU**; a GPU (PyTorch) is strongly recommended. Each
step writes its log to `/tmp/regime_<regime>/<step>.log` (see
`scripts/run_regime_pipeline.sh`), so progress can be tailed while it runs.

## Repository layout

```
channel_heads/      Python package — all pipeline/model/raster/training/viz logic
  cli/              command surface (python -m channel_heads <command>)
  pipelines/        readable top layer, one function per stage (earth/mars/poster)
  features/ pairing/ rasterization/ models/ training/ eval/ io/ viz/ mars/
tests/              pytest suite (600+ tests; maps ~1:1 to package modules)
scripts/            shell orchestrators, headless diagnostics, rendering, _archive/
notebooks/          pipeline/ (canonical 00–14) + themed + archive/ (§ below)
docs/               all project documentation (see index below)
data/               inputs + generated outputs   (gitignored; one example DEM tracked)
models/             trained model artifacts        (tracked, ~3 MB; see models/MANIFEST.md)
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

Run notebooks with the `ch-heads` conda-env kernel; if Jupyter can't find it:
`conda run -n ch-heads python -m ipykernel install --user --name ch-heads`.
A good first path through the canonical tier: `pipeline/00 → 04 → 12/13`
(setup → regime calibration → Mars results).

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

## Citation & contact

Cite the software via [CITATION.cff](CITATION.cff). The underlying method is
Goren & Shelef (2024), *Earth Surface Dynamics* 12, 1347–1369
([doi:10.5194/esurf-12-1347-2024](https://doi.org/10.5194/esurf-12-1347-2024)).
Ownership, contact details, and access to the full original data are documented
in [HANDOFF.md](HANDOFF.md).

## License

MIT — see [LICENSE](LICENSE).
