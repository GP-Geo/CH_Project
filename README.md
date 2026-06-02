# Channel Head Coupling Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TopoToolbox](https://img.shields.io/badge/TopoToolbox-0.0.6-green.svg)](https://github.com/TopoToolbox/pytopotoolbox)

Automated detection, analysis, and ML-based classification of **coupled channel
heads** in drainage networks derived from DEMs — trained on Earth, applied to
Mars valley networks. Based on Goren & Shelef (2024,
[doi:10.5194/esurf-12-1347-2024](https://doi.org/10.5194/esurf-12-1347-2024)).

The pipeline pairs channel heads that meet at confluences, detects whether their
basins are spatially coupled (touching), and predicts coupling with an XGBoost
classifier (Earth held-out test AUC ≈ 0.92) plus a 5-class CNN. Because the
features are **dimensionless**, the Earth-trained model transfers to Martian
valley networks without retraining (Earth→Mars transfer learning).

The project is **package-first**: all logic lives in `channel_heads/` and runs
through `channel_heads.pipelines`; notebooks and `scripts/cli/` are thin layers
on top. See [docs/architecture.md](docs/architecture.md).

## Install

```bash
conda env create -f env/environment.yml
conda activate ch-heads
pip install -e ".[dev,geo,viz,cnn,ml]"
python -c "from channel_heads import CouplingAnalyzer; print('OK')"
ch-analyze --help
```

## Quick start

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

Batch CLI: `ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif -o out.csv --threshold 300 -v`

## Run the Mars pipeline

```python
from channel_heads import pipelines
pipelines.run_full_mars_pipeline()   # Phase 1 -> 6C; outputs under data/Mars/
```
…or per stage: `python scripts/cli/run_mars_pipeline.py --stage topology`.
Outputs land under `data/Mars/` and `data/results/` (all regenerable; see
[docs/data_management.md](docs/data_management.md)).

## Documentation

All project documentation lives in [`docs/`](docs/):

| Doc | Contents |
|-----|----------|
| [docs/architecture.md](docs/architecture.md) | **Start here** — package-first layout + public API |
| [docs/pipeline.md](docs/pipeline.md) | Stage-by-stage flow, inputs/outputs, dependency graph |
| [docs/modeling.md](docs/modeling.md) | Model variants + the Mars operating-threshold issue |
| [docs/data_management.md](docs/data_management.md) | Data categories, safe cleanup, regeneration |
| [docs/notebooks.md](docs/notebooks.md) | Notebook catalogue + canonical notebooks |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Package API, testing, conventions |
| [docs/MARS_PIPELINE.md](docs/MARS_PIPELINE.md) · [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) · [ROADMAP_AND_RISKS.md](docs/ROADMAP_AND_RISKS.md) | Phase history, inventory, risk register |

## License

MIT — see [LICENSE](LICENSE).
