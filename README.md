# Channel Head Coupling Analysis

Automated detection and analysis of coupled channel heads in drainage networks derived from Digital Elevation Models (DEMs).

## Overview

This project identifies pairs of channel heads that converge at confluences and determines whether their drainage basins are spatially coupled (touching or overlapping). This analysis helps understand:

- Sediment connectivity in mountainous catchments
- Landscape evolution patterns
- Drainage network reorganization
- Channel head migration dynamics

## Features

- Automated drainage network extraction from DEMs using TopoToolbox
- Graph-based channel head pairing algorithm
- Spatial coupling detection (4/8-connectivity)
- Statistical summaries and CSV exports
- 2D and 3D visualization tools

## Quick Start

### Installation

```bash
# Create conda environment
conda env create -f env/environment.yml
conda activate ch-heads
```

### Basic Usage

```python
import topotoolbox as tt3
from src.first_meet_pairs_for_outlet import first_meet_pairs_for_outlet
from src.coupling_analysis import CouplingAnalyzer

# Load DEM
dem = tt3.read_tif("data/cropped_DEMs/Inyo_strm_crop.tif")

# Derive flow and stream networks
fd = tt3.FlowObject(dem)
s = tt3.StreamObject(fd, threshold=300)

# Analyze outlet
outlet_id = 5
pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)

analyzer = CouplingAnalyzer(fd, s, dem)
results = analyzer.evaluate_pairs_for_outlet(outlet_id, pairs)

print(results)
```

### Interactive Analysis

```bash
jupyter lab
# Open notebooks/00_test.ipynb
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Complete developer guide with API reference and workflows
- [improvement.md](improvement.md) - Roadmap and enhancement suggestions

## Study Areas

Included DEMs cover diverse geomorphic settings:
- Inyo Mountains, California
- Humboldt Range, Nevada
- Finisterre Range, Papua New Guinea
- Daqing Shan, China
- Luliang Mountains, China
- Kammanassie Mountains, South Africa

## Project Structure

```
channel-heads/
├── src/                    # Core analysis modules
│   ├── coupling_analysis.py
│   ├── first_meet_pairs_for_outlet.py
│   ├── stream_utils.py
│   └── plotting_utils.py
├── notebooks/              # Interactive Jupyter notebooks
├── data/                   # DEMs and outputs
├── env/                    # Conda environment specification
├── CLAUDE.md              # Developer guide
└── improvement.md         # Enhancement roadmap
```

## Requirements

- Python 3.12
- TopoToolbox 0.0.6
- See [env/environment.yml](env/environment.yml) for complete dependencies

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! See [improvement.md](improvement.md) for enhancement opportunities.
