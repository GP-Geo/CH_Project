# Channel Head Coupling Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TopoToolbox](https://img.shields.io/badge/TopoToolbox-0.0.6-green.svg)](https://github.com/TopoToolbox/pytopotoolbox)
[![Tests](https://github.com/yourusername/channel-heads/workflows/Tests/badge.svg)](https://github.com/yourusername/channel-heads/actions)

Automated detection and analysis of coupled channel heads in drainage networks derived from Digital Elevation Models (DEMs).

## Overview

This package identifies pairs of channel heads that converge at confluences and determines whether their drainage basins are spatially coupled (touching or overlapping). This analysis is fundamental for understanding:

- **Sediment connectivity** in mountainous catchments
- **Landscape evolution** patterns
- **Drainage network reorganization**
- **Channel head migration** dynamics

Based on the methodology from:
> Goren, L. and Shelef, E.: Channel concavity controls planform complexity of branching drainage networks, Earth Surf. Dynam., 12, 1347-1369, https://doi.org/10.5194/esurf-12-1347-2024, 2024.

## Features

- Automated drainage network extraction from DEMs using TopoToolbox
- Graph-based channel head pairing algorithm
- Spatial coupling detection (4/8-connectivity)
- Lengthwise asymmetry (ΔL) metric computation
- Basin configuration data from 18 mountain ranges (Goren & Shelef 2024)
- Statistical summaries and CSV exports
- 2D and 3D visualization tools
- Command-line interface for batch processing
- Comprehensive test suite with pytest

## Installation

### Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/channel-heads.git
cd channel-heads

# Create conda environment
conda env create -f env/environment.yml
conda activate ch-heads

# Install package in development mode
pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e ".[all]"
```

### Verify Installation

```bash
# Test import
python -c "from channel_heads import CouplingAnalyzer; print('OK')"

# Test CLI
ch-analyze --help
```

## Quick Start

### Python API

```python
import numpy as np
import topotoolbox as tt3
from channel_heads import (
    CouplingAnalyzer,
    first_meet_pairs_for_outlet,
    get_z_th,
    EXAMPLE_DEMS,
)

# Load DEM
dem = tt3.read_tif(str(EXAMPLE_DEMS["inyo"]))

# Apply elevation threshold
z_th = get_z_th("inyo")  # Returns 1200 m
dem.z[dem.z < z_th] = np.nan

# Derive flow and stream networks
fd = tt3.FlowObject(dem)
s = tt3.StreamObject(fd, threshold=300)

# Analyze a specific outlet
outlet_id = 5
pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)

# Compute coupling metrics
analyzer = CouplingAnalyzer(fd, s, dem, connectivity=8)
results = analyzer.evaluate_pairs_for_outlet(outlet_id, pairs)

print(results)
```

### Command-Line Interface

The `ch-analyze` CLI enables batch processing of DEMs.

#### Basic Usage

```bash
# Analyze a single DEM
ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif -o results/inyo_coupling.csv

# With verbose output
ch-analyze dem.tif -o results.csv --verbose
```

#### Full Options

```bash
ch-analyze dem.tif -o results.csv \
    --threshold 500 \           # Stream network threshold (pixels)
    --mask-below 1200 \         # Mask elevations below 1200m
    --connectivity 8 \          # 4 or 8 neighbor connectivity
    --outlets 5,12,18 \         # Analyze specific outlet IDs
    --verbose                   # Show detailed progress
```

#### CLI Options Reference

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `dem` | | Path to DEM file (GeoTIFF) | Required |
| `--output` | `-o` | Output CSV path | Required |
| `--threshold` | | Stream network area threshold (pixels) | 300 |
| `--connectivity` | | Neighbor connectivity: 4 or 8 | 8 |
| `--mask-below` | | Mask DEM elevations below this value (meters) | None |
| `--outlets` | | Comma-separated outlet node IDs to analyze | All |
| `--verbose` | `-v` | Print detailed progress information | False |

#### Batch Processing Examples

```bash
# Process all DEMs in a directory
for dem in data/cropped_DEMs/*.tif; do
    name=$(basename "$dem" _strm_crop.tif)
    ch-analyze "$dem" -o "results/${name}_coupling.csv" -v
done

# Process with basin-specific elevation thresholds
ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif \
    -o results/inyo.csv \
    --mask-below 1200 \
    --threshold 300

ch-analyze data/cropped_DEMs/Humboldt_strm_crop.tif \
    -o results/humboldt.csv \
    --mask-below 1450 \
    --threshold 300
```

#### Output Format

The CLI produces a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `outlet` | int | Outlet node ID |
| `confluence` | int | Confluence node ID |
| `head_1` | int | First channel head node ID |
| `head_2` | int | Second channel head node ID |
| `touching` | bool | Whether basins are spatially coupled |
| `overlap_px` | int | Number of overlapping pixels |
| `contact_px` | int | Number of contact boundary pixels |
| `size1_px` | int | Basin 1 size in pixels |
| `size2_px` | int | Basin 2 size in pixels |

### Interactive Analysis

```bash
conda activate ch-heads
jupyter lab
# Open notebooks/all_basins_analysis.ipynb
```

## API Reference

### CouplingAnalyzer

Detects spatial coupling between channel head drainage basins.

```python
from channel_heads import CouplingAnalyzer

analyzer = CouplingAnalyzer(fd, s, dem, connectivity=8)

# Get influence mask for a channel head
mask = analyzer.influence_mask(head_id=662)

# Test if two basins touch
result = analyzer.pair_touching(head_1=662, head_2=716)
print(f"Touching: {result.touching}, Overlap: {result.overlap_px}px")

# Evaluate all pairs at confluences for an outlet
df = analyzer.evaluate_pairs_for_outlet(outlet_id, pairs_at_confluence)

# Clear cache between outlets to manage memory
analyzer.clear_cache()
```

### first_meet_pairs_for_outlet

Identifies channel head pairs that first meet at each confluence.

```python
from channel_heads import first_meet_pairs_for_outlet

pairs_at_confluence, basin_heads = first_meet_pairs_for_outlet(s, outlet_id)

# pairs_at_confluence: Dict[int, Set[Tuple[int, int]]]
#   {confluence_id: {(head1, head2), ...}}

# basin_heads: List[int]
#   All channel head node IDs in the basin
```

### LengthwiseAsymmetryAnalyzer

Computes the lengthwise asymmetry (ΔL) metric from Goren & Shelef (2024).

```python
from channel_heads import LengthwiseAsymmetryAnalyzer, get_basin_config

config = get_basin_config("inyo")
analyzer = LengthwiseAsymmetryAnalyzer(s, dem, lat=config["lat"])

# Compute for all pairs
asym_df = analyzer.evaluate_pairs_for_outlet(outlet_id, pairs)

# Merge with coupling results
from channel_heads import merge_coupling_and_asymmetry
combined = merge_coupling_and_asymmetry(coupling_df, asym_df)
```

### Basin Configuration

Access reference data from 18 mountain ranges analyzed in Goren & Shelef (2024).

```python
from channel_heads import get_z_th, get_basin_config, get_reference_delta_L, list_basins

# Get elevation threshold for masking
z_th = get_z_th("inyo")  # Returns 1200

# Get full configuration
config = get_basin_config("inyo")
print(f"Latitude: {config['lat']}, θ: {config['theta']}")

# Get reference ΔL values for comparison
ref = get_reference_delta_L("inyo")
print(f"Reference ΔL: {ref['median']:.2f} ({ref['p25']:.2f} - {ref['p75']:.2f})")

# List all available basins
basins = list_basins()
print(basins)
```

### Visualization Functions

```python
from channel_heads.plotting_utils import (
    plot_coupled_pair,
    plot_outlet_view,
    plot_all_coupled_pairs_for_outlet,
    plot_all_coupled_pairs_for_outlet_3d
)

# Single pair visualization
plot_coupled_pair(fd, s, dem, confluence_id, head_1, head_2, view_mode="crop")

# Overview of outlet subgraph
plot_outlet_view(s, outlet_id, dem, view_mode="overview")

# All touching pairs for an outlet (2D)
plot_all_coupled_pairs_for_outlet(fd, s, dem, analyzer, df_touching, outlet_id)

# 3D visualization with DEM surface
plot_all_coupled_pairs_for_outlet_3d(fd, s, dem, analyzer, df_touching, outlet_id)
```

**View modes:**
- `"crop"` - Tight crop around subgraph (default)
- `"zoom"` - Full DEM, zoomed to subgraph
- `"overview"` - Full DEM and network, no zoom

## Project Structure

```
channel-heads/
├── channel_heads/          # Python package
│   ├── __init__.py         # Package exports
│   ├── coupling_analysis.py    # Basin coupling detection
│   ├── first_meet_pairs_for_outlet.py  # Head pairing algorithm
│   ├── lengthwise_asymmetry.py # ΔL metric computation
│   ├── stream_utils.py     # Stream network utilities
│   ├── plotting_utils.py   # Visualization functions
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Path management
│   ├── basin_config.py     # Basin parameters from paper
│   └── logging_config.py   # Logging configuration
├── tests/                  # Test suite
│   ├── conftest.py         # Pytest fixtures
│   ├── test_coupling_analysis.py
│   └── test_first_meet_pairs.py
├── notebooks/              # Interactive Jupyter notebooks
├── data/                   # DEMs and outputs
│   ├── cropped_DEMs/       # Study area DEMs
│   └── outputs/            # Analysis results
├── .github/workflows/      # CI/CD pipelines
├── env/                    # Conda environment specification
├── CLAUDE.md               # Developer guide
└── improvement.md          # Enhancement roadmap
```

## Study Areas

Included DEMs cover diverse geomorphic settings:

| Region | Location | z_th (m) | Notes |
|--------|----------|----------|-------|
| Inyo Mountains | California, USA | 1200 | Semi-arid, high relief |
| Humboldt Range | Nevada, USA | 1450 | Basin and Range |
| Clan Alpine | Nevada, USA | 1700 | Basin and Range |
| Daqing Shan | China | 1200 | Loess plateau margin |
| Luliang Mountains | China | 1100 | Continental climate |
| Kammanassie | South Africa | 630 | Cape Fold Belt |
| Finisterre Range | Papua New Guinea | 400 | Tropical, high uplift |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=channel_heads --cov-report=html

# Run specific test file
pytest tests/test_coupling_analysis.py -v
```

## Requirements

- Python 3.11+
- TopoToolbox 0.0.6+
- See [env/environment.yml](env/environment.yml) for complete dependencies

## Documentation

- [Developer Guide (CLAUDE.md)](CLAUDE.md) - Setup, API reference, workflows
- [Improvements (improvement.md)](improvement.md) - Roadmap and enhancement suggestions

## Citation

If you use this code in research, please cite:

```bibtex
@software{channel_heads,
  author = {Pinkas, Guy},
  title = {Channel Head Coupling Analysis},
  year = {2024},
  url = {https://github.com/yourusername/channel-heads}
}
```

And the underlying methodology:

```bibtex
@article{goren2024channel,
  author = {Goren, L. and Shelef, E.},
  title = {Channel concavity controls planform complexity of branching drainage networks},
  journal = {Earth Surface Dynamics},
  volume = {12},
  pages = {1347--1369},
  year = {2024},
  doi = {10.5194/esurf-12-1347-2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Run tests (`pytest tests/ -v`)
5. Format code (`black channel_heads/ tests/`)
6. Submit a pull request

See [improvement.md](improvement.md) for enhancement opportunities.
