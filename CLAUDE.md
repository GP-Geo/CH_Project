# Channel Head Coupling Analysis - Developer Guide

## Project Overview

This project analyzes **channel head coupling** in drainage networks derived from Digital Elevation Models (DEMs). It identifies pairs of channel heads that meet at confluences and determines whether their drainage basins are spatially coupled (touching or overlapping).

**Research Context:** Understanding channel head coupling helps analyze sediment connectivity, landscape evolution, and drainage network organization in mountainous terrain. The methodology is based on:

> Goren, L. and Shelef, E.: Channel concavity controls planform complexity of branching drainage networks, Earth Surf. Dynam., 12, 1347-1369, https://doi.org/10.5194/esurf-12-1347-2024, 2024.

## Architecture

### Core Components

```
channel-heads/
├── channel_heads/                # Python package
│   ├── __init__.py               # Package exports and metadata
│   ├── coupling_analysis.py      # Basin coupling detection
│   ├── first_meet_pairs_for_outlet.py  # Head pairing algorithm
│   ├── lengthwise_asymmetry.py   # ΔL metric (Goren & Shelef 2024)
│   ├── stream_utils.py           # Stream network utilities
│   ├── plotting_utils.py         # Visualization functions
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Path management
│   ├── basin_config.py           # Basin parameters from paper
│   └── logging_config.py         # Logging setup
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest fixtures (mock objects)
│   ├── test_coupling_analysis.py
│   └── test_first_meet_pairs.py
├── notebooks/                    # Interactive analysis notebooks
│   ├── basins_test.ipynb         # Basin-specific tests
│   ├── all_basins_analysis.ipynb # Multi-basin analysis
│   └── all_basins_analysis_full.ipynb
├── data/                         # Input DEMs and outputs
│   ├── cropped_DEMs/             # Processed DEM tiles
│   ├── outputs/                  # Analysis results (CSVs)
│   └── raw/                      # Original SRTM data
├── .github/workflows/            # CI/CD pipelines
│   └── tests.yml                 # GitHub Actions tests
├── env/                          # Conda environment spec
│   └── environment.yml
├── pyproject.toml                # Package configuration
├── README.md                     # User documentation
└── improvement.md                # Enhancement roadmap
```

## Environment Setup

### Prerequisites
- [Miniforge/Mambaforge](https://github.com/conda-forge/miniforge) or Anaconda
- Python 3.11+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/channel-heads.git
   cd channel-heads
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f env/environment.yml
   conda activate ch-heads
   ```

3. **Install package in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation**
   ```bash
   python -c "from channel_heads import CouplingAnalyzer; print('OK')"
   ch-analyze --help
   ```

### Key Dependencies
- **topotoolbox** (0.0.6+) - Core geospatial analysis
- **numpy** (2.0+) - Numerical computing
- **pandas** (2.0+) - Data manipulation
- **matplotlib** (3.8+) - Visualization
- **rasterio** (1.3+) - Raster I/O
- **geopandas** (1.0+) - Spatial data handling
- **scikit-image** (0.24+) - Image processing
- **pytest** (8.0+) - Testing framework

## Usage

### Command-Line Interface

The `ch-analyze` CLI enables batch processing:

```bash
# Basic usage
ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif -o results/inyo_coupling.csv

# With options
ch-analyze dem.tif -o results.csv \
    --threshold 500 \
    --mask-below 1200 \
    --connectivity 8 \
    --verbose

# Analyze specific outlets
ch-analyze dem.tif -o results.csv --outlets 5,12,18

# Batch process all DEMs
for dem in data/cropped_DEMs/*.tif; do
    name=$(basename "$dem" _strm_crop.tif)
    ch-analyze "$dem" --output "results/${name}_coupling.csv" -v
done
```

**CLI Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `dem` | Path to DEM file (GeoTIFF) | Required |
| `-o, --output` | Output CSV path | Required |
| `--threshold` | Stream network area threshold | 300 |
| `--connectivity` | Coupling detection connectivity (4 or 8) | 8 |
| `--mask-below` | Mask elevations below threshold | None |
| `--outlets` | Comma-separated outlet IDs | All |
| `-v, --verbose` | Print detailed progress | False |

### Python API

```python
import numpy as np
import topotoolbox as tt3
from channel_heads import (
    CouplingAnalyzer,
    first_meet_pairs_for_outlet,
    LengthwiseAsymmetryAnalyzer,
    get_z_th,
    get_basin_config,
    EXAMPLE_DEMS,
)

# Load DEM using config paths
dem = tt3.read_tif(str(EXAMPLE_DEMS["inyo"]))

# Apply elevation threshold from basin config
z_th = get_z_th("inyo")  # Returns 1200
dem.z[dem.z < z_th] = np.nan

# Derive flow and stream networks
fd = tt3.FlowObject(dem)
s = tt3.StreamObject(fd, threshold=300)

# Analyze a specific outlet
outlet_id = 5
pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)

# Compute coupling metrics
analyzer = CouplingAnalyzer(fd, s, dem, connectivity=8)
coupling_df = analyzer.evaluate_pairs_for_outlet(outlet_id, pairs)

# Compute lengthwise asymmetry (ΔL)
config = get_basin_config("inyo")
asym_analyzer = LengthwiseAsymmetryAnalyzer(s, dem, lat=config["lat"])
asym_df = asym_analyzer.evaluate_pairs_for_outlet(outlet_id, pairs)

# Merge results
from channel_heads import merge_coupling_and_asymmetry
combined_df = merge_coupling_and_asymmetry(coupling_df, asym_df)

print(combined_df)
```

### Interactive Analysis (Notebooks)

```bash
conda activate ch-heads
jupyter lab
# Open notebooks/all_basins_analysis.ipynb
```

## Module Reference

### `coupling_analysis.py`

**CouplingAnalyzer** - Detects spatial coupling between channel head drainage basins

```python
CouplingAnalyzer(fd, s, dem, connectivity=8)
```

**Methods:**
- `influence_grid(head_id)` - Returns GridObject mask for a channel head's basin
- `influence_mask(head_id)` - Returns numpy boolean mask (cached)
- `pair_touching(h1, h2)` - Tests if two basins touch (returns PairTouchResult)
- `evaluate_pairs_for_outlet(outlet, pairs_at_confluence)` - Returns DataFrame with coupling metrics
- `clear_cache()` - Clears the mask cache (call between outlets)
- `cache_size` - Property returning current cache size

**Output DataFrame columns:**
- `outlet`, `confluence`, `head_1`, `head_2`
- `touching` (bool), `overlap_px`, `contact_px`, `size1_px`, `size2_px`

### `first_meet_pairs_for_outlet.py`

**first_meet_pairs_for_outlet(s, outlet)** - Identifies channel head pairs

Computes which channel heads first meet at each confluence for a given outlet's drainage basin.

**Returns:**
- `pairs_at_confluence`: Dict[int, Set[Tuple[int, int]]] - {confluence_id: set of (head1, head2) pairs}
- `basin_heads`: List[int] - All channel head node IDs in the basin

**Algorithm:** Uses memoized upstream traversal to track head sets at each node, emitting pairs when branches merge at confluences.

### `lengthwise_asymmetry.py`

**LengthwiseAsymmetryAnalyzer** - Computes ΔL metric from Goren & Shelef (2024)

```python
LengthwiseAsymmetryAnalyzer(s, dem, lat=36.71)
```

The lengthwise asymmetry quantifies the difference in flow path lengths:
```
ΔL = 2|L_ij - L_ji| / (L_ij + L_ji)
```

Values range from 0 (symmetric) to 2 (maximum asymmetry).

**Methods:**
- `compute_pair_asymmetry(head_1, head_2, confluence)` - Returns PairAsymmetryResult
- `evaluate_pairs_for_outlet(outlet, pairs_at_confluence)` - Returns DataFrame

**Helper functions:**
- `compute_delta_L(L_ij, L_ji)` - Core ΔL formula
- `compute_asymmetry_statistics(delta_L_values)` - Summary statistics
- `merge_coupling_and_asymmetry(coupling_df, asymmetry_df)` - Merge results
- `compute_meters_per_degree(lat_deg)` - Coordinate conversion
- `compute_pixel_size_meters(lat_deg, cellsize_deg)` - Pixel size in meters

### `basin_config.py`

**Basin configuration data from Goren & Shelef (2024) Table A1**

Contains parameters for 18 elongated mountain ranges:
- Elevation thresholds (`z_th`)
- Reference ΔL values (median, 25th, 75th percentiles)
- Concavity index (θ)
- Location coordinates

**Functions:**
- `get_basin_config(basin_name)` - Get all parameters for a basin
- `get_z_th(basin_name)` - Get elevation threshold
- `get_reference_delta_L(basin_name)` - Get reference ΔL values
- `list_basins()` - List all available basins

**Available basins:** taiwan, clanalpine, daqing, finisterre, humboldt, inyo, kammanassie, luliang, panamint, sakhalin, vallefertil, sierramadre, sierranevada_spain, piedepalo, toano, troodos, tsugaru, yoro

### `config.py`

**Path management** - Centralized path configuration

```python
from channel_heads import (
    PROJECT_ROOT,
    DATA_DIR,
    CROPPED_DEMS_DIR,
    OUTPUTS_DIR,
    EXAMPLE_DEMS,
    get_output_dir,
    list_available_dems,
    resolve_dem_path,
)

# Use predefined paths
dem_path = EXAMPLE_DEMS["inyo"]

# Get output directory (creates if needed)
output_dir = get_output_dir("inyo")

# Resolve flexible DEM references
path = resolve_dem_path("inyo")  # Friendly name
path = resolve_dem_path("data/cropped_DEMs/custom.tif")  # Relative path
```

**Environment variables:**
- `CHANNEL_HEADS_ROOT` - Override project root
- `CHANNEL_HEADS_DATA` - Override data directory

### `logging_config.py`

**Logging setup** - Centralized logging configuration

```python
from channel_heads import get_logger, setup_logging
import logging

# Get module-specific logger
logger = get_logger(__name__)
logger.info("Processing outlet %d", outlet_id)

# Configure logging level
setup_logging(level=logging.DEBUG, console=True)
```

**Environment variables:**
- `CHANNEL_HEADS_LOG_LEVEL` - Set level (DEBUG, INFO, WARNING, ERROR)
- `CHANNEL_HEADS_LOG_FILE` - Log to file

### `plotting_utils.py`

**Visualization functions** with unified view modes:

- `view_mode="crop"` - Tight crop around subgraph (default)
- `view_mode="zoom"` - Full DEM, zoom to subgraph
- `view_mode="overview"` - Full DEM and network, no zoom

**Functions:**
- `plot_coupled_pair(fd, s, dem, confluence_id, head_i, head_j, ...)` - Single pair visualization
- `plot_outlet_view(s, outlet_id, dem, ...)` - Outlet subgraph overview
- `plot_all_coupled_pairs_for_outlet(fd, s, dem, an, df_touching, outlet_id, ...)` - 2D multi-pair view
- `plot_all_coupled_pairs_for_outlet_3d(...)` - 3D perspective with DEM surface

### `stream_utils.py`

**outlet_node_ids_from_streampoi(s)** - Extract outlet node IDs from StreamObject

Returns numpy array of outlet node indices using `s.streampoi('outlets')`.

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=channel_heads --cov-report=html

# Run specific test file
pytest tests/test_coupling_analysis.py -v

# Run specific test class
pytest tests/test_coupling_analysis.py::TestCouplingAnalyzerInit -v
```

### Test Fixtures

The test suite uses mock objects defined in `tests/conftest.py`:

- **MockGridObject** - Mock TopoToolbox GridObject
- **MockFlowObject** - Mock FlowObject with `dependencemap()`
- **MockStreamObject** - Mock StreamObject with network topology

**Available fixtures:**
- `simple_y_network` - Y-shaped network (2 heads, 1 confluence)
- `complex_network` - Multiple confluences (4 heads, 3 confluences)
- `touching_basins_network` - Network with touching drainage basins

### Adding Tests

```python
# tests/test_new_feature.py
import pytest
from channel_heads.new_module import new_function

class TestNewFeature:
    def test_basic_case(self, simple_y_network):
        """Test with simple network fixture."""
        net = simple_y_network
        result = new_function(net["s"], net["dem"])
        assert result is not None

    def test_edge_case(self):
        """Test edge case behavior."""
        with pytest.raises(ValueError):
            new_function(None, None)
```

## Continuous Integration

GitHub Actions runs on every push/PR to `main`:

1. **Tests** - pytest on Python 3.11 and 3.12
2. **Lint** - black formatting and ruff linting
3. **Type check** - mypy (informational)

See `.github/workflows/tests.yml` for configuration.

## Data

### Input Data
- **DEMs**: GeoTIFF rasters in `data/cropped_DEMs/`
- **Study areas**: Inyo, Humboldt, CalnAlpine, Daqing, Luliang, Kammanasie, Finisterre
- **Format**: SRTM-derived elevation models (meters)
- **Resolution**: 1 arc-second (~30m)

### Output Data
- **Location**: `data/outputs/<study_area>/`
- **Format**: CSV files with coupling and asymmetry metrics
- **Columns**: outlet, confluence, head_1, head_2, touching, overlap_px, contact_px, size1_px, size2_px, L_1, L_2, delta_L

## Best Practices

### Code Organization
1. **Keep notebooks clean**: Move reusable logic to `channel_heads/` modules
2. **Use config paths**: Import from `channel_heads.config` instead of hardcoding
3. **Document functions**: Include docstrings with parameters and return types
4. **Cache strategically**: Call `clear_cache()` between outlets to manage memory

### Performance Tips
1. **Clear cache**: Call `analyzer.clear_cache()` between outlets
2. **Limit DEM size**: Crop DEMs to study area before analysis
3. **Adjust threshold**: Higher stream threshold → fewer heads → faster computation
4. **Use connectivity=4**: Faster than connectivity=8 if diagonal contact isn't needed

### Visualization Guidelines
1. **Start with `view_mode="crop"`** for detailed pair analysis
2. **Use `view_mode="overview"`** for network context
3. **Limit pairs**: Set `max_pairs=10` for large outlets to avoid clutter
4. **3D plots**: Use `dem_stride=2` to reduce surface complexity for faster rendering

## Common Workflows

### Analyze a New Study Area

```python
import numpy as np
import topotoolbox as tt3
from channel_heads import (
    CouplingAnalyzer,
    LengthwiseAsymmetryAnalyzer,
    first_meet_pairs_for_outlet,
    outlet_node_ids_from_streampoi,
    merge_coupling_and_asymmetry,
    get_output_dir,
)

# 1. Load DEM
dem = tt3.read_tif("data/cropped_DEMs/NewArea_strm_crop.tif")
dem.z[dem.z < threshold_elevation] = np.nan

# 2. Derive networks
fd = tt3.FlowObject(dem)
s = tt3.StreamObject(fd, threshold=300)

# 3. Process all outlets
coupling_an = CouplingAnalyzer(fd, s, dem)
asym_an = LengthwiseAsymmetryAnalyzer(s, dem, lat=your_latitude)
outs = outlet_node_ids_from_streampoi(s)

all_results = []
for outlet_id in outs:
    pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)
    coupling_df = coupling_an.evaluate_pairs_for_outlet(outlet_id, pairs)
    asym_df = asym_an.evaluate_pairs_for_outlet(outlet_id, pairs)
    combined = merge_coupling_and_asymmetry(coupling_df, asym_df)
    all_results.append(combined)
    coupling_an.clear_cache()  # Prevent memory growth

# 4. Combine and save
import pandas as pd
df_all = pd.concat(all_results, ignore_index=True)
output_dir = get_output_dir("NewArea")
df_all.to_csv(output_dir / "coupling_results.csv", index=False)
```

### Debug a Specific Confluence

```python
from channel_heads.plotting_utils import plot_coupled_pair

# Visualize single pair
confluence_id = 831
head_1, head_2 = 662, 716

plot_coupled_pair(
    fd, s, dem,
    confluence_id, head_1, head_2,
    view_mode="crop",
    focus="masks"  # Show full drainage basins
)
```

### Compare with Reference Values

```python
from channel_heads import (
    get_reference_delta_L,
    compute_asymmetry_statistics,
)

# Get reference values from paper
ref = get_reference_delta_L("inyo")
print(f"Reference ΔL: {ref['median']:.2f} ({ref['p25']:.2f} - {ref['p75']:.2f})")

# Compute your statistics
your_stats = compute_asymmetry_statistics(df_all["delta_L"])
print(f"Your ΔL: {your_stats['median']:.2f} ({your_stats['p25']:.2f} - {your_stats['p75']:.2f})")
```

### Export Results for GIS

```python
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Add spatial coordinates to results
r, c = s.node_indices
xs, ys = s.transform * np.vstack((c, r))

df_all['conf_x'] = xs[df_all['confluence']]
df_all['conf_y'] = ys[df_all['confluence']]
df_all['head1_x'] = xs[df_all['head_1']]
df_all['head1_y'] = ys[df_all['head_1']]

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_all,
    geometry=[Point(xy) for xy in zip(df_all.conf_x, df_all.conf_y)],
    crs=dem.crs
)
gdf.to_file("outputs/coupling_results.gpkg", driver="GPKG")
```

## Troubleshooting

### Common Issues

**"DEM shape mismatch"**
- Ensure DEM and FlowObject have identical dimensions
- Check: `dem.z.shape == fd.shape`

**"Node index out of range"**
- Outlet or confluence ID may be from different StreamObject
- Verify IDs with `s.streampoi('outlets')` or `s.streampoi('confluences')`

**"Empty DataFrame returned"**
- Outlet may have no confluences (single-head basin)
- Check: `len(pairs_at_confluence) > 0`

**"Basin not found" from get_basin_config()**
- Use `list_basins()` to see available basins
- Check spelling: "calnalpine" vs "clanalpine"

**Slow performance on large DEMs**
- Crop DEM to region of interest
- Increase stream threshold to reduce network complexity
- Call `clear_cache()` between outlets

**Hardcoded paths fail**
- Use `channel_heads.config` paths:
  ```python
  from channel_heads import EXAMPLE_DEMS, resolve_dem_path
  dem_path = resolve_dem_path("inyo")
  ```

## Development

### Adding New Analysis Functions

1. **Create module in `channel_heads/`**
   ```python
   # channel_heads/new_analysis.py
   """New analysis module."""

   from typing import Dict
   import pandas as pd

   def my_analysis_function(fd, s, dem, **kwargs) -> pd.DataFrame:
       """Brief description.

       Parameters
       ----------
       fd : FlowObject
           Flow direction object
       s : StreamObject
           Stream network
       dem : GridObject
           Digital elevation model

       Returns
       -------
       pd.DataFrame
           Analysis results
       """
       # Implementation
       pass
   ```

2. **Export from `__init__.py`**
   ```python
   from .new_analysis import my_analysis_function
   __all__.append("my_analysis_function")
   ```

3. **Add tests in `tests/`**
   ```python
   # tests/test_new_analysis.py
   def test_my_analysis(simple_y_network):
       from channel_heads.new_analysis import my_analysis_function
       net = simple_y_network
       result = my_analysis_function(net["fd"], net["s"], net["dem"])
       assert not result.empty
   ```

4. **Document in this guide**

### Code Style

- Follow PEP 8
- Use type hints (enforced by mypy in CI)
- Prefer numpy/pandas vectorized operations over loops
- Add docstrings to public functions (NumPy style)
- Format with `black` before committing
- Lint with `ruff`

```bash
# Format code
black channel_heads/ tests/

# Check linting
ruff check channel_heads/ tests/

# Fix auto-fixable issues
ruff check --fix channel_heads/ tests/
```

## Resources

- **TopoToolbox Python**: https://github.com/TopoToolbox/pytopotoolbox
- **Goren & Shelef (2024)**: https://doi.org/10.5194/esurf-12-1347-2024
- **Rasterio docs**: https://rasterio.readthedocs.io/
- **GeoPandas**: https://geopandas.org/

## Support

For issues or questions:
1. Check this guide and [improvement.md](improvement.md)
2. Review existing notebook outputs
3. Inspect intermediate results (masks, pairs) with visualizations
4. Consult TopoToolbox documentation for stream network concepts

## Version Information

- **Package name**: channel-heads
- **Version**: 0.1.0
- **Environment name**: ch-heads
- **Python version**: 3.11+
- **TopoToolbox version**: 0.0.6+
- **Last updated**: 2026-02-01
