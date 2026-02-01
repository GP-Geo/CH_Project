# Channel Head Coupling Analysis - Developer Guide

## Project Overview

This project analyzes **channel head coupling** in drainage networks derived from Digital Elevation Models (DEMs). It identifies pairs of channel heads that meet at confluences and determines whether their drainage basins are spatially coupled (touching or overlapping).

**Research Context:** Understanding channel head coupling helps analyze sediment connectivity, landscape evolution, and drainage network organization in mountainous terrain.

## Architecture

### Core Components

```
channel-heads/
├── src/                          # Core analysis modules
│   ├── coupling_analysis.py      # Basin coupling detection
│   ├── first_meet_pairs_for_outlet.py  # Head pairing algorithm
│   ├── stream_utils.py           # Stream network utilities
│   └── plotting_utils.py         # Visualization functions
├── notebooks/                    # Interactive analysis notebooks
│   ├── 00_test.ipynb            # Main analysis workflow
│   └── basins_test.ipynb        # Basin-specific tests
├── data/                         # Input DEMs and outputs
│   ├── cropped_DEMs/            # Processed DEM tiles
│   ├── outputs/                 # Analysis results (CSVs)
│   └── raw/                     # Original SRTM data
└── env/                          # Conda environment spec
    └── environment.yml
```

## Environment Setup

### Prerequisites
- [Miniforge/Mambaforge](https://github.com/conda-forge/miniforge) or Anaconda
- Python 3.12

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/channel-heads
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f env/environment.yml
   conda activate ch-heads
   ```

3. **Verify installation**
   ```bash
   python -c "import topotoolbox as tt3; print(tt3.__version__)"
   ```

### Key Dependencies
- **topotoolbox** (0.0.6) - Core geospatial analysis
- **numpy** (2.3.3) - Numerical computing
- **pandas** (2.3.2) - Data manipulation
- **matplotlib** (3.10.6) - Visualization
- **rasterio** (1.4.3) - Raster I/O
- **geopandas** (1.1.1) - Spatial data handling
- **scikit-image** (0.25.2) - Image processing
- **jupyterlab** (4.4.7) - Interactive notebooks

## Usage

### Running Analyses

The primary workflow is through Jupyter notebooks:

```bash
# Activate environment
conda activate ch-heads

# Launch JupyterLab
jupyter lab

# Open notebooks/00_test.ipynb
```

### Main Workflow (00_test.ipynb)

1. **Load DEM**
   ```python
   import topotoolbox as tt3
   dem = tt3.read_tif("data/cropped_DEMs/Inyo_strm_crop.tif")
   dem.z[dem.z < 1200] = np.nan  # Mask low elevations
   ```

2. **Derive Flow and Stream Networks**
   ```python
   fd = tt3.FlowObject(dem)
   s = tt3.StreamObject(fd, threshold=300)
   ```

3. **Compute Channel Head Pairs**
   ```python
   from src.first_meet_pairs_for_outlet import first_meet_pairs_for_outlet

   outlet_id = 5
   pairs_at_confluence, basin_heads = first_meet_pairs_for_outlet(s, outlet_id)
   ```

4. **Evaluate Coupling**
   ```python
   from src.coupling_analysis import CouplingAnalyzer

   an = CouplingAnalyzer(fd, s, dem, connectivity=8)
   df = an.evaluate_pairs_for_outlet(outlet_id, pairs_at_confluence)
   ```

5. **Visualize Results**
   ```python
   from src.plotting_utils import plot_all_coupled_pairs_for_outlet

   plot_all_coupled_pairs_for_outlet(fd, s, dem, an, df, outlet_id)
   ```

### Python Module Usage

The `src/` modules can be imported directly in Python scripts:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))  # Add project root to path

from src.coupling_analysis import CouplingAnalyzer
from src.first_meet_pairs_for_outlet import first_meet_pairs_for_outlet
from src.stream_utils import outlet_node_ids_from_streampoi
from src.plotting_utils import (
    plot_coupled_pair,
    plot_outlet_view,
    plot_all_coupled_pairs_for_outlet,
    plot_all_coupled_pairs_for_outlet_3d
)
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

## Data

### Input Data
- **DEMs**: GeoTIFF rasters in `data/cropped_DEMs/`
- **Study areas**: Inyo, Humboldt, CalnAlpine, Daqing, Luliang, Kammanasie, Finisterre
- **Format**: SRTM-derived elevation models (meters)

### Output Data
- **Location**: `data/outputs/<study_area>/`
- **Format**: CSV files with coupling metrics
- **Example**: `coupling_touching.csv` contains all touching pairs with spatial metrics

## Testing

Currently, the project uses exploratory Jupyter notebooks for validation. No formal test suite exists yet (see improvement.md for recommendations).

### Manual Testing Approach
1. Run `notebooks/00_test.ipynb` end-to-end
2. Verify output CSV files in `data/outputs/`
3. Inspect visualizations for spatial correctness
4. Check coupling statistics (touching rate ~30-40% is typical)

## Best Practices

### Code Organization
1. **Keep notebooks clean**: Move reusable logic to `src/` modules
2. **Use relative paths**: Avoid hardcoding absolute paths
3. **Document functions**: Include docstrings with parameters and return types
4. **Cache strategically**: CouplingAnalyzer uses `_mask_cache` - clear per outlet to manage memory

### Performance Tips
1. **Precompute masks**: Use `warmup=True` in `compute_coupling_all_outlets()`
2. **Limit DEM size**: Crop DEMs to study area before analysis
3. **Adjust threshold**: Lower stream network threshold → more heads → slower computation
4. **Use connectivity=4**: Faster than connectivity=8 if diagonal contact isn't needed

### Visualization Guidelines
1. **Start with `view_mode="crop"`** for detailed pair analysis
2. **Use `view_mode="overview"`** for network context
3. **Limit pairs**: Set `max_pairs=10` for large outlets to avoid clutter
4. **3D plots**: Use `dem_stride=2` to reduce surface complexity for faster rendering

## Common Workflows

### Analyze a New Study Area

```python
# 1. Load DEM
dem = tt3.read_tif("data/cropped_DEMs/NewArea_strm_crop.tif")
dem.z[dem.z < threshold_elevation] = np.nan

# 2. Derive networks
fd = tt3.FlowObject(dem)
s = tt3.StreamObject(fd, threshold=300)

# 3. Process all outlets
from src.coupling_analysis import CouplingAnalyzer

an = CouplingAnalyzer(fd, s, dem)
outs = outlet_node_ids_from_streampoi(s)

all_results = []
for outlet_id in outs:
    pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)
    df = an.evaluate_pairs_for_outlet(outlet_id, pairs)
    all_results.append(df)

# 4. Combine and save
df_all = pd.concat(all_results, ignore_index=True)
df_all.to_csv("data/outputs/NewArea/coupling_results.csv", index=False)
```

### Debug a Specific Confluence

```python
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

### Export Results for GIS

```python
# Add spatial coordinates to results
r, c = s.node_indices
xs, ys = s.transform * np.vstack((c, r))

df_all['conf_x'] = xs[df_all['confluence']]
df_all['conf_y'] = ys[df_all['confluence']]
df_all['head1_x'] = xs[df_all['head_1']]
df_all['head1_y'] = ys[df_all['head_1']]

# Convert to GeoDataFrame
import geopandas as gpd
from shapely.geometry import Point

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

**Slow performance on large DEMs**
- Crop DEM to region of interest: `dem.crop(...)`
- Increase stream threshold to reduce network complexity
- Enable cache warmup: `warmup=True`

**Hardcoded paths fail**
- Use pathlib for cross-platform compatibility:
  ```python
  from pathlib import Path
  project_root = Path(__file__).parent
  dem_path = project_root / "data" / "cropped_DEMs" / "area.tif"
  ```

## Development

### Adding New Analysis Functions

1. **Create module in `src/`**
   ```python
   # src/new_analysis.py
   def my_analysis_function(fd, s, dem, **kwargs):
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
       result : DataFrame
           Analysis results
       """
       # Implementation
       pass
   ```

2. **Add to notebook imports**
   ```python
   from src.new_analysis import my_analysis_function
   ```

3. **Document in this guide**

### Code Style
- Follow PEP 8
- Use type hints where helpful
- Prefer numpy/pandas vectorized operations over loops
- Add docstrings to public functions

## Resources

- **TopoToolbox Python**: https://github.com/TopoToolbox/pytopotoolbox
- **TopoToolbox MATLAB** (reference): https://topotoolbox.wordpress.com/
- **Rasterio docs**: https://rasterio.readthedocs.io/
- **GeoPandas**: https://geopandas.org/

## Support

For issues or questions:
1. Check this guide and improvement.md
2. Review existing notebook outputs
3. Inspect intermediate results (masks, pairs) with visualizations
4. Consult TopoToolbox documentation for stream network concepts

## Version Information

- **Environment name**: ch-heads
- **Python version**: 3.12.11
- **TopoToolbox version**: 0.0.6
- **Last updated**: 2026-02-01
