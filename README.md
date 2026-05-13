# Channel Head Coupling Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TopoToolbox](https://img.shields.io/badge/TopoToolbox-0.0.6-green.svg)](https://github.com/TopoToolbox/pytopotoolbox)
[![Tests](https://github.com/yourusername/channel-heads/workflows/Tests/badge.svg)](https://github.com/yourusername/channel-heads/actions)

Automated detection, analysis, and ML-based classification of coupled channel heads in drainage networks derived from Digital Elevation Models (DEMs).

## Overview

This package identifies pairs of channel heads that converge at confluences, determines whether their drainage basins are spatially coupled (touching or overlapping), and predicts coupling using a trained XGBoost classifier (test AUC 0.920 on 17 basins).

The analysis pipeline is fundamental for understanding:

- **Sediment connectivity** in mountainous catchments
- **Landscape evolution** patterns
- **Drainage network reorganization**
- **Channel head migration** dynamics

Based on the methodology from:
> Goren, L. and Shelef, E.: Channel concavity controls planform complexity of branching drainage networks, Earth Surf. Dynam., 12, 1347-1369, https://doi.org/10.5194/esurf-12-1347-2024, 2024.

## Features

**Geomorphic analysis**
- Automated drainage network extraction from DEMs using TopoToolbox
- Graph-based channel head pairing algorithm
- Spatial coupling detection (4/8-connectivity, parallel-safe)
- Lengthwise asymmetry (ΔL) metric — Equation 4 from Goren & Shelef (2024)
- Basin configuration data from 18 mountain ranges

**Geometric feature engineering**
- Confluence angle between upstream branches
- Orientation similarity of downstream flow directions
- Head-head Euclidean distance (raw and flow-path-normalized)
- Tortuosity difference between branches
- Strahler stream order difference

**ML classification**
- XGBoost classifier trained on 10,868 labeled pairs across 17 basins
- Leave-one-basin-out cross-validation: CV AUC 0.884 ± 0.064
- Held-out test (Taiwan basin): AUC 0.920, accuracy 0.852
- End-to-end pipeline: DEM → features → dataset → trained model

**Infrastructure**
- Statistical summaries and CSV exports
- 2D and 3D visualization tools
- Command-line interface for batch processing
- Comprehensive test suite with pytest (114 tests)

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
ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif -o data/results/inyo_coupling.csv

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
    ch-analyze "$dem" -o "data/results/${name}_coupling.csv" -v
done

# Process with basin-specific elevation thresholds
ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif \
    -o data/results/inyo.csv \
    --mask-below 1200 \
    --threshold 300

ch-analyze data/cropped_DEMs/Humboldt_strm_crop.tif \
    -o data/results/humboldt.csv \
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
| `contact_px` | int | Number of contact boundary pixels |
| `size1_px` | int | Basin 1 size in pixels |
| `size2_px` | int | Basin 2 size in pixels |

### Interactive Analysis

```bash
conda activate ch-heads
jupyter lab
# Classic analysis: notebooks/analysis/03_all_basins.ipynb
# ML pipeline:      notebooks/ml/02_train_classifier.ipynb
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
print(f"Touching: {result.touching}, Contact: {result.contact_px}px")

# Evaluate all pairs at confluences for an outlet (serial or parallel)
df = analyzer.evaluate_pairs_for_outlet(outlet_id, pairs_at_confluence)
df = analyzer.evaluate_pairs_for_outlet_parallel(outlet_id, pairs_at_confluence, n_workers=4)

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

### GeometricFeaturesAnalyzer

Computes geometric properties of each channel head pair for use as ML features.

```python
from channel_heads import GeometricFeaturesAnalyzer, merge_geometric_features

geom_an = GeometricFeaturesAnalyzer(s, dem)
geom_df = geom_an.evaluate_pairs_for_outlet(outlet_id, pairs)

# Merge with coupling + asymmetry results
from channel_heads import merge_coupling_and_asymmetry, merge_geometric_features
combined = merge_coupling_and_asymmetry(coupling_df, asym_df)
combined = merge_geometric_features(combined, geom_df)
```

Features computed per pair: confluence angle, orientation similarity, head-head distance (raw + normalized), tortuosity difference, Strahler order difference. See [Feature Reference](#feature-reference) for detailed descriptions.

### ML Classifier

Run the trained XGBoost classifier on a feature DataFrame.

```python
import xgboost as xgb
from channel_heads.config import PROJECT_ROOT

model = xgb.XGBClassifier()
model.load_model(str(PROJECT_ROOT / "models/xgb_touching_classifier.json"))
feature_cols = (PROJECT_ROOT / "models/feature_columns.txt").read_text().splitlines()

X = combined[feature_cols].dropna()
combined.loc[X.index, "predicted_touching"] = model.predict(X)
combined.loc[X.index, "touching_prob"] = model.predict_proba(X)[:, 1]
```

See `notebooks/ml/02_train_classifier.ipynb` for training details and evaluation.

## Project Structure

```
channel-heads/
├── channel_heads/              # Python package
│   ├── coupling_analysis.py    # Basin coupling detection (parallel-safe)
│   ├── first_meet_pairs_for_outlet.py  # Head pairing algorithm
│   ├── geometric_analysis.py   # All geometric analysis (asymmetry, features, CSV enrichment)
│   ├── plotting_utils.py       # 2D/3D visualization
│   ├── cli.py                  # ch-analyze command
│   ├── config.py               # Centralized path management
│   ├── basin_config.py         # 18-basin reference data
│   └── logging_config.py       # Logging setup
├── tests/                      # 115 tests across 5 files
├── notebooks/
│   ├── analysis/               # Classic geomorphic analysis (01–04)
│   ├── ml/                     # ML dataset & classifier (00–02)
│   └── experiments/            # Threshold sensitivity experiments
├── data/
│   ├── cropped_DEMs/           # 18 study area DEMs (gitignored)
│   └── results/                # Analysis outputs (gitignored)
├── models/                     # Trained ML models (gitignored)
├── env/environment.yml         # Conda environment
└── pyproject.toml              # Package configuration
```

## Study Areas

All 18 mountain ranges from Goren & Shelef (2024) are included:

| Region | Location | z_th (m) | Climate |
|--------|----------|----------|---------|
| Taiwan Central Range | Taiwan | 80 | Humid tropical |
| Clan Alpine Mountains | Nevada, USA | 1700 | Arid |
| Daqing Shan | China | 1200 | Semi-arid |
| Finisterre Range | Papua New Guinea | 400 | Humid tropical |
| Humboldt Range | Nevada, USA | 1450 | Arid |
| Inyo Mountains | California, USA | 1200 | Arid |
| Kammanassie Mountains | South Africa | 630 | Semi-arid |
| Luliang Mountains | China | 1100 | Semi-arid |
| Panamint Range | California, USA | 800 | Hyper-arid |
| Sakhalin Mountains | Russia | 60 | Humid |
| Sierra del Valle Fértil | Argentina | 1050 | Arid |
| Sierra Madre del Sur | Mexico | 380 | Semi-humid |
| Sierra Nevada | Spain | 1200 | Semi-arid |
| Toano Range | Nevada, USA | 1710 | Arid |
| Troodos Mountains | Cyprus | 200 | Semi-arid |
| Tsugaru Peninsula | Japan | 30 | Humid |
| Yoro Mountains | Japan | 130 | Humid |

See `basin_config.py` for complete parameters including θ, ΔL, and aridity index values.

## Key Results

### ML Classifier Performance

| Metric | Value |
|--------|-------|
| CV AUC (leave-one-basin-out, 17 folds) | 0.884 ± 0.064 |
| Test AUC (held-out basin: Taiwan) | **0.920** |
| Test accuracy | 0.852 |
| Training set | 10,868 pairs, 17 basins |
| Top predictor | `strahler_order_diff` (importance 0.278) |

### Feature Importances (top 5)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `strahler_order_diff` | 0.278 |
| 2 | `delta_L` | 0.189 |
| 3 | `head_dist_norm` | 0.142 |
| 4 | `size1_px` | 0.118 |
| 5 | `confluence_angle` | 0.097 |

The trained model is saved at `models/xgb_touching_classifier.json` and can be reloaded with `xgb.XGBClassifier().load_model(...)`. To retrain from scratch, run `notebooks/ml/02_train_classifier.ipynb`.

## Feature Reference

All 10 ML features are computed per (head_1, head_2, confluence) triple. They come from three modules and describe different aspects of the spatial relationship between paired channel heads.

### Coupling features — `coupling_analysis.py`

These describe the spatial relationship between the two drainage basins upstream of a confluence.

| Column | Type | Description |
|--------|------|-------------|
| `touching` | bool | **Classification label.** `True` if the two upstream drainage basins are spatially adjacent — i.e., they share at least one 4- or 8-connected pixel border. This is what the model predicts. |
| `contact_px` | int | Number of border pixels at which the two basins touch. `0` for non-touching pairs; larger values indicate more extensive shared boundary. |
| `size1_px` | int | Total pixel count of head_1's upstream drainage basin (all DEM cells that drain through head_1 to the confluence). |
| `size2_px` | int | Total pixel count of head_2's upstream drainage basin. |

### Lengthwise asymmetry — `geometric_analysis.py`

Implements Equation 4 from Goren & Shelef (2024).

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `L_1` | float | ≥ 0 m | Along-stream flow path distance from head_1 to their shared confluence, in meters. |
| `L_2` | float | ≥ 0 m | Along-stream flow path distance from head_2 to their shared confluence, in meters. |
| `delta_L` | float | [0, 2] | **Normalized lengthwise asymmetry:** `ΔL = 2 · |L_1 − L_2| / (L_1 + L_2)`. A value of `0` means both heads are equidistant from the confluence (perfectly symmetric pair); a value close to `2` means one head is much farther upstream than the other. Symmetric pairs (small `delta_L`) are more likely to have touching basins because they imply similar-sized catchments on either side of the divide. **2nd most important feature.** |

### Geometric features — `geometric_analysis.py`

Computed from stream path coordinates. Direction vectors are estimated by tracing ~500 m along each branch.

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `confluence_angle_deg` | float | [0°, 180°] | Angle between the two upstream branch directions at the junction. Estimated by tracing ~500 m up each branch from the confluence and computing the angle between their weighted-average direction vectors. Small angles (< 30°) indicate nearly parallel co-flowing branches that tend to share a divide; large angles (> 120°) indicate opposing branches that typically do not touch. **5th most important feature.** |
| `orientation_diff_deg` | float | [0°, 180°] | Absolute difference in the initial downstream flow azimuths of the two heads. Each azimuth is estimated from the first ~500 m of each head's downstream path toward the confluence. `0°` = heads flow in the same direction (likely parallel ridges); `180°` = heads flow in opposite directions (typical of symmetric divide crossings). |
| `head_dist_norm` | float | ≥ 0 | Euclidean (straight-line) planform distance between the two channel head nodes **divided by** the total flow path length `L_1 + L_2`. Dimensionless; corrects for overall basin scale. Values near `0` mean the heads are very close relative to path length, which is common for touching pairs. **3rd most important feature.** |
| `headhead_dist_m` | float | ≥ 0 m | Raw Euclidean planform distance between channel heads in meters (the un-normalized version of `head_dist_norm`). |
| `tortuosity_diff` | float | ≥ 0 | Absolute difference `|τ₁ − τ₂|` in path sinuosity, where τ = flow path length / straight-line distance head→confluence. A straight channel has τ = 1; more sinuous channels have τ > 1. A large difference means one branch is significantly more winding than the other, which can indicate asymmetric hillslope geometry. |
| `strahler_order_diff` | float | ≥ 0 | Absolute difference in Strahler stream order between the two branches immediately upstream of the confluence: `|order(branch_1) − order(branch_2)|`. `0` = symmetric junction (same-order branches merging, typical of headwater bifurcations); `1` or more = asymmetric junction (a tributary joining a higher-order stem). High values strongly predict non-touching because a small tributary draining into a large stem rarely shares a continuous divide with it. **Most important feature** (importance 0.278 in XGBoost). |

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
- xgboost 3.2+, scikit-learn 1.7+, seaborn 0.13+ (ML pipeline)
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
  year = {2026},
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
