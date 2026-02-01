# Project Improvement Recommendations

This document outlines concrete improvements to enhance the channel-heads project's maintainability, usability, and scientific reproducibility.

## Priority Matrix

| Priority | Category | Effort | Impact |
|----------|----------|--------|--------|
| 🔴 High | Testing | Medium | High |
| 🔴 High | Package Structure | Low | High |
| 🔴 High | Git Hygiene | Low | Medium |
| 🟡 Medium | Documentation | Medium | Medium |
| 🟡 Medium | CLI Interface | Medium | High |
| 🟡 Medium | Path Management | Low | Medium |
| 🟢 Low | Performance | Medium | Medium |
| 🟢 Low | Code Quality | Medium | Low |

---

## 🔴 High Priority

### 1. Add Testing Infrastructure

**Current state:** No tests exist. Validation relies on manual notebook inspection.

**Recommendation:** Implement pytest-based test suite

**Implementation:**

```bash
# Install test dependencies
conda install pytest pytest-cov

# Create test structure
mkdir tests
touch tests/__init__.py
touch tests/test_coupling_analysis.py
touch tests/test_first_meet_pairs.py
touch tests/test_stream_utils.py
touch tests/test_plotting_utils.py
```

**Example test file** (`tests/test_coupling_analysis.py`):

```python
import pytest
import numpy as np
from src.coupling_analysis import CouplingAnalyzer, PairTouchResult

@pytest.fixture
def mock_dem():
    """Create minimal DEM for testing."""
    import topotoolbox as tt3
    # Create 100x100 synthetic DEM
    z = np.random.randn(100, 100) * 10 + 1000
    return tt3.GridObject(z)

@pytest.fixture
def mock_flow_stream(mock_dem):
    """Create flow and stream objects."""
    import topotoolbox as tt3
    fd = tt3.FlowObject(mock_dem)
    s = tt3.StreamObject(fd, threshold=50)
    return fd, s

def test_coupling_analyzer_init(mock_flow_stream, mock_dem):
    """Test CouplingAnalyzer initialization."""
    fd, s = mock_flow_stream
    an = CouplingAnalyzer(fd, s, mock_dem, connectivity=8)
    assert an.connectivity == 8
    assert len(an._mask_cache) == 0

def test_pair_touching_result_structure():
    """Test PairTouchResult dataclass."""
    result = PairTouchResult(
        touching=True,
        overlap_px=5,
        contact_px=10,
        size1_px=100,
        size2_px=120
    )
    assert result.touching is True
    assert result.overlap_px == 5

def test_normalize_pair():
    """Test pair normalization."""
    from src.first_meet_pairs_for_outlet import _normalize_pair
    assert _normalize_pair(5, 3) == (3, 5)
    assert _normalize_pair(3, 5) == (3, 5)
```

**Run tests:**
```bash
pytest tests/ -v --cov=src --cov-report=html
```

**Impact:** Prevents regressions, enables confident refactoring, validates edge cases

---

### 2. Proper Package Structure

**Current state:** Project is a collection of scripts, not an installable package

**Recommendation:** Convert to installable Python package with `pyproject.toml`

**Implementation:**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "channel-heads"
version = "0.1.0"
description = "Coupled channel-head detection and basin-level analysis"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Guy Pinkas", email = "your.email@example.com"}
]
keywords = ["geomorphology", "hydrology", "drainage networks", "channel heads"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: GIS",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "numpy>=2.0",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "rasterio>=1.3",
    "geopandas>=1.0",
    "scikit-image>=0.24",
    "topotoolbox>=0.0.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "black>=24.0",
    "ruff>=0.3",
    "jupyterlab>=4.0",
]

[project.scripts]
ch-analyze = "channel_heads.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.black]
line-length = 100
target-version = ['py312']

[tool.ruff]
line-length = 100
target-version = "py312"
```

**Rename `src/` to `channel_heads/`** for standard Python naming:

```bash
mv src channel_heads
```

Update `channel_heads/__init__.py`:

```python
"""Channel head coupling analysis package."""

__version__ = "0.1.0"

from .coupling_analysis import CouplingAnalyzer, PairTouchResult
from .first_meet_pairs_for_outlet import first_meet_pairs_for_outlet
from .stream_utils import outlet_node_ids_from_streampoi

__all__ = [
    "CouplingAnalyzer",
    "PairTouchResult",
    "first_meet_pairs_for_outlet",
    "outlet_node_ids_from_streampoi",
]
```

**Install package in development mode:**

```bash
pip install -e ".[dev]"
```

**Impact:** Simplifies imports, enables version management, supports pip installation

---

### 3. Git Hygiene

**Current state:** Untracked data files, PDFs, outputs pollute repository

**Recommendation:** Update `.gitignore` and clean up tracked artifacts

**Update `.gitignore`:**

```gitignore
# Python
.DS_Store
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data files (large, regeneratable)
data/raw/*.tif
data/cropped_DEMs/*.tif
data/outputs/**/*.csv
data/outputs/**/*.png
*.aux.xml

# QGIS project files (user-specific)
*.qgz
*.qgz~

# Papers/PDFs (use references instead)
*.pdf

# Temporary/local directories
GuyPinkasLiran/
outputs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
env/
.env
venv/
```

**Clean up:**

```bash
# Remove tracked .DS_Store
find . -name .DS_Store -type f -delete
git rm --cached -r .DS_Store

# Stage .gitignore updates
git add .gitignore
git commit -m "Improve .gitignore to exclude data artifacts and PDFs"
```

**Impact:** Cleaner repository, faster clones, avoids committing large binary files

---

## 🟡 Medium Priority

### 4. Enhanced Documentation

**Current state:** Minimal README, code comments only

**Recommendation:** Expand README with examples, add scientific context

**Improved `README.md`:**

```markdown
# Channel Head Coupling Analysis

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated detection and analysis of coupled channel heads in drainage networks derived from Digital Elevation Models (DEMs).

## Overview

This package identifies pairs of channel heads that converge at confluences and determines whether their drainage basins are spatially coupled (touching or overlapping). This analysis is fundamental for understanding:

- Sediment connectivity in mountainous catchments
- Landscape evolution patterns
- Drainage network reorganization
- Channel head migration dynamics

## Features

- 🗺️ Automated drainage network extraction from DEMs
- 🔗 Graph-based channel head pairing algorithm
- 📊 Spatial coupling detection (4/8-connectivity)
- 📈 Statistical summaries and CSV exports
- 🎨 2D and 3D visualization tools

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/channel-heads.git
cd channel-heads

# Create conda environment
conda env create -f env/environment.yml
conda activate ch-heads

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
import topotoolbox as tt3
from channel_heads import CouplingAnalyzer, first_meet_pairs_for_outlet

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

## Documentation

- [Developer Guide](CLAUDE.md) - Setup, API reference, workflows
- [Improvements](improvement.md) - Roadmap and enhancement suggestions

## Study Areas

Included DEMs cover diverse geomorphic settings:
- Inyo Mountains, California
- Humboldt Range, Nevada
- Finisterre Range, Papua New Guinea
- Daqing Shan, China
- And more...

## Citation

If you use this code in research, please cite:

```
[Add paper reference when published]
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

Guy Pinkas - [your.email@example.com]
```

**Add `LICENSE` file:**

```bash
# Create MIT license
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Guy Pinkas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[... standard MIT license text ...]
EOF
```

**Impact:** Easier onboarding, clearer project goals, attribution for research

---

### 5. Command-Line Interface

**Current state:** Notebook-only workflows, no batch processing

**Recommendation:** Add CLI for common analysis tasks

**Implementation:**

Create `channel_heads/cli.py`:

```python
"""Command-line interface for channel head coupling analysis."""

import argparse
from pathlib import Path
import pandas as pd
import topotoolbox as tt3

from .coupling_analysis import CouplingAnalyzer
from .first_meet_pairs_for_outlet import first_meet_pairs_for_outlet
from .stream_utils import outlet_node_ids_from_streampoi


def main():
    parser = argparse.ArgumentParser(
        description="Analyze channel head coupling in drainage networks"
    )
    parser.add_argument("dem", type=Path, help="Path to DEM file (GeoTIFF)")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output CSV path for coupling results")
    parser.add_argument("--threshold", type=int, default=300,
                        help="Stream network area threshold (default: 300)")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8,
                        help="Connectivity for coupling detection (default: 8)")
    parser.add_argument("--mask-below", type=float,
                        help="Mask DEM elevations below this threshold")
    parser.add_argument("--outlets", type=str,
                        help="Comma-separated outlet IDs to analyze (default: all)")

    args = parser.parse_args()

    # Load DEM
    print(f"Loading DEM: {args.dem}")
    dem = tt3.read_tif(str(args.dem))

    if args.mask_below:
        dem.z[dem.z < args.mask_below] = float('nan')

    # Derive networks
    print("Deriving flow and stream networks...")
    fd = tt3.FlowObject(dem)
    s = tt3.StreamObject(fd, threshold=args.threshold)

    # Select outlets
    if args.outlets:
        outlet_ids = [int(x.strip()) for x in args.outlets.split(",")]
    else:
        outlet_ids = outlet_node_ids_from_streampoi(s)

    print(f"Analyzing {len(outlet_ids)} outlets...")

    # Analyze
    an = CouplingAnalyzer(fd, s, dem, connectivity=args.connectivity)
    all_results = []

    for i, outlet_id in enumerate(outlet_ids, 1):
        print(f"  [{i}/{len(outlet_ids)}] Outlet {outlet_id}")
        pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)
        df = an.evaluate_pairs_for_outlet(outlet_id, pairs)
        if not df.empty:
            all_results.append(df)

    # Save
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(args.output, index=False)
        print(f"\n✓ Saved {len(df_all)} pair results to {args.output}")
        print(f"  Touching pairs: {df_all['touching'].sum()} ({df_all['touching'].mean():.1%})")
    else:
        print("\n⚠ No pairs found in any outlet")


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Analyze single DEM
ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif \
    --output results/inyo_coupling.csv \
    --threshold 300 \
    --mask-below 1200

# Batch process all DEMs
for dem in data/cropped_DEMs/*.tif; do
    name=$(basename "$dem" _strm_crop.tif)
    ch-analyze "$dem" --output "results/${name}_coupling.csv"
done
```

**Impact:** Enables automated batch processing, reproducible analysis pipelines

---

### 6. Path Management

**Current state:** Hardcoded absolute paths in notebooks

**Recommendation:** Use pathlib and environment variables

**Implementation:**

Create `channel_heads/config.py`:

```python
"""Project configuration and path management."""

from pathlib import Path
import os

# Auto-detect project root (supports both installed package and local dev)
if os.getenv("CHANNEL_HEADS_ROOT"):
    PROJECT_ROOT = Path(os.getenv("CHANNEL_HEADS_ROOT"))
elif __file__:
    # Installed package
    PROJECT_ROOT = Path(__file__).parent.parent
else:
    # Fallback
    PROJECT_ROOT = Path.cwd()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CROPPED_DEMS_DIR = DATA_DIR / "cropped_DEMs"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Create if missing
for dir in [DATA_DIR, RAW_DATA_DIR, CROPPED_DEMS_DIR, OUTPUTS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Example DEMs
EXAMPLE_DEMS = {
    "inyo": CROPPED_DEMS_DIR / "Inyo_strm_crop.tif",
    "humboldt": CROPPED_DEMS_DIR / "Humboldt_strm_crop.tif",
    # ... more ...
}
```

**Update notebooks:**

```python
# Instead of:
# dem = tt3.read_tif("/Users/guypi/Projects/channel-heads/data/cropped_DEMs/Inyo_strm_crop.tif")

# Use:
from channel_heads.config import EXAMPLE_DEMS
dem = tt3.read_tif(EXAMPLE_DEMS["inyo"])
```

**Impact:** Cross-platform compatibility, easier deployment, cleaner notebooks

---

## 🟢 Low Priority

### 7. Performance Optimization

**Current state:** Single-threaded processing, full DEM loading

**Recommendations:**

#### 7a. Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def analyze_outlet(outlet_id, s, fd, dem, connectivity):
    """Worker function for parallel processing."""
    an = CouplingAnalyzer(fd, s, dem, connectivity=connectivity)
    pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)
    return an.evaluate_pairs_for_outlet(outlet_id, pairs)

def compute_coupling_parallel(s, fd, dem, outlets=None, n_workers=4):
    """Parallel version of compute_coupling_all_outlets."""
    if outlets is None:
        outlets = outlet_node_ids_from_streampoi(s)

    worker = partial(analyze_outlet, s=s, fd=fd, dem=dem, connectivity=8)

    with Pool(n_workers) as pool:
        results = pool.map(worker, outlets)

    return pd.concat([r for r in results if not r.empty], ignore_index=True)
```

#### 7b. Lazy Loading with Dask

```python
import dask.array as da
import rasterio

def load_dem_lazy(path):
    """Load DEM as dask array for memory-efficient processing."""
    with rasterio.open(path) as src:
        dem_lazy = da.from_array(src.read(1), chunks=(1024, 1024))
    return dem_lazy
```

#### 7c. Spatial Indexing

```python
from rtree import index

def build_spatial_index(s):
    """Build R-tree index for fast confluence queries."""
    idx = index.Index()
    r, c = s.node_indices
    xs, ys = s.transform * np.vstack((c, r))

    conf_mask = s.streampoi('confluences')
    for i in np.flatnonzero(conf_mask):
        idx.insert(i, (xs[i], ys[i], xs[i], ys[i]))

    return idx
```

**Impact:** 3-5x speedup for large DEMs, reduced memory footprint

---

### 8. Code Quality Improvements

#### 8a. Type Hints

**Add comprehensive type annotations:**

```python
from typing import Dict, Set, Tuple, List
import numpy.typing as npt

def first_meet_pairs_for_outlet(
    s: StreamObject,
    outlet: int
) -> Tuple[Dict[int, Set[Tuple[int, int]]], List[int]]:
    """Compute first-meet pairs with full type safety."""
    ...

def influence_mask(self, head_id: int) -> npt.NDArray[np.bool_]:
    """Return boolean mask with numpy type hint."""
    ...
```

#### 8b. Error Handling

**Add validation and informative errors:**

```python
class CouplingAnalyzer:
    def __init__(self, fd, s, dem, connectivity: int = 8):
        # Existing checks...

        # Add descriptive errors
        if not hasattr(fd, 'dependencemap'):
            raise TypeError(
                f"fd must be a FlowObject with .dependencemap() method, "
                f"got {type(fd).__name__}"
            )

        if not hasattr(s, 'streampoi'):
            raise TypeError(
                f"s must be a StreamObject with .streampoi() method, "
                f"got {type(s).__name__}"
            )
```

#### 8c. Logging

**Replace print() with logging:**

```python
import logging

logger = logging.getLogger(__name__)

def compute_coupling_all_outlets(...):
    logger.info(f"Processing {len(outlets)} outlets")
    for idx, o in enumerate(outlets, 1):
        logger.debug(f"[{idx}/{len(outlets)}] outlet={o}")
        ...
```

**Impact:** Better debugging, cleaner code, IDE support

---

## Additional Recommendations

### 9. Continuous Integration

**Set up GitHub Actions for automated testing:**

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: conda-incubator/setup-miniconda@v2
        with:
          environment-file: env/environment.yml
          activate-environment: ch-heads
      - name: Run tests
        run: |
          conda activate ch-heads
          pytest tests/ --cov=channel_heads --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 10. Data Versioning

**Use DVC (Data Version Control) for large DEMs:**

```bash
pip install dvc dvc-gdrive

# Track data files
dvc add data/cropped_DEMs/
dvc add data/outputs/

# Commit DVC metadata (not the data)
git add data/.gitignore data/cropped_DEMs.dvc data/outputs.dvc
git commit -m "Track data with DVC"

# Push data to remote storage
dvc remote add -d myremote gdrive://folder_id
dvc push
```

### 11. Notebook Best Practices

**Convert notebooks to Python scripts for version control:**

```bash
# Install jupytext
conda install jupytext

# Pair notebooks with .py files
jupytext --set-formats ipynb,py:percent notebooks/00_test.ipynb

# Now git tracks the .py version (easier diffs)
git add notebooks/00_test.py
```

### 12. Scientific Reproducibility

**Add environment snapshots:**

```bash
# Exact versions for reproducibility
conda list --export > env/conda-lock.txt
pip freeze > env/requirements-lock.txt

# Document system info
python -c "import platform; print(platform.platform())" > env/system-info.txt
```

---

## Implementation Roadmap

### Phase 1 (Week 1): Foundation
- [ ] Add testing infrastructure (tests/ directory, basic tests)
- [ ] Create pyproject.toml and make package installable
- [ ] Update .gitignore and clean repository
- [ ] Add LICENSE file

### Phase 2 (Week 2): Usability
- [ ] Implement CLI interface
- [ ] Add path management (config.py)
- [ ] Expand README with examples
- [ ] Create contribution guidelines

### Phase 3 (Week 3): Quality
- [ ] Add comprehensive type hints
- [ ] Implement logging
- [ ] Add error handling and validation
- [ ] Set up CI/CD pipeline

### Phase 4 (Ongoing): Optimization
- [ ] Profile code for bottlenecks
- [ ] Implement parallel processing
- [ ] Add spatial indexing
- [ ] Optimize memory usage

---

## Success Metrics

After implementing improvements:

- ✅ Test coverage > 80%
- ✅ Installable via `pip install -e .`
- ✅ CLI enables batch processing
- ✅ Documentation covers all use cases
- ✅ Repository size < 50 MB (excluding DVC-tracked data)
- ✅ Cross-platform compatibility (Linux, macOS, Windows)
- ✅ CI pipeline passes on all commits

---

## Questions?

Refer to:
- [Developer Guide](CLAUDE.md) for current capabilities
- [GitHub Issues](https://github.com/yourusername/channel-heads/issues) for tracking improvements
