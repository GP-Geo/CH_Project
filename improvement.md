# Project Improvement Recommendations

This document tracks improvements to the channel-heads project. It serves as both a changelog for completed work and a roadmap for future enhancements.

**Last updated:** 2026-02-01

---

## Completed Improvements

### Testing Infrastructure

**Status:** Implemented

- Created `tests/` directory with pytest-based test suite
- Added `tests/conftest.py` with mock objects (MockGridObject, MockFlowObject, MockStreamObject)
- Implemented `tests/test_coupling_analysis.py` with tests for:
  - PairTouchResult dataclass
  - CouplingAnalyzer initialization and cache management
  - Influence mask computation
  - Pair touching detection
  - evaluate_pairs_for_outlet method
- Implemented `tests/test_first_meet_pairs.py` with tests for:
  - Pair normalization
  - Node ID list conversion
  - Parent adjacency construction
  - Basin node collection
  - First meet pairs algorithm (simple and complex networks)
  - Edge cases (single head, empty edges)

**Run tests:**
```bash
pytest tests/ -v --cov=channel_heads --cov-report=html
```

---

### Package Structure

**Status:** Implemented

- Renamed `src/` to `channel_heads/` for standard Python package naming
- Created `pyproject.toml` with:
  - Package metadata and versioning
  - Dependencies and optional dev dependencies
  - CLI entry point (`ch-analyze`)
  - Tool configurations (pytest, black, ruff)
- Updated `channel_heads/__init__.py` with comprehensive exports
- Package is now installable via `pip install -e ".[dev]"`

---

### Command-Line Interface

**Status:** Implemented

Created `channel_heads/cli.py` with:
- `ch-analyze` command for batch DEM processing
- Options: `--threshold`, `--connectivity`, `--mask-below`, `--outlets`, `--verbose`
- Proper logging integration
- Error handling and exit codes
- Progress reporting

**Usage:**
```bash
ch-analyze data/cropped_DEMs/Inyo_strm_crop.tif -o results/inyo.csv --mask-below 1200 -v
```

---

### Path Management

**Status:** Implemented

Created `channel_heads/config.py` with:
- Auto-detection of project root
- Centralized path constants (PROJECT_ROOT, DATA_DIR, CROPPED_DEMS_DIR, OUTPUTS_DIR)
- EXAMPLE_DEMS dictionary for quick DEM access
- `resolve_dem_path()` for flexible path resolution
- `get_output_dir()` for study-area-specific output directories
- `list_available_dems()` for discovering DEMs
- Environment variable overrides (CHANNEL_HEADS_ROOT, CHANNEL_HEADS_DATA)

---

### Logging Configuration

**Status:** Implemented

Created `channel_heads/logging_config.py` with:
- Centralized logging setup
- `get_logger()` for module-specific loggers
- `setup_logging()` for configuration
- Environment variable support (CHANNEL_HEADS_LOG_LEVEL, CHANNEL_HEADS_LOG_FILE)
- Console and file handler support

---

### Basin Configuration Data

**Status:** Implemented

Created `channel_heads/basin_config.py` with:
- Configuration data from Goren & Shelef (2024) Table A1
- 18 mountain ranges with parameters:
  - Elevation thresholds (z_th)
  - Reference ΔL values (median, 25th, 75th percentiles)
  - Concavity index (θ)
  - Location coordinates (lat, lon)
  - Area, aridity index, number of pairs
- Helper functions: `get_basin_config()`, `get_z_th()`, `get_reference_delta_L()`, `list_basins()`
- Mapping between local DEM names and paper basin names

---

### Lengthwise Asymmetry Metric

**Status:** Implemented

Created `channel_heads/lengthwise_asymmetry.py` with:
- `LengthwiseAsymmetryAnalyzer` class using TopoToolbox's `upstream_distance()`
- `compute_delta_L()` function implementing Equation 4 from Goren & Shelef (2024)
- Automatic coordinate conversion (degrees to meters) for geographic DEMs
- `compute_asymmetry_statistics()` for summary statistics
- `merge_coupling_and_asymmetry()` for combining results
- `PairAsymmetryResult` dataclass

---

### Continuous Integration

**Status:** Implemented

Created `.github/workflows/tests.yml` with:
- Test job: pytest on Python 3.11 and 3.12
- Lint job: black formatting and ruff linting
- Type check job: mypy (informational, non-blocking)
- Coverage upload to Codecov
- Runs on push to main/develop and pull requests

---

### Documentation

**Status:** Implemented

- Updated `CLAUDE.md` with:
  - New package structure
  - All new modules documented
  - Testing section with fixtures
  - CI/CD section
  - Updated workflows and examples
- Updated `README.md` with:
  - Comprehensive CLI guide
  - API reference for all major classes
  - Basin configuration usage
  - Study areas table with z_th values
  - Testing instructions

---

### Git Hygiene

**Status:** Implemented

- Updated `.gitignore` to exclude:
  - Data files (*.tif, *.csv in outputs)
  - Jupyter checkpoints
  - Python cache files
  - IDE files
  - Environment files

---

## Future Improvements

### Performance Optimization

**Priority:** Medium | **Effort:** Medium | **Impact:** Medium

#### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def analyze_outlet(outlet_id, s, fd, dem, connectivity):
    """Worker function for parallel processing."""
    an = CouplingAnalyzer(fd, s, dem, connectivity=connectivity)
    pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)
    return an.evaluate_pairs_for_outlet(outlet_id, pairs)

def compute_coupling_parallel(s, fd, dem, outlets=None, n_workers=4):
    """Parallel version of coupling analysis."""
    if outlets is None:
        outlets = outlet_node_ids_from_streampoi(s)

    worker = partial(analyze_outlet, s=s, fd=fd, dem=dem, connectivity=8)

    with Pool(n_workers) as pool:
        results = pool.map(worker, outlets)

    return pd.concat([r for r in results if not r.empty], ignore_index=True)
```

**Impact:** 2-4x speedup for multi-core systems

#### Spatial Indexing

Use R-tree for fast confluence queries:

```python
from rtree import index

def build_spatial_index(s):
    """Build R-tree index for fast spatial queries."""
    idx = index.Index()
    r, c = s.node_indices
    xs, ys = s.transform * np.vstack((c, r))

    conf_mask = s.streampoi('confluences')
    for i in np.flatnonzero(conf_mask):
        idx.insert(i, (xs[i], ys[i], xs[i], ys[i]))

    return idx
```

---

### Extended Testing

**Priority:** Medium | **Effort:** Medium | **Impact:** High

#### Integration Tests with Real DEMs

```python
# tests/test_integration.py
import pytest
from channel_heads import EXAMPLE_DEMS

@pytest.mark.integration
@pytest.mark.skipif(not EXAMPLE_DEMS["inyo"].exists(), reason="DEM not available")
def test_full_workflow_inyo():
    """Integration test with real Inyo DEM."""
    import topotoolbox as tt3
    from channel_heads import CouplingAnalyzer, first_meet_pairs_for_outlet, get_z_th

    dem = tt3.read_tif(str(EXAMPLE_DEMS["inyo"]))
    dem.z[dem.z < get_z_th("inyo")] = np.nan

    fd = tt3.FlowObject(dem)
    s = tt3.StreamObject(fd, threshold=300)

    # Test a known outlet
    pairs, heads = first_meet_pairs_for_outlet(s, outlet_id=5)
    assert len(heads) > 0

    an = CouplingAnalyzer(fd, s, dem)
    df = an.evaluate_pairs_for_outlet(5, pairs)
    assert not df.empty
```

#### Tests for Lengthwise Asymmetry

```python
# tests/test_lengthwise_asymmetry.py
def test_compute_delta_L():
    from channel_heads import compute_delta_L

    assert compute_delta_L(1000, 1000) == 0.0  # Symmetric
    assert abs(compute_delta_L(1000, 2000) - 0.6667) < 0.001
    assert compute_delta_L(0, 1000) == 2.0  # Maximum asymmetry

def test_meters_per_degree():
    from channel_heads import compute_meters_per_degree

    # At equator
    m = compute_meters_per_degree(0)
    assert 110000 < m < 112000

    # At 45 degrees
    m = compute_meters_per_degree(45)
    assert 90000 < m < 100000
```

---

### CLI Enhancements

**Priority:** Low | **Effort:** Low | **Impact:** Medium

#### Add Asymmetry Computation to CLI

```python
# In cli.py
parser.add_argument(
    "--compute-asymmetry",
    action="store_true",
    help="Also compute lengthwise asymmetry (ΔL)"
)
parser.add_argument(
    "--lat",
    type=float,
    help="Latitude for meter conversion (required with --compute-asymmetry)"
)
```

#### Progress Bar

```python
from tqdm import tqdm

for outlet_id in tqdm(outlet_ids, desc="Processing outlets"):
    # ...
```

---

### Data Versioning

**Priority:** Low | **Effort:** Medium | **Impact:** Medium

Use DVC (Data Version Control) for large DEMs:

```bash
pip install dvc dvc-gdrive

# Track data files
dvc add data/cropped_DEMs/
dvc add data/outputs/

# Commit DVC metadata
git add data/.gitignore data/cropped_DEMs.dvc
git commit -m "Track data with DVC"

# Push to remote storage
dvc remote add -d myremote gdrive://folder_id
dvc push
```

---

### Additional Basin Configuration

**Priority:** Low | **Effort:** Low | **Impact:** Low

Add DEMs for remaining basins from Goren & Shelef (2024):
- Taiwan (Central Mountain Range)
- Panamint Range, California
- Sakhalin Mountains, Russia
- Sierra Madre del Sur, Mexico
- Sierra Nevada, Spain
- Sierra Pie de Palo, Argentina
- Toano Range, Nevada
- Troodos Mountains, Cyprus
- Tsugaru Peninsula, Japan
- Yoro Mountains, Japan

---

### Notebook Improvements

**Priority:** Low | **Effort:** Low | **Impact:** Low

#### Convert to Jupytext

```bash
# Install jupytext
conda install jupytext

# Pair notebooks with .py files
jupytext --set-formats ipynb,py:percent notebooks/*.ipynb

# Track .py versions in git (easier diffs)
git add notebooks/*.py
```

#### Parameterized Notebooks with Papermill

```bash
pip install papermill

# Run notebook with parameters
papermill notebooks/all_basins_analysis.ipynb outputs/inyo_analysis.ipynb \
    -p basin_name "inyo" \
    -p threshold 300
```

---

### Scientific Reproducibility

**Priority:** Low | **Effort:** Low | **Impact:** Medium

#### Environment Snapshots

```bash
# Exact versions for reproducibility
conda list --export > env/conda-lock.txt
pip freeze > env/requirements-lock.txt

# Document system info
python -c "import platform; print(platform.platform())" > env/system-info.txt
```

#### Seed Management

```python
# In analysis scripts
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

---

## Implementation Roadmap

### Completed (Phase 1-2)

- [x] Testing infrastructure with pytest and mock objects
- [x] Package structure with pyproject.toml
- [x] CLI interface (`ch-analyze`)
- [x] Path management (config.py)
- [x] Logging configuration
- [x] Basin configuration from paper
- [x] Lengthwise asymmetry metric
- [x] CI/CD with GitHub Actions
- [x] Documentation updates (CLAUDE.md, README.md)
- [x] Git hygiene (.gitignore)

### Future (Phase 3+)

- [ ] Parallel processing for large DEMs
- [ ] Integration tests with real DEMs
- [ ] Tests for lengthwise asymmetry module
- [ ] CLI asymmetry computation flag
- [ ] Progress bar in CLI
- [ ] DVC for data versioning
- [ ] Additional basin DEMs
- [ ] Jupytext notebook pairing
- [ ] Papermill parameterization

---

## Success Metrics

Current status:

- [x] Test coverage > 50% (core modules)
- [x] Installable via `pip install -e .`
- [x] CLI enables batch processing
- [x] Documentation covers all use cases
- [ ] Test coverage > 80% (all modules)
- [ ] Repository size < 50 MB (excluding DVC-tracked data)
- [x] Cross-platform compatibility (Linux, macOS)
- [x] CI pipeline passes on all commits

---

## Questions?

Refer to:
- [Developer Guide](CLAUDE.md) for current capabilities
- [GitHub Issues](https://github.com/yourusername/channel-heads/issues) for tracking improvements
