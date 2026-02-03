"""Channel head coupling analysis package.

This package provides tools for analyzing channel head coupling in drainage networks
derived from Digital Elevation Models (DEMs). It identifies pairs of channel heads
that meet at confluences and determines whether their drainage basins are spatially
coupled (touching or overlapping).

Main components:
- CouplingAnalyzer: Detects spatial coupling between channel head drainage basins
- first_meet_pairs_for_outlet: Identifies channel head pairs for a given outlet
- LengthwiseAsymmetryAnalyzer: Computes lengthwise asymmetry (ΔL) metric
- Basin configuration data from Goren & Shelef (2024)
- Visualization utilities for 2D and 3D plotting

Example:
    >>> import topotoolbox as tt3
    >>> from channel_heads import CouplingAnalyzer, first_meet_pairs_for_outlet
    >>> from channel_heads import LengthwiseAsymmetryAnalyzer, get_z_th
    >>>
    >>> # Get elevation threshold for basin
    >>> z_th = get_z_th("inyo")  # 1200 m
    >>>
    >>> dem = tt3.read_tif("path/to/dem.tif")
    >>> dem.z[dem.z < z_th] = np.nan  # Apply threshold
    >>> fd = tt3.FlowObject(dem)
    >>> s = tt3.StreamObject(fd, threshold=300)
    >>>
    >>> pairs, heads = first_meet_pairs_for_outlet(s, outlet_id=5)
    >>> analyzer = CouplingAnalyzer(fd, s, dem)
    >>> results = analyzer.evaluate_pairs_for_outlet(5, pairs)
    >>>
    >>> # Compute lengthwise asymmetry with proper meter conversion
    >>> config = get_basin_config("inyo")
    >>> asym = LengthwiseAsymmetryAnalyzer(s, dem, lat=config["lat"])
    >>> asym_results = asym.evaluate_pairs_for_outlet(5, pairs)
"""

__version__ = "0.1.0"
__author__ = "Guy Pinkas"
__license__ = "MIT"

# Basin configuration data from Goren & Shelef (2024)
from .basin_config import (
    BASIN_CONFIG,
    LOCAL_TO_PAPER_BASIN,
    get_basin_config,
    get_reference_delta_L,
    get_z_th,
    list_basins,
)
from .config import (
    CROPPED_DEMS_DIR,
    DATA_DIR,
    EXAMPLE_DEMS,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    get_experiment_output_dir,
    get_output_dir,
    list_available_dems,
    resolve_dem_path,
)
from .coupling_analysis import CouplingAnalyzer, PairTouchResult
from .first_meet_pairs_for_outlet import first_meet_pairs_for_outlet

# Lengthwise asymmetry (Goren & Shelef 2024)
from .lengthwise_asymmetry import (
    LengthwiseAsymmetryAnalyzer,
    PairAsymmetryResult,
    compute_asymmetry_statistics,
    compute_delta_L,
    compute_meters_per_degree,
    compute_pixel_size_meters,
    merge_coupling_and_asymmetry,
)
from .logging_config import get_logger, setup_logging
from .stream_utils import outlet_node_ids_from_streampoi

__all__ = [
    # Core analysis
    "CouplingAnalyzer",
    "PairTouchResult",
    "first_meet_pairs_for_outlet",
    "outlet_node_ids_from_streampoi",
    # Lengthwise asymmetry
    "LengthwiseAsymmetryAnalyzer",
    "PairAsymmetryResult",
    "compute_delta_L",
    "compute_asymmetry_statistics",
    "merge_coupling_and_asymmetry",
    "compute_meters_per_degree",
    "compute_pixel_size_meters",
    # Basin configuration
    "BASIN_CONFIG",
    "LOCAL_TO_PAPER_BASIN",
    "get_basin_config",
    "get_z_th",
    "list_basins",
    "get_reference_delta_L",
    # Path management
    "PROJECT_ROOT",
    "DATA_DIR",
    "CROPPED_DEMS_DIR",
    "OUTPUTS_DIR",
    "EXAMPLE_DEMS",
    "get_output_dir",
    "get_experiment_output_dir",
    "list_available_dems",
    "resolve_dem_path",
    # Logging
    "get_logger",
    "setup_logging",
    # Metadata
    "__version__",
]
