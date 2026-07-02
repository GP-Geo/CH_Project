"""Channel head coupling analysis package.

This package provides tools for analyzing channel head coupling in drainage networks
derived from Digital Elevation Models (DEMs). It identifies pairs of channel heads
that meet at confluences and determines whether their drainage basins are spatially
coupled (touching or overlapping).

Main components:
- CouplingAnalyzer: Detects spatial coupling between channel head drainage basins
- first_meet_pairs_for_outlet: Identifies channel head pairs for a given outlet
- LengthwiseAsymmetryAnalyzer: Computes lengthwise asymmetry (ΔL) metric
- GeometricFeaturesAnalyzer: Computes geometric features (confluence angle, etc.)
- generate_labeled_dataset: Creates labeled datasets for classification
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
    >>> pairs, heads = first_meet_pairs_for_outlet(s, outlet=5)
    >>> analyzer = CouplingAnalyzer(fd, s, dem)
    >>> results = analyzer.evaluate_pairs_for_outlet(5, pairs)
    >>>
    >>> # Compute lengthwise asymmetry with proper meter conversion
    >>> config = get_basin_config("inyo")
    >>> asym = LengthwiseAsymmetryAnalyzer(s, dem, lat=config["lat"])
    >>> asym_results = asym.evaluate_pairs_for_outlet(5, pairs)

Note:
    Only ``io``, ``mars``, ``models``, ``pipelines`` and ``rasterization`` are
    imported eagerly. The ``viz``, ``eval``, ``training``, ``features``,
    ``pairing`` and ``cli`` subpackages are intentionally *not* imported here
    (they pull optional heavy dependencies such as geopandas, scikit-learn and
    torch) — import them explicitly, e.g. ``from channel_heads import viz``.
"""

__version__ = "0.1.0"
__author__ = "Guy Pinkas"
__license__ = "MIT"

# Basin configuration data from Goren & Shelef (2024)
from . import io, mars, models, pipelines, rasterization
from .basin_config import (
    BASIN_CONFIG,
    LOCAL_TO_PAPER_BASIN,
    get_basin_config,
    get_reference_delta_L,
    get_z_th,
    list_basins,
)
from .coupling_analysis import CouplingAnalyzer, PairTouchResult

# Geometric analysis (asymmetry, geometric features, CSV enrichment).
# Canonical implementations live in ``channel_heads.features.*`` and are
# re-exported here. (The former ``channel_heads.geometric_analysis`` shim has
# been removed — import from the package root or the ``features`` submodules.)
from .features.asymmetry import (
    LengthwiseAsymmetryAnalyzer,
    PairAsymmetryResult,
    compute_asymmetry_statistics,
    compute_delta_L,
    merge_coupling_and_asymmetry,
)
from .features.earth_enrichment import add_geometric_features_to_csv
from .features.earth_geometry import (
    GEOM_FEATURE_COLS,
    GeometricFeaturesAnalyzer,
    PairGeometricResult,
    merge_geometric_features,
)
from .io.paths import (
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
from .logging_config import get_logger, setup_logging
from .pairing.earth import first_meet_pairs_for_outlet
from .pruning import apply_strategy, build_stream_graph, prune_by_order_gap
from .rasterization.schema import (
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    NUM_CLASSES,
    OTHER_STREAMS,
)
from .training.labeling import filter_hard_negatives, generate_labeled_dataset
from .units import compute_meters_per_degree, compute_pixel_size_meters, km2_to_cells

# Rasterization (no PyTorch dependency)
try:
    from .rasterization.earth_batch import precompute_raster_dataset
    from .rasterization.earth_patches import (
        raster_quality_flags,
        rasterize_outlet_pair,
    )
except ModuleNotFoundError as exc:
    if exc.name != "skimage":
        raise
    precompute_raster_dataset = raster_quality_flags = rasterize_outlet_pair = None
from .stream_utils import outlet_node_ids_from_streampoi

# CNN modules (optional, require PyTorch)
try:
    from .models.cnn import OutletCNN, OutletPairDataset, encode_raster_onehot
    from .models.cnn_features import CNN_FEATURE_COLS, extract_embeddings, merge_cnn_features

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

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
    # Geometric features
    "GeometricFeaturesAnalyzer",
    "PairGeometricResult",
    "generate_labeled_dataset",
    "filter_hard_negatives",
    "merge_geometric_features",
    # CSV enrichment
    "add_geometric_features_to_csv",
    "GEOM_FEATURE_COLS",
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
    # IO layer (paths, tables, geopackage, cleanup)
    "io",
    # Mars cross-planet pipeline logic
    "mars",
    # Curated model + rasterization layers
    "models",
    "rasterization",
    # High-level pipeline API
    "pipelines",
    # Rasterizer
    "rasterize_outlet_pair",
    "raster_quality_flags",
    "precompute_raster_dataset",
    "BACKGROUND",
    "BRANCH_A",
    "BRANCH_B",
    "OTHER_STREAMS",
    "CONFLUENCE_MARKER",
    "NUM_CLASSES",
    # CNN (optional, require PyTorch)
    "OutletCNN",
    "OutletPairDataset",
    "encode_raster_onehot",
    "extract_embeddings",
    "merge_cnn_features",
    "CNN_FEATURE_COLS",
    # Logging
    "get_logger",
    "setup_logging",
    # Pruning
    "apply_strategy",
    "build_stream_graph",
    "prune_by_order_gap",
    # Metadata
    "__version__",
]
