"""Channel head coupling analysis package.

This package provides tools for analyzing channel head coupling in drainage networks
derived from Digital Elevation Models (DEMs). It identifies pairs of channel heads
that meet at confluences and determines whether their drainage basins are spatially
coupled (touching or overlapping).

Main components:
- CouplingAnalyzer: Detects spatial coupling between channel head drainage basins
- first_meet_pairs_for_outlet: Identifies channel head pairs for a given outlet
- Visualization utilities for 2D and 3D plotting

Example:
    >>> import topotoolbox as tt3
    >>> from channel_heads import CouplingAnalyzer, first_meet_pairs_for_outlet
    >>>
    >>> dem = tt3.read_tif("path/to/dem.tif")
    >>> fd = tt3.FlowObject(dem)
    >>> s = tt3.StreamObject(fd, threshold=300)
    >>>
    >>> pairs, heads = first_meet_pairs_for_outlet(s, outlet_id=5)
    >>> analyzer = CouplingAnalyzer(fd, s, dem)
    >>> results = analyzer.evaluate_pairs_for_outlet(5, pairs)
"""

__version__ = "0.1.0"
__author__ = "Guy Pinkas"
__license__ = "MIT"

from .coupling_analysis import CouplingAnalyzer, PairTouchResult
from .first_meet_pairs_for_outlet import first_meet_pairs_for_outlet
from .stream_utils import outlet_node_ids_from_streampoi

__all__ = [
    "CouplingAnalyzer",
    "PairTouchResult",
    "first_meet_pairs_for_outlet",
    "outlet_node_ids_from_streampoi",
    "__version__",
]
