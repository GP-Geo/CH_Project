"""Stream network utility functions.

This module provides helper functions for working with TopoToolbox StreamObjects.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from topotoolbox import StreamObject


def outlet_node_ids_from_streampoi(s: Any) -> npt.NDArray[np.intp]:
    """Extract outlet node IDs from a StreamObject.

    Parameters
    ----------
    s : StreamObject
        Stream network object from TopoToolbox with a streampoi() method.

    Returns
    -------
    npt.NDArray[np.intp]
        Array of outlet node indices (linear indices into the stream network).

    Example
    -------
    >>> import topotoolbox as tt3
    >>> dem = tt3.read_tif("dem.tif")
    >>> fd = tt3.FlowObject(dem)
    >>> s = tt3.StreamObject(fd, threshold=300)
    >>> outlets = outlet_node_ids_from_streampoi(s)
    >>> print(f"Found {len(outlets)} outlets")
    """
    outlet_mask = s.streampoi('outlets')
    return np.flatnonzero(outlet_mask)
