"""Stream network utility functions.

This module provides helper functions for working with TopoToolbox StreamObjects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def line_pixels(
    r0: int, c0: int, r1: int, c1: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Integer pixel coordinates along the straight line ``(r0, c0) -> (r1, c1)``.

    A faithful, dependency-free port of ``skimage.draw.line``'s Bresenham
    algorithm. Using it everywhere (rather than ``skimage.draw.line`` when
    available and a different fallback when not) guarantees the rasterized
    line — and therefore the stream-crossing gate that depends on it — is
    identical regardless of whether scikit-image is installed.

    Parameters
    ----------
    r0, c0 : int
        Start pixel (row, col).
    r1, c1 : int
        End pixel (row, col).

    Returns
    -------
    tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
        ``(rr, cc)`` integer arrays of length ``max(|r1-r0|, |c1-c0|) + 1``,
        including both endpoints, forming an 8-connected line.
    """
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if (r1 - r0) > 0 else -1
    sc = 1 if (c1 - c0) > 0 else -1
    r, c = r0, c0

    # Mirror skimage: iterate along the major axis (steep => swap roles).
    steep = False
    if dr > dc:
        steep = True
        c, r = r, c
        dc, dr = dr, dc
        sc, sr = sr, sc

    rr = np.zeros(dc + 1, dtype=np.intp)
    cc = np.zeros(dc + 1, dtype=np.intp)
    d = (2 * dr) - dc
    for i in range(dc):
        if steep:
            rr[i] = c
            cc[i] = r
        else:
            rr[i] = r
            cc[i] = c
        while d >= 0:
            r += sr
            d -= 2 * dc
        c += sc
        d += 2 * dr
    rr[dc] = r1
    cc[dc] = c1
    return rr, cc


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
    outlet_mask = s.streampoi("outlets")
    return np.flatnonzero(outlet_mask)
