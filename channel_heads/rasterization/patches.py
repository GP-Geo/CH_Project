"""5-class CNN patch generation (Earth + Mars compatible).

Re-exports the patch API from :mod:`channel_heads.rasterizer`. The 5-class
encoding (BACKGROUND/BRANCH_A/BRANCH_B/OTHER_STREAMS/CONFLUENCE_MARKER) is a
**frozen contract** — Mars patches must match the Earth-trained CNN, so the
underlying implementation is preserved as-is.
"""

from __future__ import annotations

from channel_heads.rasterization.schema import (
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CLASS_LABELS,
    CONFLUENCE_MARKER,
    NUM_CLASSES,
    OTHER_STREAMS,
)
from channel_heads.rasterizer import (
    precompute_raster_dataset,
    raster_quality_flags,
    rasterize_outlet_pair,
)

__all__ = [
    "rasterize_outlet_pair",
    "precompute_raster_dataset",
    "raster_quality_flags",
    "BACKGROUND",
    "BRANCH_A",
    "BRANCH_B",
    "OTHER_STREAMS",
    "CONFLUENCE_MARKER",
    "NUM_CLASSES",
    "CLASS_LABELS",
]
