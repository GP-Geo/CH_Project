"""5-class CNN patch generation (Earth + Mars compatible).

Re-exports the patch API from :mod:`channel_heads.rasterizer`. The 5-class
encoding (BACKGROUND/BRANCH_A/BRANCH_B/OTHER_STREAMS/CONFLUENCE_MARKER) is a
**frozen contract** — Mars patches must match the Earth-trained CNN, so the
underlying implementation is preserved as-is.
"""

from __future__ import annotations

from channel_heads.rasterizer import (
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    NUM_CLASSES,
    OTHER_STREAMS,
    precompute_raster_dataset,
    raster_quality_flags,
    rasterize_outlet_pair,
)

CLASS_LABELS = {
    BACKGROUND: "background",
    BRANCH_A: "branch_a",
    BRANCH_B: "branch_b",
    OTHER_STREAMS: "other_streams",
    CONFLUENCE_MARKER: "confluence_marker",
}

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
