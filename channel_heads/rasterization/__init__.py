"""Rasterization layer: patch generation + shared drawing primitives.

Earth/Mars-compatible 5-class 128x128 patches for the CNN. The implementation
lives in :mod:`channel_heads.rasterizer` (preserved as the frozen contract);
this subpackage is the curated public surface.
"""

from channel_heads.rasterization import drawing, manifest, mars_patches, patches
from channel_heads.rasterization.drawing import bresenham_line
from channel_heads.rasterization.manifest import (
    build_patch_manifest,
    validate_patch_manifest,
)
from channel_heads.rasterization.mars_patches import (
    build_mars_cnn_patches,
    render_pair_patch,
)
from channel_heads.rasterization.patches import (
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

__all__ = [
    "drawing",
    "patches",
    "manifest",
    "mars_patches",
    "bresenham_line",
    "rasterize_outlet_pair",
    "precompute_raster_dataset",
    "raster_quality_flags",
    # Mars patch generation (Phase 4)
    "render_pair_patch",
    "build_mars_cnn_patches",
    "build_patch_manifest",
    "validate_patch_manifest",
    "BACKGROUND",
    "BRANCH_A",
    "BRANCH_B",
    "OTHER_STREAMS",
    "CONFLUENCE_MARKER",
    "NUM_CLASSES",
]
