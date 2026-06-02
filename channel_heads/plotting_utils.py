"""Compatibility shim for Earth plotting utilities.

The implementation moved to :mod:`channel_heads.viz.earth`. Keep this module
for notebooks and user code that still import the historical path.
"""

from __future__ import annotations

from .viz.earth import (
    BBox,
    ViewMode,
    _bbox_from_pair_masks,
    _bbox_from_points,
    _clamp_bbox,
    _get_rc,
    _maybe_crop_dem,
    _plot_segments,
    _stream_bbox,
    _xy_all_nodes,
    plot_all_coupled_pairs_for_outlet,
    plot_all_coupled_pairs_for_outlet_3d,
    plot_coupled_pair,
    plot_outlet_view,
)

__all__ = [
    "ViewMode",
    "BBox",
    "plot_coupled_pair",
    "plot_outlet_view",
    "plot_all_coupled_pairs_for_outlet",
    "plot_all_coupled_pairs_for_outlet_3d",
    "_get_rc",
    "_xy_all_nodes",
    "_plot_segments",
    "_stream_bbox",
    "_clamp_bbox",
    "_maybe_crop_dem",
    "_bbox_from_pair_masks",
    "_bbox_from_points",
]
