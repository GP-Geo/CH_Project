"""Visualization helpers shared by scripts and notebooks.

Earth DEM/basin plotting lives in :mod:`channel_heads.viz.earth`. Mars vector
contact-sheet rendering lives in :mod:`channel_heads.viz.contact_sheet`.
"""

from __future__ import annotations

from .calibration import plot_calibration_results
from .contact_sheet import render_contact_sheet, render_pair_panel
from .curves import roc_curve_panel
from .earth import (
    plot_all_coupled_pairs_for_outlet,
    plot_all_coupled_pairs_for_outlet_3d,
    plot_coupled_pair,
    plot_outlet_view,
)
from .per_outlet import make_palette, render_outlet_touching_pairs
from .stream_crossing import plot_removed_pair, removed_pair_legend_handles

__all__ = [
    "make_palette",
    "plot_calibration_results",
    "plot_all_coupled_pairs_for_outlet",
    "plot_all_coupled_pairs_for_outlet_3d",
    "plot_coupled_pair",
    "plot_outlet_view",
    "plot_removed_pair",
    "removed_pair_legend_handles",
    "render_contact_sheet",
    "render_outlet_touching_pairs",
    "render_pair_panel",
    "roc_curve_panel",
]
