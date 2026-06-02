"""Visualization helpers shared by the Mars rendering scripts and notebooks.

Vector contact-sheet rendering (the preferred polyline network style). See
:mod:`channel_heads.viz.contact_sheet`.
"""

from __future__ import annotations

from .contact_sheet import render_contact_sheet, render_pair_panel
from .curves import roc_curve_panel
from .per_outlet import make_palette, render_outlet_touching_pairs
from .stream_crossing import plot_removed_pair, removed_pair_legend_handles

__all__ = [
    "make_palette",
    "plot_removed_pair",
    "removed_pair_legend_handles",
    "render_contact_sheet",
    "render_outlet_touching_pairs",
    "render_pair_panel",
    "roc_curve_panel",
]
