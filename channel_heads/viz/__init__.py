"""Visualization helpers shared by scripts and notebooks.

Earth DEM/basin plotting lives in :mod:`channel_heads.viz.earth`. Mars vector
contact-sheet rendering lives in :mod:`channel_heads.viz.contact_sheet`.
"""

from __future__ import annotations

from .calibration import plot_calibration_results
from .contact_sheet import render_contact_sheet, render_pair_panel
from .curves import pr_curve_panel, roc_curve_panel
from .earth import (
    largest_outlet_bbox,
    plot_all_coupled_pairs_for_outlet,
    plot_all_coupled_pairs_for_outlet_3d,
    plot_coupled_pair,
    plot_earth_outlet_map,
    plot_earth_pair_map,
    plot_network_complexity_comparison,
    plot_outlet_view,
)
from .per_outlet import make_palette, render_outlet_touching_pairs
from .poster import (
    add_geographic_ticks,
    apply_poster_style,
    assign_channel_head_labels,
    colored_hillshade,
    drainage_density_regime_panel,
    draw_channel_head_labels,
    earth_simplification_vs_mars,
    feature_distribution_panel,
    feature_schematic_panel,
    format_degree_axes,
    frame_only,
    hillshade,
    main_synthesis_figure,
    mars_marker_legend_handles,
    mars_prediction_summary_bars,
    mars_regime_summary_panel,
    model_comparison_bars,
    network_legend_handles,
    pair_definition_concept,
    patch_cmap_norm,
    patch_legend_handles,
    pipeline_flowchart,
    plot_5class_patch,
    plot_dem_hillshade,
    plot_network_on_hillshade,
    plot_network_overview,
    predicted_pair_list,
    save_figure,
    select_representative_networks,
)
from .stream_crossing import plot_removed_pair, removed_pair_legend_handles

__all__ = [
    "add_geographic_ticks",
    "apply_poster_style",
    "assign_channel_head_labels",
    "colored_hillshade",
    "drainage_density_regime_panel",
    "draw_channel_head_labels",
    "earth_simplification_vs_mars",
    "feature_distribution_panel",
    "feature_schematic_panel",
    "format_degree_axes",
    "frame_only",
    "hillshade",
    "largest_outlet_bbox",
    "main_synthesis_figure",
    "make_palette",
    "mars_marker_legend_handles",
    "network_legend_handles",
    "mars_prediction_summary_bars",
    "mars_regime_summary_panel",
    "model_comparison_bars",
    "pair_definition_concept",
    "patch_cmap_norm",
    "patch_legend_handles",
    "pipeline_flowchart",
    "plot_5class_patch",
    "plot_calibration_results",
    "plot_all_coupled_pairs_for_outlet",
    "plot_all_coupled_pairs_for_outlet_3d",
    "plot_coupled_pair",
    "plot_dem_hillshade",
    "plot_earth_outlet_map",
    "plot_earth_pair_map",
    "plot_network_complexity_comparison",
    "plot_network_on_hillshade",
    "plot_network_overview",
    "plot_outlet_view",
    "plot_removed_pair",
    "pr_curve_panel",
    "predicted_pair_list",
    "removed_pair_legend_handles",
    "render_contact_sheet",
    "render_outlet_touching_pairs",
    "render_pair_panel",
    "roc_curve_panel",
    "save_figure",
    "select_representative_networks",
]
