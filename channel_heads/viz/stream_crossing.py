"""QA rendering for pairs removed by the Mars stream-crossing filter.

One panel per removed pair: the network in grey, the crossed segment(s) in red,
both branch polylines, the dashed head-to-head straight line, and
head/confluence/outlet markers. Extracted verbatim from
``scripts/diagnostics/qa_mars_stream_crossing_filter.py``.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString


def plot_removed_pair(
    ax,
    sample_row: pd.Series,
    network_segments: gpd.GeoDataFrame,
    path_a_geom: LineString,
    path_b_geom: LineString,
    h1_xy: tuple[float, float],
    h2_xy: tuple[float, float],
    conf_xy: tuple[float, float],
    outlet_xy: tuple[float, float] | None,
    crossed_segment_ids: list[int],
    title: str,
    title_fontsize: int = 9,
) -> None:
    """Render one removed-pair QA panel onto ``ax``."""
    network_segments.plot(ax=ax, color="#cccccc", linewidth=0.6, zorder=1)

    if crossed_segment_ids:
        cs = network_segments[network_segments["segment_id"].isin(crossed_segment_ids)]
        if not cs.empty:
            cs.plot(ax=ax, color="#d62728", linewidth=2.2, alpha=0.85, zorder=2)

    gpd.GeoSeries([path_a_geom]).plot(ax=ax, color="#e6550d", linewidth=2.0, zorder=3)
    gpd.GeoSeries([path_b_geom]).plot(ax=ax, color="#1f78b4", linewidth=2.0, zorder=3)

    straight = LineString([h1_xy, h2_xy])
    gpd.GeoSeries([straight]).plot(ax=ax, color="#8b0000", linewidth=1.4, linestyle="--", zorder=4)

    ax.scatter(*h1_xy, c="black", s=42, marker="o", zorder=5)
    ax.scatter(*h2_xy, c="black", s=42, marker="o", zorder=5)
    ax.scatter(
        *conf_xy,
        c="#ff7f00",
        s=55,
        marker="s",
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
    )
    if outlet_xy is not None:
        ax.scatter(
            *outlet_xy,
            c="red",
            s=140,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    union = MultiLineString([path_a_geom, path_b_geom, straight])
    minx, miny, maxx, maxy = union.bounds
    pad = max(maxx - minx, maxy - miny) * 0.15 + 1.0
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


def removed_pair_legend_handles() -> list:
    """Legend handles describing the removed-pair QA panel colours/markers."""
    return [
        Line2D([0], [0], color="#cccccc", lw=1.5, label="other channels"),
        Line2D([0], [0], color="#d62728", lw=2.2, label="crossed segment(s)"),
        Line2D([0], [0], color="#e6550d", lw=2.0, label="branch A (head_1 → conf)"),
        Line2D([0], [0], color="#1f78b4", lw=2.0, label="branch B (head_2 → conf)"),
        Line2D(
            [0], [0], color="#8b0000", lw=1.4, linestyle="--", label="head-to-head straight line"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=8,
            label="head",
            lw=0,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#ff7f00",
            markeredgecolor="black",
            markersize=8,
            label="confluence",
            lw=0,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=12,
            label="outlet",
            lw=0,
        ),
    ]
