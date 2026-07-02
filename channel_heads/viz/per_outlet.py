"""Per-outlet figure: all predicted-touching pairs of one Mars network.

The full valley network in grey, each touching pair's two branches in a distinct
colour, with head/confluence/outlet markers and a pair legend. Extracted
verbatim from ``scripts/rendering/render_mars_outlet_touching_pairs.py`` (module
constants made parameters; ``output`` made optional for inline display).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import LineString

from .poster import assign_channel_head_labels, draw_channel_head_labels, frame_only

logger = logging.getLogger(__name__)


def make_palette(n: int) -> list:
    """Build a list of n visually distinct colours by stitching several
    qualitative palettes together.
    """
    cmaps = ["tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3"]
    colors: list = []
    for cmap_name in cmaps:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            colors.extend(list(cmap.colors))
        else:
            colors.extend([cmap(i) for i in np.linspace(0, 1, 20)])
        if len(colors) >= n:
            break
    return colors[:n] if len(colors) >= n else colors * (n // len(colors) + 1)


def render_outlet_touching_pairs(
    nid: int,
    df_net: pd.DataFrame,
    segs_n: gpd.GeoDataFrame,
    nodes_n: gpd.GeoDataFrame,
    outlet_xy: tuple[float, float] | None,
    paths_lookup: dict[tuple[str, str], LineString],
    output: Path | None = None,
    touching_col: str = "pred_touching_emb",
    prob_col: str = "prob_touching_emb",
    model_name: str = "geom_plus_cnn_emb",
    number_heads: bool = False,
    head_col_1: str = "head_node_id_1",
    head_col_2: str = "head_node_id_2",
):
    """Render one network's predicted-touching pairs.

    If ``output`` is given the figure is saved there and closed (returns
    ``None``); if ``output`` is ``None`` the figure is returned for inline
    display. Returns ``None`` (and warns) when the network has no touching pairs.

    When ``number_heads`` is true each participating channel head is given a
    unique ``C#`` label and the legend lists every predicted pair (sorted by
    probability) as ``Pair k: Ca – Cb  p=…`` so individual coupled pairs can be
    identified on a poster. This needs ``head_col_1`` / ``head_col_2`` (channel
    head node ids) in ``df_net``.
    """
    n_total_pairs = len(df_net)
    touching = df_net[df_net[touching_col] == 1].copy()
    n_touch = len(touching)
    if n_touch == 0:
        logger.warning("Network %d has no touching pairs — skipping", nid)
        return None

    # Order touching pairs by prob descending so the highest-confidence
    # pair gets the first (most distinct) palette colour and lowest pair number.
    touching = touching.sort_values(prob_col, ascending=False).reset_index(drop=True)

    # Unique labels (C1, C2, …) for the channel heads that take part in a pair.
    label_heads = (
        number_heads and head_col_1 in touching.columns and head_col_2 in touching.columns
    )
    label_map: dict[int, str] = {}
    heads_gdf = None
    if label_heads:
        ids = pd.concat([touching[head_col_1], touching[head_col_2]]).astype(int).unique()
        label_map, heads_gdf = assign_channel_head_labels(nodes_n, only_node_ids=ids)

    palette = make_palette(max(n_touch, 1))
    fig, ax = plt.subplots(figsize=(11, 10))
    segs_n.plot(ax=ax, color="#cccccc", linewidth=0.7, zorder=1)

    # Draw each touching pair's two branches in a distinct colour
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    legend_cap = n_touch if number_heads else 20
    for i, (_, r) in enumerate(touching.iterrows()):
        pair_id = str(r["pair_id"])
        num = i + 1
        color = palette[i % len(palette)]
        path_a = paths_lookup.get((pair_id, "A"))
        path_b = paths_lookup.get((pair_id, "B"))
        if path_a is None or path_b is None:
            continue
        gpd.GeoSeries([path_a]).plot(ax=ax, color=color, linewidth=2.4, zorder=3, alpha=0.92)
        gpd.GeoSeries([path_b]).plot(ax=ax, color=color, linewidth=2.4, zorder=3, alpha=0.92)
        if i < legend_cap:
            legend_handles.append(Line2D([0], [0], color=color, lw=2.4))
            if label_heads:
                la = label_map.get(int(r[head_col_1]), "?")
                lb = label_map.get(int(r[head_col_2]), "?")
                legend_labels.append(f"Pair {num}: {la} – {lb}  p={r[prob_col]:.3f}")
            else:
                legend_labels.append(f"{pair_id}  p={r[prob_col]:.3f}")

    # All channel heads (small black dots) + confluences (orange squares),
    # taken from the network's node layer (not only paired ones).
    heads = nodes_n[nodes_n["node_type"] == "channel_head"]
    confluences = nodes_n[nodes_n["node_type"] == "confluence"]
    if not heads.empty:
        ax.scatter(
            [pt.x for pt in heads.geometry],
            [pt.y for pt in heads.geometry],
            c="black",
            s=18,
            marker="o",
            zorder=5,
            label="_nolegend_",
        )
    if not confluences.empty:
        ax.scatter(
            [pt.x for pt in confluences.geometry],
            [pt.y for pt in confluences.geometry],
            c="#ff7f00",
            s=24,
            marker="s",
            edgecolors="black",
            linewidths=0.4,
            zorder=6,
            label="_nolegend_",
        )
    if outlet_xy is not None:
        ax.scatter(
            *outlet_xy,
            c="red",
            s=180,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=7,
            label="_nolegend_",
        )

    # Unique C# labels on the participating channel heads (note: only those in a
    # predicted pair, to keep the map readable).
    if label_heads and heads_gdf is not None:
        draw_channel_head_labels(ax, heads_gdf, label_map, fontsize=7)

    ax.set_aspect("equal")
    frame_only(ax)
    pct_touch = 100.0 * n_touch / n_total_pairs if n_total_pairs else 0
    ax.set_title(
        f"Mars network {nid} — outlet view\n"
        f"{n_touch} of {n_total_pairs} pairs predicted touching ({pct_touch:.1f}%)",
        fontsize=12,
    )

    if legend_handles:
        marker_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=7,
                label="channel head",
                linestyle="",
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
                linestyle="",
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
                linestyle="",
            ),
        ]
        marker_labels = ["channel head", "confluence", "outlet"]
        if number_heads:
            legend_title = (
                f"predicted pairs (all {n_touch}, by probability)\n"
                f"Cn = channel head  ·  p = probability"
            )
        else:
            legend_title = (
                f"touching pairs (top {min(20, n_touch)} of {n_touch})\n"
                f"colour = pair  ·  p = emb probability"
            )
        ax.legend(
            legend_handles + marker_handles,
            legend_labels + marker_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            title=legend_title,
            title_fontsize=8,
            frameon=False,
        )

    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote: %s", output)
        return None
    return fig
