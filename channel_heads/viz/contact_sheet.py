"""Vector contact-sheet rendering for Mars channel-head pairs.

Draws each pair in the preferred **vector network style** (grey network +
coloured branch polylines + head/confluence/outlet markers), and lays panels out
into a contact-sheet grid. This logic was duplicated byte-for-byte across the
Mars rendering scripts; the canonical version lives here so those scripts and
the ``notebooks/presentation/`` workflow call the same implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

logger = logging.getLogger(__name__)


def render_pair_panel(
    ax,
    segs_n: gpd.GeoDataFrame,
    path_a: LineString,
    path_b: LineString,
    h1_xy: tuple[float, float],
    h2_xy: tuple[float, float],
    conf_xy: tuple[float, float],
    outlet_xy: tuple[float, float] | None,
    title: str,
    title_fontsize: int = 7,
) -> None:
    """Draw one pair on ``ax``: network context + both branch polylines +
    head (circle), confluence (square) and optional outlet (star) markers."""
    segs_n.plot(ax=ax, color="#cccccc", linewidth=0.55, zorder=1)
    gpd.GeoSeries([path_a]).plot(ax=ax, color="#e6550d", linewidth=1.8, zorder=3)
    gpd.GeoSeries([path_b]).plot(ax=ax, color="#1f78b4", linewidth=1.8, zorder=3)
    ax.scatter(*h1_xy, c="black", s=24, marker="o", zorder=5)
    ax.scatter(*h2_xy, c="black", s=24, marker="o", zorder=5)
    ax.scatter(
        *conf_xy,
        c="#ff7f00",
        s=34,
        marker="s",
        edgecolors="black",
        linewidths=0.4,
        zorder=6,
    )
    if outlet_xy is not None:
        ax.scatter(
            *outlet_xy,
            c="red",
            s=90,
            marker="*",
            edgecolors="black",
            linewidths=0.4,
            zorder=7,
        )
    union = MultiLineString([path_a, path_b])
    minx, miny, maxx, maxy = union.bounds
    pad = max(maxx - minx, maxy - miny) * 0.15 + 1.0
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_fontsize)


def render_contact_sheet(
    rows: pd.DataFrame,
    nodes_by_nid: dict[int, gpd.GeoDataFrame],
    segs_by_nid: dict[int, gpd.GeoDataFrame],
    outlets_by_nid: dict[int, tuple[float, float]],
    paths_lookup: dict[tuple[str, str], LineString],
    title: str,
    subtitle_fn: Callable[[pd.Series], str],
    output: Path | None = None,
    cols: int = 5,
):
    """Render ``rows`` (one pair each) into a contact-sheet grid.

    If ``output`` is given, the figure is saved there and closed (returns
    ``None``). If ``output`` is ``None``, the figure is returned (e.g. for inline
    display in a notebook) and left open.
    """
    n = len(rows)
    if n == 0:
        logger.warning("No rows to render — skipping")
        return None
    n_rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3.0, n_rows * 3.0))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes_flat, rows.iterrows()):
        nid = int(r["network_id"])
        pair_id = str(r["pair_id"])
        h1 = int(r["head_node_id_1"])
        h2 = int(r["head_node_id_2"])
        conf = int(r["confluence_node_id"])

        nodes_n = nodes_by_nid[nid]
        segs_n = segs_by_nid[nid]
        node_xy = {
            int(nr["node_id"]): (float(nr.geometry.x), float(nr.geometry.y))
            for _, nr in nodes_n.iterrows()
        }

        path_a = paths_lookup.get((pair_id, "A"))
        path_b = paths_lookup.get((pair_id, "B"))
        if path_a is None or path_b is None:
            ax.set_visible(False)
            continue

        render_pair_panel(
            ax,
            segs_n=segs_n,
            path_a=path_a,
            path_b=path_b,
            h1_xy=node_xy[h1],
            h2_xy=node_xy[h2],
            conf_xy=node_xy[conf],
            outlet_xy=outlets_by_nid.get(nid),
            title=subtitle_fn(r),
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote: %s", output)
        return None
    return fig
