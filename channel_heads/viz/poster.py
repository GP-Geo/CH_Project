"""Poster / report figure primitives.

Reusable, presentation-grade plotting helpers shared by the poster figure
inventory notebook (``notebooks/presentation/14_poster_figure_inventory.ipynb``)
and any headless figure-export scripts. These deliberately contain *no* pipeline
or model logic — they only render artifacts that the caller has already loaded.

Design goals (poster-ready output):

* clean white background, large readable fonts, consistent line widths;
* vector-first export (``save_figure`` writes both ``.svg`` and ``.png``);
* a small number of composable helpers rather than one monolithic dashboard.

The 5-class CNN patch legend uses the **frozen** schema from
:mod:`channel_heads.rasterization.schema` (``0`` background, ``1`` branch A,
``2`` branch B, ``3`` other streams, ``4`` confluence marker) so figures never
silently disagree with the trained patch/model contract.
"""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Patch

from channel_heads.rasterization.schema import CLASS_LABELS, NUM_CLASSES

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Shared style
# --------------------------------------------------------------------------- #
PRODUCTION_GEOM_FEATURES: list[str] = [
    "orientation_diff_deg",
    "headhead_dist_norm",
    "apex_angle_deg",
    "strahler_order_diff",
    "proximity_profile_norm",
]
"""The five production geometric features, in canonical model order."""

POSTER_RCPARAMS: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",  # keep text as text in SVG (editable, small files)
}
"""Matplotlib rcParams for a clean, large-font, white poster style."""

# Phase colours for the pipeline flowchart and group-coded figures.
PHASE_COLORS: dict[str, str] = {
    "earth": "#2c7fb4",        # blue   — Earth training inputs/processing
    "calibration": "#a6761d",  # ochre  — Earth↔Mars comparability calibration
    "model": "#6a51a3",        # purple — learned models
    "mars": "#d94801",         # orange — Mars application
    "result": "#238b45",       # green  — final coupled result
}

# Discrete colours for the frozen 5-class patch encoding (index == class id).
PATCH_CLASS_COLORS: list[str] = [
    "#f7f7f7",  # 0 background
    "#e6550d",  # 1 branch A
    "#1f78b4",  # 2 branch B
    "#bdbdbd",  # 3 other streams
    "#000000",  # 4 confluence marker
]


def apply_poster_style() -> None:
    """Apply :data:`POSTER_RCPARAMS` to the global matplotlib rcParams."""
    plt.rcParams.update(POSTER_RCPARAMS)


def frame_only(ax, color: str = "#444444", linewidth: float = 1.1) -> None:
    """Strip x/y ticks and show a plain rectangular frame (all four spines).

    The poster style hides the top/right spines (fine for plots with axes); map
    panels instead want a clean rectangular border with no tick labels, which is
    what this restores.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor(color)


def _rect_spines(ax, color: str = "#444444", linewidth: float = 1.1) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor(color)


def _fmt_lon(v: float) -> str:
    return f"{abs(v):.2f}°{'E' if v >= 0 else 'W'}"


def _fmt_lat(v: float) -> str:
    return f"{abs(v):.2f}°{'N' if v >= 0 else 'S'}"


def format_degree_axes(ax, nticks: int = 5) -> None:
    """Label axes already in lon/lat degrees (e.g. an EPSG:4326 Earth map).

    Adds a small number of nice degree ticks with N/E/S/W suffixes, restores the
    rectangular frame and sets ``Longitude`` / ``Latitude`` labels.
    """
    from matplotlib.ticker import MaxNLocator

    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    lon_ticks = [t for t in MaxNLocator(nticks).tick_values(x0, x1) if x0 <= t <= x1]
    lat_ticks = [t for t in MaxNLocator(nticks).tick_values(y0, y1) if y0 <= t <= y1]
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([_fmt_lon(t) for t in lon_ticks])
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([_fmt_lat(t) for t in lat_ticks])
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=9)
    _rect_spines(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def add_geographic_ticks(ax, plot_crs, geographic_crs, nticks: int = 5) -> None:
    """Add lon/lat degree ticks to a map drawn in a *projected* CRS.

    Transforms the current view extent to ``geographic_crs`` to choose nice
    lon/lat ticks, then maps those back to ``plot_crs`` positions (along the view
    centre lines). Suitable for small extents (e.g. one Mars basin on the Robinson
    MOLA hillshade), where the graticule is effectively rectilinear.
    """
    from matplotlib.ticker import MaxNLocator
    from pyproj import Transformer

    to_geo = Transformer.from_crs(plot_crs, geographic_crs, always_xy=True)
    to_proj = Transformer.from_crs(geographic_crs, plot_crs, always_xy=True)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = [x0, x1, x0, x1]
    ys = [y0, y0, y1, y1]
    lons, lats = to_geo.transform(xs, ys)
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    lon_mid, lat_mid = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2

    lon_ticks = [t for t in MaxNLocator(nticks).tick_values(lon_min, lon_max)
                 if lon_min <= t <= lon_max]
    lat_ticks = [t for t in MaxNLocator(nticks).tick_values(lat_min, lat_max)
                 if lat_min <= t <= lat_max]
    ax.set_xticks([to_proj.transform(t, lat_mid)[0] for t in lon_ticks])
    ax.set_xticklabels([_fmt_lon(t) for t in lon_ticks])
    ax.set_yticks([to_proj.transform(lon_mid, t)[1] for t in lat_ticks])
    ax.set_yticklabels([_fmt_lat(t) for t in lat_ticks])
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=9)
    _rect_spines(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


# --------------------------------------------------------------------------- #
# Channel-head labelling shared by the prediction maps and the synthesis figure
# --------------------------------------------------------------------------- #
def assign_channel_head_labels(
    nodes_n: Any,
    only_node_ids: Any | None = None,
    prefix: str = "C",
) -> tuple[dict[int, str], Any]:
    """Assign a unique label (``C1``, ``C2``, …) to each channel head.

    Returns ``(label_map, heads_gdf)`` where ``label_map`` maps ``node_id`` →
    label. When ``only_node_ids`` is given only those heads are labelled (e.g.
    just the heads that participate in a predicted-touching pair), keeping the
    map readable.
    """
    heads = nodes_n[nodes_n["node_type"] == "channel_head"].copy()
    if only_node_ids is not None:
        keep = {int(x) for x in only_node_ids}
        heads = heads[heads["node_id"].astype(int).isin(keep)]
    heads = heads.sort_values("node_id")
    label_map = {int(nid): f"{prefix}{i + 1}"
                 for i, nid in enumerate(heads["node_id"].astype(int))}
    return label_map, heads


def draw_channel_head_labels(ax, heads_gdf: Any, label_map: Mapping[int, str],
                             fontsize: float = 7.0) -> None:
    """Draw the ``C#`` label beside each channel head in ``heads_gdf``."""
    for _, h in heads_gdf.iterrows():
        lbl = label_map.get(int(h["node_id"]))
        if lbl is None:
            continue
        ax.annotate(
            lbl, (h.geometry.x, h.geometry.y), fontsize=fontsize, fontweight="bold",
            color="black", zorder=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.5", lw=0.5, alpha=0.9),
        )


def predicted_pair_list(
    touching_df: pd.DataFrame,
    label_map: Mapping[int, str],
    head_col_1: str = "head_node_id_1",
    head_col_2: str = "head_node_id_2",
    prob_col: str = "prob_touching",
) -> list[tuple[int, str, str, float]]:
    """Return ``(rank, label_a, label_b, prob)`` for each touching pair, prob desc."""
    rows = touching_df.sort_values(prob_col, ascending=False)
    out = []
    for k, (_, r) in enumerate(rows.iterrows(), 1):
        a = label_map.get(int(r[head_col_1]), "?")
        b = label_map.get(int(r[head_col_2]), "?")
        out.append((k, a, b, float(r[prob_col])))
    return out


def save_figure(
    fig: Figure,
    out_dir: Path,
    stem: str,
    formats: Sequence[str] = ("svg", "png"),
    dpi: int = 200,
    close: bool = True,
) -> dict[str, Path]:
    """Save *fig* under *out_dir* as ``<stem>.<fmt>`` for each requested format.

    Returns a mapping ``{fmt: path}`` of the files written. ``out_dir`` is
    created on demand. When ``close`` is true the figure is closed afterwards
    (the default — convenient for batch export in a notebook loop).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, format=fmt)
        written[fmt] = path
        logger.info("Wrote: %s", path)
    if close:
        plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# 1. Pipeline flowchart
# --------------------------------------------------------------------------- #
DEFAULT_PIPELINE_STAGES: list[tuple[str, str]] = [
    ("Earth DEMs", "earth"),
    ("Earth–Mars DD calibration (threshold + pruning)", "calibration"),
    ("Earth valley networks (regime-pruned)", "earth"),
    ("Pair extraction", "earth"),
    ("Handcrafted features", "earth"),
    ("CNN patches", "earth"),
    ("CNN embeddings", "model"),
    ("XGBoost / combined model", "model"),
    ("Mars inputs", "mars"),
    ("Mars predictions", "mars"),
    ("Coupled channel heads", "result"),
]
"""Default stage sequence (``label``, ``group``) for :func:`pipeline_flowchart`.

Includes the Earth↔Mars drainage-density calibration step (choice of
stream-extraction threshold and pruning strategy) that makes the Earth training
networks comparable to the Martian networks.
"""


def pipeline_flowchart(
    stages: Sequence[tuple[str, str]] | None = None,
    title: str | None = "Channel-head coupling pipeline",
    max_per_row: int = 5,
    figsize: tuple[float, float] = (16.0, 7.0),
    box_w: float = 1.7,
    box_h: float = 0.9,
) -> Figure:
    """Draw a clean serpentine box-and-arrow pipeline flowchart.

    ``stages`` is a sequence of ``(label, group)`` where ``group`` keys into
    :data:`PHASE_COLORS`. Boxes are laid out left→right and wrap into a
    boustrophedon (snake) every ``max_per_row`` boxes, with arrows following the
    flow direction. Returns the :class:`~matplotlib.figure.Figure`.
    """
    stages = list(stages if stages is not None else DEFAULT_PIPELINE_STAGES)
    n = len(stages)

    dx, dy = box_w + 1.0, box_h + 1.2
    centers: list[tuple[float, float]] = []
    for i in range(n):
        row = i // max_per_row
        col = i % max_per_row
        if row % 2 == 1:  # snake: reverse direction on odd rows
            col = max_per_row - 1 - col
        centers.append((col * dx, -row * dy))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    for (label, group), (cx, cy) in zip(stages, centers):
        color = PHASE_COLORS.get(group, "#888888")
        box = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            cx,
            cy,
            "\n".join(textwrap.wrap(label, 16)),
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="#222222",
            zorder=3,
        )

    # Arrows along the flow path.
    for i in range(n - 1):
        (x0, y0), (x1, y1) = centers[i], centers[i + 1]
        if abs(y0 - y1) < 1e-6:  # same row → horizontal arrow
            start = (x0 + np.sign(x1 - x0) * box_w / 2, y0)
            end = (x1 - np.sign(x1 - x0) * box_w / 2, y1)
        else:  # row break → vertical arrow
            start = (x0, y0 - box_h / 2)
            end = (x1, y1 + box_h / 2)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=1.8,
                color="#555555",
                zorder=1,
            )
        )

    # Legend: phase → colour.
    phase_labels = {
        "earth": "Earth (training)",
        "calibration": "Earth↔Mars calibration",
        "model": "Learned models",
        "mars": "Mars (application)",
        "result": "Result",
    }
    used = list(dict.fromkeys(g for _, g in stages))
    handles = [
        Patch(facecolor=PHASE_COLORS[g], alpha=0.35, edgecolor=PHASE_COLORS[g],
              label=phase_labels.get(g, g))
        for g in used
        if g in PHASE_COLORS
    ]
    ax.legend(handles=handles, loc="lower center", ncol=len(handles),
              frameon=False, bbox_to_anchor=(0.5, -0.04))

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    ax.set_xlim(min(xs) - box_w, max(xs) + box_w)
    ax.set_ylim(min(ys) - box_h - 0.8, max(ys) + box_h)
    ax.set_aspect("equal")
    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. Pair-definition concept figure
# --------------------------------------------------------------------------- #
def _draw_concept_pair(
    ax,
    head1: tuple[float, float],
    head2: tuple[float, float],
    confluence: tuple[float, float],
    outlet: tuple[float, float],
    title: str,
    branch_color_a: str = "#e6550d",
    branch_color_b: str = "#1f78b4",
) -> None:
    """Draw one synthetic outlet→confluence→two-head pair on *ax*."""
    cx, cy = confluence
    # Branches (head -> confluence) and trunk (confluence -> outlet).
    ax.plot([head1[0], cx], [head1[1], cy], color=branch_color_a, lw=3.2, zorder=3)
    ax.plot([head2[0], cx], [head2[1], cy], color=branch_color_b, lw=3.2, zorder=3)
    ax.plot([cx, outlet[0]], [cy, outlet[1]], color="#444444", lw=3.2, zorder=2)

    # Markers.
    ax.scatter(*head1, s=160, c=branch_color_a, edgecolor="k", zorder=5)
    ax.scatter(*head2, s=160, c=branch_color_b, edgecolor="k", zorder=5)
    ax.scatter(cx, cy, marker="s", s=180, c="#ff7f00", edgecolor="k", zorder=6)
    ax.scatter(*outlet, marker="*", s=420, c="gold", edgecolor="k", zorder=7)

    # Labels.
    ax.annotate("channel head\n(branch 1)", head1, textcoords="offset points",
                xytext=(-4, 12), ha="center", fontsize=11, color=branch_color_a)
    ax.annotate("channel head\n(branch 2)", head2, textcoords="offset points",
                xytext=(4, 12), ha="center", fontsize=11, color=branch_color_b)
    ax.annotate("confluence\n(first meet)", (cx, cy), textcoords="offset points",
                xytext=(26, -2), ha="left", fontsize=11, color="#cc6600")
    ax.annotate("outlet", outlet, textcoords="offset points",
                xytext=(0, -22), ha="center", fontsize=12, fontweight="bold")

    ax.set_title(title, fontsize=15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def pair_definition_concept(figsize: tuple[float, float] = (13.0, 6.0)) -> Figure:
    """Synthetic two-panel diagram explaining the first-meet pair concept.

    Left panel: a *paired* (touching) pair (channel heads close, narrow apex).
    Right panel: a *non-paired* (non-touching) pair (heads far apart, wide apex).
    Both label the outlet, channel heads, confluence and the two branches. Pure
    synthetic geometry — no data loaded.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Paired: heads close together, sharp apex.
    _draw_concept_pair(
        axes[0],
        head1=(-0.6, 3.0),
        head2=(0.6, 3.0),
        confluence=(0.0, 1.6),
        outlet=(0.0, 0.0),
        title="Paired\n(adjacent heads, narrow apex)",
    )
    # Non-paired: heads far apart, wide apex.
    _draw_concept_pair(
        axes[1],
        head1=(-2.4, 3.0),
        head2=(2.4, 3.0),
        confluence=(0.0, 1.0),
        outlet=(0.0, 0.0),
        title="Non-paired\n(distant heads, wide apex)",
    )
    fig.suptitle(
        "First-meet channel-head pair: outlet · confluence · two branches",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. CNN 5-class patch rendering
# --------------------------------------------------------------------------- #
def patch_cmap_norm() -> tuple[ListedColormap, BoundaryNorm]:
    """Return a discrete ``(cmap, norm)`` for the frozen 5-class patch encoding."""
    cmap = ListedColormap(PATCH_CLASS_COLORS[:NUM_CLASSES])
    norm = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 0.5, 1.0), cmap.N)
    return cmap, norm


def patch_legend_handles() -> list[Patch]:
    """Legend handles for the 5-class patch encoding (canonical schema labels)."""
    return [
        Patch(facecolor=PATCH_CLASS_COLORS[cid], edgecolor="0.4",
              label=f"{cid} {CLASS_LABELS[cid]}")
        for cid in range(NUM_CLASSES)
    ]


def plot_5class_patch(ax, patch: np.ndarray, title: str | None = None) -> None:
    """Render one 5-class patch on *ax* with nearest-neighbour (no smoothing)."""
    cmap, norm = patch_cmap_norm()
    ax.imshow(np.asarray(patch), cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)


# --------------------------------------------------------------------------- #
# 4. Feature explanation panel
# --------------------------------------------------------------------------- #
def _feature_geometry_schematic(ax) -> None:
    """Small synthetic schematic illustrating the geometric pair features."""
    h1, h2, conf = (-1.0, 2.0), (1.4, 1.7), (0.0, 0.4)
    ax.plot([h1[0], conf[0]], [h1[1], conf[1]], color="#e6550d", lw=3)
    ax.plot([h2[0], conf[0]], [h2[1], conf[1]], color="#1f78b4", lw=3)
    ax.scatter(*h1, s=120, c="#e6550d", edgecolor="k", zorder=5)
    ax.scatter(*h2, s=120, c="#1f78b4", edgecolor="k", zorder=5)
    ax.scatter(*conf, marker="s", s=140, c="#ff7f00", edgecolor="k", zorder=6)
    # head-to-head distance
    ax.annotate("", h2, xytext=h1,
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.4))
    ax.text((h1[0] + h2[0]) / 2, max(h1[1], h2[1]) + 0.25, "head–head dist",
            ha="center", fontsize=10, color="0.2")
    ax.text(conf[0] + 0.15, conf[1] - 0.05, "apex angle", fontsize=10, color="#cc6600")
    ax.set_title("geometry schematic", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def feature_distribution_panel(
    df: pd.DataFrame,
    label_col: str = "y",
    features: Sequence[str] | None = None,
    figsize: tuple[float, float] = (15.0, 8.5),
    bins: int = 30,
    include_schematic: bool = True,
) -> Figure:
    """Panel of touching-vs-non-touching distributions for the geometric features.

    Lays the (up to five) ``features`` out in a grid of histograms. ``df`` must
    contain ``label_col`` with binary 0/1 labels. Missing/empty feature columns
    are skipped gracefully. When ``include_schematic`` is true a small geometry
    schematic occupies the trailing cell (for a dedicated, fuller schematic use
    :func:`feature_schematic_panel`).
    """
    features = list(features if features is not None else PRODUCTION_GEOM_FEATURES)
    present = [f for f in features if f in df.columns]
    n_cells = len(present) + (1 if include_schematic else 0)
    ncols = 3
    nrows = int(np.ceil(n_cells / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    has_labels = label_col in df.columns and df[label_col].nunique() > 1
    for ax, feat in zip(axes_flat, present):
        vals = pd.to_numeric(df[feat], errors="coerce")
        if has_labels:
            pos = vals[df[label_col] == 1].dropna()
            neg = vals[df[label_col] == 0].dropna()
            lo, hi = np.nanpercentile(vals.dropna(), [1, 99])
            rng = (lo, hi) if hi > lo else None
            ax.hist(neg, bins=bins, range=rng, color="#1f78b4", alpha=0.55,
                    density=True, label="non-touching")
            ax.hist(pos, bins=bins, range=rng, color="#e6550d", alpha=0.55,
                    density=True, label="touching")
            ax.legend(fontsize=9, frameon=False)
        else:
            ax.hist(vals.dropna(), bins=bins, color="#666666", alpha=0.7, density=True)
        ax.set_title(feat, fontsize=12)
        ax.set_ylabel("density")

    if include_schematic:
        _feature_geometry_schematic(axes_flat[len(present)])
    for ax in axes_flat[n_cells:]:
        ax.set_visible(False)

    title = (
        "Production geometric features — touching vs non-touching (Earth pairs)"
        if has_labels
        else "Production geometric features — distributions"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Feature schematic — one mini-diagram per production geometric feature
# --------------------------------------------------------------------------- #
def _pair_polylines() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic two-branch pair (head→confluence polylines) + outlet point."""
    a = np.array([[-1.5, 2.6], [-1.1, 1.7], [-0.5, 0.8], [0.0, 0.0]])
    b = np.array([[1.6, 2.7], [1.15, 1.7], [0.55, 0.8], [0.0, 0.0]])
    outlet = np.array([0.15, -1.3])
    return a, b, outlet


def _interp_poly(poly: np.ndarray, t: float) -> np.ndarray:
    """Point at fractional arc-length ``t`` (0=first vertex) along a polyline."""
    seg = np.diff(poly, axis=0)
    d = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(d)])
    target = t * cum[-1]
    i = int(np.clip(np.searchsorted(cum, target) - 1, 0, len(seg) - 1))
    f = (target - cum[i]) / (d[i] if d[i] > 0 else 1.0)
    return poly[i] + f * seg[i]


def _draw_pair_base(ax, a, b, outlet, lw: float = 3.0, alpha: float = 1.0,
                    markers: bool = True) -> None:
    ax.plot(a[:, 0], a[:, 1], color="#e6550d", lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=3)
    ax.plot(b[:, 0], b[:, 1], color="#1f78b4", lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=3)
    conf = a[-1]
    ax.plot([conf[0], outlet[0]], [conf[1], outlet[1]], color="#555555",
            lw=lw, alpha=alpha, zorder=2)
    if markers:
        ax.scatter(*a[0], s=80, c="#e6550d", edgecolor="k", zorder=5)
        ax.scatter(*b[0], s=80, c="#1f78b4", edgecolor="k", zorder=5)
        ax.scatter(*conf, marker="s", s=70, c="#ff7f00", edgecolor="k", zorder=6)
        ax.scatter(*outlet, marker="*", s=160, c="gold", edgecolor="k", zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-2.4, 2.5)
    ax.set_ylim(-1.7, 4.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _mini_orientation(ax) -> None:
    a, b, outlet = _pair_polylines()
    _draw_pair_base(ax, a, b, outlet, lw=2.2, alpha=0.45)
    for poly, col in ((a, "#e6550d"), (b, "#1f78b4")):
        d = poly[1] - poly[0]
        d = d / np.linalg.norm(d)
        ax.annotate("", xy=poly[0] + d * 1.05, xytext=poly[0],
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6))
    ax.text(0.05, 3.7, "initial downstream\nflow azimuths → Δ", ha="center",
            fontsize=9, color="0.2")
    ax.set_title("orientation_diff_deg", fontsize=11)


def _mini_apex(ax) -> None:
    a, b, outlet = _pair_polylines()
    _draw_pair_base(ax, a, b, outlet, lw=2.0, alpha=0.4)
    conf = a[-1]
    ax.plot([conf[0], a[0, 0]], [conf[1], a[0, 1]], color="#e6550d", lw=2.8, zorder=4)
    ax.plot([conf[0], b[0, 0]], [conf[1], b[0, 1]], color="#1f78b4", lw=2.8, zorder=4)
    a1 = np.degrees(np.arctan2(a[0, 1] - conf[1], a[0, 0] - conf[0]))
    a2 = np.degrees(np.arctan2(b[0, 1] - conf[1], b[0, 0] - conf[0]))
    ax.add_patch(Arc(conf, 1.5, 1.5, angle=0.0, theta1=min(a1, a2),
                     theta2=max(a1, a2), color="black", lw=1.6))
    ax.text(conf[0], conf[1] + 1.0, "apex angle\n(at confluence)", ha="center",
            fontsize=9, color="0.2")
    ax.set_title("apex_angle_deg", fontsize=11)


def _mini_headdist(ax) -> None:
    a, b, outlet = _pair_polylines()
    _draw_pair_base(ax, a, b, outlet, lw=2.6)
    ax.plot([a[0, 0], b[0, 0]], [a[0, 1], b[0, 1]], "k--", lw=1.7, zorder=4)
    mid = (a[0] + b[0]) / 2
    ax.text(mid[0], mid[1] + 0.2, "head–head distance d", ha="center", fontsize=9)
    pa, pb = _interp_poly(a, 0.55), _interp_poly(b, 0.55)
    ax.text(pa[0] - 0.5, pa[1], "L₁", color="#e6550d", fontsize=11, fontweight="bold")
    ax.text(pb[0] + 0.3, pb[1], "L₂", color="#1f78b4", fontsize=11, fontweight="bold")
    ax.set_title("headhead_dist_norm = d / (L₁+L₂)", fontsize=10)


def _mini_strahler(ax) -> None:
    a, b, outlet = _pair_polylines()
    _draw_pair_base(ax, a, b, outlet, lw=2.6)
    j = _interp_poly(b, 0.5)  # small tributary → branch 2 is higher order
    tip = j + np.array([0.75, 0.45])
    ax.plot([j[0], tip[0]], [j[1], tip[1]], color="#1f78b4", lw=2.0, zorder=3)
    ax.scatter(*tip, s=40, c="#1f78b4", edgecolor="k", zorder=5)
    ax.text(_interp_poly(a, 0.55)[0] - 0.6, _interp_poly(a, 0.55)[1], "order 1",
            color="#e6550d", fontsize=9, fontweight="bold")
    ax.text(_interp_poly(b, 0.6)[0] + 0.25, _interp_poly(b, 0.6)[1], "order 2",
            color="#1f78b4", fontsize=9, fontweight="bold")
    ax.set_title("strahler_order_diff = |o₁ − o₂|", fontsize=10)


def _mini_proximity(ax) -> None:
    a, b, outlet = _pair_polylines()
    _draw_pair_base(ax, a, b, outlet, lw=2.6)
    for t in (0.2, 0.4, 0.6, 0.8):
        pa, pb = _interp_poly(a, t), _interp_poly(b, t)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="0.35", lw=1.1, ls=":", zorder=4)
    ax.text(0.05, 3.7, "gap along the channels\n(1 = parallel, <1 = convergent)",
            ha="center", fontsize=8.5, color="0.2")
    ax.set_title("proximity_profile_norm", fontsize=11)


_MINI_FEATURE_DRAWERS = {
    "orientation_diff_deg": _mini_orientation,
    "headhead_dist_norm": _mini_headdist,
    "apex_angle_deg": _mini_apex,
    "strahler_order_diff": _mini_strahler,
    "proximity_profile_norm": _mini_proximity,
}


def feature_schematic_panel(
    features: Sequence[str] | None = None,
    figsize: tuple[float, float] = (15.0, 8.0),
) -> Figure:
    """Dedicated schematic: one synthetic mini-diagram per production feature.

    Each panel draws the same synthetic two-branch pair (branch 1 / branch 2 →
    confluence → outlet) and annotates one of the five production geometric
    features so the definitions read clearly. A shared marker legend fills the
    trailing cell.
    """
    feats = list(features if features is not None else PRODUCTION_GEOM_FEATURES)
    drawers = [(f, _MINI_FEATURE_DRAWERS[f]) for f in feats if f in _MINI_FEATURE_DRAWERS]
    ncols = 3
    nrows = int(np.ceil((len(drawers) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (_, fn) in zip(axes_flat, drawers):
        fn(ax)

    leg_ax = axes_flat[len(drawers)]
    leg_ax.set_axis_off()
    leg_ax.legend(
        handles=[
            Line2D([0], [0], color="#e6550d", lw=3, label="branch 1 (head 1 → confluence)"),
            Line2D([0], [0], color="#1f78b4", lw=3, label="branch 2 (head 2 → confluence)"),
            Line2D([0], [0], color="#555555", lw=3, label="trunk → outlet"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f00",
                   markeredgecolor="k", markersize=11, linestyle="", label="confluence"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markeredgecolor="k", markersize=15, linestyle="", label="outlet"),
        ],
        loc="center", frameon=False, fontsize=11,
    )
    for ax in axes_flat[len(drawers) + 1:]:
        ax.set_visible(False)

    fig.suptitle("Production geometric features — schematic definitions",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. Model comparison bars
# --------------------------------------------------------------------------- #
def model_comparison_bars(
    metrics_df: pd.DataFrame,
    label_col: str = "model",
    metric_cols: Sequence[str] = ("roc_auc", "pr_auc", "F1"),
    title: str = "Earth model comparison (held-out test)",
    figsize: tuple[float, float] = (12.0, 6.5),
) -> Figure:
    """Grouped bar chart comparing models across the given ``metric_cols``.

    ``metrics_df`` has one row per model (``label_col``) and one column per
    metric. Metrics not present in the frame are skipped. Bars are annotated
    with their values.
    """
    metrics = [m for m in metric_cols if m in metrics_df.columns]
    labels = metrics_df[label_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=figsize)
    palette = ["#2c7fb4", "#6a51a3", "#238b45", "#d94801"]
    for i, m in enumerate(metrics):
        vals = pd.to_numeric(metrics_df[m], errors="coerce").to_numpy()
        bars = ax.bar(x + i * width, vals, width, label=m, color=palette[i % len(palette)])
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=len(metrics))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6. Mars prediction summary
# --------------------------------------------------------------------------- #
def mars_prediction_summary_bars(
    per_model_df: pd.DataFrame,
    agreement_df: pd.DataFrame | None = None,
    model_col: str = "model_variant",
    figsize: tuple[float, float] = (14.0, 6.0),
) -> Figure:
    """Compact Mars prediction summary: per-model counts (+ optional agreement).

    ``per_model_df`` is the ``per_model_counts`` slice of
    ``mars_model_comparison_summary.csv`` (columns ``model_variant``,
    ``n_pairs``, ``n_touching``, ``n_high_confidence_prob_ge_0.80``). When
    ``agreement_df`` (the ``agreements`` slice) is supplied a second panel shows
    pairwise agreement counts and disagreements.
    """
    n_panels = 1 if agreement_df is None or agreement_df.empty else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    labels = per_model_df[model_col].astype(str).tolist()
    x = np.arange(len(labels))
    hc_col = next((c for c in per_model_df.columns if c.startswith("n_high_confidence")), None)
    series = [("n_touching", "predicted touching", "#d94801")]
    if hc_col:
        series.append((hc_col, "high-confidence (p≥0.80)", "#6a51a3"))
    width = 0.8 / len(series)
    for i, (col, lab, color) in enumerate(series):
        vals = pd.to_numeric(per_model_df[col], errors="coerce").to_numpy()
        bars = ax.bar(x + i * width, vals, width, label=lab, color=color)
        ax.bar_label(bars, fmt="%.0f", fontsize=9, padding=2)
    n_pairs = int(pd.to_numeric(per_model_df["n_pairs"], errors="coerce").max())
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("number of Mars pairs")
    ax.set_title(f"Mars predicted touching by model\n(total pairs = {n_pairs})")
    ax.legend(frameon=False)

    if n_panels == 2:
        ax2 = axes[1]
        alabels = agreement_df[model_col].astype(str).str.replace("agreement_", "")
        agree = pd.to_numeric(agreement_df.get("agreement_fraction"), errors="coerce")
        disagree = pd.to_numeric(agreement_df.get("n_changes"), errors="coerce")
        xx = np.arange(len(alabels))
        bars = ax2.bar(xx, agree, color="#238b45")
        ax2.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
        for xi, dval in zip(xx, disagree):
            if pd.notna(dval):
                ax2.text(xi, 0.02, f"Δ={int(dval)}", ha="center", va="bottom",
                         fontsize=9, color="white", fontweight="bold")
        ax2.set_xticks(xx)
        ax2.set_xticklabels(alabels, rotation=20, ha="right")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("agreement fraction")
        ax2.set_title("Pairwise model agreement\n(Δ = number of disagreeing pairs)")

    fig.tight_layout()
    return fig


def mars_regime_summary_panel(
    totals: Mapping[str, float],
    interp_df: pd.DataFrame | None = None,
    selected_regime: str = "regC",
    regime_col: str = "regime",
    figsize: tuple[float, float] = (14.0, 6.0),
) -> Figure:
    """Mars prediction summary for the *selected regime* (counts + regime context).

    Left panel: total scored pairs, predicted-touching and high-confidence counts
    for the selected regime (``totals`` with keys ``n_pairs``, ``n_touching``,
    ``n_high_conf``). Right panel (optional): per-regime touching fraction from
    ``interp_df`` (needs ``regime_col`` + ``touching_frac``; ``high_conf_frac_*``
    if present), with the selected regime highlighted.
    """
    n_panels = 1 if interp_df is None or interp_df.empty else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    keys = [("n_pairs", "scored pairs", "#9e9e9e"),
            ("n_touching", "predicted touching", "#d94801"),
            ("n_high_conf", "high-confidence", "#6a51a3")]
    keys = [(k, lab, c) for k, lab, c in keys if k in totals]
    vals = [float(totals[k]) for k, _, _ in keys]
    bars = ax.bar([lab for _, lab, _ in keys], vals,
                  color=[c for _, _, c in keys])
    ax.bar_label(bars, fmt="%.0f", fontsize=11, padding=2)
    ax.set_ylabel("number of Mars pairs")
    ax.set_title(f"Mars predictions — {selected_regime}\n(geom+cnn_emb regime model)")
    ax.tick_params(axis="x", rotation=15)

    if n_panels == 2:
        ax2 = axes[1]
        df = interp_df.copy()
        labels = df[regime_col].astype(str).tolist()
        x = np.arange(len(labels))
        touch = pd.to_numeric(df.get("touching_frac"), errors="coerce") * 100.0
        hc_col = next((c for c in df.columns if c.startswith("high_conf_frac")), None)
        colors = ["#d94801" if lbl == selected_regime else "#f6b27a" for lbl in labels]
        bars = ax2.bar(x, touch, color=colors, label="touching %")
        ax2.bar_label(bars, fmt="%.0f%%", fontsize=10, padding=2)
        if hc_col:
            hc = pd.to_numeric(df[hc_col], errors="coerce") * 100.0
            ax2.plot(x, hc, "o--", color="#6a51a3", label="high-conf %")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("% of Mars pairs")
        ax2.set_ylim(0, 100)
        ax2.set_title(f"Touching fraction by regime\n({selected_regime} selected)")
        ax2.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 7. Representative network selection (no model logic — pure summarisation)
# --------------------------------------------------------------------------- #
def select_representative_networks(
    preds: pd.DataFrame,
    network_col: str = "network_id",
    pred_col: str = "pred_touching_emb",
    prob_col: str = "prob_touching_emb",
    disagreement_col: str = "emb_logit_disagreement",
    high_conf_prob: float = 0.80,
    min_pairs: int = 6,
) -> dict[str, int]:
    """Pick a few representative networks for poster maps from a predictions frame.

    Returns a category→``network_id`` mapping covering, where derivable:
    ``"dense"`` (most pairs), ``"simple"`` (fewest pairs above ``min_pairs``),
    ``"high_confidence"`` (most p≥``high_conf_prob`` touching pairs) and
    ``"disagreement"`` (most emb-vs-logit disagreements). Categories that cannot
    be derived from the available columns are omitted. Purely descriptive — no
    inference is performed here.
    """
    g = preds.groupby(network_col)
    summary = pd.DataFrame({"n_pairs": g.size()})
    if prob_col in preds.columns:
        summary["n_high_conf"] = g.apply(
            lambda d: int((d[prob_col] >= high_conf_prob).sum()), include_groups=False
        )
    if disagreement_col in preds.columns:
        summary["n_disagree"] = g[disagreement_col].sum()

    eligible = summary[summary["n_pairs"] >= min_pairs]
    if eligible.empty:
        eligible = summary

    picks: dict[str, int] = {}
    taken: set[int] = set()

    def _pick(category: str, column: str, *, largest: bool, positive: bool = False) -> None:
        if column not in eligible.columns:
            return
        avail = eligible[~eligible.index.isin(taken)]
        if positive:
            avail = avail[avail[column] > 0]
        if avail.empty:
            return
        nid = int(avail[column].idxmax() if largest else avail[column].idxmin())
        picks[category] = nid
        taken.add(nid)

    # Priority order; each category takes the best *still-available* network so
    # the categories stay distinct and 3-4 networks are usually returned.
    _pick("dense", "n_pairs", largest=True)
    _pick("simple", "n_pairs", largest=False)
    _pick("high_confidence", "n_high_conf", largest=True, positive=True)
    _pick("disagreement", "n_disagree", largest=True, positive=True)
    return picks


# --------------------------------------------------------------------------- #
# 8. DEM hillshade (Earth overview)
# --------------------------------------------------------------------------- #
def hillshade(
    z: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    vert_exag: float = 1.0,
) -> np.ndarray:
    """Compute a normalised [0, 1] hillshade from an elevation array.

    Standard Horn-style illumination model; ``NaN`` cells are propagated then
    filled with the array mean so ``imshow`` renders cleanly.
    """
    z = np.asarray(z, dtype=float)
    if np.isnan(z).any():
        z = np.where(np.isnan(z), np.nanmean(z), z)
    dy, dx = np.gradient(z * vert_exag)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    lo, hi = np.nanmin(shaded), np.nanmax(shaded)
    return (shaded - lo) / (hi - lo) if hi > lo else np.zeros_like(shaded)


def plot_dem_hillshade(ax, dem_path: Path, title: str | None = None) -> tuple | None:
    """Read a (small, cropped) GeoTIFF DEM and draw its hillshade on *ax*.

    Read-only: opens the raster, computes a numpy hillshade and shows it. Returns
    the ``(left, right, bottom, top)`` extent, or ``None`` if rasterio is
    unavailable or the file cannot be read (a message is printed in that case).
    """
    try:
        import rasterio
    except ImportError:  # pragma: no cover - depends on optional geo extra
        print("[missing capability] rasterio not installed — skipping DEM hillshade")
        return None
    dem_path = Path(dem_path)
    if not dem_path.exists():
        print(f"[missing artifact] DEM not found: {dem_path}")
        return None
    with rasterio.open(dem_path) as src:
        # Cast to float *before* filling so nodata can become NaN even for
        # integer-typed DEMs (e.g. int16 SRTM).
        band = src.read(1, masked=True)
        z = np.ma.filled(band.astype("float64"), np.nan)
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
    colored_hillshade(ax, z, extent)
    ax.set_aspect("equal")
    frame_only(ax)
    if title:
        ax.set_title(title, fontsize=14)
    return extent


def colored_hillshade(
    ax,
    z: np.ndarray,
    extent,
    cmap: str = "terrain",
    hs_alpha: float = 0.5,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> None:
    """Draw a TopoToolbox-style coloured hillshade (terrain colours × shading).

    Renders the elevation with a terrain colormap and overlays a translucent
    grey hillshade. ``NaN`` cells stay transparent so masked-out areas read as
    background rather than flat grey.
    """
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)
    ax.imshow(z, cmap=cmap, extent=extent, origin="upper", zorder=0)
    hs = hillshade(z, azimuth=azimuth, altitude=altitude)
    hs_masked = np.ma.masked_where(~finite, hs)
    ax.imshow(hs_masked, cmap="gray", alpha=hs_alpha, extent=extent,
              origin="upper", zorder=1)


# --------------------------------------------------------------------------- #
# 9. Vector network overview map (Mars / generic)
# --------------------------------------------------------------------------- #
def network_legend_handles(
    stream_color: str = "#1565ff",
    network_label: str = "valley network",
) -> list[Line2D]:
    """Shared 4-entry network key (valley-network line + channel-head /
    confluence / outlet markers).

    Used by both the Earth outlet map and the Mars overview so the two panels
    read against one identical legend; the Mars vectors are coloured to match
    (``stream_color`` line, black channel heads, orange confluences, red outlet).
    """
    return [
        Line2D([0], [0], color=stream_color, lw=2.4, label=network_label),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=8, linestyle="", label="channel head"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f00",
               markeredgecolor="k", markersize=9, linestyle="", label="confluence"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="k", markersize=14, linestyle="", label="outlet"),
    ]


def mars_marker_legend_handles() -> list[Line2D]:
    """Marker legend handles: channel head, confluence, outlet."""
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=8, linestyle="", label="channel head"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f00",
               markeredgecolor="black", markersize=9, linestyle="", label="confluence"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=14, linestyle="", label="outlet"),
    ]


def plot_network_overview(
    ax,
    segs_n: Any,
    nodes_n: Any | None = None,
    outlet_xy: tuple[float, float] | None = None,
    title: str | None = None,
    network_color: str = "#9e9e9e",
    network_lw: float = 0.9,
) -> None:
    """Draw one valley network (grey lines + head/confluence/outlet markers).

    ``segs_n`` is a GeoDataFrame of segment geometries; ``nodes_n`` (optional)
    must carry a ``node_type`` column (``channel_head`` / ``confluence``).
    """
    segs_n.plot(ax=ax, color=network_color, linewidth=network_lw, zorder=1)
    if nodes_n is not None and "node_type" in nodes_n.columns:
        heads = nodes_n[nodes_n["node_type"] == "channel_head"]
        confs = nodes_n[nodes_n["node_type"] == "confluence"]
        if not heads.empty:
            ax.scatter([p.x for p in heads.geometry], [p.y for p in heads.geometry],
                       c="black", s=14, marker="o", zorder=4)
        if not confs.empty:
            ax.scatter([p.x for p in confs.geometry], [p.y for p in confs.geometry],
                       c="#ff7f00", s=20, marker="s", edgecolors="black",
                       linewidths=0.3, zorder=5)
    if outlet_xy is not None:
        ax.scatter(*outlet_xy, c="red", s=180, marker="*", edgecolors="black",
                   linewidths=0.5, zorder=6)
    ax.set_aspect("equal")
    frame_only(ax)
    if title:
        ax.set_title(title, fontsize=13)


def plot_network_on_hillshade(
    ax,
    segs_n: Any,
    raster_path: Path,
    nodes_n: Any | None = None,
    outlet_geom: Any | None = None,
    pad_mult: float = 2.0,
    pad_min: float = 5000.0,
    view_frac: float = 0.7,
    network_color: str = "#111111",
    network_lw: float = 1.2,
    head_color: str = "yellow",
    title: str | None = None,
    geographic_axes: bool = False,
    shape_true: bool = True,
    dem_source: bool = False,
    vert_exag: float = 2.0,
) -> tuple | None:
    """Overlay a vector network on a hillshade raster as regional context.

    The vectors (``segs_n`` and optional ``nodes_n``/``outlet_geom``) and a padded
    window of the raster around the network are reprojected to a **local conformal
    projection** (azimuthal stereographic centred on the network) before drawing,
    so circular ground features (e.g. impact craters) are *not* distorted. Global
    compromise projections such as the Robinson MOLA hillshade are neither
    conformal nor equal-area and, far from their central meridian, additionally
    *shear* the view — which ``set_aspect("equal")`` cannot undo. Set
    ``shape_true=False`` to plot in the raster's native CRS instead.

    When ``dem_source`` is true, ``raster_path`` is read as an *elevation* raster
    (e.g. the equirectangular MOLA DEM) and a grey hillshade is computed from the
    reprojected window with ``vert_exag`` vertical exaggeration; otherwise the
    raster is assumed to already be a (grey) hillshade.

    The view is tightened to the network plus a ``view_frac`` margin. Returns the
    shown extent (in the plot CRS), or ``None`` (with a printed message) if
    rasterio or the raster file is unavailable.
    """
    try:
        import rasterio
        from rasterio.transform import array_bounds
        from rasterio.warp import Resampling, calculate_default_transform, reproject
        from rasterio.windows import bounds as window_bounds
        from rasterio.windows import from_bounds
        from rasterio.windows import transform as window_transform
    except ImportError:  # pragma: no cover - optional geo extra
        print("[missing capability] rasterio not installed — skipping hillshade overlay")
        return None
    raster_path = Path(raster_path)
    if not raster_path.exists():
        print(f"[missing artifact] hillshade not found: {raster_path}")
        return None

    from pyproj import CRS

    geo_crs = segs_n.crs.geodetic_crs
    centroid = segs_n.to_crs(geo_crs).union_all().centroid

    with rasterio.open(raster_path) as src:
        native_crs = src.crs
        segs_native = segs_n.to_crs(native_crs)
        minx, miny, maxx, maxy = segs_native.total_bounds
        padx = (maxx - minx) * pad_mult + pad_min
        pady = (maxy - miny) * pad_mult + pad_min
        win = from_bounds(minx - padx, miny - pady, maxx + padx, maxy + pady,
                          src.transform)
        arr = src.read(1, window=win, boundless=True,
                       fill_value=src.nodata if src.nodata is not None else 0)
        wb = window_bounds(win, src.transform)
        win_transform = window_transform(win, src.transform)
        nodata = src.nodata
        try:
            radius = CRS(native_crs).ellipsoid.semi_major_metre or 3396190.0
        except Exception:  # pragma: no cover - non-standard ellipsoid
            radius = 3396190.0

    arr = arr.astype("float32")
    if dem_source and nodata is not None:
        arr[arr == nodata] = np.nan

    if shape_true:
        # Azimuthal stereographic is conformal everywhere, so a tightly-windowed
        # view around ``centroid`` has neither shape distortion nor shear.
        plot_crs = CRS.from_proj4(
            f"+proj=stere +lat_0={centroid.y} +lon_0={centroid.x} +k=1 "
            f"+x_0=0 +y_0=0 +R={radius} +units=m +no_defs")
        dst_transform, dw, dh = calculate_default_transform(
            native_crs, plot_crs, arr.shape[1], arr.shape[0],
            left=wb[0], bottom=wb[1], right=wb[2], top=wb[3])
        dst = np.full((dh, dw), np.nan, dtype="float32")
        reproject(
            source=arr, destination=dst,
            src_transform=win_transform, src_crs=native_crs,
            dst_transform=dst_transform, dst_crs=plot_crs,
            src_nodata=np.nan if dem_source else None, dst_nodata=np.nan,
            resampling=Resampling.bilinear)
        arr = dst
        b = array_bounds(dh, dw, dst_transform)  # (left, bottom, right, top)
        extent = (b[0], b[2], b[1], b[3])
        px, py = abs(dst_transform.a), abs(dst_transform.e)
    else:
        plot_crs = native_crs
        extent = (wb[0], wb[2], wb[1], wb[3])
        px, py = abs(win_transform.a), abs(win_transform.e)

    if dem_source:
        from matplotlib.colors import LightSource
        filled = np.nan_to_num(arr, nan=float(np.nanmin(arr)))
        gray = LightSource(azdeg=315, altdeg=45).hillshade(
            filled, dx=px, dy=py, vert_exag=vert_exag)
        gray = np.where(np.isnan(arr), np.nan, gray)
    else:
        gray = arr

    segs_r = segs_n.to_crs(plot_crs)
    nodes_r = nodes_n.to_crs(plot_crs) if nodes_n is not None else None
    outlet_r = (gpd_geoseries_to_crs(outlet_geom, segs_n.crs, plot_crs)
                if outlet_geom is not None else None)

    ax.imshow(gray, cmap="gray", extent=extent, origin="upper", zorder=0)
    segs_r.plot(ax=ax, color=network_color, linewidth=network_lw, zorder=2)
    if nodes_r is not None and "node_type" in nodes_r.columns:
        heads = nodes_r[nodes_r["node_type"] == "channel_head"]
        confs = nodes_r[nodes_r["node_type"] == "confluence"]
        if not heads.empty:
            ax.scatter([p.x for p in heads.geometry], [p.y for p in heads.geometry],
                       c=head_color, s=16, marker="o", edgecolors="black",
                       linewidths=0.3, zorder=4)
        if not confs.empty:
            ax.scatter([p.x for p in confs.geometry], [p.y for p in confs.geometry],
                       c="#ff7f00", s=22, marker="s", edgecolors="black",
                       linewidths=0.3, zorder=5)
    if outlet_r is not None:
        ax.scatter(outlet_r.x, outlet_r.y, c="red", s=190, marker="*",
                   edgecolors="black", linewidths=0.5, zorder=6)

    nminx, nminy, nmaxx, nmaxy = segs_r.total_bounds
    sx, sy = (nmaxx - nminx), (nmaxy - nminy)
    vx = sx * view_frac + pad_min * 0.3
    vy = sy * view_frac + pad_min * 0.3
    ax.set_xlim(nminx - vx, nmaxx + vx)
    ax.set_ylim(nminy - vy, nmaxy + vy)
    ax.set_aspect("equal")
    if geographic_axes:
        add_geographic_ticks(ax, plot_crs, geo_crs)
    else:
        frame_only(ax)
    if title:
        ax.set_title(title, fontsize=13)
    return extent


def gpd_geoseries_to_crs(geom: Any, src_crs: Any, dst_crs: Any):
    """Reproject a single shapely geometry from ``src_crs`` to ``dst_crs``."""
    import geopandas as gpd
    return gpd.GeoSeries([geom], crs=src_crs).to_crs(dst_crs).iloc[0]


# --------------------------------------------------------------------------- #
# 10. Main result synthesis (3-panel) figure
# --------------------------------------------------------------------------- #
def _distinct_colors(n: int) -> list:
    """``n`` visually distinct colours by stitching qualitative colormaps."""
    colors: list = []
    for name in ("tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3"):
        cmap = plt.get_cmap(name)
        colors.extend(list(getattr(cmap, "colors", [])))
        if len(colors) >= n:
            break
    if len(colors) < n:
        colors = colors * (n // max(len(colors), 1) + 1)
    return colors[:n]


def main_synthesis_figure(
    segs_n: Any,
    nodes_n: Any,
    df_net: pd.DataFrame,
    paths_lookup: Mapping[tuple[str, str], Any],
    outlet_xy: tuple[float, float] | None,
    network_id: int,
    pred_col: str = "pred_touching_emb",
    prob_col: str = "prob_touching_emb",
    head_col_1: str = "head_node_id_1",
    head_col_2: str = "head_node_id_2",
    high_conf_prob: float = 0.80,
    figsize: tuple[float, float] = (18.0, 7.5),
) -> Figure:
    """Three-panel synthesis for one network: raw → candidate pairs → predicted.

    Panel 1 shows the raw valley network; panel 2 overlays every candidate
    first-meet pair (model input); panel 3 colours each predicted-touching pair,
    labels its two channel heads (``C#``) and lists the pairs (sorted by
    probability) as ``Pair k: Ca – Cb, p=…``. All panels share spatial limits.
    Composition only — no inference is performed.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    import geopandas as gpd  # local import keeps module import light

    # Panel 1 — raw network.
    plot_network_overview(axes[0], segs_n, nodes_n, outlet_xy,
                          title="1 · Raw valley network")

    # Panel 2 — all candidate pairs (model input).
    plot_network_overview(axes[1], segs_n, nodes_n, outlet_xy,
                          title="2 · Candidate head pairs (model input)")
    for _, r in df_net.iterrows():
        pid = str(r["pair_id"])
        for branch, color in (("A", "#e6550d"), ("B", "#1f78b4")):
            path = paths_lookup.get((pid, branch))
            if path is not None:
                gpd.GeoSeries([path]).plot(ax=axes[1], color=color, linewidth=1.4,
                                           alpha=0.7, zorder=3)

    # Panel 3 — predicted coupled heads, with numbered heads + pair list.
    plot_network_overview(axes[2], segs_n, nodes_n, outlet_xy,
                          title="3 · Predicted coupled channel heads")
    n_total = len(df_net)
    touch = df_net[df_net[pred_col] == 1]
    if prob_col in touch.columns:
        touch = touch.sort_values(prob_col, ascending=False)
    n_touch = len(touch)
    n_hc = int((df_net[prob_col] >= high_conf_prob).sum()) if prob_col in df_net else 0

    have_heads = head_col_1 in touch.columns and head_col_2 in touch.columns
    label_map, heads_gdf = {}, None
    if have_heads and n_touch and "node_type" in nodes_n.columns:
        ids = pd.concat([touch[head_col_1], touch[head_col_2]]).astype(int).unique()
        label_map, heads_gdf = assign_channel_head_labels(nodes_n, only_node_ids=ids)

    palette = _distinct_colors(max(n_touch, 1))
    for i, (_, r) in enumerate(touch.iterrows()):
        pid = str(r["pair_id"])
        color = palette[i % len(palette)]
        for branch in ("A", "B"):
            path = paths_lookup.get((pid, branch))
            if path is not None:
                gpd.GeoSeries([path]).plot(ax=axes[2], color=color, linewidth=2.4,
                                           alpha=0.92, zorder=4)
    if heads_gdf is not None:
        draw_channel_head_labels(axes[2], heads_gdf, label_map, fontsize=7)

    # Pair list (sorted by probability) to the right of panel 3.
    pct = 100.0 * n_touch / n_total if n_total else 0.0
    header = (f"network {network_id}\n"
              f"{n_touch}/{n_total} pairs touching ({pct:.0f}%)\n"
              f"high-confidence (p≥{high_conf_prob:g}): {n_hc}\n")
    if have_heads and label_map:
        rows = predicted_pair_list(touch, label_map, head_col_1, head_col_2, prob_col)
        listing = "\n".join(f"Pair {k}: {a} – {b}, p={p:.3f}" for k, a, b, p in rows)
    else:
        listing = ""
    axes[2].text(
        1.03, 1.0, header + "\n" + listing, transform=axes[2].transAxes,
        va="top", ha="left", fontsize=7.0, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#bbbbbb"),
    )

    # Share spatial limits across panels.
    minx, miny, maxx, maxy = segs_n.total_bounds
    for ax in axes:
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    fig.suptitle(
        f"Mars network {network_id}: from raw network to predicted coupled channel heads",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    return fig


# --------------------------------------------------------------------------- #
# 11. Drainage-density regime calibration panel
# --------------------------------------------------------------------------- #
def drainage_density_regime_panel(
    earth_sweep: pd.DataFrame,
    threshold_col: str,
    earth_dd_col: str,
    mars_dd_median: float | None = None,
    regime_thresholds: Mapping[str, float] | None = None,
    title: str = "Earth–Mars drainage-density calibration",
    figsize: tuple[float, float] = (11.0, 6.5),
) -> Figure:
    """Earth drainage-density vs threshold with the Mars reference overlaid.

    ``earth_sweep`` must contain ``threshold_col`` and ``earth_dd_col``. A
    horizontal Mars-median reference line and optional vertical regime-threshold
    markers contextualise the chosen operating point(s).
    """
    df = earth_sweep.sort_values(threshold_col)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[threshold_col], df[earth_dd_col], "o-", color="#2c7fb4",
            label="Earth median Dd")
    if mars_dd_median is not None:
        ax.axhline(mars_dd_median, color="#d94801", linestyle="--", linewidth=2.2,
                   label=f"Mars median Dd = {mars_dd_median:.2f}")
    if regime_thresholds:
        colors = ["#238b45", "#6a51a3", "#cc6600", "#888888"]
        for (name, thr), c in zip(regime_thresholds.items(), colors):
            ax.axvline(thr, color=c, linestyle=":", linewidth=1.8,
                       label=f"{name} (T={thr:g})")
    ax.set_xscale("log")
    ax.set_xlabel("Stream-extraction threshold (km²)")
    ax.set_ylabel("Drainage density (km / km²)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 12. Earth network simplification vs Mars (extraction-protocol calibration)
# --------------------------------------------------------------------------- #
# Colours tie into the poster palette: Earth streams blue, Mars valleys orange,
# convex hull ochre, basin outline muted green, pruned-away segments faint.
_SIMPL_EARTH = PHASE_COLORS["earth"]
_SIMPL_MARS = PHASE_COLORS["mars"]
_SIMPL_HULL = PHASE_COLORS["calibration"]
_SIMPL_BASIN = "#557f63"
_SIMPL_REMOVED = "#d9c9b8"
_SIMPL_SCALEBARS = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])


def _simpl_bounds(*arrays: Any) -> tuple[np.ndarray, float]:
    """Square map centre + half-extent (metres) covering all point arrays."""
    pts = [np.asarray(a).reshape(-1, 2) for a in arrays if a is not None and np.asarray(a).size]
    if not pts:
        return np.array([0.0, 0.0]), 1.0
    stacked = np.vstack(pts)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    center = (lo + hi) / 2.0
    half = max(float((hi - lo).max()) / 2.0, 1.0) * 1.04
    return center, half


def _simpl_finish(ax, half_km: float, title: str) -> None:
    ax.set_xlim(-half_km, half_km)
    ax.set_ylim(-half_km, half_km)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="normal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    scale = _SIMPL_SCALEBARS[np.argmin(np.abs(_SIMPL_SCALEBARS - half_km / 2.5))]
    x0, y0 = -half_km * 0.90, -half_km * 0.90
    ax.plot([x0, x0 + scale], [y0, y0], color="black", lw=2.0)
    ax.text(x0 + scale / 2, y0 + half_km * 0.035, f"{scale:g} km",
            ha="center", va="bottom", fontsize=7)


def _simpl_draw_earth(ax, *, stream_seg, hull_xy, removed_seg, basin_xy, outlet_xy,
                      title, center, half) -> None:
    def to_km(a: Any) -> np.ndarray:
        return (np.asarray(a) - center) / 1000.0

    if basin_xy is not None and len(basin_xy):
        b = to_km(basin_xy)
        ax.plot(b[:, 0], b[:, 1], color=_SIMPL_BASIN, lw=1.0, alpha=0.95)
    if removed_seg is not None and len(removed_seg):
        ax.add_collection(LineCollection(to_km(removed_seg), colors=_SIMPL_REMOVED,
                                         linewidths=0.5, alpha=0.9))
    if stream_seg is not None and len(stream_seg):
        lw = float(np.clip(42.0 / np.sqrt(max(len(stream_seg), 1)), 0.18, 1.1))
        ax.add_collection(LineCollection(to_km(stream_seg), colors=_SIMPL_EARTH,
                                         linewidths=lw, alpha=0.95))
    if hull_xy is not None and len(hull_xy):
        h = to_km(np.vstack([hull_xy, hull_xy[:1]]))
        ax.plot(h[:, 0], h[:, 1], "--", color=_SIMPL_HULL, lw=1.2)
        ax.fill(h[:, 0], h[:, 1], color=_SIMPL_HULL, alpha=0.08)
    if outlet_xy is not None:
        p = to_km(np.asarray(outlet_xy).reshape(1, 2))[0]
        ax.scatter([p[0]], [p[1]], s=22, color="black", zorder=5)
    _simpl_finish(ax, half / 1000.0, title)


def _simpl_draw_mars(ax, *, lines, hull_xy, title) -> None:
    parts = [np.asarray(ln)[:, :2] for ln in lines if np.asarray(ln).size]
    center, half = _simpl_bounds(np.vstack(parts) if parts else None, hull_xy)

    def to_km(a: Any) -> np.ndarray:
        return (np.asarray(a) - center) / 1000.0

    segs = [to_km(np.asarray(ln)[:, :2]) for ln in lines if np.asarray(ln).shape[0] >= 2]
    if segs:
        lw = float(np.clip(12.0 / np.sqrt(max(len(segs), 1)), 0.7, 1.8))
        ax.add_collection(LineCollection(segs, colors=_SIMPL_MARS, linewidths=lw, alpha=0.96))
    if hull_xy is not None and len(hull_xy):
        h = to_km(np.vstack([hull_xy, hull_xy[:1]]))
        ax.plot(h[:, 0], h[:, 1], "--", color=_SIMPL_HULL, lw=1.2)
        ax.fill(h[:, 0], h[:, 1], color=_SIMPL_HULL, alpha=0.08)
    _simpl_finish(ax, half / 1000.0, title)


def _simpl_pick_mars_network(
    mars_gpkg: Path | str, pin_id: int | None = None
) -> tuple[int, float, dict[str, Any]]:
    """Pick a large, branch-complex Mars network with a higher (denser) Dd_hull.

    When ``pin_id`` is given that exact ``network_id`` is returned. Otherwise
    selection favours networks at/above the Mars **median** Dd_hull (denser, not
    sprawling), then ranks the large ones by segment count, Strahler order and
    total length. Footprint shape is not constrained.
    """
    from channel_heads.dd_calibration import (
        MARS_DD_STATS,
        STATUS_OK,
        mars_network_geometry,
        mars_network_strahler,
        mars_network_table,
    )

    tbl = mars_network_table(mars_gpkg)
    tbl = tbl[tbl["status"] == STATUS_OK].copy()
    geom = mars_network_geometry(mars_gpkg)
    tbl = tbl[tbl["network_id"].isin(geom.keys())].copy()

    if pin_id is not None:
        if int(pin_id) not in geom:
            raise ValueError(f"Mars network {pin_id} not found / not usable in {mars_gpkg}.")
        dd = tbl.loc[tbl["network_id"] == int(pin_id), "dd_hull_km_km2"]
        return int(pin_id), (float(dd.iloc[0]) if len(dd) else float("nan")), geom[int(pin_id)]

    try:
        st = mars_network_strahler(mars_gpkg)
        st = st[st["status"] == STATUS_OK][["network_id", "max_strahler"]]
        tbl = tbl.merge(st, on="network_id", how="left")
    except Exception:  # noqa: BLE001 - Strahler is best-effort metadata
        tbl["max_strahler"] = np.nan
    tbl["n_segments"] = [len(geom[int(n)]["lines"]) for n in tbl["network_id"]]

    big = tbl["length_km"] >= tbl["length_km"].quantile(0.75)
    pool = tbl[(tbl["dd_hull_km_km2"] >= MARS_DD_STATS["median"]) & big].copy()
    if pool.empty:
        pool = tbl[big].copy()
    if pool.empty:
        pool = tbl.copy()
    pool["score"] = (
        pool["n_segments"].rank(pct=True)
        + pool["max_strahler"].fillna(0).rank(pct=True)
        + 0.5 * pool["length_km"].rank(pct=True)
    )
    row = pool.sort_values("score", ascending=False).iloc[0]
    nid = int(row["network_id"])
    return nid, float(row["dd_hull_km_km2"]), geom[nid]


def earth_simplification_vs_mars(
    basin_name: str,
    outlet_node: int,
    *,
    threshold_km2: float = 0.2,
    mars_gpkg: Path | str | None = None,
    mars_network_id: int | None = None,
    figsize: tuple[float, float] = (9.6, 9.9),
) -> Figure:
    """Square 2x2 calibration figure: Mars VN beside one Earth outlet simplified.

    At a **fixed** extraction threshold (default 0.2 km²) one Earth outlet is
    drawn as its full drainage network and under two Strahler-pruning levels —
    drop 1st-order (keep ``S>=2``) and drop ``<=`` 2nd-order (keep ``S>=3``) —
    each labelled with its per-outlet convex-hull drainage density ``Dd_hull``.
    A large, branch-complex mapped Martian valley network (auto-selected toward
    higher/denser Dd_hull, or pinned via ``mars_network_id``) is shown in the
    fourth panel as the calibration target. The panels use independent local
    extents, so the comparison is about network *form* and simplification, not
    matched geography.

    Parameters mirror the cached complexity sweep: ``outlet_node`` is the
    threshold-dependent StreamObject node id at ``threshold_km2`` (the value in
    ``dd_master_sweep_complexity.csv``). Returns the Matplotlib ``Figure``.
    """
    from channel_heads.basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
    from channel_heads.dd_calibration import (
        DEFAULT_MARS_VALLEYS_GPKG,
        STATUS_OK,
        collect_dd_hull_trim_for_dem,
    )
    from channel_heads.io.paths import EXAMPLE_DEMS

    gpkg = Path(mars_gpkg) if mars_gpkg is not None else DEFAULT_MARS_VALLEYS_GPKG
    cfg = get_basin_config(LOCAL_TO_PAPER_BASIN.get(basin_name, basin_name))
    recs = collect_dd_hull_trim_for_dem(
        dem_path=EXAMPLE_DEMS[basin_name],
        basin_name=basin_name,
        threshold_km2=threshold_km2,
        z_th=cfg["z_th"],
        lat_deg=float(cfg["lat"]),
        min_outlet_basin_pixels=1000,
        max_outlets=25,
        geometry_for_outlets={outlet_node},
    )
    match = [r for r in recs if int(r.get("outlet_node", -1)) == outlet_node]
    if not match:
        with_geom = [r for r in recs if r.get("stream_seg_full") is not None]
        if not with_geom:
            raise ValueError(
                f"No outlet geometry returned for {basin_name} at T={threshold_km2} km²."
            )
        rec = max(with_geom, key=lambda r: r["basin_area_km2"])
    else:
        rec = match[0]
    for sfx in ("full", "trim", "ge3"):
        if rec.get(f"status_{sfx}") != STATUS_OK:
            raise ValueError(
                f"{basin_name} outlet {rec['outlet_node']}: variant '{sfx}' is not valid "
                f"(status={rec.get(f'status_{sfx}')})."
            )

    mars_nid, mars_dd, mars_geom = _simpl_pick_mars_network(gpkg, pin_id=mars_network_id)

    center, half = _simpl_bounds(
        rec.get("stream_seg_full"), rec.get("basin_boundary_xy"), rec.get("outlet_xy"),
    )

    # Square 2x2: Earth simplification reads full -> drop-1st -> drop-<=2nd
    # across the top-left/top-right/bottom-left, ending at the Mars analogue.
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    earth_specs = [
        (axes[0, 0], "full", "Earth — full network"),
        (axes[0, 1], "trim", "drop 1st order  (keep Strahler ≥ 2)"),
        (axes[1, 0], "ge3", "drop ≤2nd order  (keep Strahler ≥ 3)"),
    ]
    for ax, sfx, label in earth_specs:
        smax = int(rec[f"max_strahler_{sfx}"])
        _simpl_draw_earth(
            ax,
            stream_seg=rec.get(f"stream_seg_{sfx}"),
            hull_xy=rec.get(f"hull_xy_{sfx}"),
            removed_seg=None if sfx == "full" else rec.get(f"removed_seg_{sfx}"),
            basin_xy=rec.get("basin_boundary_xy"),
            outlet_xy=rec.get("outlet_xy"),
            title=f"{label}\nDd_hull = {rec[f'dd_hull_{sfx}']:.2f} km/km²  (max Strahler {smax})",
            center=center, half=half,
        )
    _simpl_draw_mars(
        axes[1, 1], lines=mars_geom["lines"], hull_xy=mars_geom["hull_xy"],
        title=f"Mars mapped VN  (net {mars_nid})\nDd_hull = {mars_dd:.2f} km/km²",
    )
    fig.suptitle(
        f"{basin_name.capitalize()} — outlet {int(rec['outlet_node'])} · drainage-network "
        f"simplification at Threshold = {threshold_km2:g} km² vs Mars",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


__all__ = [
    "PRODUCTION_GEOM_FEATURES",
    "POSTER_RCPARAMS",
    "PHASE_COLORS",
    "PATCH_CLASS_COLORS",
    "DEFAULT_PIPELINE_STAGES",
    "apply_poster_style",
    "frame_only",
    "format_degree_axes",
    "add_geographic_ticks",
    "assign_channel_head_labels",
    "draw_channel_head_labels",
    "predicted_pair_list",
    "save_figure",
    "pipeline_flowchart",
    "pair_definition_concept",
    "patch_cmap_norm",
    "patch_legend_handles",
    "plot_5class_patch",
    "feature_distribution_panel",
    "feature_schematic_panel",
    "model_comparison_bars",
    "mars_prediction_summary_bars",
    "mars_regime_summary_panel",
    "select_representative_networks",
    "hillshade",
    "colored_hillshade",
    "plot_dem_hillshade",
    "network_legend_handles",
    "mars_marker_legend_handles",
    "plot_network_overview",
    "plot_network_on_hillshade",
    "gpd_geoseries_to_crs",
    "main_synthesis_figure",
    "drainage_density_regime_panel",
    "earth_simplification_vs_mars",
]
