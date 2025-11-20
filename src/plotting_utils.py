"""
plotting_utils.py
=================
Unified visual utilities for TopoToolbox-like StreamObject and FlowObject analysis.

All main plotting functions accept:
    view_mode: "crop" | "zoom" | "overview"
        - "crop": crop DEM tightly around relevant subgraph (default)
        - "zoom": full DEM background, zoom to subgraph extent
        - "overview": full DEM + full network, no zoom
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------

def _get_rc(stream_obj):
    ni = stream_obj.node_indices
    if callable(ni):
        r, c = ni()
    else:
        r, c = ni
    return np.asarray(r), np.asarray(c)

def _xy_all_nodes(stream_obj):
    r, c = _get_rc(stream_obj)
    xs, ys = stream_obj.transform * np.vstack((c, r))
    return np.asarray(xs), np.asarray(ys)

def _plot_segments(ax, segments, **kwargs):
    lines = [np.asarray(seg) for seg in segments if len(seg) > 1]
    lc = LineCollection(lines, **({"color": "steelblue", "linewidth": 0.9} | kwargs))
    ax.add_collection(lc)
    return lc

def _stream_bbox(s_up):
    segs = s_up.xy()
    xs = np.concatenate([np.fromiter((p[0] for p in seg), float) for seg in segs if seg])
    ys = np.concatenate([np.fromiter((p[1] for p in seg), float) for seg in segs if seg])
    return xs.min(), xs.max(), ys.min(), ys.max()

def _clamp_bbox(L, R, B, T, dem):
    dL, dR, dB, dT = dem.extent
    L = max(L, dL); R = min(R, dR)
    B = max(B, dB); T = min(T, dT)
    return L, R, B, T

def _maybe_crop_dem(dem, s_up, view_mode="crop", pad_frac=0.05):
    xmin, xmax, ymin, ymax = _stream_bbox(s_up)
    dx, dy = xmax - xmin, ymax - ymin
    px, py = max(dx * pad_frac, 1e-9), max(dy * pad_frac, 1e-9)
    L, R, B, T = xmin - px, xmax + px, ymin - py, ymax + py
    L, R, B, T = _clamp_bbox(L, R, B, T, dem)

    if view_mode == "crop":
        return dem.crop(left=L, right=R, top=T, bottom=B, mode="coordinate"), (L, R, B, T)
    else:
        return dem, (L, R, B, T)

def _bbox_from_pair_masks(dem, s, dep_i, dep_j, confluence_id, pad_frac):
    U = np.asarray(dep_i.z | dep_j.z, dtype=bool)
    yy, xx = np.nonzero(U)
    r, c = _get_rc(s)
    xs_all, ys_all = s.transform * np.vstack((c, r))
    cx, cy = xs_all[confluence_id], ys_all[confluence_id]
    if yy.size == 0:
        eps = 1e-6
        return cx - eps, cx + eps, cy - eps, cy + eps
    Xs, Ys = s.transform * np.vstack((xx, yy))
    xmin, xmax = min(Xs.min(), cx), max(Xs.max(), cx)
    ymin, ymax = min(Ys.min(), cy), max(Ys.max(), cy)
    dx = max(xmax - xmin, 1e-9); dy = max(ymax - ymin, 1e-9)
    px, py = dx * pad_frac, dy * pad_frac
    return xmin - px, xmax + px, ymin - py, ymax + py

def _bbox_from_points(s, head_i, head_j, confluence_id, pad_frac):
    r, c = _get_rc(s)
    xs, ys = s.transform * np.vstack((c, r))
    xpts = np.array([xs[head_i], xs[head_j], xs[confluence_id]])
    ypts = np.array([ys[head_i], ys[head_j], ys[confluence_id]])
    xmin, xmax = xpts.min(), xpts.max()
    ymin, ymax = ypts.min(), ypts.max()
    dx = max(xmax - xmin, 1e-9); dy = max(ymax - ymin, 1e-9)
    px, py = dx * pad_frac, dy * pad_frac
    return xmin - px, xmax + px, ymin - py, ymax + py

# ---------------------------------------------------------------------
# 1. Single coupled pair
# ---------------------------------------------------------------------

def plot_coupled_pair(fd, s, dem, confluence_id, head_i, head_j,
                      view_mode="crop", pad_frac=0.05, alpha=0.35,
                      focus="points"):
    """
    Plot two coupled heads + confluence with unified view modes.
    view_mode: 'crop' (default) | 'zoom' | 'overview'
    focus: 'points' (tight box around heads+confluence) | 'masks' (union of dependence maps)
    """
    # seeds & dependence maps
    r, c = _get_rc(s)
    seed_i = dem.duplicate_with_new_data(np.zeros_like(dem.z, bool))
    seed_j = dem.duplicate_with_new_data(np.zeros_like(dem.z, bool))
    seed_i.z[int(r[head_i]), int(c[head_i])] = True
    seed_j.z[int(r[head_j]), int(c[head_j])] = True
    dep_i = fd.dependencemap(seed_i)
    dep_j = fd.dependencemap(seed_j)
    overlap = dep_i.z & dep_j.z

    # bbox
    if focus == "masks":
        L, R, B, T = _bbox_from_pair_masks(dem, s, dep_i, dep_j, confluence_id, pad_frac)
    else:  # 'points'
        L, R, B, T = _bbox_from_points(s, head_i, head_j, confluence_id, pad_frac)
    L, R, B, T = _clamp_bbox(L, R, B, T, dem)

    # DEM / overlay extent
    if view_mode == "crop":
        dem_used = dem.crop(left=L, right=R, top=T, bottom=B, mode="coordinate")
        overlay_extent = dem_used.extent
    else:
        dem_used = dem
        overlay_extent = dem.extent

    # draw
    fig, ax = plt.subplots(figsize=(8, 6))
    dem_used.plot(ax=ax, cmap="terrain", alpha=0.95)
    _plot_segments(ax, s.xy(), color="black", alpha=0.5)

    # overlays — use the chosen extent so crops look tight
    ax.imshow(dep_i.z, cmap="Blues",  alpha=alpha, extent=overlay_extent, origin="upper")
    ax.imshow(dep_j.z, cmap="Reds",   alpha=alpha, extent=overlay_extent, origin="upper")
    if np.any(overlap):
        ax.imshow(overlap, cmap="Purples", alpha=0.5, extent=overlay_extent, origin="upper")

    xs_all, ys_all = s.transform * np.vstack((c, r))
    ax.scatter(xs_all[head_i], ys_all[head_i], s=60, c="blue", edgecolor="k", zorder=5)
    ax.scatter(xs_all[head_j], ys_all[head_j], s=60, c="red",  edgecolor="k", zorder=5)
    ax.scatter(xs_all[confluence_id], ys_all[confluence_id], marker="*", s=140,
               edgecolor="k", facecolor="gold", zorder=6)

    if view_mode in ("crop", "zoom"):
        ax.set_xlim(L, R); ax.set_ylim(B, T)

    ax.set_aspect("equal", "box")
    ax.set_title(f"Heads {head_i}, {head_j} | Confluence {confluence_id} — {view_mode}, focus={focus}")
    plt.tight_layout()
    return fig, ax
# ---------------------------------------------------------------------
# 2. Outlet view (crop / zoom / overview)
# ---------------------------------------------------------------------

def plot_outlet_view(s, outlet_id, dem=None, by_outlet=None,
                     view_mode="crop", pad_frac=0.05):
    """Unified outlet visualization."""
    if by_outlet and outlet_id in by_outlet:
        s_up = by_outlet[outlet_id]["s_up"]
    else:
        om = s.streampoi("outlets")
        m = np.zeros_like(om, bool); m[outlet_id] = True
        s_up = s.upstreamto(m)

    dem_used, (L, R, B, T) = _maybe_crop_dem(dem, s_up, view_mode, pad_frac)
    fig, ax = plt.subplots(figsize=(7, 6))

    if dem is not None:
        dem_used.plot(ax=ax, cmap="terrain")

    _plot_segments(ax, s_up.xy())

    conf = s_up.streampoi("confluences")
    heads = s_up.streampoi("channelheads")
    outs = s_up.streampoi("outlets")
    xs, ys = _xy_all_nodes(s_up)
    ax.scatter(xs[outs], ys[outs], marker="*", s=160, edgecolor="k", facecolor="gold", zorder=5)
    ax.scatter(xs[conf], ys[conf], s=28, c="red", zorder=4)
    ax.scatter(xs[heads], ys[heads], s=24, c="limegreen", zorder=4)

    ax.set_xlim(L, R); ax.set_ylim(B, T)
    ax.set_aspect("equal", "box")
    ax.set_title(f"Outlet {outlet_id} ({view_mode})")
    plt.tight_layout()
    return fig, ax

# ---------------------------------------------------------------------
# 3. All coupled pairs overview (gold confluences)
# ---------------------------------------------------------------------

def plot_all_coupled_pairs_for_outlet(fd, s, dem, an, df_touching, outlet_id,
                                      view_mode="crop", pad_frac=0.05,
                                      alpha=0.25, max_pairs=None):
    """Plot all touching pairs for one outlet, with unified view modes."""
    df_o = df_touching[df_touching["outlet"] == outlet_id]
    if df_o.empty:
        print(f"No touching pairs for outlet {outlet_id}.")
        return None, None
    if max_pairs and len(df_o) > max_pairs:
        df_o = df_o.head(max_pairs)

    om = s.streampoi("outlets")
    m = np.zeros_like(om, bool); m[outlet_id] = True
    s_up = s.upstreamto(m)
    dem_used, (L, R, B, T) = _maybe_crop_dem(dem, s_up, view_mode, pad_frac)

    fig, ax = plt.subplots(figsize=(10, 8))
    dem_used.plot(ax=ax, cmap="terrain", alpha=0.95)
    _plot_segments(ax, s_up.xy(), color="k", alpha=0.5)

    r, c = _get_rc(s)
    xs, ys = s.transform * np.vstack((c, r))
    cmap = plt.cm.get_cmap("tab20", len(df_o))
    patches = []

    for idx, row in enumerate(df_o.itertuples(index=False)):
        h1, h2, conf = int(row.head_1), int(row.head_2), int(row.confluence)
        color = cmap(idx)
        G1, G2 = an.influence_grid(h1), an.influence_grid(h2)
        mask = np.asarray(G1.z | G2.z, bool)
        ax.imshow(np.ma.masked_where(~mask, mask),
                  cmap=plt.cm.colors.ListedColormap([color]),
                  alpha=alpha, extent=dem.extent)
        ax.scatter(xs[h1], ys[h1], c=[color], s=45, edgecolor="k", zorder=5)
        ax.scatter(xs[h2], ys[h2], c=[color], s=45, edgecolor="k", zorder=5)
        ax.scatter(xs[conf], ys[conf], marker="*", s=120, edgecolor="k",
                   facecolor="gold", zorder=6)
        patches.append(mpatches.Patch(color=color, label=f"Pair {idx+1} ({h1},{h2})"))

    out_mask = s_up.streampoi("outlets")
    xs_up, ys_up = _xy_all_nodes(s_up)
    ax.scatter(xs_up[out_mask], ys_up[out_mask], marker="*", s=200,
               edgecolor="k", facecolor="gold", zorder=6)

    ax.set_xlim(L, R); ax.set_ylim(B, T)
    ax.set_aspect("equal", "box")
    ax.legend(handles=patches, frameon=True, fontsize=8, loc="upper right")
    ax.set_title(f"Outlet {outlet_id} — {len(df_o)} coupled pairs ({view_mode})")
    plt.tight_layout()
    return fig, ax

def plot_all_coupled_pairs_for_outlet_3d(fd, s, dem, an, df_touching, outlet_id,
                                         view_mode="crop", pad_frac=0.05,
                                         alpha=0.25, max_pairs=None,
                                         dem_stride=2,
                                         surface_kwargs=None,
                                         stream_kwargs=None,
                                         z_exaggeration=1.0):
    """
    3D version: plot DEM surface + stream network + coupled channel heads,
    keeping the same inputs/behavior as your 2D function where possible.
    """
    if surface_kwargs is None:
        surface_kwargs = dict(cmap="terrain", linewidth=0, antialiased=True, alpha=0.95)
    if stream_kwargs is None:
        stream_kwargs = dict(color="k", linewidth=0.6, alpha=0.6)

    df_o = df_touching[df_touching["outlet"] == outlet_id]
    if df_o.empty:
        print(f"No touching pairs for outlet {outlet_id}.")
        return None, None
    if max_pairs and len(df_o) > max_pairs:
        df_o = df_o.head(max_pairs)

    # Upstream subnetwork for the chosen outlet
    om = s.streampoi("outlets")
    m = np.zeros_like(om, bool); m[outlet_id] = True
    s_up = s.upstreamto(m)

    # Crop DEM if requested
    dem_used, (L, R, B, T) = _maybe_crop_dem(dem, s_up, view_mode, pad_frac)

    # Prepare 3D figure
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # --- DEM surface (optionally subsampled for speed) ---
    X, Y = dem_used.coordinates
    Z = dem_used.z * z_exaggeration
    if dem_stride and dem_stride > 1:
        ax.plot_surface(X[::dem_stride, ::dem_stride],
                        Y[::dem_stride, ::dem_stride],
                        Z[::dem_stride, ::dem_stride],
                        **surface_kwargs)
    else:
        dem_used.plot_surface(ax=ax, **surface_kwargs)  # uses self.z internally (no exaggeration)
        if z_exaggeration != 1.0:
            # If you want strict exaggeration when not striding, re-plot with Z*factor as an overlay:
            ax.plot_surface(X, Y, Z, alpha=0.0)  # keeps axes z-scale consistent

    # --- Stream network as 3D polylines ---
    r_up, c_up = _get_rc(s_up)
    # World coords for nodes (same transform you used in 2D)
    xs_up, ys_up = s_up.transform * np.vstack((c_up, r_up))
    # Elevation at nodes (use full DEM to avoid indexing mismatch post-crop)
    z_nodes_up = dem.z[r_up, c_up] * z_exaggeration

    if hasattr(s_up, "source") and hasattr(s_up, "target"):
        for a, b in zip(s_up.source, s_up.target):
            ax.plot([xs_up[a], xs_up[b]],
                    [ys_up[a], ys_up[b]],
                    [z_nodes_up[a], z_nodes_up[b]],
                    **stream_kwargs)

    # --- Coupled heads, influence masks, and confluences ---
    cmap = plt.cm.get_cmap("tab20", len(df_o))
    patches = []

    for idx, row in enumerate(df_o.itertuples(index=False)):
        h1, h2, conf = int(row.head_1), int(row.head_2), int(row.confluence)
        color = cmap(idx)

        # Influence mask overlay as a semi-transparent surface
        G1, G2 = an.influence_grid(h1), an.influence_grid(h2)
        mask = np.asarray(G1.z | G2.z, bool)

        # Align mask to the (possibly) cropped DEM grid:
        # dem_used.z has same shape as X,Y here; mask must match that shape.
        # If influence_grid already matches the DEM grid, just mask outside crop:
        # Build a masked Z surface where only influence==True is plotted
        Z_mask = np.where(mask, dem.z, np.nan)  # base on full dem
        # Crop Z_mask to current view using dem_used's window indices:
        # dem_used carries a view of dem; safest is to rebuild via dem_used.z and mask reindexed.
        # If G1/G2 are same shape as dem_used.z, this is enough:
        try:
            Zm_local = np.where(mask, dem_used.z, np.nan) * z_exaggeration
        except ValueError:
            # Fallback: if mask is full-size while dem_used is cropped, rebuild a local mask from extent
            # (assumes nearest-neighbor semantics via extents)
            Zm_local = np.where(mask, dem.z, np.nan) * z_exaggeration  # may be a bit heavier

        ax.plot_surface(X, Y, Zm_local, color=color, alpha=alpha, linewidth=0)

        # Heads & confluence markers
        ax.scatter(xs_up[h1], ys_up[h1], z_nodes_up[h1],
                   color=color, s=70, edgecolor="k", depthshade=True)
        ax.scatter(xs_up[h2], ys_up[h2], z_nodes_up[h2],
                   color=color, s=70, edgecolor="k", depthshade=True)
        ax.scatter(xs_up[conf], ys_up[conf], z_nodes_up[conf],
                   marker="*", s=180, edgecolor="k", facecolor="gold")

        patches.append(mpatches.Patch(color=color, label=f"Pair {idx+1} ({h1},{h2})"))

    # --- Upstream outlet marker ---
    out_mask_up = s_up.streampoi("outlets")
    ax.scatter(xs_up[out_mask_up], ys_up[out_mask_up], z_nodes_up[out_mask_up],
               marker="*", s=260, edgecolor="k", facecolor="gold")

    # Same spatial framing you used in 2D
    ax.set_xlim(L, R)
    ax.set_ylim(B, T)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.view_init(elev=35, azim=-65)
    ax.set_title(f"Outlet {outlet_id} — {len(df_o)} coupled pairs (3D, {view_mode})")
    ax.legend(handles=patches, frameon=True, fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig, ax
