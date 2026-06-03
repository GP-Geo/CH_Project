"""Mars CNN raster patch generation (Phase 4) — 5-class, 128x128, uint8.

Moved from ``scripts/build_mars_cnn_patches_5class.py``. Produces one patch per
Mars model-ready pair, matching the Earth CNN pipeline exactly so that
``models/cnn_outlet_final.pt`` runs on Mars patches **without retraining**.

Frozen contract (must match Earth ``channel_heads/rasterizer.py``):

* 128x128 uint8, 5 classes — BACKGROUND=0, BRANCH_A=1, BRANCH_B=2,
  OTHER_STREAMS=3, CONFLUENCE_MARKER=4.
* confluence-centred, heads-up rotation (head midpoint to image top).
* tight bbox over branch-A ∪ branch-B after rotation, padding_frac=0.2.
* draw directly into the final 128x128 grid (nearest-neighbour), augment OFF,
  no normalization, branch-A/B mutual protection.

Earth works in DEM-pixel space; Mars geometry is in MOLA-projected metres, so
coordinates are scaled by ``MOLA_CELL_SIZE_M`` (200 m/px) before the shared
rotate/crop/scale/draw algorithm.

Public entry points: :func:`render_pair_patch` (one patch) and
:func:`build_mars_cnn_patches` (full Phase-4 run + manifest).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.io import paths
from channel_heads.logging_config import get_logger
from channel_heads.rasterization.drawing import (
    compute_rotation_angle,
    draw_polyline,
    linestring_to_rc,
    rotate_rc,
    xy_to_rc,
)
from channel_heads.rasterization.manifest import (
    build_patch_manifest,
    empty_flags,
    validate_patch_manifest,
)
from channel_heads.rasterization.schema import (
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    OTHER_STREAMS,
)
from channel_heads.rasterization.earth_patches import raster_quality_flags

log = get_logger("rasterization.mars_patches")

# Patch spec — must match Earth.
TARGET_SIZE = 128
PADDING_FRAC = 0.2
PATCH_ENCODING = "earth_5class"
MOLA_CELL_SIZE_M = 200.0

# QA contact-sheet selection.
QA_RANDOM_N = 30
QA_HIGH_CONF_N = 30
QA_UNCERTAIN_N = 30
QA_HIGH_CONF_PROB_MIN = 0.80
QA_UNCERTAIN_PROB_MIN = 0.45
QA_UNCERTAIN_PROB_MAX = 0.70
QA_SEED = 42

CLASS_COLORS: dict[int, tuple[float, float, float]] = {
    BACKGROUND: (0.95, 0.95, 0.95),
    BRANCH_A: (0.85, 0.33, 0.10),
    BRANCH_B: (0.10, 0.45, 0.82),
    OTHER_STREAMS: (0.70, 0.70, 0.70),
    CONFLUENCE_MARKER: (0.90, 0.80, 0.00),
}


# --------------------------------------------------------------------------- #
# Core rasterizer
# --------------------------------------------------------------------------- #
def render_pair_patch(
    h1_xy: tuple[float, float],
    h2_xy: tuple[float, float],
    conf_xy: tuple[float, float],
    path_a,
    path_b,
    network_segments: Iterable,
    cell_size_m: float = MOLA_CELL_SIZE_M,
    target_size: int = TARGET_SIZE,
    padding_frac: float = PADDING_FRAC,
) -> np.ndarray:
    """Produce a 128x128 uint8 5-class patch for one Mars pair.

    Mirrors ``channel_heads.rasterizer.rasterize_outlet_pair``.
    """
    h1_rc = xy_to_rc(*h1_xy, cell_size_m=cell_size_m)
    h2_rc = xy_to_rc(*h2_xy, cell_size_m=cell_size_m)
    conf_rc = xy_to_rc(*conf_xy, cell_size_m=cell_size_m)

    angle = compute_rotation_angle(h1_rc, h2_rc, conf_rc)

    rs_a, cs_a = linestring_to_rc(path_a, cell_size_m)
    rs_b, cs_b = linestring_to_rc(path_b, cell_size_m)
    rot_rs_a, rot_cs_a = rotate_rc(rs_a, cs_a, conf_rc[0], conf_rc[1], angle)
    rot_rs_b, rot_cs_b = rotate_rc(rs_b, cs_b, conf_rc[0], conf_rc[1], angle)

    all_r = np.concatenate([rot_rs_a, rot_rs_b])
    all_c = np.concatenate([rot_cs_a, rot_cs_b])
    r_min, r_max = float(all_r.min()), float(all_r.max())
    c_min, c_max = float(all_c.min()), float(all_c.max())
    r_span = max(r_max - r_min, 2.0)
    c_span = max(c_max - c_min, 2.0)
    pad_r = r_span * padding_frac
    pad_c = c_span * padding_frac
    r_min -= pad_r
    r_max += pad_r
    c_min -= pad_c
    c_max += pad_c

    r_span = max(float(r_max - r_min), 1e-9)
    c_span = max(float(c_max - c_min), 1e-9)

    def project_to_target(rs, cs):
        out_r = (rs - r_min) / r_span * (target_size - 1)
        out_c = (cs - c_min) / c_span * (target_size - 1)
        return out_r, out_c

    raster = np.zeros((target_size, target_size), dtype=np.uint8)

    # 1. Other streams of the same network.
    for seg in network_segments:
        rs_o, cs_o = linestring_to_rc(seg, cell_size_m)
        rot_rs_o, rot_cs_o = rotate_rc(rs_o, cs_o, conf_rc[0], conf_rc[1], angle)
        out_rs_o, out_cs_o = project_to_target(rot_rs_o, rot_cs_o)
        draw_polyline(raster, out_rs_o, out_cs_o, 0.0, 0.0, OTHER_STREAMS)

    # 2/3. Branch A then B, with mutual protection.
    out_rs_a, out_cs_a = project_to_target(rot_rs_a, rot_cs_a)
    out_rs_b, out_cs_b = project_to_target(rot_rs_b, rot_cs_b)
    draw_polyline(raster, out_rs_a, out_cs_a, 0.0, 0.0, BRANCH_A, protect=(BRANCH_B,))
    draw_polyline(raster, out_rs_b, out_cs_b, 0.0, 0.0, BRANCH_B, protect=(BRANCH_A,))

    # 4. Confluence marker (confluence is the rotation centre).
    conf_out_r, conf_out_c = project_to_target(
        np.array([conf_rc[0]], dtype=float), np.array([conf_rc[1]], dtype=float)
    )
    cr = int(round(float(conf_out_r[0])))
    cc = int(round(float(conf_out_c[0])))
    if 0 <= cr < target_size and 0 <= cc < target_size:
        raster[cr, cc] = CONFLUENCE_MARKER

    return raster


# Backwards-compatible alias (Earth-parallel name used by tests/old callers).
rasterize_mars_pair = render_pair_patch


# --------------------------------------------------------------------------- #
# Visualisation (best-effort)
# --------------------------------------------------------------------------- #
def patch_to_rgb(raster: np.ndarray) -> np.ndarray:
    h, w = raster.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for cls_val, color in CLASS_COLORS.items():
        mask = raster == cls_val
        for ch in range(3):
            rgb[:, :, ch][mask] = color[ch]
    return rgb


def render_contact_sheet(patches: list[tuple[np.ndarray, str]], title: str, output: Path, cols: int = 6) -> None:
    import matplotlib.pyplot as plt

    if not patches:
        log.warning("No patches to render for %s — skipping", output.name)
        return
    n = len(patches)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (patch, subtitle) in zip(axes_flat, patches):
        ax.imshow(patch_to_rgb(patch), interpolation="nearest")
        ax.set_title(subtitle, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_qa_sheets(idx_df, output_dir, predictions_parquet) -> None:
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(QA_SEED)
    ok_idx = idx_df[idx_df["patch_status"] == "ok"].reset_index(drop=True)
    if ok_idx.empty:
        return

    def load_patch(rel_path):
        return np.load(paths.PROJECT_ROOT / rel_path)

    n_random = min(QA_RANDOM_N, len(ok_idx))
    rand_pick = ok_idx.sample(n=n_random, random_state=int(rng.integers(0, 2**31 - 1)))
    render_contact_sheet(
        [(load_patch(r["patch_path"]), f"net={r['network_id']}\n{r['pair_id']}") for _, r in rand_pick.iterrows()],
        title=f"Random {n_random} Mars CNN patches (5-class, 128x128)",
        output=figures_dir / "contact_sheet_random_30.png",
    )

    predictions_parquet = Path(predictions_parquet)
    if not predictions_parquet.exists():
        log.warning("Predictions not found (%s); skipping prediction-driven sheets", predictions_parquet)
        return
    preds = pd.read_parquet(predictions_parquet)[["pair_id", "xgb_prob_touching", "xgb_pred_touching"]]
    merged = ok_idx.merge(preds, on="pair_id", how="left")
    hc = merged[merged["xgb_prob_touching"] >= QA_HIGH_CONF_PROB_MIN].sort_values(
        "xgb_prob_touching", ascending=False
    ).head(QA_HIGH_CONF_N)
    render_contact_sheet(
        [(load_patch(r["patch_path"]), f"net={r['network_id']}  p={r['xgb_prob_touching']:.3f}") for _, r in hc.iterrows()],
        title=f"{len(hc)} high-confidence touching patches (prob >= {QA_HIGH_CONF_PROB_MIN:.2f})",
        output=figures_dir / "contact_sheet_high_conf_touching_30.png",
    )
    unc_pool = merged[
        (merged["xgb_prob_touching"] >= QA_UNCERTAIN_PROB_MIN)
        & (merged["xgb_prob_touching"] <= QA_UNCERTAIN_PROB_MAX)
    ].copy()
    if len(unc_pool) > QA_UNCERTAIN_N:
        unc_pool = unc_pool.sort_values("xgb_prob_touching")
        idxs = np.linspace(0, len(unc_pool) - 1, QA_UNCERTAIN_N).round().astype(int)
        unc = unc_pool.iloc[idxs]
    else:
        unc = unc_pool.sort_values("xgb_prob_touching")
    render_contact_sheet(
        [(load_patch(r["patch_path"]), f"net={r['network_id']}  p={r['xgb_prob_touching']:.3f}") for _, r in unc.iterrows()],
        title=f"{len(unc)} uncertain patches (prob in [{QA_UNCERTAIN_PROB_MIN:.2f}, {QA_UNCERTAIN_PROB_MAX:.2f}])",
        output=figures_dir / "contact_sheet_uncertain_30.png",
    )


# --------------------------------------------------------------------------- #
# High-level entry point
# --------------------------------------------------------------------------- #
def build_mars_cnn_patches(
    features_parquet=paths.MARS_MODEL_INPUTS_DIR / "mars_pair_features_5feat_model_ready.parquet",
    pairs_gpkg=paths.MARS_PAIRS_GPKG,
    topology_gpkg=paths.MARS_TOPOLOGY_GPKG,
    output_dir=paths.MARS_CNN_PATCHES_DIR,
    index_parquet=paths.MARS_CNN_PATCH_INDEX,
    predictions_parquet=paths.MARS_MODEL_OUTPUTS_DIR / "mars_xgb_predictions_5feat.parquet",
    *,
    write: bool = True,
    make_figures: bool = True,
) -> dict:
    """Phase 4: rasterize one 5-class patch per model-ready pair + write manifest.

    Returns ``{"manifest", "n_ok", "n_invalid", "n_failed", "paths"}``.
    """
    from channel_heads.io import read_gpkg, write_table

    output_dir = Path(output_dir)
    df_pairs = pd.read_parquet(features_parquet)
    paths_gdf = read_gpkg(pairs_gpkg, layer="mars_pair_paths")
    nodes_gdf = read_gpkg(topology_gpkg, layer="mars_nodes")
    segments_gdf = read_gpkg(topology_gpkg, layer="mars_segments")

    nodes_by_nid = dict(tuple(nodes_gdf.groupby("network_id")))
    segs_by_nid = dict(tuple(segments_gdf.groupby("network_id")))
    pair_paths_lookup: dict[tuple[str, str], object] = {}
    for _, r in paths_gdf.iterrows():
        pair_paths_lookup[(str(r["pair_id"]), str(r["branch"]))] = r.geometry

    log.info("Input model-ready pairs: %d (%d networks)", len(df_pairs), df_pairs["network_id"].nunique())

    index_rows: list[dict] = []
    n_ok = n_invalid = n_fail = 0

    for _, row in df_pairs.iterrows():
        nid = int(row["network_id"])
        pair_id = str(row["pair_id"])
        h1 = int(row["head_node_id_1"])
        h2 = int(row["head_node_id_2"])
        conf = int(row["confluence_node_id"])

        nodes_n = nodes_by_nid.get(nid)
        segs_n = segs_by_nid.get(nid)
        path_a = pair_paths_lookup.get((pair_id, "A"))
        path_b = pair_paths_lookup.get((pair_id, "B"))
        status, qa_reason = "ok", ""
        patch_flags = empty_flags()
        node_xy: dict[int, tuple[float, float]] = {}

        if nodes_n is None or segs_n is None:
            status, qa_reason = "skipped", "missing_network_topology"
        elif path_a is None or path_b is None:
            status, qa_reason = "skipped", "missing_pair_path"
        else:
            node_xy = {
                int(r["node_id"]): (float(r.geometry.x), float(r.geometry.y))
                for _, r in nodes_n.iterrows()
            }
            if h1 not in node_xy or h2 not in node_xy or conf not in node_xy:
                status, qa_reason = "skipped", "missing_node_coord"

        patch_path: Path | None = None
        if status == "ok":
            try:
                patch = render_pair_patch(
                    h1_xy=node_xy[h1], h2_xy=node_xy[h2], conf_xy=node_xy[conf],
                    path_a=path_a, path_b=path_b, network_segments=segs_n.geometry,
                )
                if write:
                    patch_path = output_dir / str(nid) / f"{pair_id}.npy"
                    patch_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(patch_path, patch)
                patch_flags = raster_quality_flags(patch)
                if patch_flags["branches_connected"]:
                    n_ok += 1
                else:
                    status = "invalid"
                    qa_reason = "qa_failed:" + ",".join(k for k, v in patch_flags.items() if not v)
                    n_invalid += 1
            except Exception as exc:  # recorded in the manifest
                status = "failed"
                qa_reason = f"rasterizer_exception:{type(exc).__name__}:{exc}"
                n_fail += 1

        index_rows.append(
            {
                "network_id": nid,
                "pair_id": pair_id,
                "head_id_1": int(row["head_id_1"]),
                "head_id_2": int(row["head_id_2"]),
                "head_node_id_1": h1,
                "head_node_id_2": h2,
                "confluence_id": int(row["confluence_id"]),
                "confluence_node_id": conf,
                "patch_path": (str(patch_path.relative_to(paths.PROJECT_ROOT)) if patch_path else ""),
                "patch_shape": f"{TARGET_SIZE},{TARGET_SIZE}",
                "patch_dtype": "uint8",
                "patch_encoding": PATCH_ENCODING,
                "patch_status": status,
                "patch_qa_reason": qa_reason,
                **patch_flags,
            }
        )

    idx_df = build_patch_manifest(index_rows)
    validate_patch_manifest(idx_df, target_size=TARGET_SIZE)
    log.info("Patches: ok=%d invalid=%d failed=%d total=%d", n_ok, n_invalid, n_fail, len(idx_df))

    written: dict[str, Path] = {}
    if write:
        written["index"] = write_table(idx_df, index_parquet)
        if make_figures:
            try:
                _render_qa_sheets(idx_df, output_dir, predictions_parquet)
            except Exception as exc:  # diagnostics only
                log.warning("QA contact sheets failed: %s", exc)

    return {
        "manifest": idx_df,
        "n_ok": n_ok,
        "n_invalid": n_invalid,
        "n_failed": n_fail,
        "paths": written,
    }


__all__ = [
    "render_pair_patch",
    "rasterize_mars_pair",
    "build_mars_cnn_patches",
    "patch_to_rgb",
    "render_contact_sheet",
    "TARGET_SIZE",
    "PADDING_FRAC",
    "MOLA_CELL_SIZE_M",
    "PATCH_ENCODING",
    "raster_quality_flags",
    "BACKGROUND",
    "BRANCH_A",
    "BRANCH_B",
    "OTHER_STREAMS",
    "CONFLUENCE_MARKER",
]
