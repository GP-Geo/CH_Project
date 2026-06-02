#!/usr/bin/env python
"""Phase 4 — Generate Mars CNN raster patches (5-class, 128×128, uint8).

Produces one patch per Mars model-ready pair, matching the Earth CNN
pipeline exactly so that ``models/cnn_outlet_final.pt`` can run inference
on Mars patches without retraining.

Spec reproduced from ``channel_heads/rasterizer.py``:

  - 128 × 128, uint8
  - 5 classes:
      0 BACKGROUND       (non-stream pixels)
      1 BRANCH_A         (head_1 → confluence path)
      2 BRANCH_B         (head_2 → confluence path)
      3 OTHER_STREAMS    (all other channels of the same network)
      4 CONFLUENCE_MARKER (single pixel marker at the confluence)
  - rotation: confluence at the bottom of the image, midpoint of the
    two heads at the top
  - crop bounding box: tight bbox over the union of branch-A + branch-B
    after rotation, expanded by padding_frac = 0.2 on each side
  - final draw: rotate/crop/scale coordinates, then draw directly into 128 × 128
  - augment OFF (inference)
  - no per-pixel normalization

Earth operates in DEM-pixel space. Mars geometry is in MOLA-projected metres,
so this script converts metres to "MOLA-pixel" units using
``MOLA_CELL_SIZE_M`` before applying Earth's rotate/crop/scale/draw algorithm.
Drawing happens in the final 128×128 grid so branch connectivity and the
confluence marker do not depend on a later downsampling step.

Outputs:
  data/Mars/model_inputs/cnn_patches_5class/{network_id}/{pair_id}.npy
  data/Mars/model_inputs/mars_cnn_patch_index.parquet
  data/Mars/model_inputs/mars_cnn_patch_index.csv
  data/Mars/model_inputs/cnn_patches_5class/figures/
      contact_sheet_random_30.png
      contact_sheet_high_conf_touching_30.png
      contact_sheet_uncertain_30.png

Phase 4 only — no CNN embeddings, no retraining, no XGBoost rerun.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

# Reuse Earth class constants so any future change in the rasterizer
# propagates here automatically.
from channel_heads.rasterizer import (  # noqa: E402
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    OTHER_STREAMS,
    bresenham_line,
    raster_quality_flags,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_inputs/mars_pair_features_5feat_model_ready.parquet"
)
PREDICTIONS_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_outputs/mars_xgb_predictions_5feat.parquet"
)
PAIRS_GPKG = PROJECT_ROOT / "data/Mars/topology/mars_vn_pairs.gpkg"
TOPOLOGY_GPKG = (
    PROJECT_ROOT
    / "data/Mars/topology/mars_vn_topology_model_ready.gpkg"
)

OUTPUT_DIR = PROJECT_ROOT / "data/Mars/model_inputs/cnn_patches_5class"
FIGURES_DIR = OUTPUT_DIR / "figures"
INDEX_PARQUET = (
    PROJECT_ROOT / "data/Mars/model_inputs/mars_cnn_patch_index.parquet"
)
INDEX_CSV = (
    PROJECT_ROOT / "data/Mars/model_inputs/mars_cnn_patch_index.csv"
)

# Patch spec — must match Earth (channel_heads/rasterizer.py)
TARGET_SIZE = 128
PADDING_FRAC = 0.2
PATCH_ENCODING = "earth_5class"

# Mars-specific: temp-grid scale. 200 m/px is the MOLA resolution and
# matches the natural sampling of the Phase-1 reprojected DEM. Choosing
# this value puts a Mars pair at the same temp-pixel resolution that an
# Earth pair would be at on a 200 m DEM.
MOLA_CELL_SIZE_M = 200.0

# QA contact-sheet selection
QA_RANDOM_N = 30
QA_HIGH_CONF_N = 30
QA_UNCERTAIN_N = 30
QA_HIGH_CONF_PROB_MIN = 0.80
QA_UNCERTAIN_PROB_MIN = 0.45
QA_UNCERTAIN_PROB_MAX = 0.70
QA_SEED = 42

# Class colour map (mirrors notebooks/training/04_cnn_embeddings.ipynb cell 7)
CLASS_COLORS: dict[int, tuple[float, float, float]] = {
    BACKGROUND: (0.95, 0.95, 0.95),
    BRANCH_A: (0.85, 0.33, 0.10),
    BRANCH_B: (0.10, 0.45, 0.82),
    OTHER_STREAMS: (0.70, 0.70, 0.70),
    CONFLUENCE_MARKER: (0.90, 0.80, 0.00),
}

log = logging.getLogger("mars_cnn_patches")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def xy_to_rc(x_m: float, y_m: float, cell_size_m: float) -> tuple[float, float]:
    """Mars projected metres → (row, col) in MOLA-pixel image coords.

    Row increases downward in image space, so we negate the northing.
    """
    return (-y_m / cell_size_m, x_m / cell_size_m)


def linestring_to_rc(
    line: LineString, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(line.coords, dtype=float)
    rs = -coords[:, 1] / cell_size_m
    cs = coords[:, 0] / cell_size_m
    return rs, cs


def compute_rotation_angle(
    h1_rc: tuple[float, float],
    h2_rc: tuple[float, float],
    conf_rc: tuple[float, float],
) -> float:
    """Port of channel_heads/rasterizer.py:_compute_rotation_angle."""
    mid_r = (h1_rc[0] + h2_rc[0]) / 2.0
    mid_c = (h1_rc[1] + h2_rc[1]) / 2.0
    dr = mid_r - conf_rc[0]
    dc = mid_c - conf_rc[1]
    if abs(dr) < 1e-9 and abs(dc) < 1e-9:
        return 0.0
    current_angle = math.atan2(dc, dr)
    target_angle = math.pi  # pointing in -row direction (image up)
    return target_angle - current_angle


def rotate_rc(
    rs: np.ndarray,
    cs: np.ndarray,
    cr: float,
    cc: float,
    angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of channel_heads/rasterizer.py:_rotate_coordinates."""
    dr = rs - cr
    dc = cs - cc
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_r = dr * cos_a - dc * sin_a + cr
    new_c = dr * sin_a + dc * cos_a + cc
    return new_r, new_c


# ---------------------------------------------------------------------------
# Polyline drawing helpers (port of _draw_path_on_raster / _draw_edges_on_raster)
# ---------------------------------------------------------------------------
def draw_polyline(
    raster: np.ndarray,
    rs: np.ndarray,
    cs: np.ndarray,
    r_min: float,
    c_min: float,
    value: int,
    protect: tuple[int, ...] = (),
) -> None:
    """Draw a Mars polyline (one segment's chained LineString) onto raster.

    Each consecutive vertex pair becomes one Bresenham line. Vertex
    pixels are always written; interior pixels are protected from
    overwriting the values listed in ``protect`` (mirrors Earth's
    branch-A vs branch-B protection in ``_draw_path_on_raster``).
    """
    H, W = raster.shape
    if len(rs) < 2:
        # single point: just mark it
        ir = int(round(rs[0] - r_min))
        ic = int(round(cs[0] - c_min))
        if 0 <= ir < H and 0 <= ic < W:
            raster[ir, ic] = value
        return

    # Pre-compute integer vertex positions and collect the vertex set.
    irs = np.rint(rs - r_min).astype(int)
    ics = np.rint(cs - c_min).astype(int)
    vertex_set: set[tuple[int, int]] = set()
    for ir, ic in zip(irs, ics):
        if 0 <= ir < H and 0 <= ic < W:
            vertex_set.add((int(ir), int(ic)))

    for i in range(len(rs) - 1):
        r0, c0 = int(irs[i]), int(ics[i])
        r1, c1 = int(irs[i + 1]), int(ics[i + 1])
        for lr, lc in bresenham_line(r0, c0, r1, c1):
            if 0 <= lr < H and 0 <= lc < W:
                is_vertex = (lr, lc) in vertex_set
                if is_vertex or raster[lr, lc] not in protect:
                    raster[lr, lc] = value


# ---------------------------------------------------------------------------
# Core Mars rasterizer
# ---------------------------------------------------------------------------
def rasterize_mars_pair(
    h1_xy: tuple[float, float],
    h2_xy: tuple[float, float],
    conf_xy: tuple[float, float],
    path_a: LineString,
    path_b: LineString,
    network_segments: Iterable[LineString],
    cell_size_m: float = MOLA_CELL_SIZE_M,
    target_size: int = TARGET_SIZE,
    padding_frac: float = PADDING_FRAC,
) -> np.ndarray:
    """Produce a 128×128 uint8 raster patch for one Mars pair.

    Pipeline mirrors channel_heads/rasterizer.py:rasterize_outlet_pair.
    """
    # Convert head/confluence to (r, c) image coords (in MOLA-pixel units)
    h1_rc = xy_to_rc(*h1_xy, cell_size_m=cell_size_m)
    h2_rc = xy_to_rc(*h2_xy, cell_size_m=cell_size_m)
    conf_rc = xy_to_rc(*conf_xy, cell_size_m=cell_size_m)

    angle = compute_rotation_angle(h1_rc, h2_rc, conf_rc)

    rs_a, cs_a = linestring_to_rc(path_a, cell_size_m)
    rs_b, cs_b = linestring_to_rc(path_b, cell_size_m)
    rot_rs_a, rot_cs_a = rotate_rc(rs_a, cs_a, conf_rc[0], conf_rc[1], angle)
    rot_rs_b, rot_cs_b = rotate_rc(rs_b, cs_b, conf_rc[0], conf_rc[1], angle)

    # Bounding box over branch-A ∪ branch-B (rotated), with padding
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

    def project_to_target(rs: np.ndarray, cs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out_r = (rs - r_min) / r_span * (target_size - 1)
        out_c = (cs - c_min) / c_span * (target_size - 1)
        return out_r, out_c

    raster = np.zeros((target_size, target_size), dtype=np.uint8)

    # 1. Draw all OTHER STREAMS in the same network first (rotated)
    for seg in network_segments:
        rs_o, cs_o = linestring_to_rc(seg, cell_size_m)
        rot_rs_o, rot_cs_o = rotate_rc(rs_o, cs_o, conf_rc[0], conf_rc[1], angle)
        out_rs_o, out_cs_o = project_to_target(rot_rs_o, rot_cs_o)
        draw_polyline(raster, out_rs_o, out_cs_o, 0.0, 0.0, OTHER_STREAMS)

    # 2. Draw branch A (protect B from being overwritten by A's interpolation)
    out_rs_a, out_cs_a = project_to_target(rot_rs_a, rot_cs_a)
    out_rs_b, out_cs_b = project_to_target(rot_rs_b, rot_cs_b)
    draw_polyline(
        raster, out_rs_a, out_cs_a, 0.0, 0.0, BRANCH_A, protect=(BRANCH_B,)
    )
    # 3. Draw branch B (protect A symmetrically)
    draw_polyline(
        raster, out_rs_b, out_cs_b, 0.0, 0.0, BRANCH_B, protect=(BRANCH_A,)
    )

    # 4. Confluence marker — confluence is the rotation center so its
    #    rotated position equals its original (r, c).
    conf_out_r, conf_out_c = project_to_target(
        np.array([conf_rc[0]], dtype=float),
        np.array([conf_rc[1]], dtype=float),
    )
    cr = int(round(float(conf_out_r[0])))
    cc = int(round(float(conf_out_c[0])))
    if 0 <= cr < target_size and 0 <= cc < target_size:
        raster[cr, cc] = CONFLUENCE_MARKER

    return raster


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def patch_to_rgb(raster: np.ndarray) -> np.ndarray:
    h, w = raster.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for cls_val, color in CLASS_COLORS.items():
        mask = raster == cls_val
        for ch in range(3):
            rgb[:, :, ch][mask] = color[ch]
    return rgb


def render_contact_sheet(
    patches: list[tuple[np.ndarray, str]],
    title: str,
    output: Path,
    cols: int = 6,
) -> None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ---------------------------------------------------
    log.info("Loading model-ready features: %s", FEATURES_PARQUET)
    df_pairs = pd.read_parquet(FEATURES_PARQUET)
    log.info("Loading Phase 2B pairs/paths: %s", PAIRS_GPKG)
    paths_gdf = gpd.read_file(PAIRS_GPKG, layer="mars_pair_paths")
    log.info("Loading topology nodes + segments: %s", TOPOLOGY_GPKG)
    nodes_gdf = gpd.read_file(TOPOLOGY_GPKG, layer="mars_nodes")
    segments_gdf = gpd.read_file(TOPOLOGY_GPKG, layer="mars_segments")

    # --- Lookups -------------------------------------------------------
    nodes_by_nid = dict(tuple(nodes_gdf.groupby("network_id")))
    segs_by_nid = dict(tuple(segments_gdf.groupby("network_id")))
    pair_paths_lookup: dict[tuple[str, str], LineString] = {}
    for _, r in paths_gdf.iterrows():
        pair_paths_lookup[(str(r["pair_id"]), str(r["branch"]))] = r.geometry

    log.info(
        "Input model-ready pairs: %d (across %d networks)",
        len(df_pairs),
        df_pairs["network_id"].nunique(),
    )

    # --- Generate patches ---------------------------------------------
    index_rows: list[dict] = []
    n_ok = 0
    n_invalid = 0
    n_fail = 0

    # Cache rotated network polylines per (network_id, conf, angle)? Not
    # worth the complexity — each pair has its own confluence center and
    # angle. We do cache per-network nodes_xy and segments to avoid
    # re-grouping the GeoDataFrame on every pair.

    for _, row in tqdm(
        df_pairs.iterrows(), total=len(df_pairs), desc="Rasterizing"
    ):
        nid = int(row["network_id"])
        pair_id = str(row["pair_id"])
        h1 = int(row["head_node_id_1"])
        h2 = int(row["head_node_id_2"])
        conf = int(row["confluence_node_id"])

        nodes_n = nodes_by_nid.get(nid)
        segs_n = segs_by_nid.get(nid)
        path_a = pair_paths_lookup.get((pair_id, "A"))
        path_b = pair_paths_lookup.get((pair_id, "B"))
        status = "ok"
        qa_reason = ""
        patch_flags = {
            "has_branch_a": False,
            "has_branch_b": False,
            "has_confluence": False,
            "branch_a_connected": False,
            "branch_b_connected": False,
            "branches_connected": False,
        }

        if nodes_n is None or segs_n is None:
            status = "skipped"
            qa_reason = "missing_network_topology"
        elif path_a is None or path_b is None:
            status = "skipped"
            qa_reason = "missing_pair_path"
        else:
            node_xy = {
                int(r["node_id"]): (float(r.geometry.x), float(r.geometry.y))
                for _, r in nodes_n.iterrows()
            }
            if h1 not in node_xy or h2 not in node_xy or conf not in node_xy:
                status = "skipped"
                qa_reason = "missing_node_coord"

        patch_path: Path | None = None
        if status == "ok":
            try:
                patch = rasterize_mars_pair(
                    h1_xy=node_xy[h1],
                    h2_xy=node_xy[h2],
                    conf_xy=node_xy[conf],
                    path_a=path_a,
                    path_b=path_b,
                    network_segments=segs_n.geometry,
                )
                patch_path = OUTPUT_DIR / str(nid) / f"{pair_id}.npy"
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(patch_path, patch)
                patch_flags = raster_quality_flags(patch)
                if patch_flags["branches_connected"]:
                    n_ok += 1
                else:
                    status = "invalid"
                    failed_flags = [key for key, value in patch_flags.items() if not value]
                    qa_reason = "qa_failed:" + ",".join(failed_flags)
                    n_invalid += 1
            except Exception as exc:  # narrow exception caught in QA reason
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
                "patch_path": (
                    str(patch_path.relative_to(PROJECT_ROOT))
                    if patch_path is not None
                    else ""
                ),
                "patch_shape": f"{TARGET_SIZE},{TARGET_SIZE}",
                "patch_dtype": "uint8",
                "patch_encoding": PATCH_ENCODING,
                "patch_status": status,
                "patch_qa_reason": qa_reason,
                **patch_flags,
            }
        )

    log.info(
        "Patches: created=%d, failed=%d, total=%d",
        n_ok,
        n_fail,
        len(index_rows),
    )
    if n_invalid:
        log.warning("Invalid patches failing structural QA: %d", n_invalid)

    # --- Write index --------------------------------------------------
    idx_df = pd.DataFrame(index_rows)
    idx_df.to_parquet(INDEX_PARQUET, index=False)
    idx_df.to_csv(INDEX_CSV, index=False)
    log.info("Wrote patch index: %s", INDEX_PARQUET)
    log.info("Wrote patch index: %s", INDEX_CSV)
    log.info(
        "Patch status counts: %s",
        idx_df["patch_status"].value_counts().to_dict(),
    )

    # --- QA contact sheets --------------------------------------------
    rng = np.random.default_rng(QA_SEED)
    ok_idx = idx_df[idx_df["patch_status"] == "ok"].reset_index(drop=True)

    # Random sample
    n_random = min(QA_RANDOM_N, len(ok_idx))
    rand_pick = ok_idx.sample(
        n=n_random, random_state=int(rng.integers(0, 2**31 - 1))
    )

    def load_patch(rel_path: str) -> np.ndarray:
        return np.load(PROJECT_ROOT / rel_path)

    rand_patches = [
        (
            load_patch(r["patch_path"]),
            f"net={r['network_id']}\n{r['pair_id']}",
        )
        for _, r in rand_pick.iterrows()
    ]
    render_contact_sheet(
        rand_patches,
        title=f"Random {n_random} Mars CNN patches (5-class, 128×128)",
        output=FIGURES_DIR / "contact_sheet_random_30.png",
    )

    # Predictions-driven samples
    if PREDICTIONS_PARQUET.exists():
        preds = pd.read_parquet(PREDICTIONS_PARQUET)[
            ["pair_id", "xgb_prob_touching", "xgb_pred_touching"]
        ]
        merged = ok_idx.merge(preds, on="pair_id", how="left")

        hc = (
            merged[merged["xgb_prob_touching"] >= QA_HIGH_CONF_PROB_MIN]
            .sort_values("xgb_prob_touching", ascending=False)
            .head(QA_HIGH_CONF_N)
        )
        hc_patches = [
            (
                load_patch(r["patch_path"]),
                f"net={r['network_id']}  p={r['xgb_prob_touching']:.3f}",
            )
            for _, r in hc.iterrows()
        ]
        render_contact_sheet(
            hc_patches,
            title=(
                f"{len(hc_patches)} high-confidence touching patches "
                f"(prob ≥ {QA_HIGH_CONF_PROB_MIN:.2f})"
            ),
            output=FIGURES_DIR / "contact_sheet_high_conf_touching_30.png",
        )

        unc_pool = merged[
            (merged["xgb_prob_touching"] >= QA_UNCERTAIN_PROB_MIN)
            & (merged["xgb_prob_touching"] <= QA_UNCERTAIN_PROB_MAX)
        ].copy()
        if len(unc_pool) > QA_UNCERTAIN_N:
            unc_pool = unc_pool.sort_values("xgb_prob_touching")
            idxs = np.linspace(
                0, len(unc_pool) - 1, QA_UNCERTAIN_N
            ).round().astype(int)
            unc = unc_pool.iloc[idxs]
        else:
            unc = unc_pool.sort_values("xgb_prob_touching")
        unc_patches = [
            (
                load_patch(r["patch_path"]),
                f"net={r['network_id']}  p={r['xgb_prob_touching']:.3f}",
            )
            for _, r in unc.iterrows()
        ]
        render_contact_sheet(
            unc_patches,
            title=(
                f"{len(unc_patches)} uncertain patches "
                f"(prob in [{QA_UNCERTAIN_PROB_MIN:.2f}, "
                f"{QA_UNCERTAIN_PROB_MAX:.2f}])"
            ),
            output=FIGURES_DIR / "contact_sheet_uncertain_30.png",
        )
    else:
        log.warning(
            "Predictions parquet not found at %s — skipping the two "
            "predictions-driven contact sheets",
            PREDICTIONS_PARQUET,
        )

    log.info("Outputs in: %s", OUTPUT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
