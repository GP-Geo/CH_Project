"""Labeled-dataset assembly and hard-negative filtering for Earth training.

This module is the canonical home for the training-dataset post-processing
applied to Earth channel-head pairs:

    - ``generate_labeled_dataset`` merges coupling, asymmetry, and geometric
      features and attaches the binary ``y`` label, and
    - ``filter_hard_negatives`` keeps only "hard" negatives matched to the
      positives (plus the optional stream-crossing geometric filter).

The implementation was extracted verbatim from
:mod:`channel_heads.geometric_analysis`; that module re-exports these names
(including the private ``_line_crosses_stream`` / ``_build_stream_mask``
helpers) for backward compatibility.

References:
    Goren, L. and Shelef, E.: Channel concavity controls planform complexity
    of branching drainage networks, Earth Surf. Dynam., 12, 1347-1369,
    https://doi.org/10.5194/esurf-12-1347-2024, 2024.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..features.earth_geometry import GEOM_FEATURE_COLS
from ..stream_utils import line_pixels

# =============================================================================
# Stream-Crossing Helpers
# =============================================================================


def _line_crosses_stream(
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    stream_mask: npt.NDArray[np.bool_],
) -> bool:
    """Return True if the rasterized line from (r1,c1) to (r2,c2) passes
    through any stream pixel, excluding the two endpoint pixels themselves.

    Uses Bresenham's line algorithm (skimage.draw.line) to enumerate
    intermediate pixels.
    """
    rr, cc = line_pixels(r1, c1, r2, c2)
    # Clip to valid array bounds
    valid = (rr >= 0) & (rr < stream_mask.shape[0]) & (cc >= 0) & (cc < stream_mask.shape[1])
    rr, cc = rr[valid], cc[valid]
    if len(rr) <= 2:
        # Only endpoints, no intermediate pixels
        return False
    # Exclude first and last pixel (the channel heads themselves)
    interior_rr, interior_cc = rr[1:-1], cc[1:-1]
    return bool(stream_mask[interior_rr, interior_cc].any())


def _build_stream_mask(
    s: Any,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray, npt.NDArray]:
    """Build a binary stream mask and return (mask, r_nodes, c_nodes).

    mask shape is inferred from s.shape if available, else from max node indices.
    """
    node_indices = (
        s.node_indices() if callable(getattr(s, "node_indices", None)) else s.node_indices
    )
    r_nodes, c_nodes = np.asarray(node_indices[0]), np.asarray(node_indices[1])

    shape = getattr(s, "shape", None)
    if shape is None:
        shape = (int(r_nodes.max()) + 1, int(c_nodes.max()) + 1)

    mask = np.zeros(shape, dtype=bool)
    mask[r_nodes, c_nodes] = True
    return mask, r_nodes, c_nodes


# =============================================================================
# Labeling Functions
# =============================================================================


def generate_labeled_dataset(
    coupling_df: pd.DataFrame,
    asymmetry_df: pd.DataFrame,
    geometric_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all features and add label column.

    Creates a labeled dataset for classification where:
    - y=1 (positive): pairs where touching=True (coupled)
    - y=0 (negative): pairs where touching=False (hard negatives)

    Parameters
    ----------
    coupling_df : pd.DataFrame
        From CouplingAnalyzer with 'touching' column.
    asymmetry_df : pd.DataFrame
        From LengthwiseAsymmetryAnalyzer with L_1, L_2, delta_L.
    geometric_df : pd.DataFrame
        From GeometricFeaturesAnalyzer with features 2-5.

    Returns
    -------
    pd.DataFrame
        Combined features with 'y' column (1=touching/positive, 0=not-touching/negative).
    """
    merge_keys = ["outlet", "confluence", "head_1", "head_2"]

    # Start with coupling (has the label)
    df = coupling_df.copy()

    # Add label column
    df["y"] = df["touching"].astype(int)

    # Merge asymmetry features
    if not asymmetry_df.empty:
        asymmetry_cols = ["L_1", "L_2", "delta_L"]
        cols_to_merge = [c for c in asymmetry_cols if c in asymmetry_df.columns]
        if cols_to_merge:
            asymmetry_subset = asymmetry_df[merge_keys + cols_to_merge]
            df = pd.merge(df, asymmetry_subset, on=merge_keys, how="left")

    # Merge geometric features (use GEOM_FEATURE_COLS so new features are included)
    if not geometric_df.empty:
        geom_cols = list(GEOM_FEATURE_COLS)
        cols_to_merge = [c for c in geom_cols if c in geometric_df.columns]
        if cols_to_merge:
            geom_subset = geometric_df[merge_keys + cols_to_merge]
            df = pd.merge(df, geom_subset, on=merge_keys, how="left")

    return df


def filter_hard_negatives(
    labeled_df: pd.DataFrame,
    max_L_ratio: float = 3.0,
    max_dist_ratio: float = 5.0,
    group_col: str | None = None,
    s: Any | None = None,
) -> pd.DataFrame:
    """Filter negatives to keep only 'hard' negatives matched to positives.

    Hard negatives are pairs at the same confluence with:
    - Similar L-scale (L_1 + L_2 within max_L_ratio of median positive)
    - Relatively close in planform (head-head distance within max_dist_ratio)
    - Head-to-head vector does NOT cross any stream pixel (geometric criterion)

    This avoids trivially far-away or trivially separated negatives. When ``s``
    is provided, pairs whose straight-line vector between the two channel heads
    crosses a stream pixel are removed as trivially non-touching.

    .. warning::
        **Data leakage risk:** The filtering thresholds are derived from the
        positives in ``labeled_df``. Call this function separately on each
        cross-validation fold's training set, not on the combined dataset, to
        prevent positives from the test fold from influencing the thresholds
        used to filter training negatives.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Labeled dataset with 'y', 'L_1', 'L_2', 'headhead_dist_m' columns.
    max_L_ratio : float, optional
        Maximum ratio of negative L_sum to median positive L_sum (default: 3.0).
    max_dist_ratio : float, optional
        Maximum ratio of negative distance to median positive distance (default: 5.0).
    group_col : str, optional
        Column name to group by (e.g., 'basin') for per-group threshold
        computation. If None, uses global thresholds across all data.
    s : StreamObject, optional
        TopoToolbox StreamObject. When provided, negatives whose head-to-head
        vector crosses a stream pixel are removed (geometric intersection filter).

    Returns
    -------
    pd.DataFrame
        Filtered dataset with positives and hard negatives only.
    """
    if labeled_df.empty:
        return labeled_df

    # Per-group filtering: compute thresholds within each group
    if group_col is not None and group_col in labeled_df.columns:
        parts = []
        for _, group_df in labeled_df.groupby(group_col):
            parts.append(filter_hard_negatives(group_df, max_L_ratio, max_dist_ratio, s=s))
        result = pd.concat(parts, ignore_index=True)
        result.sort_values(
            ["outlet", "confluence", "head_1", "head_2"],
            inplace=True,
            ignore_index=True,
        )
        return result

    # Separate positives and negatives
    positives = labeled_df[labeled_df["y"] == 1].copy()
    negatives = labeled_df[labeled_df["y"] == 0].copy()

    if positives.empty or negatives.empty:
        return labeled_df

    # Compute L_sum if L_1 and L_2 exist
    has_L = "L_1" in labeled_df.columns and "L_2" in labeled_df.columns
    has_dist = "headhead_dist_m" in labeled_df.columns

    if has_L:
        positives["L_sum"] = positives["L_1"] + positives["L_2"]
        negatives["L_sum"] = negatives["L_1"] + negatives["L_2"]

        median_pos_L = positives["L_sum"].median()

        if not pd.isna(median_pos_L) and median_pos_L > 0:
            L_threshold = median_pos_L * max_L_ratio
            negatives = negatives[(negatives["L_sum"].isna()) | (negatives["L_sum"] <= L_threshold)]

        # Clean up temp column
        positives.drop(columns=["L_sum"], inplace=True)
        if "L_sum" in negatives.columns:
            negatives.drop(columns=["L_sum"], inplace=True)

    if has_dist:
        median_pos_dist = positives["headhead_dist_m"].median()

        if not pd.isna(median_pos_dist) and median_pos_dist > 0:
            dist_threshold = median_pos_dist * max_dist_ratio
            negatives = negatives[
                (negatives["headhead_dist_m"].isna())
                | (negatives["headhead_dist_m"] <= dist_threshold)
            ]

    # Geometric stream-crossing filter: remove pairs whose head-to-head vector
    # crosses a stream pixel — these are trivially non-touching.
    # NOTE: Apply this AFTER train/test split to avoid cross-contamination of thresholds.
    if (
        s is not None
        and not negatives.empty
        and "head_1" in negatives.columns
        and "head_2" in negatives.columns
    ):
        stream_mask, r_nodes, c_nodes = _build_stream_mask(s)

        def _does_not_cross(h1: int, h2: int) -> bool:
            try:
                return not _line_crosses_stream(
                    int(r_nodes[h1]),
                    int(c_nodes[h1]),
                    int(r_nodes[h2]),
                    int(c_nodes[h2]),
                    stream_mask,
                )
            except (IndexError, KeyError):
                return True  # keep on error (conservative)

        keep_mask = [
            _does_not_cross(int(h1), int(h2))
            for h1, h2 in zip(negatives["head_1"], negatives["head_2"])
        ]
        negatives = negatives[keep_mask]

    # Combine
    result = pd.concat([positives, negatives], ignore_index=True)
    result.sort_values(
        ["outlet", "confluence", "head_1", "head_2"], inplace=True, ignore_index=True
    )

    return result


__all__ = [
    "generate_labeled_dataset",
    "filter_hard_negatives",
]
