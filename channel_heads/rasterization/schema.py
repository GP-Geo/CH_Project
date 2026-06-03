"""Shared raster patch schema constants.

The 5-class encoding is a frozen CNN contract shared by Earth and Mars patches.
Keep these values unchanged unless the trained patch/model artifacts are
explicitly rebuilt.
"""

from __future__ import annotations

BACKGROUND = 0
BRANCH_A = 1
BRANCH_B = 2
OTHER_STREAMS = 3
CONFLUENCE_MARKER = 4
NUM_CLASSES = 5

CLASS_LABELS = {
    BACKGROUND: "background",
    BRANCH_A: "branch_a",
    BRANCH_B: "branch_b",
    OTHER_STREAMS: "other_streams",
    CONFLUENCE_MARKER: "confluence_marker",
}

PATCH_FLAG_COLUMNS = [
    "has_branch_a",
    "has_branch_b",
    "has_confluence",
    "branch_a_connected",
    "branch_b_connected",
    "branches_connected",
]

__all__ = [
    "BACKGROUND",
    "BRANCH_A",
    "BRANCH_B",
    "OTHER_STREAMS",
    "CONFLUENCE_MARKER",
    "NUM_CLASSES",
    "CLASS_LABELS",
    "PATCH_FLAG_COLUMNS",
]
