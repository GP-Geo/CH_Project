"""
coupling_analysis.py
====================
Utilities to compute and summarize *coupling* (first-meet + touching) of channel heads
for a single outlet, keeping notebooks clean.

Dependencies
------------
- numpy
- pandas

Assumptions
-----------
- You already have `first_meet_pairs_for_outlet(s, outlet)` available (e.g., in src/first_meet_pairs_for_outlet.py).
- `fd` is a FlowObject with: .dependencemap(GridObject), .unravel_index(idxs), .shape
- `s`  is a StreamObject with: .node_indices (tuple of (row_idx, col_idx))

Main API
--------
- CouplingAnalyzer(fd, s, dem, connectivity=8)
    .influence_grid(head_id) -> GridObject
    .influence_mask(head_id) -> np.ndarray[bool]
    .pair_touching(h1, h2)   -> dict (touching, overlap_px, contact_px, size1_px, size2_px)
    .evaluate_pairs_for_outlet(outlet, pairs_at_confluence) -> pandas.DataFrame
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd

NodeId = int
HeadId = int
HeadPair = Tuple[HeadId, HeadId]


@dataclass
class PairTouchResult:
    touching: bool
    overlap_px: int
    contact_px: int  # 4- or 8-connected contact pixels (excludes direct overlap)
    size1_px: int
    size2_px: int


class CouplingAnalyzer:
    """
    Compute dependence maps via FlowObject.dependencemap and test touching robustly (no wrap-around).
    Caches per-head influence masks to avoid recomputation across many pairs.
    """

    def __init__(self, fd, s, dem, connectivity: int = 8):
        # Alignment check
        if getattr(dem, "z", None) is None:
            raise ValueError("`dem` must be a GridObject with a .z array")
        if tuple(dem.z.shape) != tuple(fd.shape):
            raise ValueError(f"DEM shape {dem.z.shape} != FlowObject shape {fd.shape}")

        # Store references
        self.fd = fd
        self.s = s
        # StreamObject.node_indices can be attribute (tuple)) or method; handle both.
        self._node_indices = s.node_indices() if callable(getattr(s, "node_indices", None)) else s.node_indices
        if not isinstance(self._node_indices, tuple) or len(self._node_indices) != 2:
            raise TypeError("StreamObject.node_indices must be a (rows, cols) tuple")
        self._r_nodes, self._c_nodes = self._node_indices

        self.dem = dem
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.connectivity = connectivity

        # Simple cache: head_id -> np.ndarray[bool] mask (same shape as dem.z)
        self._mask_cache: Dict[int, np.ndarray] = {}

    # ---------- low-level helpers ----------

    def _rc_for_head(self, h: int) -> Tuple[int, int]:
        """
        Resolve a head identifier to (row, col) in the DEM/Flow grid.
        First try interpreting `h` as a *stream node id* (index into node_indices).
        If that fails, treat `h` as a *linear pixel index* in the flow grid and use fd.unravel_index.
        """
        try:
            rr = int(self._r_nodes[h]); cc = int(self._c_nodes[h])
            return rr, cc
        except Exception:
            rr_arr, cc_arr = self.fd.unravel_index(int(h))
            rr = int(np.asarray(rr_arr).item()) if np.ndim(rr_arr) else int(rr_arr)
            cc = int(np.asarray(cc_arr).item()) if np.ndim(cc_arr) else int(cc_arr)
            return rr, cc

    def _seed_grid_for_head(self, h: int):
        seed = self.dem.duplicate_with_new_data(np.zeros_like(self.dem.z, dtype=bool))
        rr, cc = self._rc_for_head(int(h))
        seed.z[rr, cc] = True
        return seed

    # ---------- public API ----------

    def influence_grid(self, head_id: int):
        """Return GridObject logical mask (dependence map) for a head, using FlowObject.dependencemap."""
        seed = self._seed_grid_for_head(int(head_id))
        return self.fd.dependencemap(seed)

    def influence_mask(self, head_id: int) -> np.ndarray:
        """Return boolean numpy mask for a head; uses cache to avoid recomputation."""
        hid = int(head_id)
        if hid not in self._mask_cache:
            G = self.influence_grid(hid)
            self._mask_cache[hid] = np.asarray(G.z, dtype=bool)
        return self._mask_cache[hid]

    def pair_touching(self, h1: int, h2: int) -> PairTouchResult:
        """
        Check if the dependence masks of two heads touch (4/8-connected) or overlap.
        No wrap-around; avoids np.roll.
        """
        A = self.influence_mask(int(h1))
        B = self.influence_mask(int(h2))

        # Direct overlap (strongest)
        overlap = A & B
        overlap_px = int(overlap.sum())
        if overlap_px > 0:
            return PairTouchResult(True, overlap_px, 0, int(A.sum()), int(B.sum()))

        # 4-connected contact (exclude overlap by construction)
        contact_px = 0
        # vertical neighbors
        contact_px += int((A[1:, :] & B[:-1, :]).sum())
        contact_px += int((A[:-1, :] & B[1:,  :]).sum())
        # horizontal neighbors
        contact_px += int((A[:, 1:] & B[:, :-1]).sum())
        contact_px += int((A[:, :-1] & B[:, 1:]).sum())

        if self.connectivity == 8:
            # diagonals
            contact_px += int((A[1:, 1:]  & B[:-1, :-1]).sum())
            contact_px += int((A[1:, :-1] & B[:-1,  1:]).sum())
            contact_px += int((A[:-1, 1:] & B[1:,  :-1]).sum())
            contact_px += int((A[:-1, :-1]& B[1:,   1:]).sum())

        touching = (contact_px > 0)
        return PairTouchResult(touching, overlap_px, contact_px, int(A.sum()), int(B.sum()))

    def evaluate_pairs_for_outlet(self, outlet: int, pairs_at_confluence: Dict[int, Set[HeadPair]]) -> pd.DataFrame:
        """
        Build a tidy DataFrame with one row per pair in the given outlet's confluences.
        Columns:
            outlet, confluence, head_1, head_2, touching, overlap_px, contact_px, size1_px, size2_px
        """
        rows = []
        out = int(outlet)
        for conf, pairs in pairs_at_confluence.items():
            if not pairs:
                continue
            for (h1, h2) in pairs:
                res = self.pair_touching(h1, h2)
                rows.append({
                    "outlet": out,
                    "confluence": int(conf),
                    "head_1": int(min(h1, h2)),
                    "head_2": int(max(h1, h2)),
                    "touching": bool(res.touching),
                    "overlap_px": int(res.overlap_px),
                    "contact_px": int(res.contact_px),
                    "size1_px": int(res.size1_px),
                    "size2_px": int(res.size2_px),
                })
        df = pd.DataFrame(rows, columns=[
            "outlet","confluence","head_1","head_2","touching","overlap_px","contact_px","size1_px","size2_px"
        ])
        # Optional: stable ordering for readability
        if not df.empty:
            df.sort_values(["confluence","head_1","head_2"], inplace=True, ignore_index=True)
        return df
