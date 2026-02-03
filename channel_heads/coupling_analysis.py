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
- CouplingAnalyzer(fd, s, dem, connectivity=8, threshold=300, prefilter_multiplier=2.0)
    .influence_grid(head_id) -> GridObject
    .influence_mask(head_id) -> np.ndarray[bool]
    .pair_touching(h1, h2)   -> dict (touching, overlap_px, contact_px, size1_px, size2_px)
    .evaluate_pairs_for_outlet(outlet, pairs_at_confluence) -> pandas.DataFrame
    .evaluate_pairs_for_outlet_parallel(outlet, pairs_at_confluence, n_workers=4) -> pandas.DataFrame
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    pass

# Type aliases for clarity
NodeId = int
HeadId = int
HeadPair = tuple[HeadId, HeadId]
BoolMask = npt.NDArray[np.bool_]


@dataclass(slots=True)
class PairTouchResult:
    """Result of basin contact detection between two channel heads.

    Attributes
    ----------
    touching : bool
        Whether the two basins are touching (contact or overlap).
    overlap_px : int
        Number of pixels where the two basins directly overlap.
    contact_px : int
        Number of 4- or 8-connected contact pixels (excludes direct overlap).
    size1_px : int
        Total pixels in the first basin.
    size2_px : int
        Total pixels in the second basin.
    """

    touching: bool
    overlap_px: int
    contact_px: int
    size1_px: int
    size2_px: int


class CouplingAnalyzer:
    """Compute basin coupling between channel heads.

    Uses FlowObject.dependencemap to compute influence masks and tests for
    spatial contact (touching or overlap) between basin pairs. Caches masks
    to avoid recomputation across pairs.

    Parameters
    ----------
    fd : FlowObject
        Flow direction object from TopoToolbox.
    s : StreamObject
        Stream network object from TopoToolbox.
    dem : GridObject
        Digital elevation model as a GridObject.
    connectivity : int, optional
        Pixel connectivity for contact detection: 4 or 8 (default: 8).
    threshold : int, optional
        Stream network area threshold in pixels (default: 300). Used for
        spatial pre-filtering: pairs farther than prefilter_multiplier * sqrt(threshold)
        pixels apart are skipped as they cannot have touching basins.
    prefilter_multiplier : float, optional
        Multiplier for pre-filter distance calculation (default: 2.0).
        Distance threshold = prefilter_multiplier * sqrt(threshold).

    Attributes
    ----------
    fd : FlowObject
        Flow direction object.
    s : StreamObject
        Stream network object.
    dem : GridObject
        Digital elevation model.
    connectivity : int
        Connectivity setting (4 or 8).
    threshold : int
        Stream threshold for pre-filtering.
    prefilter_multiplier : float
        Multiplier for distance threshold.

    Example
    -------
    >>> analyzer = CouplingAnalyzer(fd, s, dem, connectivity=8, threshold=300)
    >>> result = analyzer.pair_touching(head_1=662, head_2=716)
    >>> print(f"Touching: {result.touching}")
    """

    fd: Any  # FlowObject
    s: Any  # StreamObject
    dem: Any  # GridObject
    connectivity: int
    threshold: int
    prefilter_multiplier: float
    _mask_cache: dict[int, BoolMask]
    _cache_lock: threading.Lock

    def __init__(
        self,
        fd: Any,  # FlowObject
        s: Any,  # StreamObject
        dem: Any,  # GridObject
        connectivity: int = 8,
        threshold: int = 300,
        prefilter_multiplier: float = 2.0,
    ) -> None:
        # Alignment check
        if getattr(dem, "z", None) is None:
            raise ValueError("`dem` must be a GridObject with a .z array")
        if tuple(dem.z.shape) != tuple(fd.shape):
            raise ValueError(f"DEM shape {dem.z.shape} != FlowObject shape {fd.shape}")

        # Store references
        self.fd = fd
        self.s = s
        # StreamObject.node_indices can be attribute (tuple)) or method; handle both.
        self._node_indices = (
            s.node_indices() if callable(getattr(s, "node_indices", None)) else s.node_indices
        )
        if not isinstance(self._node_indices, tuple) or len(self._node_indices) != 2:
            raise TypeError("StreamObject.node_indices must be a (rows, cols) tuple")
        self._r_nodes, self._c_nodes = self._node_indices

        self.dem = dem
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.connectivity = connectivity

        # Pre-filtering parameters
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        if prefilter_multiplier <= 0:
            raise ValueError("prefilter_multiplier must be positive")
        self.prefilter_multiplier = prefilter_multiplier
        # Pre-compute distance threshold for faster checks
        self._prefilter_distance = prefilter_multiplier * math.sqrt(threshold)

        # Thread-safe cache: head_id -> np.ndarray[bool] mask (same shape as dem.z)
        self._mask_cache: dict[int, np.ndarray] = {}
        self._cache_lock = threading.Lock()

    def clear_cache(self) -> int:
        """Clear the mask cache to free memory.

        This should be called between outlets when processing in batch to
        prevent unbounded memory growth. Each mask has shape (dem_rows, dem_cols),
        so processing many outlets without clearing can consume significant memory.

        This method is thread-safe.

        Returns
        -------
        int
            Number of cached masks that were cleared.

        Example
        -------
        >>> for outlet_id in outlets:
        ...     pairs, heads = first_meet_pairs_for_outlet(s, outlet_id)
        ...     df = analyzer.evaluate_pairs_for_outlet(outlet_id, pairs)
        ...     analyzer.clear_cache()  # Free memory before next outlet
        """
        with self._cache_lock:
            n_cleared = len(self._mask_cache)
            self._mask_cache.clear()
        return n_cleared

    @property
    def cache_size(self) -> int:
        """Return the number of cached masks.

        This property is thread-safe.
        """
        with self._cache_lock:
            return len(self._mask_cache)

    # ---------- low-level helpers ----------

    def _rc_for_head(self, h: int) -> tuple[int, int]:
        """Resolve a head identifier to (row, col) in the DEM/Flow grid.

        First tries interpreting `h` as a stream node ID (index into node_indices).
        If that fails due to index bounds, falls back to treating `h` as a linear
        pixel index in the flow grid using fd.unravel_index.

        Parameters
        ----------
        h : int
            Head identifier (either stream node ID or linear pixel index).

        Returns
        -------
        Tuple[int, int]
            Row and column indices in the DEM/Flow grid.
        """
        h = int(h)
        # Try stream node ID first (most common case)
        if 0 <= h < len(self._r_nodes):
            return int(self._r_nodes[h]), int(self._c_nodes[h])

        # Fall back to linear pixel index
        rr_arr, cc_arr = self.fd.unravel_index(h)
        rr = int(np.asarray(rr_arr).item()) if np.ndim(rr_arr) else int(rr_arr)
        cc = int(np.asarray(cc_arr).item()) if np.ndim(cc_arr) else int(cc_arr)
        return rr, cc

    def _seed_grid_for_head(self, h: int):
        seed = self.dem.duplicate_with_new_data(np.zeros_like(self.dem.z, dtype=bool))
        rr, cc = self._rc_for_head(int(h))
        seed.z[rr, cc] = True
        return seed

    def _heads_can_touch(self, h1: int, h2: int) -> bool:
        """Quick spatial check if two heads could possibly have touching basins.

        Uses Euclidean distance between channel head positions to determine if
        basins could touch. Two heads can only have touching basins if they are
        within prefilter_multiplier * sqrt(threshold) pixels of each other.

        Parameters
        ----------
        h1 : int
            Node ID of the first channel head.
        h2 : int
            Node ID of the second channel head.

        Returns
        -------
        bool
            True if heads are close enough that their basins could touch,
            False if they are too far apart.
        """
        r1, c1 = self._rc_for_head(int(h1))
        r2, c2 = self._rc_for_head(int(h2))

        # Euclidean distance between heads
        distance = math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

        return distance <= self._prefilter_distance

    # ---------- public API ----------

    def influence_grid(self, head_id: int):
        """Return GridObject logical mask (dependence map) for a head, using FlowObject.dependencemap."""
        seed = self._seed_grid_for_head(int(head_id))
        return self.fd.dependencemap(seed)

    def influence_mask(self, head_id: int) -> BoolMask:
        """Return boolean numpy mask for a head.

        Uses internal cache to avoid recomputation for repeated queries.
        This method is thread-safe using double-checked locking.

        Parameters
        ----------
        head_id : int
            Node ID of the channel head.

        Returns
        -------
        BoolMask
            Boolean mask array (same shape as DEM) where True indicates
            cells in the head's drainage basin.
        """
        hid = int(head_id)
        # First check without lock (fast path)
        if hid in self._mask_cache:
            return self._mask_cache[hid]

        # Compute mask outside lock to avoid blocking other threads
        G = self.influence_grid(hid)
        mask = np.asarray(G.z, dtype=bool)

        # Second check with lock (thread-safe update)
        with self._cache_lock:
            # Check again in case another thread computed it while we were computing
            if hid not in self._mask_cache:
                self._mask_cache[hid] = mask
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
        contact_px += int((A[:-1, :] & B[1:, :]).sum())
        # horizontal neighbors
        contact_px += int((A[:, 1:] & B[:, :-1]).sum())
        contact_px += int((A[:, :-1] & B[:, 1:]).sum())

        if self.connectivity == 8:
            # diagonals
            contact_px += int((A[1:, 1:] & B[:-1, :-1]).sum())
            contact_px += int((A[1:, :-1] & B[:-1, 1:]).sum())
            contact_px += int((A[:-1, 1:] & B[1:, :-1]).sum())
            contact_px += int((A[:-1, :-1] & B[1:, 1:]).sum())

        touching = contact_px > 0
        return PairTouchResult(touching, overlap_px, contact_px, int(A.sum()), int(B.sum()))

    def evaluate_pairs_for_outlet(
        self,
        outlet: int,
        pairs_at_confluence: dict[int, set[HeadPair]],
        use_prefilter: bool = True,
    ) -> pd.DataFrame:
        """
        Build a tidy DataFrame with one row per pair in the given outlet's confluences.

        Parameters
        ----------
        outlet : int
            Outlet node ID.
        pairs_at_confluence : dict[int, set[HeadPair]]
            Mapping from confluence ID to set of (head1, head2) pairs.
        use_prefilter : bool, optional
            If True (default), skip pairs where heads are too far apart to touch.
            Skipped pairs have touching=False, overlap_px=0, contact_px=0,
            size1_px=None, size2_px=None, and skipped_prefilter=True.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: outlet, confluence, head_1, head_2, touching,
            overlap_px, contact_px, size1_px, size2_px, skipped_prefilter
        """
        rows = []
        out = int(outlet)
        for conf, pairs in pairs_at_confluence.items():
            if not pairs:
                continue
            for h1, h2 in pairs:
                h1_norm, h2_norm = int(min(h1, h2)), int(max(h1, h2))

                # Check if pre-filtering should skip this pair
                if use_prefilter and not self._heads_can_touch(h1, h2):
                    rows.append(
                        {
                            "outlet": out,
                            "confluence": int(conf),
                            "head_1": h1_norm,
                            "head_2": h2_norm,
                            "touching": False,
                            "overlap_px": 0,
                            "contact_px": 0,
                            "size1_px": None,
                            "size2_px": None,
                            "skipped_prefilter": True,
                        }
                    )
                else:
                    res = self.pair_touching(h1, h2)
                    rows.append(
                        {
                            "outlet": out,
                            "confluence": int(conf),
                            "head_1": h1_norm,
                            "head_2": h2_norm,
                            "touching": bool(res.touching),
                            "overlap_px": int(res.overlap_px),
                            "contact_px": int(res.contact_px),
                            "size1_px": int(res.size1_px),
                            "size2_px": int(res.size2_px),
                            "skipped_prefilter": False,
                        }
                    )
        df = pd.DataFrame(
            rows,
            columns=[
                "outlet",
                "confluence",
                "head_1",
                "head_2",
                "touching",
                "overlap_px",
                "contact_px",
                "size1_px",
                "size2_px",
                "skipped_prefilter",
            ],
        )
        # Optional: stable ordering for readability
        if not df.empty:
            df.sort_values(["confluence", "head_1", "head_2"], inplace=True, ignore_index=True)
        return df

    def evaluate_pairs_for_outlet_parallel(
        self,
        outlet: int,
        pairs_at_confluence: dict[int, set[HeadPair]],
        n_workers: int = 4,
        use_prefilter: bool = True,
    ) -> pd.DataFrame:
        """
        Thread-safe parallel evaluation of pairs for a given outlet.

        This method processes pairs in parallel using ThreadPoolExecutor while
        maintaining thread safety for the mask cache.

        Parameters
        ----------
        outlet : int
            Outlet node ID.
        pairs_at_confluence : dict[int, set[HeadPair]]
            Mapping from confluence ID to set of (head1, head2) pairs.
        n_workers : int, optional
            Number of worker threads (default: 4).
        use_prefilter : bool, optional
            If True (default), skip pairs where heads are too far apart to touch.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: outlet, confluence, head_1, head_2, touching,
            overlap_px, contact_px, size1_px, size2_px, skipped_prefilter
        """
        out = int(outlet)

        # Flatten pairs with confluence info for parallel processing
        work_items: list[tuple[int, int, int]] = []  # (confluence, h1, h2)
        for conf, pairs in pairs_at_confluence.items():
            if not pairs:
                continue
            for h1, h2 in pairs:
                work_items.append((int(conf), int(h1), int(h2)))

        if not work_items:
            return pd.DataFrame(
                columns=[
                    "outlet",
                    "confluence",
                    "head_1",
                    "head_2",
                    "touching",
                    "overlap_px",
                    "contact_px",
                    "size1_px",
                    "size2_px",
                    "skipped_prefilter",
                ]
            )

        def process_pair(
            item: tuple[int, int, int],
        ) -> dict[str, int | bool | None]:
            """Process a single pair (thread-safe)."""
            conf, h1, h2 = item
            h1_norm, h2_norm = min(h1, h2), max(h1, h2)

            # Check pre-filter
            if use_prefilter and not self._heads_can_touch(h1, h2):
                return {
                    "outlet": out,
                    "confluence": conf,
                    "head_1": h1_norm,
                    "head_2": h2_norm,
                    "touching": False,
                    "overlap_px": 0,
                    "contact_px": 0,
                    "size1_px": None,
                    "size2_px": None,
                    "skipped_prefilter": True,
                }

            # Full evaluation (thread-safe via influence_mask locking)
            res = self.pair_touching(h1, h2)
            return {
                "outlet": out,
                "confluence": conf,
                "head_1": h1_norm,
                "head_2": h2_norm,
                "touching": bool(res.touching),
                "overlap_px": int(res.overlap_px),
                "contact_px": int(res.contact_px),
                "size1_px": int(res.size1_px),
                "size2_px": int(res.size2_px),
                "skipped_prefilter": False,
            }

        # Process pairs in parallel
        rows: list[dict[str, int | bool | None]] = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_pair, item): item for item in work_items}
            for future in as_completed(futures):
                rows.append(future.result())

        df = pd.DataFrame(
            rows,
            columns=[
                "outlet",
                "confluence",
                "head_1",
                "head_2",
                "touching",
                "overlap_px",
                "contact_px",
                "size1_px",
                "size2_px",
                "skipped_prefilter",
            ],
        )
        # Stable ordering for readability
        if not df.empty:
            df.sort_values(["confluence", "head_1", "head_2"], inplace=True, ignore_index=True)
        return df
