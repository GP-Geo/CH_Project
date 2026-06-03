"""Sampling helpers for stream-crossing filter QA.

Called by ``scripts/diagnostics/qa_mars_stream_crossing_filter.py`` and by
``notebooks/diagnostics/stream_crossing_qa.ipynb``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def stratified_sample_removed_pairs(
    df_all: pd.DataFrame,
    n_samples: int,
    size_buckets: list[tuple[str, int, int, int]],
    seed: int,
) -> pd.DataFrame:
    """Return a stratified sample of pairs removed by the stream-crossing filter.

    Stratification is by per-network pair count (size_buckets). Prefers one
    removed pair per network to maximise visual diversity.

    Parameters
    ----------
    df_all:
        Full pair DataFrame; must contain ``stream_crossing_drop`` (bool),
        ``network_id``, and ``pair_id`` columns.
    n_samples:
        Maximum total pairs to return.
    size_buckets:
        List of ``(label, lo, hi, n_target)`` tuples defining network size
        ranges and how many pairs to pick from each bucket.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled rows with extra columns: ``size_bucket``, ``sample_index``,
        ``network_n_pairs``, ``network_n_dropped``, ``network_drop_rate``.
    """
    df_removed = df_all[df_all["stream_crossing_drop"]].copy()

    per_net = df_all.groupby("network_id").agg(
        n_pairs=("pair_id", "count"),
        n_dropped=("stream_crossing_drop", "sum"),
    )
    per_net["drop_rate"] = per_net["n_dropped"] / per_net["n_pairs"]

    rng = np.random.default_rng(seed)
    picks: list[pd.Series] = []
    used_networks: set[int] = set()

    for label, lo, hi, n_target in size_buckets:
        bucket_nets = per_net[
            (per_net["n_pairs"] >= lo)
            & (per_net["n_pairs"] <= hi)
            & (per_net["n_dropped"] > 0)
        ].index.tolist()
        bucket_nets = [n for n in bucket_nets if n not in used_networks]

        if not bucket_nets:
            log.warning("Bucket %s: no networks available", label)
            continue

        bucket_sorted = per_net.loc[bucket_nets].sort_values("drop_rate").index.tolist()
        if len(bucket_sorted) <= n_target:
            chosen_nets = bucket_sorted
        else:
            ranks = np.linspace(0, len(bucket_sorted) - 1, n_target).round().astype(int)
            chosen_nets = [bucket_sorted[r] for r in ranks]

        for nid in chosen_nets:
            cand = df_removed[df_removed["network_id"] == nid]
            if cand.empty:
                continue
            row = cand.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
            r = row.iloc[0].copy()
            r["size_bucket"] = label
            r["network_n_pairs"] = int(per_net.loc[nid, "n_pairs"])
            r["network_n_dropped"] = int(per_net.loc[nid, "n_dropped"])
            r["network_drop_rate"] = float(per_net.loc[nid, "drop_rate"])
            picks.append(r)
            used_networks.add(int(nid))
            if len(picks) >= n_samples:
                break
        if len(picks) >= n_samples:
            break

    # Top-up from any remaining networks if needed
    if len(picks) < n_samples:
        remaining = df_removed[~df_removed["network_id"].isin(used_networks)]
        if not remaining.empty:
            extra = remaining.sample(
                n=min(n_samples - len(picks), len(remaining)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            for _, r in extra.iterrows():
                rc = r.copy()
                nid = int(rc["network_id"])
                rc["size_bucket"] = "topup"
                rc["network_n_pairs"] = int(per_net.loc[nid, "n_pairs"])
                rc["network_n_dropped"] = int(per_net.loc[nid, "n_dropped"])
                rc["network_drop_rate"] = float(per_net.loc[nid, "drop_rate"])
                picks.append(rc)

    out = pd.DataFrame(picks).reset_index(drop=True)
    out["sample_index"] = np.arange(1, len(out) + 1)
    return out


__all__ = ["stratified_sample_removed_pairs"]
