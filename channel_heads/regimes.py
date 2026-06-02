"""Regime presets for the Mars-calibration pipeline.

A *regime* bundles the stream-extraction area threshold (km²) and the
network-pruning parameters (Strahler pre-removal + order-gap) used to re-run the
Earth→Mars workflow under alternative network-complexity settings.

Previously these presets lived inside ``scripts/build_earth_features_regime.py``
and were imported by the other regime scripts via a fragile
``sys.path.insert(...)`` sibling import. They now live here so notebooks,
scripts, and tests can all ``from channel_heads.regimes import REGIMES, Regime``.

Regime presets
--------------
- ``regA``: T = 0.05 km², pre_remove ≤ 2 (drop 1st+2nd order), order_gap ≥ 4
- ``regB``: T = 0.25 km², pre_remove ≤ 1 (drop 1st order only), order_gap ≥ 4
- ``regC``: T = 0.10 km², pre_remove ≤ 1, order_gap ≥ 4
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Regime:
    name: str
    threshold_km2: float
    pre_remove_max_order: int
    order_gap_to_prune: int
    # Worker threads for evaluate_pairs_for_outlet_parallel.
    coupling_n_workers: int = 4
    # Floor for the per-outlet prefilter distance (pixels). Even a tiny
    # outlet's basin yields at least this prefilter, so we don't auto-skip
    # everything in pathologically small basins. 30 px ≈ production's
    # 2 * sqrt(300) so this matches the original effective floor.
    min_prefilter_px: float = 30.0


REGIMES: dict[str, Regime] = {
    "regA": Regime(
        name="regA",
        threshold_km2=0.05,
        pre_remove_max_order=2,
        order_gap_to_prune=4,
    ),
    "regB": Regime(
        name="regB",
        threshold_km2=0.25,
        pre_remove_max_order=1,
        order_gap_to_prune=4,
    ),
    "regC": Regime(
        name="regC",
        threshold_km2=0.1,
        pre_remove_max_order=1,
        order_gap_to_prune=4,
    ),
}
