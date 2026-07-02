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
These are the **top-3 data-driven regimes** selected in
``notebooks/pipeline/04_earth_mars_regime_calibration.ipynb`` (ranked by
``scientific_score``, lower = better). All three are plain ``trim`` — drop 1st
order only, no order-gap delta pruning:

- ``regA``: T = 0.20 km², pre_remove ≤ 1 (drop 1st order only), no order-gap pruning
- ``regB``: T = 0.25 km², pre_remove ≤ 1 (drop 1st order only), no order-gap pruning
- ``regC``: T = 0.15 km², pre_remove ≤ 1 (drop 1st order only), no order-gap pruning

.. note::
    These definitions replace the prior hand-frozen presets
    (regA T=0.05/pre_remove=2/order_gap=4; regB T=0.25/pre_remove=1/order_gap=4;
    regC T=0.10/pre_remove=1/order_gap=4). The full ``*_reg{A,B,C}`` artifact
    chain (CNN, XGBoost, optimal thresholds, Mars predictions) was **retrained
    on these presets on 2026-06-13** — operating thresholds regA 0.769133 /
    regB 0.773238 / regC 0.810635 (``models/optimal_threshold_*_reg*.txt`` are
    authoritative) — so on-disk regime artifacts are valid. See
    ``docs/REGIME_SELECTION.md``.
"""

from __future__ import annotations

import csv as _csv
from dataclasses import dataclass
from pathlib import Path


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
        threshold_km2=0.20,
        pre_remove_max_order=1,
        order_gap_to_prune=0,
    ),
    "regB": Regime(
        name="regB",
        threshold_km2=0.25,
        pre_remove_max_order=1,
        order_gap_to_prune=0,
    ),
    "regC": Regime(
        name="regC",
        threshold_km2=0.15,
        pre_remove_max_order=1,
        order_gap_to_prune=0,
    ),
}


def default_selected_regimes_csv() -> Path:
    """Canonical path of the CSV-selected regimes (regA..regE).

    Lazy import of ``io.paths`` keeps this module import-light and avoids a
    circular import at module load.
    """
    from .io.paths import RESULTS_DIR

    return (
        RESULTS_DIR
        / "drainage_density_calibration"
        / "regime_optimization"
        / "selected_regimes_AE.csv"
    )


def load_selected_regimes(csv_path: str | Path | None = None) -> dict[str, Regime]:
    """Load CSV-selected regimes (regA..regE) into ``{name: Regime}``.

    Additive companion to the frozen :data:`REGIMES` dict: it reads the regime
    table written by Stage-4
    ``notebooks/pipeline/04_earth_mars_regime_calibration.ipynb`` (regA–regE
    selection; the archived ``notebooks/archive/regime/03_optimize_regime_candidates.ipynb``
    is the historical source) so downstream stages can
    iterate over the *selected* regimes without hard-coding definitions. The
    frozen :data:`REGIMES` are left untouched.

    The CSV must carry ``name``, ``threshold_km2``, ``pre_remove_max_order`` and
    ``order_gap_to_prune`` columns; a blank/``NA`` order-gap means no order-gap
    pruning (0).

    .. warning::
        Selected regimes may differ from the frozen ``regA/regB/regC`` the
        production ``*_reg{A,B,C}`` artifacts were trained on. Models must be
        retrained for these definitions before inference results are valid.
    """
    path = Path(csv_path) if csv_path is not None else default_selected_regimes_csv()
    regimes: dict[str, Regime] = {}
    with open(path, newline="") as fh:
        for row in _csv.DictReader(fh):
            raw_gap = (row.get("order_gap_to_prune") or "").strip()
            gap = 0 if raw_gap.lower() in ("", "na", "nan", "none") else int(float(raw_gap))
            regimes[row["name"]] = Regime(
                name=row["name"],
                threshold_km2=float(row["threshold_km2"]),
                pre_remove_max_order=int(float(row["pre_remove_max_order"])),
                order_gap_to_prune=gap,
            )
    return regimes
