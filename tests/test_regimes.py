"""Tests for channel_heads.regimes — regime presets extracted from the scripts."""

from __future__ import annotations

import dataclasses

import pytest

from channel_heads.regimes import REGIMES, Regime


def test_expected_regimes_present():
    assert set(REGIMES) == {"regA", "regB", "regC"}


@pytest.mark.parametrize(
    "name,threshold_km2,pre_remove,order_gap",
    [
        ("regA", 0.05, 2, 4),
        ("regB", 0.25, 1, 4),
        ("regC", 0.10, 1, 4),
    ],
)
def test_regime_preset_values(name, threshold_km2, pre_remove, order_gap):
    r = REGIMES[name]
    assert r.name == name
    assert r.threshold_km2 == threshold_km2
    assert r.pre_remove_max_order == pre_remove
    assert r.order_gap_to_prune == order_gap


def test_regime_defaults():
    r = REGIMES["regA"]
    assert r.coupling_n_workers == 4
    assert r.min_prefilter_px == 30.0


def test_regime_is_frozen_dataclass():
    assert dataclasses.is_dataclass(Regime)
    with pytest.raises(dataclasses.FrozenInstanceError):
        REGIMES["regA"].threshold_km2 = 1.0  # type: ignore[misc]
