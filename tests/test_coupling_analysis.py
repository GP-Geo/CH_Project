"""Tests for coupling_analysis module."""

import pytest
import numpy as np
from channel_heads.coupling_analysis import PairTouchResult


class TestPairTouchResult:
    """Test PairTouchResult dataclass."""

    def test_creation(self):
        """Test creating a PairTouchResult."""
        result = PairTouchResult(
            touching=True,
            overlap_px=5,
            contact_px=10,
            size1_px=100,
            size2_px=120
        )
        assert result.touching is True
        assert result.overlap_px == 5
        assert result.contact_px == 10
        assert result.size1_px == 100
        assert result.size2_px == 120

    def test_no_touching(self):
        """Test non-touching pair result."""
        result = PairTouchResult(
            touching=False,
            overlap_px=0,
            contact_px=0,
            size1_px=100,
            size2_px=120
        )
        assert result.touching is False
        assert result.overlap_px == 0
        assert result.contact_px == 0


# Note: Full integration tests with TopoToolbox objects require
# either mock objects or small test datasets. See improvement.md
# for recommendations on setting up comprehensive test fixtures.
