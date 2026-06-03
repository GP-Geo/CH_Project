"""Shared rasterization primitive: bresenham_line."""

from __future__ import annotations

from channel_heads.rasterization.earth_patches import bresenham_line


class TestBresenham:
    def test_endpoints_inclusive(self):
        pts = bresenham_line(0, 0, 0, 3)
        assert pts[0] == (0, 0)
        assert pts[-1] == (0, 3)

    def test_horizontal(self):
        assert bresenham_line(2, 0, 2, 2) == [(2, 0), (2, 1), (2, 2)]

    def test_diagonal(self):
        assert bresenham_line(0, 0, 2, 2) == [(0, 0), (1, 1), (2, 2)]

    def test_single_point(self):
        assert bresenham_line(5, 5, 5, 5) == [(5, 5)]

    def test_8_connected(self):
        # consecutive pixels never jump more than 1 in each axis
        pts = bresenham_line(0, 0, 7, 3)
        for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
            assert abs(r1 - r0) <= 1 and abs(c1 - c0) <= 1
