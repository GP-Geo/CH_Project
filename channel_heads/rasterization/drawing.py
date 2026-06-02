"""Shared raster drawing primitives.

Thin, stable surface over :mod:`channel_heads.rasterizer` so callers import
``channel_heads.rasterization.drawing`` for low-level line drawing rather than
the historical module name.
"""

from __future__ import annotations

from channel_heads.rasterizer import bresenham_line

__all__ = ["bresenham_line"]
