"""Compatibility shim — torch device selection moved to ``channel_heads.models.device``.

The canonical implementation now lives in :mod:`channel_heads.models.device`.
This module is preserved so existing imports
(``channel_heads.inference.device`` and
``from channel_heads.inference import pick_device``), older notebooks, and
scripts keep working unchanged. New code should import from
:mod:`channel_heads.models.device`.
"""

from __future__ import annotations

from channel_heads.models.device import pick_device

__all__ = ["pick_device"]
