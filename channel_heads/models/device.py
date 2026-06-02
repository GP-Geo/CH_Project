"""Torch device selection for the model/inference layer.

Canonical home of :func:`pick_device`, used by the CNN-using inference and
embedding paths. Torch is imported lazily inside the function so importing this
module never requires PyTorch.

The historical import location :mod:`channel_heads.inference.device` re-exports
this as a compatibility shim.
"""

from __future__ import annotations


def pick_device() -> str:
    """Return the best available torch device: ``mps`` > ``cuda`` > ``cpu``."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


__all__ = ["pick_device"]
