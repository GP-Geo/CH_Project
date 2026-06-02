"""Torch device selection shared by the CNN-using inference scripts."""

from __future__ import annotations


def pick_device() -> str:
    """Return the best available torch device: ``mps`` > ``cuda`` > ``cpu``."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
