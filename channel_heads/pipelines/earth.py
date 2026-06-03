"""Earth training pipeline — produces the models Mars inference consumes.

Read top to bottom for the training flow. The production artifacts
(``models/xgb_touching_classifier.json``, ``models/cnn_outlet_final.pt``) are
preserved as-is; a rebuild writes research/regime variants unless explicitly
intended (see ``docs/modeling.md``).
"""

from __future__ import annotations

import runpy
import sys

from channel_heads.io import paths
from channel_heads.logging_config import get_logger

log = get_logger("pipelines.earth")


def _run_cli(script_name: str, argv: list[str] | None = None) -> None:
    script = paths.PROJECT_ROOT / "scripts" / "cli" / script_name
    if not script.exists():
        raise FileNotFoundError(f"CLI script not found: {script}")
    log.info("running scripts/cli/%s", script_name)
    old_argv = sys.argv
    sys.argv = [str(script), *(argv or [])]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def train_earth_cnn() -> None:
    """Train + persist the Earth outlet CNN (models/cnn_outlet_final.pt)."""
    _run_cli("train_cnn_baseline.py", ["-v"])


def train_earth_xgb_variants() -> None:
    """Train the 3 Earth XGBoost variants (geom-only / +emb / +logit)."""
    _run_cli("train_combined_xgb_phase6b.py")


def train_earth_models() -> None:
    """Full Earth training: CNN then the XGBoost variants."""
    train_earth_cnn()
    train_earth_xgb_variants()


__all__ = ["train_earth_cnn", "train_earth_xgb_variants", "train_earth_models"]
