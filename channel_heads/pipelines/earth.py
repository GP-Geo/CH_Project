"""Earth training pipeline — produces the models Mars inference consumes.

Read top to bottom for the training flow. The production artifacts
(``models/xgb_touching_classifier.json``, ``models/cnn_outlet_final.pt``) are
preserved as-is; a rebuild writes research/regime variants unless explicitly
intended (see ``docs/modeling.md``).
"""

from __future__ import annotations

from channel_heads.pipelines._delegate import run_script


def train_earth_cnn() -> None:
    """Train + persist the Earth outlet CNN (models/cnn_outlet_final.pt).

    TRANSITIONAL — runs ``scripts/train_cnn_baseline.py``.
    """
    run_script("train_cnn_baseline.py", ["-v"])


def train_earth_xgb_variants() -> None:
    """Train the 3 Earth XGBoost variants (geom-only / +emb / +logit).

    TRANSITIONAL — runs ``scripts/train_combined_xgb_phase6b.py``.
    """
    run_script("train_combined_xgb_phase6b.py")


def train_earth_models() -> None:
    """Full Earth training: CNN then the XGBoost variants."""
    train_earth_cnn()
    train_earth_xgb_variants()


__all__ = ["train_earth_cnn", "train_earth_xgb_variants", "train_earth_models"]
