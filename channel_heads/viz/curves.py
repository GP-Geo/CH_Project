"""ROC/PR-curve plotting shared by the result-figure and diagnostics workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

RocEntry = tuple[str, npt.NDArray, npt.NDArray]  # (label, y_true, proba)


def roc_curve_panel(ax, entries: Sequence[RocEntry], title: str) -> None:
    """Draw one or more ROC curves (with AUC in the legend) on ``ax``.

    ``entries`` is a sequence of ``(label, y_true, proba)``. A chance diagonal is
    added and the axes are set square.
    """
    for name, y, p in entries:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")


def pr_curve_panel(ax, entries: Sequence[RocEntry], title: str) -> None:
    """Draw one or more precision-recall curves (with AP in the legend) on ``ax``.

    ``entries`` is a sequence of ``(label, y_true, proba)``. The no-skill
    baseline (positive prevalence) of the first entry is drawn as a dashed line.
    """
    for i, (name, y, p) in enumerate(entries):
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")
        if i == 0:
            ax.axhline(float(y.mean()), color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
