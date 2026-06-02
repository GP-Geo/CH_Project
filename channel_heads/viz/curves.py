"""ROC-curve plotting shared by the result-figure and diagnostics workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy.typing as npt
from sklearn.metrics import roc_auc_score, roc_curve

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
