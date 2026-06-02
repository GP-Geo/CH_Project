"""Grouped train/test splitting shared by the regime training and retune steps.

The combined-model evaluation holds out whole outlets (``basin__outlet``) so no
outlet's pairs straddle the train/test boundary. This ``GroupShuffleSplit``
(test_size=0.20, random_state=42, grouped by ``basin__outlet``) was duplicated
in ``train_combined_xgb_regime.py`` and ``retune_threshold_regime.py``; the
canonical version lives here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

RANDOM_STATE = 42
TEST_SIZE = 0.20


def outlet_group_holdout(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    basin_col: str = "basin",
    outlet_col: str = "outlet",
    label_col: str = "y",
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Return ``(train_idx, test_idx)`` from a single grouped shuffle split.

    Groups are ``f"{basin}__{outlet}"`` so all pairs of an outlet land on the
    same side of the split. Matches the regime training/retune protocol.
    """
    groups = df[basin_col].astype(str) + "__" + df[outlet_col].astype(str)
    y = df[label_col].astype(int).to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    return train_idx, test_idx


def leave_one_group_out_oof(
    df: pd.DataFrame,
    features: list[str],
    fit_predict: Callable[[pd.DataFrame, npt.NDArray, pd.DataFrame], npt.NDArray],
    group_col: str = "basin",
    label_col: str = "y",
) -> tuple[npt.NDArray, npt.NDArray, pd.Series, list[float]]:
    """Leave-one-group-out out-of-fold probabilities.

    Each group (e.g. one basin) is held out in turn; ``fit_predict`` trains on
    the rest and predicts the held-out rows. Model-agnostic — the caller passes
    ``fit_predict(X_train, y_train, X_test) -> proba_test``.

    Returns ``(oof_proba, y, groups, fold_aucs)`` where ``oof_proba`` aligns
    positionally with ``df`` and ``fold_aucs`` holds per-fold ROC-AUC for folds
    whose held-out rows contain both classes.
    """
    X = df[features]
    y = df[label_col].astype(int).to_numpy()
    groups = df[group_col]
    oof = np.full(len(df), np.nan)
    fold_aucs: list[float] = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        p = fit_predict(X.iloc[tr], y[tr], X.iloc[te])
        oof[te] = p
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(float(roc_auc_score(y[te], p)))
    return oof, y, groups, fold_aucs
