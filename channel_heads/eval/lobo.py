"""Leave-one-basin-out (LOBO) XGBoost report for the geom+CNN-embedding model.

Owns the LOBO-CV diagnostic that was inline in ``scripts/eval_lobo_cv.py``:
the per-config dataset map, the per-fold XGBoost fit, the leave-one-basin-out
out-of-fold probabilities, and the pooled-metrics summary. The generic
leave-one-group-out splitter and metric primitives stay in
:mod:`channel_heads.eval.splitting` / :mod:`channel_heads.eval.metrics`.

The fold AUC list includes only held-out basins that contain both classes
(handled by :func:`channel_heads.eval.splitting.leave_one_group_out_oof`).

This module deliberately uses its own XGBoost configuration — note it has **no**
``n_jobs`` or ``tree_method`` argument, unlike the training recipe in
:mod:`channel_heads.training.xgboost` — to preserve the exact LOBO behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from channel_heads.eval.metrics import (
    classification_metrics,
    f1_optimal_threshold,
    max_precision_threshold,
)
from channel_heads.eval.splitting import leave_one_group_out_oof

# Five geom features + four CNN embeddings, in the model's feature order.
GEOM = [
    "orientation_diff_deg",
    "headhead_dist_norm",
    "apex_angle_deg",
    "strahler_order_diff",
    "proximity_profile_norm",
]
EMB = [f"emb_{i}" for i in range(4)]
FEATURES = GEOM + EMB

N_ESTIMATORS, MAX_DEPTH, LR, SEED = 200, 4, 0.1, 42


def lobo_dataset_paths(root: str | Path) -> dict[str, Path]:
    """Return the ``config -> dataset CSV path`` map used by the LOBO report."""
    root = Path(root)
    return {
        "baseline": root / "data/results/master_dataset_v4_cnn_full.csv",
        "regA": root / "data/results/master_dataset_regA_with_emb.csv",
        "regB": root / "data/results/master_dataset_regB_with_emb.csv",
        "regC": root / "data/results/master_dataset_regC_with_emb.csv",
    }


def _fit_predict(X_tr: pd.DataFrame, y_tr: npt.NDArray, X_te: pd.DataFrame) -> np.ndarray:
    """Train the geom+emb XGBoost on a fold and predict P(touching)."""
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LR,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=SEED,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def lobo_xgb_report(df: pd.DataFrame) -> dict:
    """Leave-one-basin-out report for one dataset.

    Drops rows with NaN in ``FEATURES + ["y", "basin"]``, computes
    leave-one-basin-out OOF probabilities, picks the max-precision threshold
    (``min_recall=0.5``), and returns the pooled-metrics summary dict.
    """
    df = df.dropna(subset=FEATURES + ["y", "basin"]).reset_index(drop=True)
    oof, y, groups, fold_aucs = leave_one_group_out_oof(df, FEATURES, _fit_predict)
    thr = max_precision_threshold(y, oof)
    m = classification_metrics(y, oof, thr)
    return {
        "n": len(df),
        "n_basins": groups.nunique(),
        "pooled_auc": m["roc_auc"],
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "threshold": thr,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
    }


# =============================================================================
# True leave-one-basin-out engine (leakage-audited)
# =============================================================================
#
# The engine above (``lobo_xgb_report``) reads *precomputed* embeddings and tunes
# its threshold on pooled out-of-fold predictions, so it is **leakage-prone** for
# the regime datasets (the global CNN that produced ``emb_*`` was trained on 16
# of 17 basins, and the threshold sees the held-out data). The functions below
# implement a *true* LOBO: a model-agnostic engine that
#
#   * holds out one whole basin per fold,
#   * asserts zero train/test overlap of basin / outlet / pair / raster ids,
#   * tunes the decision threshold on the training basins only (inner grouped
#     split), and
#   * records the embedding provenance so CNN-embedding leakage is *flagged*
#     when it cannot be removed (precomputed mode) and *removed* when a per-fold
#     CNN factory is supplied.
#
# The engine is intentionally torch-free; the heavy per-fold CNN retraining lives
# in :mod:`channel_heads.eval.lobo_cnn` and is injected as a ``ModelFactory``.

# Columns that identify a candidate pair for leakage auditing.
PAIR_ID_COLS = ("basin", "outlet", "head_1", "head_2")

# Provenance strings describing how the CNN embedding features were produced.
PROV_NO_CNN = "geom_only (no CNN features)"
PROV_PRECOMPUTED = (
    "precomputed_global_cnn — LEAKAGE-PRONE: emb_* came from a CNN trained on "
    "16/17 basins, so the held-out basin's embeddings are contaminated"
)
PROV_PER_FOLD_CNN = "per_fold_cnn — CNN retrained per fold excluding the held-out basin"


class FoldModel(Protocol):
    """A fitted model that can score an arbitrary frame of pairs."""

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - protocol
        ...


# A factory fits a fresh model on the given training frame and returns it. The
# factory owns *all* model-specific preprocessing (including any per-fold CNN
# retrain + embedding extraction), so the engine never has to touch torch.
ModelFactory = Callable[[pd.DataFrame], FoldModel]


def make_pair_id(df: pd.DataFrame) -> pd.Series:
    """Stable per-pair id ``basin__outlet__minhead_maxhead``.

    Pairs are stored in normalized ``(min, max)`` head order already, but we sort
    defensively so the id is invariant to head order. Requires the columns in
    :data:`PAIR_ID_COLS`.
    """
    missing = [c for c in PAIR_ID_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"make_pair_id needs columns {PAIR_ID_COLS}; missing {missing}")
    h1 = df["head_1"].astype("int64").to_numpy()
    h2 = df["head_2"].astype("int64").to_numpy()
    lo = np.minimum(h1, h2).astype(str)
    hi = np.maximum(h1, h2).astype(str)
    return pd.Series(
        df["basin"].astype(str).to_numpy()
        + "__"
        + df["outlet"].astype(str).to_numpy()
        + "__"
        + lo
        + "_"
        + hi,
        index=df.index,
    )


@dataclass
class FoldAudit:
    """Per-fold record proving train/test disjointness."""

    test_basin: str
    n_train: int
    n_test: int
    n_train_basins: int
    basins_disjoint: bool
    outlets_disjoint: bool
    pairs_disjoint: bool
    rasters_disjoint: bool
    embedding_provenance: str

    @property
    def clean(self) -> bool:
        return (
            self.basins_disjoint
            and self.outlets_disjoint
            and self.pairs_disjoint
            and self.rasters_disjoint
        )


def audit_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    test_basin: str,
    embedding_provenance: str,
    assert_clean: bool = True,
) -> FoldAudit:
    """Audit one LOBO fold and (by default) assert zero train/test id overlap.

    Checks disjointness of basin ids, ``basin__outlet`` ids, pair ids, and
    ``raster_path`` records. Raises ``AssertionError`` on any overlap when
    ``assert_clean`` is true. Note: this proves *row-level* separation; it
    **cannot** detect contamination baked into precomputed embedding columns —
    that is surfaced via ``embedding_provenance`` instead.
    """
    tr_basins = set(df_train["basin"].astype(str))
    te_basins = set(df_test["basin"].astype(str))
    basins_disjoint = tr_basins.isdisjoint(te_basins)

    def _outlet_ids(d: pd.DataFrame) -> set[str]:
        return set(d["basin"].astype(str) + "__" + d["outlet"].astype(str))

    outlets_disjoint = _outlet_ids(df_train).isdisjoint(_outlet_ids(df_test))
    pairs_disjoint = set(make_pair_id(df_train)).isdisjoint(set(make_pair_id(df_test)))

    if "raster_path" in df_train.columns and "raster_path" in df_test.columns:
        tr_r = set(df_train["raster_path"].dropna().astype(str))
        te_r = set(df_test["raster_path"].dropna().astype(str))
        rasters_disjoint = tr_r.isdisjoint(te_r)
    else:
        rasters_disjoint = True  # n/a for geom-only datasets

    audit = FoldAudit(
        test_basin=test_basin,
        n_train=int(len(df_train)),
        n_test=int(len(df_test)),
        n_train_basins=int(len(tr_basins)),
        basins_disjoint=basins_disjoint,
        outlets_disjoint=outlets_disjoint,
        pairs_disjoint=pairs_disjoint,
        rasters_disjoint=rasters_disjoint,
        embedding_provenance=embedding_provenance,
    )
    if assert_clean:
        assert basins_disjoint, f"[{test_basin}] basin id leaked across train/test"
        assert outlets_disjoint, f"[{test_basin}] outlet id leaked across train/test"
        assert pairs_disjoint, f"[{test_basin}] pair id leaked across train/test"
        assert rasters_disjoint, f"[{test_basin}] raster record leaked across train/test"
    return audit


def tune_threshold_train_only(
    df_train: pd.DataFrame,
    factory: ModelFactory,
    *,
    policy: str = "max_precision",
    min_recall: float = 0.5,
    inner_test_size: float = 0.25,
    seed: int = 42,
    group_col: str = "basin",
    label_col: str = "y",
) -> float:
    """Pick a decision threshold using the training basins only.

    Splits the training basins into inner-train / inner-val by **whole basin**
    (``GroupShuffleSplit`` on ``group_col``), fits a fresh model on inner-train,
    scores inner-val, and selects the threshold there. No held-out (outer) basin
    data is touched. Falls back to ``0.5`` when the inner split cannot yield two
    classes on both sides. ``policy`` is ``"max_precision"`` (production) or
    ``"f1"``.
    """
    groups = df_train[group_col].astype(str)
    if groups.nunique() < 2:
        return 0.5
    gss = GroupShuffleSplit(n_splits=1, test_size=inner_test_size, random_state=seed)
    itr, iva = next(gss.split(df_train, df_train[label_col], groups=groups))
    inner_train = df_train.iloc[itr]
    inner_val = df_train.iloc[iva]
    y_val = inner_val[label_col].astype(int).to_numpy()
    if inner_train[label_col].nunique() < 2 or len(np.unique(y_val)) < 2:
        return 0.5
    model = factory(inner_train)
    proba_val = model.predict_proba(inner_val)
    if policy == "f1":
        return f1_optimal_threshold(y_val, proba_val)[0]
    return max_precision_threshold(y_val, proba_val, min_recall=min_recall)


def threshold_from_fitted_model(
    model: FoldModel,
    df_train: pd.DataFrame,
    *,
    policy: str = "max_precision",
    min_recall: float = 0.5,
    inner_test_size: float = 0.25,
    seed: int = 42,
    group_col: str = "basin",
    label_col: str = "y",
) -> float:
    """Cheaper threshold: score a grouped inner slice with an *already-fitted* model.

    Used when refitting the model just to tune the threshold is too expensive
    (per-fold CNN). The inner-val basins are a subset of the training basins, so
    the held-out (outer) basin is still never touched — but the model has *seen*
    those rows in training, so the threshold is mildly optimistic. AUC / PR-AUC
    are threshold-free and therefore unaffected by this choice; only the
    thresholded metrics (precision/recall/F1/accuracy) are.
    """
    groups = df_train[group_col].astype(str)
    if groups.nunique() < 2:
        return 0.5
    gss = GroupShuffleSplit(n_splits=1, test_size=inner_test_size, random_state=seed)
    _, iva = next(gss.split(df_train, df_train[label_col], groups=groups))
    inner_val = df_train.iloc[iva]
    y_val = inner_val[label_col].astype(int).to_numpy()
    if len(np.unique(y_val)) < 2:
        return 0.5
    proba_val = model.predict_proba(inner_val)
    if policy == "f1":
        return f1_optimal_threshold(y_val, proba_val)[0]
    return max_precision_threshold(y_val, proba_val, min_recall=min_recall)


def _safe_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Threshold metrics + ranking metrics, tolerant of single-class folds."""
    pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y)) > 1
    return {
        "roc_auc": float(roc_auc_score(y, proba)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if both_classes else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


@dataclass
class LoboResult:
    predictions: pd.DataFrame
    per_basin: pd.DataFrame
    summary: dict
    audits: list[FoldAudit]
    embedding_provenance: str

    @property
    def all_folds_clean(self) -> bool:
        return all(a.clean for a in self.audits)


def run_lobo(
    df: pd.DataFrame,
    features: list[str],
    factory: ModelFactory,
    *,
    group_col: str = "basin",
    label_col: str = "y",
    embedding_provenance: str = "unspecified",
    threshold_policy: str = "max_precision",
    min_recall: float = 0.5,
    inner_test_size: float = 0.25,
    seed: int = 42,
    assert_clean: bool = True,
    refit_for_threshold: bool = True,
) -> LoboResult:
    """Run a true leave-one-basin-out evaluation.

    For each basin: hold it out, audit train/test disjointness, tune the
    threshold on the training basins only, fit ``factory`` on all training
    basins, and score the held-out basin. Returns per-row out-of-fold
    predictions, per-basin metrics, an aggregate summary (mean/std/median +
    pooled), and the fold audits.

    ``features`` is used for NaN-dropping and is the column set the ``factory``
    consumes; the factory is responsible for actually selecting them.

    ``refit_for_threshold`` (default ``True``) fits a *separate* inner model on a
    grouped slice of the training basins to pick the threshold — fully out of
    sample. Set ``False`` for expensive factories (per-fold CNN) to instead score
    the inner slice with the already-fitted fold model: ~half the compute, the
    held-out basin is still never touched, and AUC / PR-AUC are unchanged (only
    the thresholded metrics become mildly optimistic).
    """
    df = df.dropna(subset=[c for c in features if c in df.columns] + [label_col, group_col])
    df = df.reset_index(drop=True)
    basins = sorted(df[group_col].astype(str).unique())

    pred_frames: list[pd.DataFrame] = []
    per_basin_rows: list[dict] = []
    audits: list[FoldAudit] = []

    for basin in basins:
        te_mask = df[group_col].astype(str) == basin
        df_test = df[te_mask]
        df_train = df[~te_mask]

        audits.append(
            audit_fold(
                df_train,
                df_test,
                test_basin=basin,
                embedding_provenance=embedding_provenance,
                assert_clean=assert_clean,
            )
        )

        if refit_for_threshold:
            threshold = tune_threshold_train_only(
                df_train,
                factory,
                policy=threshold_policy,
                min_recall=min_recall,
                inner_test_size=inner_test_size,
                seed=seed,
                group_col=group_col,
                label_col=label_col,
            )

        model = factory(df_train)
        if not refit_for_threshold:
            threshold = threshold_from_fitted_model(
                model,
                df_train,
                policy=threshold_policy,
                min_recall=min_recall,
                inner_test_size=inner_test_size,
                seed=seed,
                group_col=group_col,
                label_col=label_col,
            )
        proba = np.asarray(model.predict_proba(df_test), dtype=float)
        y = df_test[label_col].astype(int).to_numpy()

        pred_frames.append(
            pd.DataFrame(
                {
                    "pair_id": make_pair_id(df_test).to_numpy(),
                    "basin": basin,
                    "outlet": df_test["outlet"].to_numpy(),
                    "y": y,
                    "proba": proba,
                    "fold_threshold": threshold,
                    "pred": (proba >= threshold).astype(int),
                }
            )
        )

        mets = _safe_metrics(y, proba, threshold)
        per_basin_rows.append(
            {
                "basin": basin,
                "n_pairs": int(len(y)),
                "prevalence": float(y.mean()),
                "threshold": float(threshold),
                **mets,
            }
        )

    predictions = pd.concat(pred_frames, ignore_index=True)
    per_basin = pd.DataFrame(per_basin_rows)
    summary = summarize_lobo(per_basin, predictions)
    return LoboResult(predictions, per_basin, summary, audits, embedding_provenance)


def summarize_lobo(per_basin: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Aggregate per-basin metrics (mean/std/median, NaN-safe) + pooled metrics.

    Per-fold stats summarise the per-basin distribution; pooled stats concatenate
    every fold's out-of-fold predictions, each thresholded at *its own* fold
    threshold, and score them together.
    """
    metric_cols = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
    summary: dict = {}
    for col in metric_cols:
        vals = per_basin[col].to_numpy(dtype=float)
        summary[f"{col}_mean"] = float(np.nanmean(vals))
        summary[f"{col}_std"] = float(np.nanstd(vals))
        summary[f"{col}_median"] = float(np.nanmedian(vals))

    y = predictions["y"].to_numpy()
    proba = predictions["proba"].to_numpy()
    pred = predictions["pred"].to_numpy()
    both = len(np.unique(y)) > 1
    summary["pooled_roc_auc"] = float(roc_auc_score(y, proba)) if both else float("nan")
    summary["pooled_pr_auc"] = float(average_precision_score(y, proba)) if both else float("nan")
    summary["pooled_f1"] = float(f1_score(y, pred, zero_division=0))
    summary["pooled_precision"] = float(precision_score(y, pred, zero_division=0))
    summary["pooled_recall"] = float(recall_score(y, pred, zero_division=0))
    summary["pooled_accuracy"] = float(accuracy_score(y, pred))
    summary["n_pairs"] = int(len(predictions))
    summary["n_basins"] = int(len(per_basin))
    summary["n_basins_scored"] = int(per_basin["roc_auc"].notna().sum())
    return summary


def make_xgb_factory(
    features: list[str],
    *,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = MAX_DEPTH,
    learning_rate: float = LR,
    seed: int = SEED,
    label_col: str = "y",
) -> ModelFactory:
    """A :data:`ModelFactory` that fits the geom(+emb) XGBoost on a training frame.

    The returned model selects ``features`` itself, so the engine can hand it the
    full frame for both threshold tuning and held-out scoring.
    """

    class _XGBFoldModel:
        def __init__(self, model: XGBClassifier) -> None:
            self._model = model

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return self._model.predict_proba(frame[features])[:, 1]

    def factory(df_train: pd.DataFrame) -> FoldModel:
        y = df_train[label_col].astype(int).to_numpy()
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=seed,
        )
        model.fit(df_train[features], y)
        return _XGBFoldModel(model)

    return factory


def save_lobo_result(
    result: LoboResult,
    outdir: str | Path,
    *,
    dataset_label: str,
    features: list[str],
) -> Path:
    """Persist fold assignments, predictions, metrics, and a leakage-audit report.

    Writes into ``outdir``: ``fold_assignments.csv``, ``predictions.csv``
    (+ ``.parquet`` if available), ``metrics_per_basin.csv``,
    ``metrics_summary.csv``, ``leakage_audit.json``, and ``leakage_audit.md``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fold_assignments = result.predictions[["pair_id", "basin", "outlet"]].copy()
    fold_assignments["test_fold"] = fold_assignments["basin"]
    fold_assignments.to_csv(outdir / "fold_assignments.csv", index=False)

    result.predictions.to_csv(outdir / "predictions.csv", index=False)
    try:
        result.predictions.to_parquet(outdir / "predictions.parquet", index=False)
    except Exception:  # pragma: no cover - optional parquet engine
        pass

    result.per_basin.to_csv(outdir / "metrics_per_basin.csv", index=False)
    pd.DataFrame([result.summary]).to_csv(outdir / "metrics_summary.csv", index=False)

    leakage_prone = "LEAKAGE-PRONE" in result.embedding_provenance
    audit = {
        "dataset": dataset_label,
        "features": features,
        "protocol": "leave-one-basin-out",
        "threshold_tuning": "train basins only (inner grouped split)",
        "embedding_provenance": result.embedding_provenance,
        "embedding_leakage_risk": leakage_prone,
        "all_folds_row_disjoint": result.all_folds_clean,
        "n_basins": int(len(result.audits)),
        "folds": [asdict(a) for a in result.audits],
    }
    (outdir / "leakage_audit.json").write_text(json.dumps(audit, indent=2))
    (outdir / "leakage_audit.md").write_text(_audit_markdown(audit, result))
    return outdir


def _audit_markdown(audit: dict, result: LoboResult) -> str:
    lines = [
        f"# LOBO leakage audit — {audit['dataset']}",
        "",
        f"- **Protocol:** {audit['protocol']}",
        f"- **Threshold tuning:** {audit['threshold_tuning']}",
        f"- **Embedding provenance:** {audit['embedding_provenance']}",
        f"- **Row-level disjoint (all folds):** {audit['all_folds_row_disjoint']}",
        f"- **Embedding-leakage risk:** {audit['embedding_leakage_risk']}",
        "",
    ]
    if audit["embedding_leakage_risk"]:
        lines += [
            "> ⚠️ **Embedding leakage flagged.** Row ids are disjoint, but the "
            "`emb_*` features were produced by a CNN that saw the held-out basin. "
            "Treat these scores as optimistic; use `per_fold_cnn` mode for a clean "
            "estimate.",
            "",
        ]
    lines += [
        "## Per-fold disjointness",
        "",
        "| held-out basin | n_train | n_test | basins | outlets | pairs | rasters |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in result.audits:
        lines.append(
            f"| {a.test_basin} | {a.n_train} | {a.n_test} | "
            f"{'✅' if a.basins_disjoint else '❌'} | "
            f"{'✅' if a.outlets_disjoint else '❌'} | "
            f"{'✅' if a.pairs_disjoint else '❌'} | "
            f"{'✅' if a.rasters_disjoint else '❌'} |"
        )
    s = result.summary
    lines += [
        "",
        "## Aggregate (held-out folds)",
        "",
        f"- ROC AUC: mean {s['roc_auc_mean']:.3f} ± {s['roc_auc_std']:.3f}, "
        f"median {s['roc_auc_median']:.3f} | pooled {s['pooled_roc_auc']:.3f}",
        f"- PR AUC: mean {s['pr_auc_mean']:.3f} ± {s['pr_auc_std']:.3f}, "
        f"median {s['pr_auc_median']:.3f} | pooled {s['pooled_pr_auc']:.3f}",
        f"- F1: mean {s['f1_mean']:.3f} | pooled {s['pooled_f1']:.3f}",
        f"- basins scored: {s['n_basins_scored']}/{s['n_basins']} | pairs: {s['n_pairs']}",
        "",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "GEOM",
    "EMB",
    "FEATURES",
    "lobo_dataset_paths",
    "lobo_xgb_report",
    # True LOBO engine
    "PAIR_ID_COLS",
    "PROV_NO_CNN",
    "PROV_PRECOMPUTED",
    "PROV_PER_FOLD_CNN",
    "FoldModel",
    "ModelFactory",
    "FoldAudit",
    "LoboResult",
    "make_pair_id",
    "audit_fold",
    "tune_threshold_train_only",
    "threshold_from_fitted_model",
    "run_lobo",
    "summarize_lobo",
    "make_xgb_factory",
    "save_lobo_result",
]
