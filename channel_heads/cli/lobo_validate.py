#!/usr/bin/env python
"""True leave-one-basin-out (LOBO) validation for the regime geom(+CNN) model.

For one regime, every Earth basin is held out in turn; the model trains on the
rest, tunes its decision threshold on the *training* basins only (inner grouped
split), and scores the held-out basin. Writes fold assignments, per-row
out-of-fold predictions, per-basin + aggregate metrics, and a leakage-audit
report under ``data/results/lobo/<regime>/<mode>/``.

Modes trade rigour for cost:

  * ``geom_only``       5 geometric features, no CNN — leakage-free, fast.
  * ``precomputed_emb`` geom + existing ``emb_*`` columns — **FLAGGED leakage-prone**
                        (those embeddings came from a CNN trained on the held-out
                        basins); fast, for comparison only.
  * ``per_fold_cnn``    geom + a CNN retrained inside every fold — leakage-free,
                        slow (one CNN train per fold).

Run::

    python -m channel_heads lobo-validate --regime regA --mode geom_only
    python -m channel_heads lobo-validate --regime regA --mode per_fold_cnn --epochs 40
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from channel_heads.eval import lobo
from channel_heads.io.paths import RESULTS_DIR
from channel_heads.regimes import REGIMES

log = logging.getLogger("lobo_validate")

MODES = ("geom_only", "precomputed_emb", "per_fold_cnn")


def _resolve_mode(mode: str, epochs: int):
    """Return ``(nan_drop_features, documented_features, factory, provenance)``."""
    if mode == "geom_only":
        return lobo.GEOM, lobo.GEOM, lobo.make_xgb_factory(lobo.GEOM), lobo.PROV_NO_CNN
    if mode == "precomputed_emb":
        return (
            lobo.FEATURES,
            lobo.FEATURES,
            lobo.make_xgb_factory(lobo.FEATURES),
            lobo.PROV_PRECOMPUTED,
        )
    # per_fold_cnn — heavy, leakage-free
    from channel_heads.eval.lobo_cnn import make_per_fold_cnn_factory

    return (
        lobo.GEOM,  # NaN-drop on geom only; embeddings are computed per fold
        lobo.FEATURES,
        make_per_fold_cnn_factory(epochs=epochs),
        lobo.PROV_PER_FOLD_CNN,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=sorted(REGIMES.keys()))
    parser.add_argument("--mode", choices=MODES, default="geom_only")
    parser.add_argument("--dataset", default=None, help="Override dataset CSV path.")
    parser.add_argument("--out", default=str(RESULTS_DIR / "lobo"), help="Output root.")
    parser.add_argument(
        "--threshold-policy", choices=("max_precision", "f1"), default="max_precision"
    )
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=40, help="CNN epochs (per_fold_cnn).")
    parser.add_argument(
        "--rigorous-threshold",
        action="store_true",
        help="Refit a separate inner model per fold for threshold tuning "
        "(fully out-of-sample). Default for per_fold_cnn is the cheaper in-sample "
        "threshold, which leaves AUC/PR-AUC unchanged and halves CNN trains.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    regime = REGIMES[args.regime]
    dataset = Path(args.dataset) if args.dataset else (
        RESULTS_DIR / f"master_dataset_{regime.name}_with_emb.csv"
    )
    if not dataset.exists():
        log.error("Missing dataset: %s", dataset)
        return 1

    df = pd.read_csv(dataset)
    nan_drop, documented, factory, provenance = _resolve_mode(args.mode, args.epochs)

    print(
        f"LOBO {regime.name} [{args.mode}] — {df['basin'].nunique()} basins, "
        f"{len(df)} pairs; threshold policy={args.threshold_policy}"
    )
    # Per-fold CNN is expensive — default to the in-sample threshold (AUC/PR-AUC
    # identical) unless the user asks for the rigorous refit.
    refit_for_threshold = args.rigorous_threshold or args.mode != "per_fold_cnn"
    result = lobo.run_lobo(
        df,
        nan_drop,
        factory,
        embedding_provenance=provenance,
        threshold_policy=args.threshold_policy,
        min_recall=args.min_recall,
        refit_for_threshold=refit_for_threshold,
    )

    outdir = Path(args.out) / regime.name / args.mode
    lobo.save_lobo_result(result, outdir, dataset_label=f"{regime.name}/{args.mode}", features=documented)

    s = result.summary
    print(
        f"  ROC AUC: mean {s['roc_auc_mean']:.3f}±{s['roc_auc_std']:.3f} "
        f"median {s['roc_auc_median']:.3f} pooled {s['pooled_roc_auc']:.3f}\n"
        f"  PR  AUC: mean {s['pr_auc_mean']:.3f} pooled {s['pooled_pr_auc']:.3f}\n"
        f"  F1     : mean {s['f1_mean']:.3f} pooled {s['pooled_f1']:.3f}\n"
        f"  basins scored {s['n_basins_scored']}/{s['n_basins']}; "
        f"row-disjoint={result.all_folds_clean}; provenance={provenance}\n"
        f"  -> {outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
