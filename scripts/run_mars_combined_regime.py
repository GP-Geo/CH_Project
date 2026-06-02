#!/usr/bin/env python
"""Step 6 (Mars calibration) — Run Mars inference under a regime's
combined model (geom + 4 CNN embeddings).

The Mars-side artifacts are reused as-is from Phase 1-5 (decided in the
regime selection: "Keep Mars as-is" — see ``phase_6c_combined_mars_
inference_summary.md`` for why). What changes per regime is:

  1. The CNN used to compute Mars patch embeddings:
     ``models/cnn_outlet_<regime>.pt`` (from Step 4).
  2. The combined XGBoost applied to (geom + emb):
     ``models/xgb_geom_plus_cnn_emb_<regime>.json`` (from Step 5).

The 5 dimensionless geometric features and the 5-class 128x128 patches
are taken verbatim from the existing Mars artifacts; no Mars topology
or feature derivation is re-run.

Outputs (per regime):
  data/Mars/model_outputs/mars_combined_<regime>_predictions.{parquet,csv}
  data/Mars/model_outputs/mars_combined_<regime>_predictions.gpkg
  data/Mars/model_outputs/mars_combined_<regime>_by_network.csv

Primary interface
-----------------
``notebooks/regime/01_mars_inference.ipynb`` is the primary, documented way to
understand this step; it calls the same shared package functions
(``channel_heads.inference.regime.attach_regime_embeddings`` plus the
``channel_heads.inference`` loaders / predict helpers). This script is the
headless batch wrapper that writes the per-regime prediction tables.

Run::

    python scripts/run_mars_combined_regime.py --regime regA
    python scripts/run_mars_combined_regime.py --regime regB
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from xgboost import XGBClassifier

from channel_heads.inference.regime import attach_regime_embeddings
from channel_heads.io.paths import PROJECT_ROOT
from channel_heads.models.device import pick_device
from channel_heads.models.xgboost import load_feature_columns, load_threshold
from channel_heads.regimes import REGIMES

log = logging.getLogger("run_mars_combined_regime")

# Existing Mars artifacts (unchanged across regimes).
INPUT_PARQUET = (
    PROJECT_ROOT / "data/Mars/model_inputs/mars_model_input_tabular_plus_cnn.parquet"
)
PATCH_INDEX_PARQUET = (
    PROJECT_ROOT / "data/Mars/model_inputs/mars_cnn_patch_index.parquet"
)
PAIRS_GPKG = PROJECT_ROOT / "data/Mars/topology/mars_vn_pairs.gpkg"
OUTPUT_DIR = PROJECT_ROOT / "data/Mars/model_outputs"

HIGH_CONF_PROB_MIN = 0.80

# Regime CNN embedding extraction + patch-index merge live in
# channel_heads.inference.regime (shared with notebooks/regime/).


def per_network_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-network counts: total pairs, predicted touching, high-conf, mean prob."""
    if "network_id" not in df.columns:
        log.warning("network_id missing; per-network summary skipped")
        return pd.DataFrame()
    rows = []
    for nid, sub in df.groupby("network_id"):
        rows.append(
            {
                "network_id": int(nid),
                "n_pairs": int(len(sub)),
                "n_touching": int(sub["pred_touching"].sum()),
                "n_high_conf": int((sub["prob_touching"] >= HIGH_CONF_PROB_MIN).sum()),
                "mean_prob": float(sub["prob_touching"].mean()),
                "max_prob": float(sub["prob_touching"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("network_id").reset_index(drop=True)


def write_gpkg(
    df: pd.DataFrame, gpkg_path: Path, pair_geometry: gpd.GeoDataFrame
) -> None:
    """Join predictions with Mars pair geometry and write a single-layer GPKG."""
    gdf = pair_geometry.merge(
        df[["pair_id", "prob_touching", "pred_touching"]],
        on="pair_id",
        how="inner",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        gpkg_path.unlink()
    gdf.to_file(gpkg_path, layer="pairs", driver="GPKG")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=sorted(REGIMES.keys()))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    regime = REGIMES[args.regime]

    cnn_path = PROJECT_ROOT / "models" / f"cnn_outlet_{regime.name}.pt"
    xgb_path = PROJECT_ROOT / "models" / f"xgb_geom_plus_cnn_emb_{regime.name}.json"
    feat_path = PROJECT_ROOT / "models" / f"feature_columns_geom_plus_cnn_emb_{regime.name}.txt"
    thr_path = PROJECT_ROOT / "models" / f"optimal_threshold_geom_plus_cnn_emb_{regime.name}.txt"
    for p in (cnn_path, xgb_path, feat_path, thr_path, INPUT_PARQUET, PATCH_INDEX_PARQUET):
        if not p.exists():
            log.error("Missing required artifact: %s", p)
            return 1

    device = pick_device()
    log.info("Device: %s | regime=%s", device, regime.name)

    log.info("Loading Mars tabular+CNN parquet: %s", INPUT_PARQUET)
    df_in = pd.read_parquet(INPUT_PARQUET)
    log.info(
        "Mars input: %d rows, %d networks",
        len(df_in),
        df_in["network_id"].nunique() if "network_id" in df_in.columns else -1,
    )

    df = attach_regime_embeddings(
        df_in, cnn_path, PATCH_INDEX_PARQUET, PROJECT_ROOT, device
    )

    feats = load_feature_columns(feat_path)
    threshold = load_threshold(thr_path)
    log.info("Model: %s | features=%d | threshold=%.6f", xgb_path.name, len(feats), threshold)

    missing = [c for c in feats if c not in df.columns]
    if missing:
        log.error("Feature columns missing from Mars table: %s", missing)
        return 1

    model = XGBClassifier()
    model.load_model(str(xgb_path))

    X = df[feats].to_numpy(dtype=float)
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        log.info("Feature matrix contains %d NaN cells — XGBoost handles natively.", n_nan)

    proba = model.predict_proba(X)[:, 1]
    df["prob_touching"] = proba
    df["pred_touching"] = (proba >= threshold).astype(int)

    n_touch = int(df["pred_touching"].sum())
    n_high = int((proba >= HIGH_CONF_PROB_MIN).sum())
    log.info(
        "Mars predictions: touching=%d/%d (%.1f%%), high-conf (>=%.2f)=%d (%.1f%%)",
        n_touch,
        len(df),
        100.0 * n_touch / len(df),
        HIGH_CONF_PROB_MIN,
        n_high,
        100.0 * n_high / len(df),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_pq = OUTPUT_DIR / f"mars_combined_{regime.name}_predictions.parquet"
    pred_csv = OUTPUT_DIR / f"mars_combined_{regime.name}_predictions.csv"
    by_net_csv = OUTPUT_DIR / f"mars_combined_{regime.name}_by_network.csv"
    pred_gpkg = OUTPUT_DIR / f"mars_combined_{regime.name}_predictions.gpkg"

    df.to_parquet(pred_pq, index=False)
    df.to_csv(pred_csv, index=False)
    log.info("Wrote -> %s, %s", pred_pq, pred_csv)

    by_net = per_network_summary(df)
    if not by_net.empty:
        by_net.to_csv(by_net_csv, index=False)
        log.info("Wrote per-network summary -> %s", by_net_csv)

    # Geo output: join with Mars pair polylines.
    # mars_vn_pairs.gpkg has two layers; "mars_pairs" is the MultiLineString
    # one (each row = one pair's head-to-head path).
    if PAIRS_GPKG.exists():
        try:
            pair_geo = gpd.read_file(PAIRS_GPKG, layer="mars_pairs")
            write_gpkg(df, pred_gpkg, pair_geo)
            log.info("Wrote -> %s", pred_gpkg)
        except Exception:  # noqa: BLE001
            log.exception("GPKG write failed; predictions parquet/csv are intact.")
    else:
        log.warning("Mars pairs GPKG missing (%s); skipping GPKG write.", PAIRS_GPKG)

    return 0


if __name__ == "__main__":
    _ = LineString  # silence unused-import nag (used only if/when geometry writes happen)
    raise SystemExit(main())
