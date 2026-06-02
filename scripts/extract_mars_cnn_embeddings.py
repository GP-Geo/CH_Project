#!/usr/bin/env python
"""Phase 5 — Extract CNN embeddings for Mars 5-class patches.

Loads the Earth-trained CNN (``models/cnn_outlet_final.pt``), runs Mars
Phase-4 patches through it in inference mode, and exports a per-pair
embedding table plus the tabular-+-CNN combined model-input table.

Earth pipeline pieces reused / mirrored:
  - channel_heads.cnn_model.OutletCNN
      Same architecture (5→16→32→64→4 dim embedding)
  - channel_heads.cnn_model.OutletPairDataset
      Loads ``.npy``, applies ``encode_raster_onehot`` → float32
      (5, 128, 128). ``augment=False`` for inference.
  - channel_heads.cnn_model.encode_raster_onehot
      Per-class one-hot. NO mean/std normalization.
  - channel_heads.cnn_features.extract_embeddings
      Same function: load_state_dict + model.eval() + model.embed(x).
      We call it directly so the preprocessing path is exactly Earth's.
  - channel_heads.cnn_features.CNN_FEATURE_COLS == ["emb_0","emb_1","emb_2","emb_3"]

Phase 5 only. Does NOT retrain the CNN, regenerate patches, or rerun
XGBoost.

Outputs:
  data/Mars/model_inputs/mars_cnn_embeddings.parquet
  data/Mars/model_inputs/mars_cnn_embeddings.csv
  data/Mars/model_inputs/mars_model_input_tabular_plus_cnn.parquet
  data/Mars/model_inputs/mars_model_input_tabular_plus_cnn.csv
  data/Mars/model_inputs/cnn_patches_5class/figures/emb_scatter_emb0_emb1.png
  data/Mars/model_inputs/cnn_patches_5class/figures/emb_scatter_emb2_emb3.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Reuse Earth code so preprocessing + architecture are guaranteed identical.
from channel_heads.cnn_features import CNN_FEATURE_COLS, extract_embeddings  # noqa: E402
from channel_heads.cnn_model import DEFAULT_EMBEDDING_DIM, OutletCNN  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATCH_INDEX_PARQUET = (
    PROJECT_ROOT / "data/Mars/model_inputs/mars_cnn_patch_index.parquet"
)
FEATURES_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_inputs/mars_pair_features_5feat_model_ready.parquet"
)
PREDICTIONS_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_outputs/mars_xgb_predictions_5feat.parquet"
)
MODEL_PATH = PROJECT_ROOT / "models/cnn_outlet_final.pt"

OUTPUT_EMB_PQ = (
    PROJECT_ROOT / "data/Mars/model_inputs/mars_cnn_embeddings.parquet"
)
OUTPUT_EMB_CSV = (
    PROJECT_ROOT / "data/Mars/model_inputs/mars_cnn_embeddings.csv"
)
OUTPUT_COMBINED_PQ = (
    PROJECT_ROOT
    / "data/Mars/model_inputs/mars_model_input_tabular_plus_cnn.parquet"
)
OUTPUT_COMBINED_CSV = (
    PROJECT_ROOT
    / "data/Mars/model_inputs/mars_model_input_tabular_plus_cnn.csv"
)
FIGURES_DIR = (
    PROJECT_ROOT
    / "data/Mars/model_inputs/cnn_patches_5class/figures"
)

BATCH_SIZE = 64

log = logging.getLogger("mars_cnn_emb")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_emb_dataframe(
    df_emb: pd.DataFrame, expected_n: int, emb_cols: list[str]
) -> None:
    if len(df_emb) != expected_n:
        raise RuntimeError(
            f"Embedding row count mismatch: got {len(df_emb)}, expected {expected_n}"
        )
    if df_emb["pair_id"].isna().any():
        raise RuntimeError("Some embedding rows have missing pair_id")
    if df_emb["pair_id"].duplicated().any():
        raise RuntimeError("Duplicate pair_id in embedding table")
    for c in emb_cols:
        if c not in df_emb.columns:
            raise RuntimeError(f"Missing embedding column: {c}")
        if not np.issubdtype(df_emb[c].dtype, np.number):
            raise RuntimeError(f"Embedding column {c} is non-numeric")
        arr = df_emb[c].to_numpy(dtype=float)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan or n_inf:
            raise RuntimeError(
                f"Embedding column {c} contains NaN={n_nan} / Inf={n_inf}"
            )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def scatter_embeddings(
    df_emb: pd.DataFrame,
    color_values: np.ndarray | None,
    color_label: str | None,
    xcol: str,
    ycol: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if color_values is not None:
        sc = ax.scatter(
            df_emb[xcol],
            df_emb[ycol],
            c=color_values,
            cmap="viridis",
            s=14,
            alpha=0.7,
            edgecolors="none",
        )
        cb = plt.colorbar(sc, ax=ax)
        if color_label:
            cb.set_label(color_label)
    else:
        ax.scatter(
            df_emb[xcol],
            df_emb[ycol],
            s=14,
            alpha=0.7,
            edgecolors="none",
            color="#4C78A8",
        )
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(
        f"Mars CNN embeddings: {xcol} vs {ycol} (n={len(df_emb):,})"
    )
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()

    device = pick_device()
    log.info("Device: %s", device)
    log.info("Loading patch index: %s", PATCH_INDEX_PARQUET)
    idx_df = pd.read_parquet(PATCH_INDEX_PARQUET)
    n_total = len(idx_df)
    n_ok = int((idx_df["patch_status"] == "ok").sum())
    log.info(
        "Patch index: %d rows (status: %s)",
        n_total,
        idx_df["patch_status"].value_counts().to_dict(),
    )

    usable = idx_df[idx_df["patch_status"] == "ok"].copy().reset_index(drop=True)
    log.info("Patches usable for embedding: %d", len(usable))

    # The Earth extract_embeddings() reads from manifest_df["raster_path"].
    # Map our patch_path to raster_path (as absolute paths) so we can pass
    # raster_dir=None and avoid path rewriting downstream.
    usable["raster_path"] = [
        str((PROJECT_ROOT / p).resolve()) for p in usable["patch_path"]
    ]

    # Quick sanity: confirm a sample of the .npy patches load with the
    # expected shape/dtype before sending them to the CNN.
    sample_rows = usable.head(3)
    for _, r in sample_rows.iterrows():
        p = np.load(r["raster_path"])
        if p.shape != (128, 128) or p.dtype != np.uint8:
            raise RuntimeError(
                f"Unexpected patch shape/dtype for {r['pair_id']}: "
                f"{p.shape}/{p.dtype} (expected (128,128)/uint8)"
            )

    # Confirm Earth's published embedding columns and dim.
    log.info(
        "Embedding dim from cnn_model: %d, columns: %s",
        DEFAULT_EMBEDDING_DIM,
        CNN_FEATURE_COLS,
    )
    if CNN_FEATURE_COLS != [f"emb_{i}" for i in range(DEFAULT_EMBEDDING_DIM)]:
        raise RuntimeError(
            f"Embedding column names from cnn_features.py {CNN_FEATURE_COLS} "
            f"do not match the expected emb_0..emb_{DEFAULT_EMBEDDING_DIM-1}"
        )

    # Verify CNN weights load into the expected architecture before
    # running the full pipeline.
    test_model = OutletCNN(embedding_dim=DEFAULT_EMBEDDING_DIM)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    missing, unexpected = test_model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"CNN state-dict mismatch: missing={missing} unexpected={unexpected}"
        )
    log.info("CNN state-dict loaded OK into OutletCNN(embedding_dim=%d)", DEFAULT_EMBEDDING_DIM)

    # --- Run Earth's extract_embeddings ----------------------------------
    log.info(
        "Extracting embeddings (batch_size=%d, device=%s) using %s",
        BATCH_SIZE,
        device,
        MODEL_PATH,
    )
    result_df = extract_embeddings(
        model_path=MODEL_PATH,
        raster_dir=None,
        manifest_df=usable,
        batch_size=BATCH_SIZE,
        device=device,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
    )

    # --- Assemble the embedding table -----------------------------------
    df_emb = result_df[
        [
            "network_id",
            "pair_id",
            "patch_path",
        ]
        + CNN_FEATURE_COLS
    ].copy()
    df_emb["embedding_status"] = "ok"
    df_emb["embedding_qa_reason"] = ""

    # Add rows for patches that failed to embed (none expected — all ok).
    # If usable < idx_df.size then add the skipped/failed rows back with
    # NaN embeddings and an explicit status so the table mirrors the
    # patch index 1:1.
    if len(df_emb) < n_total:
        missing_rows = idx_df[~idx_df["pair_id"].isin(df_emb["pair_id"])]
        for _, mr in missing_rows.iterrows():
            df_emb = pd.concat(
                [
                    df_emb,
                    pd.DataFrame(
                        [
                            {
                                "network_id": int(mr["network_id"]),
                                "pair_id": str(mr["pair_id"]),
                                "patch_path": str(mr["patch_path"]),
                                **{c: np.nan for c in CNN_FEATURE_COLS},
                                "embedding_status": "skipped",
                                "embedding_qa_reason": (
                                    f"patch_status={mr['patch_status']}"
                                    + (
                                        f";{mr['patch_qa_reason']}"
                                        if mr["patch_qa_reason"]
                                        else ""
                                    )
                                ),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # --- Validate (only on ok rows) -------------------------------------
    ok_rows = df_emb[df_emb["embedding_status"] == "ok"].copy()
    validate_emb_dataframe(ok_rows, expected_n=len(usable), emb_cols=CNN_FEATURE_COLS)

    # --- Embedding summary stats ----------------------------------------
    log.info("Embedding summary stats (ok rows, n=%d):", len(ok_rows))
    for c in CNN_FEATURE_COLS:
        s = ok_rows[c]
        log.info(
            "  %s: mean=%.4f std=%.4f min=%.4f median=%.4f max=%.4f",
            c,
            float(s.mean()),
            float(s.std()),
            float(s.min()),
            float(s.median()),
            float(s.max()),
        )

    # --- Persist embedding table ---------------------------------------
    df_emb = df_emb[
        [
            "network_id",
            "pair_id",
            "patch_path",
            *CNN_FEATURE_COLS,
            "embedding_status",
            "embedding_qa_reason",
        ]
    ]
    df_emb.to_parquet(OUTPUT_EMB_PQ, index=False)
    df_emb.to_csv(OUTPUT_EMB_CSV, index=False)
    log.info("Wrote: %s", OUTPUT_EMB_PQ)
    log.info("Wrote: %s", OUTPUT_EMB_CSV)

    # --- Merge with Phase 3A tabular features --------------------------
    log.info("Merging with tabular features: %s", FEATURES_PARQUET)
    tabular = pd.read_parquet(FEATURES_PARQUET)
    combined = tabular.merge(
        df_emb[
            ["pair_id", *CNN_FEATURE_COLS, "embedding_status", "embedding_qa_reason"]
        ],
        on="pair_id",
        how="left",
    )

    n_matched = int(combined["embedding_status"].eq("ok").sum())
    log.info(
        "Combined table: %d rows; %d matched with ok embeddings",
        len(combined),
        n_matched,
    )
    combined.to_parquet(OUTPUT_COMBINED_PQ, index=False)
    combined.to_csv(OUTPUT_COMBINED_CSV, index=False)
    log.info("Wrote: %s", OUTPUT_COMBINED_PQ)
    log.info("Wrote: %s", OUTPUT_COMBINED_CSV)

    # --- Scatter plots colored by Phase 3B prob (if available) ---------
    color_values: np.ndarray | None = None
    color_label: str | None = None
    if PREDICTIONS_PARQUET.exists():
        preds = pd.read_parquet(PREDICTIONS_PARQUET)[
            ["pair_id", "xgb_prob_touching"]
        ]
        scatter_df = ok_rows.merge(preds, on="pair_id", how="left")
        color_values = scatter_df["xgb_prob_touching"].to_numpy()
        color_label = "xgb_prob_touching"
    else:
        scatter_df = ok_rows
        log.warning(
            "Predictions parquet not found — plotting without color"
        )

    scatter_embeddings(
        scatter_df,
        color_values=color_values,
        color_label=color_label,
        xcol="emb_0",
        ycol="emb_1",
        output=FIGURES_DIR / "emb_scatter_emb0_emb1.png",
    )
    scatter_embeddings(
        scatter_df,
        color_values=color_values,
        color_label=color_label,
        xcol="emb_2",
        ycol="emb_3",
        output=FIGURES_DIR / "emb_scatter_emb2_emb3.png",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
