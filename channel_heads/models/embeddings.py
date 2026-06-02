"""Mars CNN embedding extraction (Phase 5).

This module moves the historical ``scripts/extract_mars_cnn_embeddings.py``
logic into the package. It keeps the Earth CNN preprocessing path intact by
calling :func:`channel_heads.cnn_features.extract_embeddings`, then assembles
the Mars embedding table and the tabular-plus-CNN model input table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.cnn_features import CNN_FEATURE_COLS, extract_embeddings
from channel_heads.cnn_model import DEFAULT_EMBEDDING_DIM, OutletCNN
from channel_heads.io import paths
from channel_heads.io.tables import write_table
from channel_heads.logging_config import get_logger
from channel_heads.models.device import pick_device

log = get_logger("models.embeddings")

DEFAULT_PATCH_INDEX_PARQUET = paths.MARS_CNN_PATCH_INDEX
DEFAULT_FEATURES_PARQUET = (
    paths.MARS_MODEL_INPUTS_DIR / "mars_pair_features_5feat_model_ready.parquet"
)
DEFAULT_PREDICTIONS_PARQUET = (
    paths.MARS_MODEL_OUTPUTS_DIR / "mars_xgb_predictions_5feat.parquet"
)
DEFAULT_OUTPUT_EMBEDDINGS = paths.MARS_MODEL_INPUTS_DIR / "mars_cnn_embeddings.parquet"
DEFAULT_OUTPUT_COMBINED = (
    paths.MARS_MODEL_INPUTS_DIR / "mars_model_input_tabular_plus_cnn.parquet"
)
DEFAULT_FIGURES_DIR = paths.MARS_CNN_PATCHES_DIR / "figures"
DEFAULT_BATCH_SIZE = 64


def _resolve_project_path(path: str | Path, project_root: Path = paths.PROJECT_ROOT) -> Path:
    """Resolve a stored artifact path relative to the project root when needed."""
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def load_cnn_model_for_embeddings(
    model_path: str | Path = paths.CNN_PRODUCTION,
    *,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    device: str = "cpu",
) -> OutletCNN:
    """Load ``cnn_outlet_final.pt`` into the frozen OutletCNN architecture."""
    import torch

    model = OutletCNN(embedding_dim=embedding_dim)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"CNN state-dict mismatch: missing={missing} unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def prepare_embedding_manifest(
    patch_index: pd.DataFrame,
    *,
    project_root: Path = paths.PROJECT_ROOT,
) -> pd.DataFrame:
    """Return the subset of Phase-4 patches usable for embedding extraction."""
    required = {"network_id", "pair_id", "patch_path", "patch_status"}
    missing = sorted(required - set(patch_index.columns))
    if missing:
        raise RuntimeError(f"Patch index missing required columns: {missing}")

    usable = patch_index[patch_index["patch_status"] == "ok"].copy().reset_index(drop=True)
    usable["raster_path"] = [
        str(_resolve_project_path(p, project_root)) for p in usable["patch_path"]
    ]
    return usable


def validate_patch_sample(
    manifest_df: pd.DataFrame,
    *,
    sample_size: int = 3,
    expected_shape: tuple[int, int] = (128, 128),
    expected_dtype=np.uint8,
) -> None:
    """Sanity-check a few rendered patches before CNN inference."""
    for _, row in manifest_df.head(sample_size).iterrows():
        raster = np.load(row["raster_path"])
        if raster.shape != expected_shape or raster.dtype != expected_dtype:
            raise RuntimeError(
                f"Unexpected patch shape/dtype for {row['pair_id']}: "
                f"{raster.shape}/{raster.dtype} "
                f"(expected {expected_shape}/{np.dtype(expected_dtype)})"
            )


def validate_embedding_table(
    df_emb: pd.DataFrame,
    *,
    expected_n: int | None = None,
    emb_cols: list[str] | None = None,
    status_col: str | None = "embedding_status",
) -> None:
    """Validate embedding table schema and finite values.

    If ``status_col`` is present, finite-value checks apply only to rows where
    ``status_col == "ok"``. Skipped rows may carry NaN embeddings, preserving
    the one-row-per-patch-index behavior of the historical script.
    """
    emb_cols = emb_cols or CNN_FEATURE_COLS
    if expected_n is not None and len(df_emb) != expected_n:
        raise RuntimeError(
            f"Embedding row count mismatch: got {len(df_emb)}, expected {expected_n}"
        )
    if "pair_id" not in df_emb.columns:
        raise RuntimeError("Embedding table missing pair_id")
    if df_emb["pair_id"].isna().any():
        raise RuntimeError("Some embedding rows have missing pair_id")
    if df_emb["pair_id"].duplicated().any():
        raise RuntimeError("Duplicate pair_id in embedding table")

    for col in emb_cols:
        if col not in df_emb.columns:
            raise RuntimeError(f"Missing embedding column: {col}")
        if not np.issubdtype(df_emb[col].dtype, np.number):
            raise RuntimeError(f"Embedding column {col} is non-numeric")

    finite_df = df_emb
    if status_col and status_col in df_emb.columns:
        finite_df = df_emb[df_emb[status_col] == "ok"]

    for col in emb_cols:
        arr = finite_df[col].to_numpy(dtype=float)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan or n_inf:
            raise RuntimeError(
                f"Embedding column {col} contains NaN={n_nan} / Inf={n_inf}"
            )


def assemble_embedding_table(
    patch_index: pd.DataFrame,
    extracted_df: pd.DataFrame,
    *,
    emb_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build the Mars embedding output table, including skipped patch rows."""
    emb_cols = emb_cols or CNN_FEATURE_COLS
    required = {"network_id", "pair_id", "patch_path", *emb_cols}
    missing = sorted(required - set(extracted_df.columns))
    if missing:
        raise RuntimeError(f"Extracted embedding dataframe missing columns: {missing}")

    df_emb = extracted_df[["network_id", "pair_id", "patch_path", *emb_cols]].copy()
    df_emb["embedding_status"] = "ok"
    df_emb["embedding_qa_reason"] = ""

    if len(df_emb) < len(patch_index):
        skipped = patch_index[~patch_index["pair_id"].isin(df_emb["pair_id"])]
        rows: list[dict] = []
        for _, row in skipped.iterrows():
            patch_qa = row.get("patch_qa_reason", "")
            reason = f"patch_status={row['patch_status']}"
            if patch_qa:
                reason += f";{patch_qa}"
            rows.append(
                {
                    "network_id": int(row["network_id"]),
                    "pair_id": str(row["pair_id"]),
                    "patch_path": str(row["patch_path"]),
                    **{col: np.nan for col in emb_cols},
                    "embedding_status": "skipped",
                    "embedding_qa_reason": reason,
                }
            )
        if rows:
            df_emb = pd.concat([df_emb, pd.DataFrame(rows)], ignore_index=True)

    final_cols = [
        "network_id",
        "pair_id",
        "patch_path",
        *emb_cols,
        "embedding_status",
        "embedding_qa_reason",
    ]
    return df_emb[final_cols]


def merge_embeddings_with_tabular_features(
    tabular: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    emb_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Merge Mars CNN embeddings into the Phase-3A model-ready feature table."""
    emb_cols = emb_cols or CNN_FEATURE_COLS
    required = {"pair_id", *emb_cols, "embedding_status", "embedding_qa_reason"}
    missing = sorted(required - set(embeddings.columns))
    if missing:
        raise RuntimeError(f"Embedding table missing merge columns: {missing}")
    if "pair_id" not in tabular.columns:
        raise RuntimeError("Tabular feature table missing pair_id")

    right = embeddings[["pair_id", *emb_cols, "embedding_status", "embedding_qa_reason"]]
    return tabular.merge(right, on="pair_id", how="left")


def scatter_embeddings(
    df_emb: pd.DataFrame,
    color_values: np.ndarray | None,
    color_label: str | None,
    xcol: str,
    ycol: str,
    output: Path,
) -> None:
    """Write one diagnostic embedding scatter plot."""
    import matplotlib.pyplot as plt

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
    ax.set_title(f"Mars CNN embeddings: {xcol} vs {ycol} (n={len(df_emb):,})")
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def render_embedding_figures(
    ok_embeddings: pd.DataFrame,
    *,
    predictions_parquet: str | Path = DEFAULT_PREDICTIONS_PARQUET,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Path]:
    """Write Phase-5 embedding scatter figures, best-effort."""
    figures_dir = Path(figures_dir)
    scatter_df = ok_embeddings
    color_values: np.ndarray | None = None
    color_label: str | None = None

    try:
        predictions_parquet = Path(predictions_parquet)
        if predictions_parquet.exists():
            preds = pd.read_parquet(predictions_parquet)[["pair_id", "xgb_prob_touching"]]
            scatter_df = ok_embeddings.merge(preds, on="pair_id", how="left")
            color_values = scatter_df["xgb_prob_touching"].to_numpy()
            color_label = "xgb_prob_touching"
        else:
            log.warning("Predictions parquet not found; plotting embeddings without color")

        outputs = {
            "emb_scatter_0_1": figures_dir / "emb_scatter_emb0_emb1.png",
            "emb_scatter_2_3": figures_dir / "emb_scatter_emb2_emb3.png",
        }
        scatter_embeddings(
            scatter_df,
            color_values=color_values,
            color_label=color_label,
            xcol="emb_0",
            ycol="emb_1",
            output=outputs["emb_scatter_0_1"],
        )
        scatter_embeddings(
            scatter_df,
            color_values=color_values,
            color_label=color_label,
            xcol="emb_2",
            ycol="emb_3",
            output=outputs["emb_scatter_2_3"],
        )
        return outputs
    except Exception as exc:  # diagnostics only
        log.warning("Embedding figure generation failed: %s", exc)
        return {}


def export_embedding_outputs(
    embeddings: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    embeddings_parquet: str | Path = DEFAULT_OUTPUT_EMBEDDINGS,
    combined_parquet: str | Path = DEFAULT_OUTPUT_COMBINED,
) -> dict[str, Path]:
    """Write Phase-5 parquet/csv outputs and return their canonical paths."""
    return {
        "embeddings": write_table(embeddings, embeddings_parquet),
        "combined": write_table(combined, combined_parquet),
    }


def extract_mars_cnn_embeddings(
    patch_index_parquet: str | Path = DEFAULT_PATCH_INDEX_PARQUET,
    features_parquet: str | Path = DEFAULT_FEATURES_PARQUET,
    predictions_parquet: str | Path = DEFAULT_PREDICTIONS_PARQUET,
    model_path: str | Path = paths.CNN_PRODUCTION,
    output_embeddings: str | Path = DEFAULT_OUTPUT_EMBEDDINGS,
    output_combined: str | Path = DEFAULT_OUTPUT_COMBINED,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    write: bool = True,
    make_figures: bool = True,
) -> dict:
    """Phase 5: extract Mars CNN embeddings and merge them into model inputs.

    Returns ``{"embeddings", "combined", "paths"}``.
    """
    device = device or pick_device()
    log.info("Device: %s", device)

    patch_index = pd.read_parquet(patch_index_parquet)
    n_total = len(patch_index)
    log.info(
        "Patch index: %d rows (status: %s)",
        n_total,
        patch_index["patch_status"].value_counts().to_dict(),
    )

    usable = prepare_embedding_manifest(patch_index)
    log.info("Patches usable for embedding: %d", len(usable))
    validate_patch_sample(usable)

    if CNN_FEATURE_COLS != [f"emb_{i}" for i in range(DEFAULT_EMBEDDING_DIM)]:
        raise RuntimeError(
            f"Embedding column names from cnn_features.py {CNN_FEATURE_COLS} "
            f"do not match expected emb_0..emb_{DEFAULT_EMBEDDING_DIM - 1}"
        )

    load_cnn_model_for_embeddings(
        model_path, embedding_dim=embedding_dim, device="cpu"
    )
    log.info(
        "CNN state-dict loaded OK into OutletCNN(embedding_dim=%d)",
        embedding_dim,
    )

    extracted = extract_embeddings(
        model_path=Path(model_path),
        raster_dir=None,
        manifest_df=usable,
        batch_size=batch_size,
        device=device,
        embedding_dim=embedding_dim,
    )
    embeddings = assemble_embedding_table(patch_index, extracted)

    ok_rows = embeddings[embeddings["embedding_status"] == "ok"].copy()
    validate_embedding_table(ok_rows, expected_n=len(usable), status_col=None)
    for col in CNN_FEATURE_COLS:
        s = ok_rows[col]
        log.info(
            "%s: mean=%.4f std=%.4f min=%.4f median=%.4f max=%.4f",
            col,
            float(s.mean()),
            float(s.std()),
            float(s.min()),
            float(s.median()),
            float(s.max()),
        )

    tabular = pd.read_parquet(features_parquet)
    combined = merge_embeddings_with_tabular_features(tabular, embeddings)
    n_matched = int(combined["embedding_status"].eq("ok").sum())
    log.info(
        "Combined table: %d rows; %d matched with ok embeddings",
        len(combined),
        n_matched,
    )

    written: dict[str, Path] = {}
    if write:
        written = export_embedding_outputs(
            embeddings,
            combined,
            embeddings_parquet=output_embeddings,
            combined_parquet=output_combined,
        )
        if make_figures:
            written.update(
                render_embedding_figures(
                    ok_rows,
                    predictions_parquet=predictions_parquet,
                    figures_dir=figures_dir,
                )
            )

    return {"embeddings": embeddings, "combined": combined, "paths": written}


__all__ = [
    "CNN_FEATURE_COLS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_OUTPUT_COMBINED",
    "DEFAULT_OUTPUT_EMBEDDINGS",
    "assemble_embedding_table",
    "export_embedding_outputs",
    "extract_mars_cnn_embeddings",
    "load_cnn_model_for_embeddings",
    "merge_embeddings_with_tabular_features",
    "prepare_embedding_manifest",
    "render_embedding_figures",
    "scatter_embeddings",
    "validate_embedding_table",
    "validate_patch_sample",
]
