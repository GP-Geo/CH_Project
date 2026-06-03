"""Visualisation helpers for Earth drainage-density calibration sweeps.

Called by ``scripts/diagnostics/calibrate_stream_threshold_by_mars_dd.py``
and by ``notebooks/diagnostics/dd_threshold_calibration.ipynb``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from channel_heads.dd_calibration import STATUS_OK


def plot_calibration_results(
    per_basin_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    mars_stats: dict[str, float] | None = None,
) -> None:
    """Write diagnostic PNG plots for a calibration sweep to *output_dir*.

    Parameters
    ----------
    per_basin_df:
        One row per (basin, threshold); must have ``status``, ``threshold_km2``,
        and ``dd_true_km_km2`` columns.
    summary_df:
        One row per threshold; must have ``threshold_km2``, ``count``,
        ``dd_true_median``, ``dd_true_q1``, ``dd_true_q3``, ``dd_hull_median``.
    output_dir:
        Directory where PNGs are written (created if absent).
    mars_stats:
        Optional dict with Mars distribution stats (``median``, ``q1``, ``q3``,
        ``mean``, ``n``). When supplied and ``summary_df`` contains
        ``score_iqr`` and ``is_best_threshold``, Mars-comparison plots are also
        written.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_df.empty:
        print("Summary is empty; skipping calibration plots.")
        return

    # 1. Earth Dd_true: median + Q1/Q3 ribbon vs threshold
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold_km2"], summary_df["dd_true_median"], "o-", label="median")
    ax.fill_between(
        summary_df["threshold_km2"],
        summary_df["dd_true_q1"],
        summary_df["dd_true_q3"],
        alpha=0.2,
        label="Q1-Q3",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (km²)")
    ax.set_ylabel("Dd_true (km / km²)")
    ax.set_title("Earth Dd_true vs threshold (per-outlet basins)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_threshold_vs_dd_true.png", dpi=120)
    plt.close(fig)

    # 2. Boxplot of Dd_true by threshold
    valid = per_basin_df[per_basin_df["status"] == STATUS_OK]
    if not valid.empty:
        thresholds = sorted(valid["threshold_km2"].unique())
        data = [
            valid.loc[valid["threshold_km2"] == t, "dd_true_km_km2"].dropna()
            for t in thresholds
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, tick_labels=[f"{t:g}" for t in thresholds], showfliers=False)
        ax.set_xlabel("Threshold (km²)")
        ax.set_ylabel("Dd_true (km / km²)")
        ax.set_title("Earth Dd_true distribution by threshold")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_dd_true_boxplot.png", dpi=120)
        plt.close(fig)

    # 3. Basin count vs threshold (sample-size diagnostic)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold_km2"], summary_df["count"], "o-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Threshold (km²)")
    ax.set_ylabel("Number of valid basins")
    ax.set_title("Sample size vs threshold")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_threshold_vs_count.png", dpi=120)
    plt.close(fig)

    # 4. Dd_hull vs Dd_true medians (secondary diagnostic)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold_km2"], summary_df["dd_hull_median"], "o-", label="Dd_hull")
    ax.plot(summary_df["threshold_km2"], summary_df["dd_true_median"], "s-", label="Dd_true")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Threshold (km²)")
    ax.set_ylabel("Median (km / km²)")
    ax.set_title("Dd_true and Dd_hull medians vs threshold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_dd_true_vs_dd_hull_median.png", dpi=120)
    plt.close(fig)

    # Optional Mars-comparison plots
    if mars_stats is None or "score_iqr" not in summary_df.columns:
        return

    best_mask = summary_df.get("is_best_threshold", pd.Series(False, index=summary_df.index))
    best_rows = summary_df[best_mask]
    if best_rows.empty:
        return
    best_thr = float(best_rows["threshold_km2"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold_km2"], summary_df["dd_hull_median"], "o-", label="Earth median")
    ax.axhline(mars_stats["median"], color="red", linestyle="--", label="Mars median")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (km²)")
    ax.set_ylabel("Dd_hull (km / km²)")
    ax.set_title("Earth median Dd_hull vs threshold (Mars overlaid)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_threshold_vs_dd_hull_with_mars.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold_km2"], summary_df["score_iqr"], "o-")
    ax.axvline(best_thr, color="green", linestyle=":", label=f"best = {best_thr:g} km²")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (km²)")
    ax.set_ylabel("score_iqr (smaller = better)")
    ax.set_title("Mars-match score vs threshold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_threshold_vs_score.png", dpi=120)
    plt.close(fig)


__all__ = ["plot_calibration_results"]
