#!/usr/bin/env python
"""Earth drainage-density calibration across stream-extraction thresholds.

For each terrestrial DEM in `channel_heads.io.paths.EXAMPLE_DEMS` (filterable via
``--basins``), this script:

  1. Sweeps a list of contributing-area thresholds (km^2).
  2. Builds a StreamObject per (DEM, threshold).
  3. Splits each DEM into per-outlet drainage basins.
  4. Computes Dd_true and Dd_hull for every basin.
  5. Summarises the Earth Dd_true distribution per threshold (primary metric)
     and Dd_hull distribution (secondary diagnostic).
  6. Writes CSVs and diagnostic plots into the output directory
     (default ``data/results/drainage_density_calibration/``).

Mars scoring (the original goal) is opt-in via ``--mars-target``. It scores the
Earth Dd_hull distribution against the published Mars distribution and picks a
``best_threshold``. By default Mars scoring is OFF because Earth's dendritic
flow-accumulation networks make Dd_hull structurally higher than the Mars
target at every threshold tested; the metric is biased rather than the data.

Example
-------
    python scripts/calibrate_stream_threshold_by_mars_dd.py \\
        --basins inyo humboldt \\
        --thresholds 1 5 10 \\
        --output data/results/drainage_density_calibration/quick_test

Primary interface: ``notebooks/diagnostics/dd_threshold_calibration.ipynb`` runs
the same calibration inline for a single basin read-only (calls
``channel_heads.dd_calibration``); this script is the headless wrapper that
sweeps all basins and writes the CSVs, GeoPackage and diagnostic plots.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
from channel_heads.dd_calibration import (
    DEFAULT_THRESHOLDS_KM2,
    MARS_DD_STATS,
    STATUS_OK,
    BasinMetrics,
    collect_basin_metrics_for_dem,
    compute_convex_hull_area_km2,
    summarize_threshold_metrics,
)
from channel_heads.io.paths import EXAMPLE_DEMS, RESULTS_DIR
from channel_heads.logging_config import setup_logging

# =============================================================================
# Output handling
# =============================================================================


def metrics_to_dataframe(rows: list[BasinMetrics]) -> pd.DataFrame:
    from dataclasses import asdict

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([asdict(m) for m in rows])


def write_geometries(
    per_basin_df: pd.DataFrame,
    best_threshold_km2: float,
    output_dir: Path,
    dem_lookup: dict[str, float],
) -> None:
    """Write extracted streams + convex hulls for the best threshold to GPKG.

    Streams are stored as one LineString per basin (concatenated edges) and the
    hull as one Polygon per basin. Coordinates are local pixel-frame metres,
    not a real CRS — these files are diagnostic, not authoritative.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Polygon
    except ImportError:
        print("geopandas/shapely not available; skipping GeoPackage outputs.")
        return

    import topotoolbox as tt3

    from channel_heads.dd_calibration import (
        _node_rowcol,
        compute_pixel_size_m_from_dem,
        compute_threshold_cells,
    )

    subset = per_basin_df[
        (per_basin_df["threshold_km2"] == best_threshold_km2)
        & (per_basin_df["status"] == STATUS_OK)
    ]
    if subset.empty:
        print("No valid basins at the chosen threshold; skipping GeoPackage outputs.")
        return

    stream_records = []
    hull_records = []

    for dem_name, basin_rows in subset.groupby("dem_name"):
        dem_path = EXAMPLE_DEMS.get(dem_name)
        if dem_path is None or not dem_path.exists():
            print(f"  [geometries] DEM for '{dem_name}' missing; skipping")
            continue
        lat = dem_lookup[dem_name]
        z_th = get_basin_config(dem_name)["z_th"]

        dem = tt3.read_tif(str(dem_path))
        if z_th is not None:
            dem.z[dem.z < z_th] = np.nan
        pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=lat)
        cells = compute_threshold_cells(best_threshold_km2, pixel_size_m)
        fd = tt3.FlowObject(dem)
        s = tt3.StreamObject(fd, threshold=cells)

        rows_all, cols_all = _node_rowcol(s)
        for _, row in basin_rows.iterrows():
            out_id = int(row["outlet_node"])
            up_mask = np.zeros(rows_all.shape[0], dtype=bool)
            up_mask[out_id] = True
            s_up = s.upstreamto(up_mask)

            rows_up, cols_up = _node_rowcol(s_up)
            if rows_up.size < 2:
                continue
            # source_indices / target_indices are (row, col) tuples already.
            src_r, src_c = s_up.source_indices
            tgt_r, tgt_c = s_up.target_indices
            segments = []
            for r0, c0, r1, c1 in zip(src_r, src_c, tgt_r, tgt_c):
                x0, y0 = float(c0 * pixel_size_m), float(r0 * pixel_size_m)
                x1, y1 = float(c1 * pixel_size_m), float(r1 * pixel_size_m)
                segments.append(LineString([(x0, -y0), (x1, -y1)]))  # flip y for orientation
            from shapely.geometry import MultiLineString
            if segments:
                stream_records.append(
                    {
                        "basin_id": row["basin_id"],
                        "dem_name": dem_name,
                        "threshold_km2": best_threshold_km2,
                        "geometry": MultiLineString(segments) if len(segments) > 1 else segments[0],
                    }
                )

            # Hull polygon
            _, hull_poly, status = compute_convex_hull_area_km2(
                rows_up.astype(float), cols_up.astype(float), pixel_size_m
            )
            if status == STATUS_OK and hull_poly is not None and len(hull_poly) >= 3:
                # hull_poly is (x_m, y_m); flip y the same way we flipped streams
                poly_xy = [(float(x), float(-y)) for x, y in hull_poly]
                hull_records.append(
                    {
                        "basin_id": row["basin_id"],
                        "dem_name": dem_name,
                        "threshold_km2": best_threshold_km2,
                        "hull_area_km2": row["hull_area_km2"],
                        "geometry": Polygon(poly_xy),
                    }
                )

    if stream_records:
        gpd.GeoDataFrame(stream_records, crs=None).to_file(
            output_dir / "extracted_streams_best_threshold.gpkg", driver="GPKG"
        )
    if hull_records:
        gpd.GeoDataFrame(hull_records, crs=None).to_file(
            output_dir / "convex_hulls_best_threshold.gpkg", driver="GPKG"
        )
    print(
        f"  GeoPackages written: streams={len(stream_records)}, hulls={len(hull_records)}"
    )


def make_diagnostic_plots(
    per_basin_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    mars_stats: dict[str, float] | None = None,
) -> None:
    """Plot Earth Dd_true (primary) and Dd_hull (secondary) per threshold.

    When `mars_stats` is provided, additional Mars-comparison plots are emitted
    (`plot_threshold_vs_score.png`, `plot_threshold_vs_dd_hull_median.png` with
    a Mars reference line, etc.).
    """
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("Summary is empty; skipping plots.")
        return

    # ---- Primary plots (Dd_true) ---------------------------------------------

    # 1. Earth Dd_true: median + Q1/Q3 vs threshold
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
    ax.set_xlabel("Threshold (km^2)")
    ax.set_ylabel("Dd_true (km / km^2)")
    ax.set_title("Earth Dd_true vs threshold (per-outlet basins)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_threshold_vs_dd_true.png", dpi=120)
    plt.close(fig)

    # 2. boxplot of Dd_true by threshold
    valid = per_basin_df[per_basin_df["status"] == STATUS_OK]
    if not valid.empty:
        thresholds = sorted(valid["threshold_km2"].unique())
        data = [
            valid.loc[valid["threshold_km2"] == t, "dd_true_km_km2"].dropna()
            for t in thresholds
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, tick_labels=[f"{t:g}" for t in thresholds], showfliers=False)
        ax.set_xlabel("Threshold (km^2)")
        ax.set_ylabel("Dd_true (km / km^2)")
        ax.set_title("Earth Dd_true distribution by threshold")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_dd_true_boxplot.png", dpi=120)
        plt.close(fig)

    # 3. basin count vs threshold (sample size of distribution)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold_km2"], summary_df["count"], "o-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Threshold (km^2)")
    ax.set_ylabel("Number of valid basins")
    ax.set_title("Sample size vs threshold")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_threshold_vs_count.png", dpi=120)
    plt.close(fig)

    # 4. Dd_hull as secondary diagnostic (no Mars line yet)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold_km2"], summary_df["dd_hull_median"], "o-", label="Dd_hull")
    ax.plot(
        summary_df["threshold_km2"], summary_df["dd_true_median"], "s-", label="Dd_true"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Threshold (km^2)")
    ax.set_ylabel("Median (km / km^2)")
    ax.set_title("Dd_true and Dd_hull medians vs threshold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_dd_true_vs_dd_hull_median.png", dpi=120)
    plt.close(fig)

    # ---- Optional Mars-comparison plots -------------------------------------

    if mars_stats is None or "score_iqr" not in summary_df.columns:
        return

    best_thr = float(
        summary_df.loc[summary_df["is_best_threshold"], "threshold_km2"].iloc[0]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        summary_df["threshold_km2"], summary_df["dd_hull_median"], "o-", label="Earth median"
    )
    ax.axhline(mars_stats["median"], color="red", linestyle="--", label="Mars median")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (km^2)")
    ax.set_ylabel("Dd_hull (km / km^2)")
    ax.set_title("Earth median Dd_hull vs threshold (Mars overlaid)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_threshold_vs_dd_hull_with_mars.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold_km2"], summary_df["score_iqr"], "o-")
    ax.axvline(best_thr, color="green", linestyle=":", label=f"best = {best_thr:g} km^2")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (km^2)")
    ax.set_ylabel("score_iqr (smaller = better)")
    ax.set_title("Mars-match score vs threshold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_threshold_vs_score.png", dpi=120)
    plt.close(fig)


def _format_summary_table(summary_df: pd.DataFrame) -> str:
    cols = [
        "threshold_km2",
        "count",
        "dd_true_median",
        "dd_true_q1",
        "dd_true_q3",
        "dd_hull_median",
    ]
    sub = summary_df[cols].copy()
    lines = ["| threshold km^2 | count | Dd_true median | Q1 | Q3 | Dd_hull median |"]
    lines.append("|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['threshold_km2']:g} | {int(r['count'])} "
            f"| {r['dd_true_median']:.4f} | {r['dd_true_q1']:.4f} "
            f"| {r['dd_true_q3']:.4f} | {r['dd_hull_median']:.4f} |"
        )
    return "\n".join(lines)


def write_readme(
    output_dir: Path,
    summary_df: pd.DataFrame,
    thresholds_km2: list[float],
    basin_names: list[str],
    mars_stats: dict[str, float] | None = None,
) -> None:
    if summary_df.empty:
        return

    mars_section = ""
    if mars_stats is not None and "score_iqr" in summary_df.columns:
        best_row = summary_df[summary_df["is_best_threshold"]].iloc[0]
        best_thr = float(best_row["threshold_km2"])
        mars_section = f"""

## Mars-matching (opt-in via --mars-target)

The Martian drainage-density proxy is computed as mapped valley length divided
by the convex-hull area of each mapped valley network. **Note (2026-05):** at
all thresholds tested, Earth `Dd_hull` remains structurally higher than Mars
because flow-accumulation streams form connected dendritic trunks that yield
narrow convex hulls. The scoring is therefore reported for reference only.

| Mars stat | value (km/km^2) |
|-----------|-----------------|
| n         | {int(mars_stats['n'])} |
| median    | {mars_stats['median']:.6f} |
| q1        | {mars_stats['q1']:.6f} |
| q3        | {mars_stats['q3']:.6f} |
| mean      | {mars_stats['mean']:.6f} |

Best (min `score_iqr`) threshold within the sweep: **{best_thr:g} km^2**
(Earth Dd_hull median {best_row['dd_hull_median']:.4f}, Mars median
{mars_stats['median']:.4f} — gap is real, not a calibration error).
"""

    readme = f"""# Earth drainage-density calibration

This output reports the Earth-side drainage density of extracted stream
networks at a range of contributing-area thresholds. The primary metric is
true watershed drainage density:

    Dd_true = stream length / basin area

computed per drainage basin (one basin per `StreamObject` outlet on each DEM).
A hull-based proxy `Dd_hull = stream length / convex_hull_area` is also stored
for reference but no longer drives threshold selection — see notes below.

## Earth Dd_true distribution by threshold

{_format_summary_table(summary_df)}

## Inputs

- Thresholds (km^2): {thresholds_km2}
- Basins: {basin_names}

## Files

- `per_basin_threshold_metrics.csv` — one row per (basin, threshold) with
  `stream_length_km`, `basin_area_km2`, `hull_area_km2`, `dd_true_km_km2`,
  `dd_hull_km_km2`, and a `status` field describing invalid rows.
- `threshold_summary.csv` — one row per threshold with full Dd_true and
  Dd_hull distribution stats; Mars-comparison scores appear only when
  `--mars-target` is passed on the command line.
- `plot_threshold_vs_dd_true.png` — Dd_true median + Q1-Q3 ribbon vs threshold.
- `plot_dd_true_boxplot.png` — Dd_true boxplot by threshold.
- `plot_threshold_vs_count.png` — number of valid basins vs threshold (sample
  size collapses at higher thresholds — important when interpreting medians).
- `plot_dd_true_vs_dd_hull_median.png` — both medians on a log axis for
  diagnostic purposes (Dd_hull stays much higher than Dd_true).

## Methodology

For each DEM:
  1. Elevation mask using `z_th` from `channel_heads.basin_config`.
  2. `FlowObject` built once per DEM.
  3. For each threshold, build `StreamObject(fd, threshold=cells_in_pixels)`
     where `cells_in_pixels = round(threshold_km2 * 1e6 / pixel_size_m**2)`
     and `pixel_size_m` is derived from the geographic-CRS cellsize using the
     basin's latitude.
  4. Split the DEM into per-outlet drainage basins via
     `FlowObject.drainagebasins` (one labelled basin per outlet).
  5. For each basin, extract the upstream subgraph (`StreamObject.upstreamto`),
     sum Euclidean stream-edge lengths for `stream_length_km`, count basin
     pixels for `basin_area_km2`, and compute a convex hull of the stream-node
     coordinates for `hull_area_km2`.
  6. `Dd_true = stream_length_km / basin_area_km2`.
     `Dd_hull = stream_length_km / hull_area_km2`.

## Why we moved away from Mars-`Dd_hull` matching

Earth flow-accumulation streams are guaranteed-connected dendritic trees. A
convex hull around such a network closely tracks the elongated trunk, giving a
narrow hull relative to channel length. The resulting `Dd_hull` is structurally
higher than the Mars target across every threshold tested, including the
"best" threshold within the sweep. Calibration via Mars `Dd_hull` therefore
does not yield a scientifically meaningful match. We instead report
`Dd_true` distributions; Mars matching can be re-enabled later with
`--mars-target` once a compatible Mars metric is available.{mars_section}
"""
    (output_dir / "README.md").write_text(readme)


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--basins",
        nargs="+",
        default=None,
        help="Subset of basin names (default: all in EXAMPLE_DEMS that exist on disk).",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS_KM2,
        help="Contributing-area thresholds in km^2.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "drainage_density_calibration",
        help="Output directory.",
    )
    p.add_argument(
        "--save-geometries",
        action="store_true",
        help="Also write GeoPackage outputs for the best threshold "
        "(requires --mars-target so a 'best' threshold exists).",
    )
    p.add_argument(
        "--min-basin-pixels",
        type=int,
        default=1,
        help="Skip outlet basins smaller than this many DEM pixels (default 1).",
    )
    p.add_argument(
        "--no-z-th",
        action="store_true",
        help="Do not apply basin z_th elevation masking.",
    )
    p.add_argument(
        "--mars-target",
        action="store_true",
        help=(
            "Enable Mars Dd_hull scoring + best-threshold selection. Off by "
            "default because the Dd_hull metric is structurally biased against "
            "Earth's dendritic networks; see README."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def resolve_basins(requested: list[str] | None) -> list[str]:
    if requested:
        names = [b.lower() for b in requested]
    else:
        names = list(EXAMPLE_DEMS.keys())
    available = [n for n in names if EXAMPLE_DEMS.get(n, Path()).exists()]
    missing = [n for n in names if n not in available]
    if missing:
        print(f"WARNING: skipping missing DEMs: {missing}")
    return available


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)

    args.output.mkdir(parents=True, exist_ok=True)
    basins = resolve_basins(args.basins)
    if not basins:
        print("No basins found on disk; aborting.")
        return 1

    print(f"Calibrating with basins: {basins}")
    print(f"Thresholds (km^2): {args.thresholds}")
    print(f"Output dir: {args.output}")

    all_rows: list[BasinMetrics] = []
    lat_lookup: dict[str, float] = {}
    t0 = time.time()
    for name in basins:
        paper_name = LOCAL_TO_PAPER_BASIN.get(name, name)
        try:
            cfg = get_basin_config(paper_name)
        except KeyError:
            print(f"  [{name}] no basin_config entry; skipping")
            continue
        lat = float(cfg["lat"])
        lat_lookup[name] = lat
        z_th = None if args.no_z_th else cfg["z_th"]

        print(f"\n[{name}] lat={lat:.2f}, z_th={z_th}")
        rows = collect_basin_metrics_for_dem(
            dem_path=EXAMPLE_DEMS[name],
            basin_name=name,
            thresholds_km2=list(args.thresholds),
            z_th=z_th,
            lat_deg=lat,
            min_outlet_basin_pixels=args.min_basin_pixels,
        )
        print(f"  -> {len(rows)} (basin, threshold) rows")
        all_rows.extend(rows)

    per_basin_df = metrics_to_dataframe(all_rows)
    if per_basin_df.empty:
        print("No metrics produced; aborting.")
        return 1

    mars_stats = MARS_DD_STATS if args.mars_target else None
    summary_df = summarize_threshold_metrics(per_basin_df, mars_stats=mars_stats)

    per_basin_df.to_csv(args.output / "per_basin_threshold_metrics.csv", index=False)
    summary_df.to_csv(args.output / "threshold_summary.csv", index=False)
    print(
        f"\nWrote per_basin_threshold_metrics.csv ({len(per_basin_df)} rows) "
        f"and threshold_summary.csv ({len(summary_df)} rows)"
    )

    make_diagnostic_plots(per_basin_df, summary_df, args.output, mars_stats=mars_stats)
    print("Wrote diagnostic plots.")

    if not summary_df.empty:
        if args.mars_target:
            best_thr = float(
                summary_df.loc[summary_df["is_best_threshold"], "threshold_km2"].iloc[0]
            )
            print(f"\nBest threshold (score_iqr, Mars-matching): {best_thr:g} km^2")
            if args.save_geometries:
                write_geometries(per_basin_df, best_thr, args.output, lat_lookup)
        elif args.save_geometries:
            print("--save-geometries requires --mars-target (no 'best threshold' otherwise); skipping.")
        write_readme(
            args.output,
            summary_df,
            list(args.thresholds),
            basins,
            mars_stats=mars_stats,
        )
        print("Wrote README.md")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
