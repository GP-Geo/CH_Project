# Data management

All of `data/` is **gitignored** (except two small tracked example files);
`models/` is **tracked in git** as of 2026-07-02 (~3 MB, see
`models/MANIFEST.md`). This doc defines what each kind of data is, what can be
deleted, and how to regenerate it. Authoritative per-path tags — and the
provenance of every raw input — are in [DATA_STATUS.md](DATA_STATUS.md);
regeneration commands are in [pipeline.md](pipeline.md).

## Categories

| Category | Examples | Policy |
|----------|----------|--------|
| **Raw / source** | `data/cropped_DEMs/`, `data/raw/`, `data/final_valleys/`, `MARS_DEM`, `MOLA hillshade` | **Never delete.** Not regenerable. |
| **Model artifacts** | `models/*.json`, `models/*.pt`, `feature_columns_*.txt`, `optimal_threshold_*.txt` | Keep unless explicitly retraining. Production `xgb_touching_classifier.json` + `cnn_outlet_final.pt` are frozen. |
| **Regenerable processed** | topology/pair GeoPackages, feature tables, CNN patches/embeddings, predictions, intermediate parquet | Safe to delete once the pipeline is stable; rebuild via `channel_heads.pipelines`. |
| **Figures / reports** | `data/exports/*.pdf`, `data/results/figures_*`, `data/Mars/.../figures_combined` | Regenerate from presentation notebooks. |
| **Temporary / debug** | old contact sheets, scratch CSVs, `*.sr.lock`, `.DS_Store`, `data/outputs/` (legacy dup) | Delete freely. |
| **Backup** | `data/_rebuild_backup_*/` | Keep off-repo; do not commit. |

## Safe cleanup tool

`channel_heads.io.cleanup` encodes this classification and **never lists raw
data or `models/`**. It is dry-run by default.

```python
from channel_heads.io import cleanup

# 1. Inspect — prints path, size, tag, regeneration hint. Deletes nothing.
print(cleanup.format_manifest(cleanup.scan()))

# 2. Archive uncertain items instead of deleting:
cleanup.clean(tags={"LEGACY"}, dry_run=False, archive_to=paths.DATA_DIR / "_archive")

# 3. Delete a specific class once the rebuild is verified:
cleanup.clean(tags={"STALE_AFTER_RASTER_FIX"}, dry_run=False)
```

Tags handled: `STALE_AFTER_RASTER_FIX` (pre-rasterizer-rewrite raster chain),
`REPORT` (figures), `LEGACY` (`data/outputs/` duplicate).

## Recommended regeneration strategy

1. Confirm raw + models present; confirm tests green (`pytest -q`).
2. `cleanup.scan()` → review manifest.
3. Archive anything uncertain (`archive_to=...`) rather than deleting.
4. Delete `STALE_AFTER_RASTER_FIX` + `REPORT`.
5. Rebuild: `pipelines.run_full_mars_pipeline()` (+ regime steps if needed).
6. Re-run presentation notebooks to regenerate figures.
7. Update [DATA_STATUS.md](DATA_STATUS.md) tags for rebuilt chains.

> Never delete raw DEMs, valley vectors, MOLA, or trained models without
> explicit intent. The cleanup tool enforces this; manual `rm` does not.
