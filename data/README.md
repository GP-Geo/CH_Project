# data/ → see docs

The data inventory and the rasterization-fix / stale-output guidance are
consolidated in **[../docs/PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md#4-data-and-models-inventory)**.
Detailed per-output status: **[../docs/DATA_STATUS.md](../docs/DATA_STATUS.md)**,
which also documents dataset provenance (SRTM GL3 via OpenTopography, NASA
MOLA, Alemanno et al. 2018). The canonical full-data copy (~18 GB, not in the
repo) lives with the owner — see [../HANDOFF.md](../HANDOFF.md).

Canonical output dir: `data/results/` (`config.RESULTS_DIR`). Never delete data
— mark legacy instead. In-situ markers remain in place:
- `data/archive/README.md` — archive policy + holding area.
- `data/outputs/README.LEGACY.md` — legacy-duplicate marker.
