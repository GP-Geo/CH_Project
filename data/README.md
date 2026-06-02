# data/ → see docs

The data inventory and the rasterization-fix / stale-output guidance are
consolidated in **[../docs/PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md#4-data-and-models-inventory)**.
Detailed per-output status will live in `docs/DATA_STATUS.md` (Phase 7).

Canonical output dir: `data/results/` (`config.RESULTS_DIR`). Never delete data
— mark legacy instead. In-situ markers remain in place:
- `data/archive/README.md` — archive policy + holding area.
- `data/outputs/README.LEGACY.md` — legacy-duplicate marker.
