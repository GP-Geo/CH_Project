# scripts/ → see docs

The script inventory, run order, path couplings, and migration status are
consolidated in **[../docs/PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md#3-scripts-inventory)**.

Quick orientation:
- `rendering/`, `diagnostics/` — moved subfolders (already organized).
- Mars / regime / training / maintenance scripts are still flat (path-coupled;
  see the migration table in the doc above).
- Pipeline entrypoints: `scripts/run_regime_pipeline.sh regA|regB` and the
  `python scripts/<phase>.py` steps listed in the doc.
