# scripts/ — thin entry points only

The project is **package-first**: real logic lives in `channel_heads/` and is
run through `channel_heads.pipelines`. This directory holds only entry points
and transitional implementations.

## `cli/` — maintained CLI wrappers (use these)
Tiny argparse shells over `channel_heads.pipelines`:

| Script | Calls |
|--------|-------|
| `cli/run_mars_pipeline.py` | `pipelines.run_full_mars_pipeline` / per-stage |
| `cli/run_mars_inference.py` | `pipelines.run_mars_combined_inference` |
| `cli/generate_poster_figures.py` | `pipelines.generate_poster_figures` |

## root `*.py` — legacy entry points
Most Mars root scripts are now thin wrappers around package-resident logic in
`channel_heads/`. Remaining transitional implementations are Earth/regime
training-oriented and are scheduled for a separate extraction slice. Do not call
root scripts in new code; call the matching `pipelines.*` function.

`diagnostics/` and `rendering/` remain notebook-backed batch helpers (each has a
primary notebook in `notebooks/`). `*.sh` files orchestrate multi-stage rebuilds.

## `_archive/` — retained but not maintained
Historical / superseded scripts kept for provenance:
- `extract_mars_outlet_candidates.py` — superseded by `channel_heads.mars.topology`
  (outlet selection is now part of the Phase-1 topology build; its standalone
  output was consumed by nothing).
- `old_experiments/exp_calibration_standardize.py` — dropped per-basin
  standardization experiment (neutral-to-negative result; no inbound refs).
