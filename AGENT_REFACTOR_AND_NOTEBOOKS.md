# CLI collapse + notebook restructure + cleanup

_Run: 2026-06-04. Branch: `refactor/package-first-architecture`._

Covers the six-item request: per-stage notebooks, notebook refreshes, CLI/shim
canonicalization, dedup, and project cleanup.

## 1. CLI collapsed into the package (#5/#6)

`scripts/cli/*` (14 wrappers) → **`channel_heads/cli/` package**: a subcommand
dispatcher (`channel_heads/cli/__init__.py`) + one module per command, plus
`channel_heads/__main__.py`. Invoke via:

```bash
python -m channel_heads <command> [args]      # e.g. train-cnn-regime --regime regA
channel-heads <command> [args]                # console script (pyproject [project.scripts])
channel-heads --help                          # list commands
```

- Removed the `channel_heads/inference/` re-export shim (only its own test used
  it); `tests/test_inference.py` now imports canonical `channel_heads.models.*`.
- Dropped redundant `run_mars_inference.py` (== `run-mars-pipeline --stage combined`).
- Scrubbed stale `channel_heads.geometric_analysis` references (file long gone).
- Rewired `run_full_rebuild.sh`, `run_regime_pipeline.sh`, and all living docs to
  the new CLI; fixed a pre-existing `scripts/cli/cli/` double-path typo.
- **Verified:** 570 pytest pass; ruff clean; all 14 CLI submodules import; a real
  `eval-lobo-cv` run via the new CLI reproduces the old output exactly.

Commit `2aa1db4`.

## 2. Per-stage pipeline notebooks (#1)

New `notebooks/pipeline/00–14` — one **lightweight, executable** notebook per
pipeline stage (Stage 0–14 of `docs/PIPELINE_DESIGN.md`). Each imports the
canonical package and loads on-disk artifacts to verify/visualize that stage (no
heavy recompute). All 15 execute clean headless. Includes `README.md` stage index
and `_build_pipeline_notebooks.py` generator. Existing themed notebooks kept as
deep-dive references. Commit `af275af`.

## 3. Refreshed existing notebooks (#2/#3/#4)

Re-executed on the fresh (post-retrain) data, all clean:
- **Foundation (1–3):** `analysis/01_earth_source_data_qa`,
  `analysis/02_earth_network_explorer`, `mars/00_mars_network_explorer`.
- **Stage 12/13:** `mars/05_mars_threshold_sensitivity`,
  `interpretation/00_scientific_summary`.
- **Presentation (14):** `presentation/{mars_contact_sheets,
  per_outlet_touching_pairs, result_figures, simple_mars_earth_dd_presentation}`.

## 4. Project cleanup (#6 + cleanup request)

- Deleted all `__pycache__`, `.pytest_cache`, `.ruff_cache` (gitignored — repo was
  already clean; these regenerate on next run).
- Removed **junk empty dirs** created by the old `parents[1]` path bug:
  `scripts/data/`, `scripts/models/`, plus empty `data/results/final_figures/` and
  the empty `channel_heads/inference/` shim dir.
- Kept `channel_heads.egg-info` (editable-install metadata) and all data/figures.

## Status

Package-first, canonical, and clean: one CLI surface (`channel_heads/cli`), no
re-export shims, an enumerated per-stage notebook set, and all analysis/figure
notebooks refreshed on current data. `scripts/` now holds only shell orchestrators,
diagnostics, and figure-render helpers.
