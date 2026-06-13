# scripts/ — shell orchestrators, diagnostics, rendering, and archive

This repository is **package-first**: all pipeline/model/raster/training logic —
**including the command-line surface** — lives in `channel_heads/`. The CLI is the
`channel_heads/cli/` package, run via:

```bash
python -m channel_heads <command> [args]   # e.g. run-mars-pipeline --stage all
channel-heads <command> [args]             # console script
channel-heads --help                       # list all commands
```

`scripts/` therefore holds **no Python CLI entry points and no implementation** —
only shell orchestrators, headless diagnostics, rendering helpers, and archive.

> The former `scripts/cli/*` wrappers and the root-level Mars/Earth/regime
> training scripts were collapsed into `channel_heads/cli/` commands (e.g.
> `run-mars-pipeline`, `train-cnn-regime`, `train-combined-xgb-phase6b`,
> `eval-lobo-cv`, `retune-threshold-regime`, `make-result-figures`,
> `generate-poster-figures`). See [`docs/PROJECT_STRUCTURE.md §3`](../docs/PROJECT_STRUCTURE.md)
> for the full inventory and the notebook ↔ command map.

## Contents

### Shell / orchestration
| Script | Notes |
|--------|-------|
| `run_regime_pipeline.sh <regA\|regB\|regC> [full\|retrain]` | Regime Steps 2→6; invokes the package CLI by command name. |
| `run_full_rebuild.sh` | Baseline + regime rebuild batch runner over the package CLI. |
| `clean-cache.sh` | Cache cleanup; used by the generated pre-push hook. |
| `setup-hooks.sh` | Installs a pre-push hook that calls `./scripts/clean-cache.sh`. |

### Diagnostics (`diagnostics/`)
Headless QA / reporting helpers; each has a read-only notebook counterpart.
| Script | Package modules |
|--------|-----------------|
| `calibrate_stream_threshold_by_mars_dd.py` | `channel_heads.dd_calibration` |
| `diag_regB_threshold.py` | `channel_heads.eval` + model loaders |
| `qa_mars_stream_crossing_filter.py` | `channel_heads.pairing`, `channel_heads.viz` |

### Rendering (`rendering/`)
Batch figure writers over `channel_heads.viz`.
| Script | Purpose |
|--------|---------|
| `render_mars_combined_contact_sheets_vector.py` | Phase 6C contact sheets (vector polylines). |
| `render_mars_high_conf_emb_contact_sheet.py` | Top-20 embedding-probability pairs. |
| `render_mars_outlet_touching_pairs.py` | Per-outlet touching-pair figures. |

### Archive (`_archive/`)
Retained for provenance, **not maintained**. Each is superseded by a
`run-mars-pipeline --stage <…>` command or `channel_heads.pipelines` function:
`build_mars_network_topology.py` (topology), `extract_mars_first_meet_pairs.py`
(pairs), `build_mars_pair_features_5feat.py` (features),
`run_mars_xgb_inference_5feat.py` (xgb), `build_mars_cnn_patches_5class.py`
(patches), `extract_mars_cnn_embeddings.py` (embeddings),
`run_mars_combined_xgb_inference.py` (combined),
`extract_mars_outlet_candidates.py` (→ `channel_heads.mars.topology`), and
`old_experiments/exp_calibration_standardize.py`.

## Policy

- Add new command-line work as a `channel_heads/cli/` command, not a script here.
- New diagnostics → `scripts/diagnostics/`; new figure writers → `scripts/rendering/`.
- The `scripts/{diagnostics,rendering}/*.py` resolve the project root via
  `Path(__file__).resolve().parents[2]`; keep them at depth 2 under `scripts/`.
- `run_regime_pipeline.sh` / `run_full_rebuild.sh` call the CLI by command name —
  they break only if a command is renamed, not if a file moves.
- Never delete scripts permanently; move to `_archive/` after confirming no live
  references.
