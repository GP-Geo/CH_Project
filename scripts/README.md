# scripts/ — CLI, diagnostics, and archive layer

This repository is package-first: real pipeline/model/raster/training logic
lives in `channel_heads/`. The `scripts/` tree is retained as a stable command
surface for rebuilds, diagnostics, rendering, maintenance, and historical
provenance.

Mars package wrappers have been archived because the maintained
`scripts/cli/run_mars_pipeline.py` entry point now covers Phases 1-6C. Earth,
regime, training, diagnostics, and shell entry points remain at their documented
paths.

## Categories

- **Maintained CLI:** preferred human-facing command entry point.
- **Wrapper over package API:** compatibility command whose logic is package-owned.
- **Diagnostics utility:** headless analysis, QA, or reporting helper.
- **Shell/orchestration entry point:** batch runner that invokes other scripts.
- **Historical/archive candidate:** retained for provenance, not maintained.
- **Unknown/manual-review:** no scripts currently fall in this category.

## Preferred CLI Entry Points

Use these for new command-line work when they fit the task.

| Script | Category | Canonical package API |
|--------|----------|-----------------------|
| `cli/run_mars_pipeline.py` | maintained CLI | `channel_heads.pipelines.run_full_mars_pipeline` and per-stage pipeline functions |
| `cli/run_mars_inference.py` | maintained CLI | `channel_heads.pipelines.run_mars_combined_inference` |
| `cli/generate_poster_figures.py` | maintained CLI | `channel_heads.pipelines.generate_poster_figures` |

## Root Compatibility Wrappers

These root scripts remain in place because they are documented and/or called by
shell orchestrators. They should stay thin; new implementation belongs in the
listed package module.

| Script | Category | Canonical package alternative |
|--------|----------|-------------------------------|
| `build_earth_features_regime.py` | wrapper over package API | `channel_heads.training.regime.build_regime_feature_dataset` |
| `build_cnn_patches_regime.py` | wrapper over package API | `channel_heads.training.regime.build_regime_patch_dataset` |
| `train_cnn_baseline.py` | wrapper over package API | `channel_heads.training.cnn.train_cnn` plus dataset helpers in `channel_heads.training.datasets` |
| `train_cnn_regime.py` | wrapper over package API | `channel_heads.training.cnn.train_cnn` plus regime/dataset helpers |
| `train_cnn_multiseed.py` | wrapper over package API | `channel_heads.training.cnn.train_cnn` plus deterministic split helpers in `channel_heads.training.datasets` |
| `train_combined_xgb_phase6b.py` | wrapper over package API | `channel_heads.training.xgboost` helpers for strict CNN extraction, model training, thresholds, and artifact writers |
| `train_combined_xgb_regime.py` | wrapper over package API | `channel_heads.training.xgboost` helpers plus regime manifest/model paths |
| `eval_lobo_cv.py` | diagnostics utility / wrapper over package API | `channel_heads.eval.lobo` |
| `run_mars_combined_regime.py` | maintained CLI / wrapper over package API | `channel_heads.models.regime`, `channel_heads.models.xgboost`, and Mars input/output helpers |

## Diagnostics and Rendering

These scripts are intentionally headless batch utilities. Most have a primary
notebook for exploratory/read-only use and a script for writing report, QA, or
figure outputs.

| Script | Category | Canonical package alternative |
|--------|----------|-------------------------------|
| `retune_threshold_regime.py` | diagnostics utility | `channel_heads.eval` threshold/metric helpers and `channel_heads.models.xgboost` loaders |
| `make_result_figures.py` | diagnostics utility | `channel_heads.viz`, `channel_heads.eval`, and model loaders |
| `diagnostics/calibrate_stream_threshold_by_mars_dd.py` | diagnostics utility | `channel_heads.dd_calibration` |
| `diagnostics/diag_regB_threshold.py` | diagnostics utility | `channel_heads.eval` and model/inference helpers |
| `diagnostics/qa_mars_stream_crossing_filter.py` | diagnostics utility | `channel_heads.pairing` and `channel_heads.viz` helpers |
| `rendering/render_mars_combined_contact_sheets_vector.py` | diagnostics/rendering utility | `channel_heads.viz.render_contact_sheet` |
| `rendering/render_mars_high_conf_emb_contact_sheet.py` | diagnostics/rendering utility | `channel_heads.viz.render_pair_panel` |
| `rendering/render_mars_outlet_touching_pairs.py` | diagnostics/rendering utility | `channel_heads.viz.render_outlet_touching_pairs` |

## Shell and Maintenance Entry Points

| Script | Category | Notes |
|--------|----------|-------|
| `run_full_rebuild.sh` | shell/orchestration entry point | Path-coupled batch runner for baseline and regime rebuild steps. Keep root script paths stable unless the shell is updated in the same slice. |
| `run_regime_pipeline.sh` | shell/orchestration entry point | Path-coupled regime Step 2-6 runner. It intentionally remains useful as a top-level batch entry point. |
| `clean-cache.sh` | shell/orchestration entry point | Maintenance helper used by the generated pre-push hook. |
| `setup-hooks.sh` | shell/orchestration entry point | Installs a hook that calls `./scripts/clean-cache.sh`; keep path stable. |

## Archive

Archived scripts are retained, not deleted. They are not maintained command
surfaces and should not be used for new workflows.

| Script | Category | Reason |
|--------|----------|--------|
| `_archive/build_mars_network_topology.py` | historical/archive candidate | Superseded by `scripts/cli/run_mars_pipeline.py --stage topology` and `channel_heads.pipelines.build_mars_topology`. |
| `_archive/extract_mars_first_meet_pairs.py` | historical/archive candidate | Superseded by `scripts/cli/run_mars_pipeline.py --stage pairs` and `channel_heads.pipelines.extract_mars_pairs`. |
| `_archive/build_mars_pair_features_5feat.py` | historical/archive candidate | Superseded by `scripts/cli/run_mars_pipeline.py --stage features` and `channel_heads.pipelines.build_mars_features`. |
| `_archive/run_mars_xgb_inference_5feat.py` | historical/archive candidate | Superseded by `scripts/cli/run_mars_pipeline.py --stage xgb` and `channel_heads.pipelines.run_mars_xgb_inference`. |
| `_archive/build_mars_cnn_patches_5class.py` | historical/archive candidate | Superseded by `scripts/cli/run_mars_pipeline.py --stage patches` and `channel_heads.pipelines.build_mars_cnn_patches`. |
| `_archive/extract_mars_cnn_embeddings.py` | historical/archive candidate | Superseded by `scripts/cli/run_mars_pipeline.py --stage embeddings` and `channel_heads.pipelines.extract_mars_cnn_embeddings`. |
| `_archive/run_mars_combined_xgb_inference.py` | historical/archive candidate | Superseded by `scripts/cli/run_mars_pipeline.py --stage combined` and `channel_heads.pipelines.run_mars_combined_inference`. |
| `_archive/extract_mars_outlet_candidates.py` | historical/archive candidate | Superseded by `channel_heads.mars.topology`; standalone output is no longer consumed. |
| `_archive/old_experiments/exp_calibration_standardize.py` | historical/archive candidate | Dropped per-basin standardization experiment retained for provenance. |

## Move Policy

- Do not move a script if docs, shell scripts, notebooks, or tests reference its
  current path.
- Do not move a script just because it is thin; root compatibility wrappers are
  allowed when they preserve documented commands.
- New general-purpose CLIs should prefer `scripts/cli/`.
- New diagnostics should prefer `scripts/diagnostics/` or `scripts/rendering/`.
- Obsolete scripts may move to `scripts/_archive/` only after `rg` confirms no
  live references need updating.
- Never delete scripts permanently in cleanup slices.
