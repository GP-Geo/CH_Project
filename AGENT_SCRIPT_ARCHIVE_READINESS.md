# AGENT_SCRIPT_ARCHIVE_READINESS.md

Read-only readiness report for root-level `scripts/*.py` wrappers.

Scope checked:
- `docs/`
- `scripts/*.sh`
- `scripts/README.md`
- `AGENT_*.md`

This report identifies what still blocks archiving root-level wrapper scripts.
No files were moved, archived, or deleted.

## Summary

Most root-level scripts are now thin wrappers over package modules, but they are
still path-coupled to docs and shell orchestrators. That makes them safe to keep
as compatibility entry points, but not safe to archive yet without a broader
reference update.

The main blockers are:
- `docs/PIPELINE_RERUN.md` and `docs/PROJECT_STRUCTURE.md` still document root
  script paths.
- `scripts/run_full_rebuild.sh` and `scripts/run_regime_pipeline.sh` still call
  root scripts by path.
- Several root scripts still mention their own root invocation in docstrings,
  which reinforces the documented path surface.

## Readiness Matrix

| Script | Current references found by `rg` | Maintained CLI / package alternative | Archive status | References to update first |
|--------|----------------------------------|--------------------------------------|----------------|----------------------------|
| `scripts/build_earth_features_regime.py` | `docs/PIPELINE_RERUN.md`, `docs/PROJECT_STRUCTURE.md`, `scripts/run_regime_pipeline.sh`, `scripts/run_full_rebuild.sh`, `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.training.regime.build_regime_feature_dataset` | Needs manual review later | Update docs and both shell orchestrators together; then retarget any script docstring examples |
| `scripts/build_cnn_patches_regime.py` | `docs/PIPELINE_RERUN.md`, `docs/PROJECT_STRUCTURE.md`, `scripts/run_regime_pipeline.sh`, `scripts/run_full_rebuild.sh`, `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.training.regime.build_regime_patch_dataset` | Needs manual review later | Same blockers as above |
| `scripts/train_cnn_baseline.py` | `docs/PIPELINE_RERUN.md`, `scripts/run_full_rebuild.sh`, `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.training.cnn.train_cnn` + `channel_heads.training.datasets` | Needs manual review later | Update `docs/PIPELINE_RERUN.md`, `scripts/run_full_rebuild.sh`, and any root-path examples |
| `scripts/train_cnn_regime.py` | `docs/PIPELINE_RERUN.md`, `docs/PROJECT_STRUCTURE.md`, `scripts/run_regime_pipeline.sh`, `scripts/run_full_rebuild.sh`, `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.training.cnn.train_cnn` + `channel_heads.training.datasets` | Needs manual review later | Update docs and both shell orchestrators together |
| `scripts/train_cnn_multiseed.py` | `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, script docstring, and a few historical audit references in `AGENT_*` files | No dedicated package CLI; behavior lives in `channel_heads.training.cnn` plus dataset helpers | Safe to archive later only if a replacement CLI is added or docs stop advertising the root path | Add or document a package-backed CLI first, then remove root-path references from `AGENT_*` if they are still current |
| `scripts/train_combined_xgb_phase6b.py` | `docs/PIPELINE_RERUN.md`, `scripts/run_full_rebuild.sh`, `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.training.xgboost` | Needs manual review later | Update `docs/PIPELINE_RERUN.md`, `scripts/run_full_rebuild.sh`, and root-path examples |
| `scripts/train_combined_xgb_regime.py` | `docs/PIPELINE_RERUN.md`, `docs/PROJECT_STRUCTURE.md`, `scripts/run_regime_pipeline.sh`, `scripts/run_full_rebuild.sh`, `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.training.xgboost` | Needs manual review later | Update docs and both shell orchestrators together |
| `scripts/eval_lobo_cv.py` | `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, `AGENT_AUDIT_EARTH_REGIME.md`, script docstring | `channel_heads.eval.lobo` | Safe to archive later only if a replacement CLI path is introduced or consumers switch to the package API | Update any user-facing docs and script examples that still point at the root wrapper |
| `scripts/build_mars_cnn_patches_5class.py` | `scripts/README.md`, `scripts/run_full_rebuild.sh`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.rasterization.mars_patches` / `channel_heads.pipelines.build_mars_cnn_patches` | Needs manual review later | Update `scripts/run_full_rebuild.sh` and the user-facing examples in the script docstring/docs |
| `scripts/build_mars_network_topology.py` | `scripts/README.md`, `scripts/run_full_rebuild.sh`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.pipelines.build_mars_topology` | Needs manual review later | Update the pipeline docs and rebuild shell if the root path is to be retired |
| `scripts/build_mars_pair_features_5feat.py` | `scripts/README.md`, `scripts/run_full_rebuild.sh`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.pipelines.build_mars_features` | Needs manual review later | Update the pipeline docs and rebuild shell first |
| `scripts/extract_mars_cnn_embeddings.py` | `docs/PIPELINE_RERUN.md`, `scripts/run_full_rebuild.sh`, `scripts/README.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.models.embeddings` / `channel_heads.pipelines.extract_mars_cnn_embeddings` | Needs manual review later | Update `docs/PIPELINE_RERUN.md`, `scripts/run_full_rebuild.sh`, and root examples |
| `scripts/extract_mars_first_meet_pairs.py` | `scripts/README.md`, `scripts/run_full_rebuild.sh`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.pipelines.extract_mars_pairs` | Needs manual review later | Update the rebuild shell and any docstring usage examples |
| `scripts/run_mars_combined_regime.py` | `docs/PIPELINE_RERUN.md`, `docs/PROJECT_STRUCTURE.md`, `scripts/run_regime_pipeline.sh`, `scripts/run_full_rebuild.sh`, `scripts/README.md`, `AGENT_STATE.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.models.regime`, `channel_heads.models.xgboost`, Mars input/output helpers | Keep as maintained CLI for now | Do not archive until the regime shell pipeline and docs are fully repointed |
| `scripts/run_mars_combined_xgb_inference.py` | `scripts/README.md`, `scripts/run_full_rebuild.sh`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.pipelines.run_mars_combined_inference` | Needs manual review later | Update rebuild docs/shell and root examples |
| `scripts/run_mars_xgb_inference_5feat.py` | `scripts/README.md`, `scripts/run_full_rebuild.sh`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.pipelines.run_mars_xgb_inference` | Needs manual review later | Update rebuild docs/shell and root examples |
| `scripts/retune_threshold_regime.py` | `docs/PIPELINE_RERUN.md`, `scripts/README.md`, `AGENT_RUN_LOG.md`, script docstring | `channel_heads.eval` + `channel_heads.models.xgboost` | Needs manual review later | Update docs and decide whether this remains a maintained diagnostic CLI |
| `scripts/make_result_figures.py` | `scripts/README.md`, `AGENT_RUN_LOG.md`, script docstring, notebook references | `channel_heads.viz` + `channel_heads.eval` | Keep as diagnostics utility, not archive | No archive action until notebook/documentation references are clarified |

## What still blocks archiving

1. Root script paths are part of the documented command surface.
2. `run_full_rebuild.sh` and `run_regime_pipeline.sh` still require the root
   scripts to exist at their current paths.
3. Several scripts still advertise their root path in docstrings, which is fine
   for compatibility, but it means archiving them would break the published
   interface.
4. `AGENT_*` history still records root-path commands as the current truth; that
   is informative, but it also shows that the cleanup has not been normalized to
   package-first command entry points yet.

## Safe now vs later

- Safe now: none of the root wrappers should be archived yet.
- Safe later: the thin Mars wrappers and the Earth/regime wrappers can be
  archived only after docs and shell runners are repointed together, or after a
  package-backed CLI takes over the published command path.
- Needs manual review: the shell entry points and the diagnostics script
  `retune_threshold_regime.py`, because their current role is partly command
  surface and partly workflow-specific utility.

## Required reference updates before archiving

1. Update `docs/PIPELINE_RERUN.md` to point at package-backed CLIs or a new
   wrapper location.
2. Update `docs/PROJECT_STRUCTURE.md` so the inventory matches the new command
   surface.
3. Update `scripts/run_full_rebuild.sh` and `scripts/run_regime_pipeline.sh` to
   call the new entry points directly.
4. Update root-script docstrings and any `AGENT_*` history entries that still
   present the old root path as the recommended command.
5. Only then move the obsolete root wrapper into `scripts/_archive/` if the
   command path is no longer documented or used.

