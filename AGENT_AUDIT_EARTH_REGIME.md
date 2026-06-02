# AGENT_AUDIT_EARTH_REGIME.md — Slice 5 audit

Date: 2026-06-02

Scope: read-only audit of Earth/regime training scripts and adjacent package
modules. No implementation code was changed. The goal is to plan future
package-first extraction while preserving scientific behavior exactly.

## Executive Summary

The Earth/regime training path is still transitional and **not disposable**.
Several scripts contain real training/orchestration logic that does not yet have
an equivalent package-resident API:

- `scripts/build_earth_features_regime.py` is the regime Earth feature builder.
- `scripts/build_cnn_patches_regime.py` is the regime-specific Earth patch
  builder.
- `scripts/train_combined_xgb_phase6b.py` is the canonical baseline combined
  XGBoost training recipe for geom-only / geom+CNN-embedding / geom+CNN-logit.
- `scripts/train_combined_xgb_regime.py` is the regime combined-XGB training
  recipe.
- `scripts/run_mars_combined_regime.py` is the headless regime Mars inference
  writer; it uses some package glue but still owns output behavior.
- `scripts/retune_threshold_regime.py` and `scripts/eval_lobo_cv.py` are
  diagnostics/decision-support wrappers, but each still owns XGBoost-specific
  fitting or artifact overwrite behavior.

The package already owns the common low-level pieces: regime presets,
CNN architecture/dataset/training loop, XGBoost inference helpers, grouped
splitting and threshold metrics, strict regime embedding attachment, and the
package Mars combined inference path. Future work should move script recipes
into package modules and leave scripts as thin CLI wrappers.

## Reference Map

| File | Current role | Real logic present |
|------|--------------|--------------------|
| `scripts/build_earth_features_regime.py` | Regime Step 2: Earth feature generation under pruning regimes | Yes: DEM discovery, threshold-cell conversion, TopoToolbox stream build, pruning, per-outlet prefilter sizing, pair/coupling/asymmetry/geometry evaluation, hard-negative filtering, stratified negative subsampling, cache/stats/master writes. |
| `scripts/build_cnn_patches_regime.py` | Regime Step 3: Earth CNN patches for regime datasets | Yes: regime stream loader that rebuilds/prunes Earth network to resolve node IDs; manifest/output-root handling. Uses package `precompute_raster_dataset`. |
| `scripts/train_cnn_regime.py` | Regime Step 4: train `cnn_outlet_<regime>.pt` | Partly: wrapper around package `train_cnn`, but owns manifest filtering, Taiwan holdout, deterministic validation split, CLI hyperparameter overrides, and artifact names. |
| `scripts/train_cnn_baseline.py` | Baseline production CNN rebuild | Partly: same wrapper pattern as regime CNN, with production manifest/model paths. |
| `scripts/train_cnn_multiseed.py` | Best-of-N CNN training utility | Yes: config map, per-seed split/torch/NumPy seeding, best-val-loss model selection, multi-seed metrics artifact. |
| `scripts/train_combined_xgb_phase6b.py` | Baseline Phase 6B Earth combined-XGB training | Yes: strict CNN emb+logit extraction, full Earth master-v4 construction, three variant definitions, grouped split, XGBoost fit, threshold tuning, metrics/artifact writes. Docs identify this as the canonical persisted training recipe. |
| `scripts/train_combined_xgb_regime.py` | Regime Step 5: geom+emb XGBoost training | Yes: strict regime CNN embedding extraction, regime `_with_emb` dataset write, grouped split, XGBoost fit, threshold tuning, metrics/artifact writes. |
| `scripts/run_mars_combined_regime.py` | Regime Step 6: Mars inference under a regime | Yes: artifact bundle resolution, Mars input loading, feature loading, model prediction, per-network summary, CSV/parquet/GPKG output. Embedding attachment itself is already package-owned. |
| `scripts/retune_threshold_regime.py` | Optional threshold retune | Partly: uses package eval helpers but owns artifact paths, model predict, threshold overwrite, and metrics CSV mutation. |
| `scripts/eval_lobo_cv.py` | LOBO-CV diagnostic batch writer | Yes: dataset map, XGBoost fold fit function, pooled/fold metrics, `models/lobo_cv_metrics.csv` write. Uses package LOBO splitter and metrics helpers. |
| `scripts/run_regime_pipeline.sh` | Shell orchestrator | Thin but path-coupled: hardcodes regA/regB only, step order, log paths, and script paths. |
| `channel_heads/regimes.py` | Canonical regime presets | Yes, already canonical. Regime values are tested. |
| `channel_heads/training/cnn.py` | Canonical generic CNN training loop | Yes, already canonical for `train_cnn`, defaults, `HOLDOUT_BASIN`, `RANDOM_STATE`, `pick_device` re-export. |
| `channel_heads/models/cnn.py` | Canonical CNN architecture/dataset/one-hot | Yes, already canonical. State-dict shape/key contract is tested. |
| `channel_heads/models/cnn_features.py` | Generic/Earth lenient embedding extraction | Yes, canonical for lenient manifest-keyed embeddings. Must stay distinct from strict regime/combined extractors. |
| `channel_heads/inference/regime.py` | Transitional regime embedding attach | Yes: strict regime CNN embedding extraction, patch-index join, drop missing patches, overwrite `emb_*`, finite checks. |
| `channel_heads/models/xgboost.py` | Canonical XGBoost inference helpers | Yes: feature columns, threshold, model load, feature matrix checks, thresholded predict. No training helpers yet. |
| `channel_heads/eval/*` | Evaluation primitives | Yes: F1-opt threshold, max-precision-at-recall threshold, metric bundle, outlet-group split, leave-one-basin/group OOF split. |
| `channel_heads/models/mars_combined.py` | Baseline package Mars combined inference | Yes: strict logit extraction, combined Mars artifacts, feature verification, summaries, GPKG/figure outputs. Closest model for future regime Mars inference. |

## Script-by-Script Classification

### Keep As Thin CLI Wrappers Later

These should become CLI wrappers over package entry points once equivalent
package functions exist:

- `scripts/build_earth_features_regime.py`
- `scripts/build_cnn_patches_regime.py`
- `scripts/train_cnn_regime.py`
- `scripts/train_cnn_baseline.py`
- `scripts/train_cnn_multiseed.py`
- `scripts/train_combined_xgb_phase6b.py`
- `scripts/train_combined_xgb_regime.py`
- `scripts/run_mars_combined_regime.py`
- `scripts/retune_threshold_regime.py`
- `scripts/eval_lobo_cv.py`
- `scripts/run_regime_pipeline.sh` (or replace with a Python/package
  orchestrator plus a shell wrapper)

### Historical / Archive Candidates

No script in this audit should be archived now. After package extraction:

- `run_regime_pipeline.sh` can become an archive candidate if replaced by a
  package/CLI orchestrator that supports regA/regB/regC and preserves log/step
  behavior.
- `train_cnn_multiseed.py` is a research utility, not a core production stage.
  Keep it as a wrapper or diagnostics helper unless docs/tests confirm it is no
  longer used.
- `retune_threshold_regime.py` and `eval_lobo_cv.py` are notebook-backed
  decision-support writers. Keep wrappers unless the project decides these
  should be notebook-only.

## Duplication Map

| Concern | Existing package owner | Duplicated / script-owned copies | Future direction |
|---------|------------------------|----------------------------------|------------------|
| Regime presets | `channel_heads.regimes` | Scripts consume it; no remaining duplicate preset map except shell `regA|regB` allow-list. | Keep `regimes.py`; update shell/CLI orchestration to include regC when preserving docs behavior allows. |
| Manifest filtering (`raster_path` present and `raster_status == "ok"`) | None centralized | `train_cnn_baseline.py`, `train_cnn_regime.py`, `train_cnn_multiseed.py`, `train_combined_xgb_phase6b.py`, `train_combined_xgb_regime.py` | Move to `channel_heads/training/datasets.py` with exact filtering semantics. |
| Taiwan holdout + validation split | Constants in `training.cnn`; split logic script-owned | `train_cnn_baseline.py`, `train_cnn_regime.py`, `train_cnn_multiseed.py` | Add `training.datasets` helpers for CV pool and deterministic val split; keep `train_cnn` as the low-level loop. |
| CNN training loop | `channel_heads.training.cnn.train_cnn` | No duplicate loop in audited scripts; scripts wrap it. | Add higher-level `train_cnn_from_manifest` / `train_best_of_seeds` without changing `train_cnn`. |
| Strict CNN embedding extraction | `inference.regime.extract_regime_embeddings`; `models.mars_combined.extract_logits` for logits | `train_combined_xgb_phase6b.extract_emb_and_logit`, `train_combined_xgb_regime.extract_emb` | Extract strict embedding/logit helpers separately from lenient `models.cnn_features.extract_embeddings`. Preserve strict state-dict checks. |
| Lenient CNN embedding extraction | `models.cnn_features.extract_embeddings` | Not used by regime strict scripts | Keep distinct; do not merge strict and lenient paths blindly. |
| Feature sets/order | None centralized for training | `GEOM_FEATURES`, `EMB_FEATURES`, `FEATURES` duplicated in combined-XGB, retune, LOBO scripts | Add constants in `training/xgboost.py` or `training/datasets.py`; tests must pin order. |
| XGBoost training hyperparameters | None | Phase6B, regime combined, LOBO scripts | Add `training/xgboost.py` config(s), preserving variants where scripts differ (`tree_method`/`n_jobs` in training scripts, absent in LOBO). |
| XGBoost inference artifact loading | `models.xgboost` | `run_mars_combined_regime.py` uses only loaders, then manual `XGBClassifier.load_model` / predict | Use `load_xgb_model`, `verify_feature_matrix`, `predict_with_threshold` in future package move, preserving current behavior first with tests. |
| Grouped split by `basin__outlet` | `eval.outlet_group_holdout` | `train_combined_xgb_phase6b.py` and `train_combined_xgb_regime.py` still inline `GroupShuffleSplit` | Reuse package splitter after tests prove identical indices. |
| Threshold policy | `eval.max_precision_threshold`; `eval.f1_optimal_threshold` | Training scripts inline max-precision-at-recall; retune uses package F1 helper | Training move should call `max_precision_threshold` only after tests pin same tie-breaking and metrics columns/source strings. |
| LOBO OOF | `eval.leave_one_group_out_oof` | `eval_lobo_cv.py` owns XGB fold training and dataset map | Move XGB-specific LOBO report to `eval/lobo.py` or `training/lobo.py`; leave generic splitter in eval. |
| Mars regime embedding attach | `inference.regime` | `run_mars_combined_regime.py` already delegates attach; still owns prediction/output | Move canonical implementation to `models/regime.py` or integrate with `models.mars_combined`; leave `inference/regime.py` as shim. |
| Paths/artifact bundles | `io.paths` for common roots; many specific artifacts absent | Scripts hardcode many suffix paths with `PROJECT_ROOT / "models"` and `data/results` | Add path/artifact-bundle helpers in `io.paths` or training modules; preserve suffixes exactly. |

## Recommended Canonical Package Ownership

### `channel_heads/training/datasets.py`

Own reusable training dataset helpers:

- Load raster manifests and filter to valid rows (`raster_path.notna()` and
  optional `raster_status == "ok"`).
- Build CNN CV pools excluding `HOLDOUT_BASIN`.
- Deterministic validation splits with `np.random.default_rng(seed)` and
  `val_size = max(int(n * val_frac), 10)`.
- Feature constants or structured feature-set descriptors:
  `GEOM_FEATURES`, `EMB_FEATURES`, `CNN_LOGIT_FEATURE`, and combined feature
  lists, if not placed in `training/xgboost.py`.
- Dataset artifact path helpers for baseline vs regime manifests and
  `_with_emb` outputs.

### `channel_heads/training/cnn.py`

Keep current low-level `train_cnn` unchanged. Add higher-level wrappers later:

- `train_cnn_from_manifest(...)` for baseline/regime one-shot training.
- `train_cnn_best_of_seeds(...)` for `train_cnn_multiseed.py` behavior.
- Return model/history/artifact metadata; CLI wrappers do file writes only if
  the package function is parameterized to do so.

Do not change architecture, `OutletPairDataset`, augmentation, loss,
`pos_weight`, optimizer, early stopping, logger name, or default constants.

### `channel_heads/training/xgboost.py`

Own Earth XGBoost training recipes:

- Strict CNN feature extraction for training datasets:
  `extract_emb_and_logit_strict(...)` and `extract_emb_strict(...)`, or call a
  dedicated strict helper from `models.cnn_features`.
- `build_master_v4(...)` baseline dataset construction.
- `train_combined_variants(...)` for phase6B geom-only / geom+emb / geom+logit.
- `train_regime_combined_emb(...)` for per-regime geom+emb only.
- Common XGBoost config: `n_estimators=200`, `max_depth=4`,
  `learning_rate=0.1`, `random_state=42`, `eval_metric="logloss"`,
  `tree_method="hist"`, `n_jobs=-1`, and `scale_pos_weight=n_neg/max(n_pos,1)`.
- Threshold-tuning metrics and artifact persistence, with exact feature column
  ordering and threshold file formatting.

### `channel_heads/training/regime.py`

Own regime workflow orchestration and Earth-regime dataset generation:

- `process_regime_basin(...)` from `build_earth_features_regime.py`.
- Regime DEM discovery (`DEM_TO_BASIN`, `_resolve_basins`) or a path helper in
  `io.paths`.
- Regime master assembly: hard-negative filtering, stratified negative
  subsampling, stats CSV generation.
- Regime patch-stream loader and patch manifest generation, or a thin adapter
  around a future rasterization package helper.
- A `run_regime_pipeline(regime, ...)` function matching the shell order.

This module will depend on existing Earth analyzers and TopoToolbox. Avoid
moving `geometric_analysis.py` or `rasterizer.py` in this slice series until
their separate audits are complete.

### `channel_heads/models/regime.py`

Recommended future canonical home for regime Mars model inference:

- Move `extract_regime_embeddings` and `attach_regime_embeddings` here, keeping
  strict state-dict load, missing-patch drop behavior, embedding overwrite, and
  finite checks.
- Add a regime artifact bundle dataclass:
  CNN model, XGB model, feature-columns file, threshold file, input parquet,
  patch index, output paths.
- Add a package `run_mars_regime_inference(...)` that can replace
  `scripts/run_mars_combined_regime.py`.

Then reduce `channel_heads/inference/regime.py` to a compatibility shim. The
`inference/` package already became a shim surface for XGBoost/device; regime
should follow that pattern once tests cover it.

Alternative: fold regime Mars inference into `models.mars_combined.py` as a
parameterized variant. That is viable, but only if tests prove the regime path
still uses one geom+emb model, overwrites baseline `emb_*`, writes the current
`mars_combined_<regime>_*` outputs, and does not add baseline emb/logit columns
or comparison outputs unexpectedly.

### `channel_heads/eval/lobo.py`

Own LOBO report generation:

- Keep generic `leave_one_group_out_oof` in `eval.splitting`.
- Move XGBoost-specific LOBO report logic out of `scripts/eval_lobo_cv.py` to
  `eval/lobo.py` or `training/lobo.py`. Prefer `eval/lobo.py` because the
  workflow is model evaluation/diagnostics, not artifact training.
- Preserve per-fold AUC inclusion rule: only folds with both classes contribute
  to `fold_aucs`.

## What Should Happen To `channel_heads/inference/regime.py`

Short term: leave it untouched. It is already package code used by
`run_mars_combined_regime.py` and regime notebooks.

Future slice: move it to `channel_heads/models/regime.py` and leave
`channel_heads/inference/regime.py` as a pure re-export shim. Before moving,
add/expand tests for:

- Strict `load_state_dict(..., strict=True)` behavior.
- `patch_status == "ok"` filtering.
- Dropping rows with no patch.
- Absolute vs project-relative patch path resolution.
- Overwriting existing `emb_0..emb_3` values.
- Finite-value checks.
- Returned DataFrame dropping `patch_path_abs`.

Do not merge it with lenient `models.cnn_features.extract_embeddings`.

## Behavior That Must Be Preserved Exactly

### Regime Feature Generation

- Regime presets:
  - `regA`: `threshold_km2=0.05`, `pre_remove_max_order=2`,
    `order_gap_to_prune=4`.
  - `regB`: `threshold_km2=0.25`, `pre_remove_max_order=1`,
    `order_gap_to_prune=4`.
  - `regC`: `threshold_km2=0.10`, `pre_remove_max_order=1`,
    `order_gap_to_prune=4`.
  - Defaults: `coupling_n_workers=4`, `min_prefilter_px=30.0`.
- DEM thresholding: basin `z_th` mask, geographic-aware pixel size, km² to
  cells via `compute_threshold_cells`.
- Pruning order: `pre_remove_max_order` then `order_gap_to_prune`.
- Pair construction and feature order/columns produced by the existing
  analyzers.
- Runtime filters: `min_basin_px=500`, `max_outlets=40` by default.
- Per-outlet prefilter distance:
  `max(regime.min_prefilter_px, 2 * sqrt(outlet_basin_px))`.
- `CONNECTIVITY=8`.
- Hard-negative filter parameters:
  `max_L_ratio=3.0`, `max_dist_ratio=5.0`.
- Stratified negative subsampling:
  target neg:pos ratio `3.0`, RNG seed `42`, basin-proportional targets with
  largest-basin adjustment, final sort by `basin`, `outlet`, `confluence`.
- Existing cache behavior: load `full_features_<regime>.csv` unless `--force`.

### Patch Generation

- Regime patch stream loader must rebuild the same pruned stream as feature
  generation so node IDs resolve.
- Patch size default remains `128`.
- Patches remain 5-class `uint8` rasters using the existing rasterizer contract.
- Existing behavior writes to `RESULTS_DIR / f"_rasters_{regime.name}"` and
  writes `RESULTS_DIR / f"raster_manifest_{regime.name}.csv"`.
  Preserve actual behavior even though the script docstring still mentions
  `data/results/{basin}_<regime>/rasters/...`.

### CNN Training

- Baseline manifest: `data/results/raster_manifest.csv`.
- Regime manifest: `data/results/raster_manifest_<regime>.csv`.
- Valid rows: `raster_path` present and, if available, `raster_status == "ok"`.
- Holdout basin: `taiwan`.
- Validation split: exclude Taiwan, then `np.random.default_rng(seed)` with
  `val_size = max(int(len(df_cv) * val_frac), 10)`.
- Default hyperparameters from `training.cnn`: epochs `60`, lr `1e-3`,
  weight decay `1e-4`, batch size `64`, dropout `0.3`, patience `12`.
- CNN artifact paths:
  - Baseline: `models/cnn_outlet_final.pt`,
    `models/cnn_outlet_final_history.csv`.
  - Regime: `models/cnn_outlet_<regime>.pt`,
    `models/cnn_outlet_<regime>_history.csv`.
  - Multi-seed metrics: `models/<model_stem>_multiseed.csv`.
- Multi-seed behavior: seeds `0..N-1`, both `torch.manual_seed(seed)` and
  `np.random.seed(seed)`, validation split RNG seeded per seed, keep lowest
  validation loss state.

### Combined XGBoost Training

- Geometric feature order:
  1. `orientation_diff_deg`
  2. `headhead_dist_norm`
  3. `apex_angle_deg`
  4. `strahler_order_diff`
  5. `proximity_profile_norm`
- Embedding feature order: `emb_0`, `emb_1`, `emb_2`, `emb_3`.
- Baseline variants:
  - `geom_only`
  - `geom_plus_cnn_emb`
  - `geom_plus_cnn_logit`
- Regime variant: `geom_plus_cnn_emb_<regime>` only.
- Strict CNN state loading in training feature extraction:
  `load_state_dict(..., strict=True)` with explicit missing/unexpected check.
- Baseline `master_dataset_v4_cnn_full.csv` is rebuilt from all valid manifest
  rows, includes all 17 basins, and contains `emb_0..emb_3` plus `cnn_logit`.
- Regime `_with_emb` datasets overwrite/add `emb_0..emb_3` from the
  regime-specific CNN.
- Group split: `GroupShuffleSplit(n_splits=1, test_size=0.20,
  random_state=42)`, grouped by `f"{basin}__{outlet}"`.
- XGBoost config in training scripts:
  `n_estimators=200`, `max_depth=4`, `learning_rate=0.1`,
  `scale_pos_weight=n_neg/max(n_pos,1)`, `random_state=42`, `n_jobs=-1`,
  `eval_metric="logloss"`, `tree_method="hist"`.
- Threshold policy: choose threshold maximizing precision where recall >= 0.50;
  fallback to `0.5` if none. Preserve tie behavior from current NumPy code.
- Threshold files write exactly `"{threshold:.6f}\n"`.
- Feature columns files write one feature per line in training order and a final
  newline.
- Metrics column names and `threshold_source` strings:
  `max_precision_at_recall>=0.50`, `fallback_default_0.5`.

### Threshold Retune

- Retune is F1-optimal, not max-precision-at-recall.
- It reproduces `outlet_group_holdout(df)` on
  `master_dataset_<regime>_with_emb.csv`.
- It overwrites
  `models/optimal_threshold_geom_plus_cnn_emb_<regime>.txt`.
- It mutates the first row of
  `models/xgb_geom_plus_cnn_emb_<regime>_metrics.csv` when present.
- `threshold_source` becomes `F1_optimal_retune` in appended metrics.

### LOBO-CV

- Dataset map:
  - baseline: `data/results/master_dataset_v4_cnn_full.csv`
  - regA/B/C: `data/results/master_dataset_reg{A,B,C}_with_emb.csv`
- Features: 5 geom + 4 embeddings in the exact order above.
- Drop rows with NaN in `FEATURES + ["y", "basin"]`.
- Leave-one-basin-out OOF probabilities.
- Fold AUC list includes only held-out basins containing both labels.
- Threshold uses `max_precision_threshold` default `min_recall=0.5`.
- Output path: `models/lobo_cv_metrics.csv`.

### Mars Regime Inference

- Mars topology/features/patches are reused as-is from baseline Mars artifacts.
- Existing input artifacts:
  - `data/Mars/model_inputs/mars_model_input_tabular_plus_cnn.parquet`
  - `data/Mars/model_inputs/mars_cnn_patch_index.parquet`
  - `data/Mars/topology/mars_vn_pairs.gpkg`
- Regime model artifacts:
  - `models/cnn_outlet_<regime>.pt`
  - `models/xgb_geom_plus_cnn_emb_<regime>.json`
  - `models/feature_columns_geom_plus_cnn_emb_<regime>.txt`
  - `models/optimal_threshold_geom_plus_cnn_emb_<regime>.txt`
- `attach_regime_embeddings` filters patch index to `patch_status == "ok"`,
  drops Mars pairs without patches, and overwrites existing `emb_*`.
- Mars XGBoost prediction currently allows NaNs but logs the count; future
  package move should decide whether to preserve this exact manual behavior or
  intentionally adopt `verify_feature_matrix` after test coverage.
- High-confidence threshold remains `0.80`.
- Outputs:
  - `data/Mars/model_outputs/mars_combined_<regime>_predictions.parquet`
  - `data/Mars/model_outputs/mars_combined_<regime>_predictions.csv`
  - `data/Mars/model_outputs/mars_combined_<regime>_by_network.csv`
  - `data/Mars/model_outputs/mars_combined_<regime>_predictions.gpkg`
- Current GPKG behavior unlinks an existing output and writes one `pairs` layer.

## Proposed Implementation Slices

1. **Pin More Behavior With Tests (no moves yet).**
   Add tests for manifest filtering/splits, feature constants/order, strict CNN
   extraction, training threshold tie/fallback behavior, artifact path builders,
   and regime Mars output schema. This should be first because the scripts carry
   scientific behavior.

2. **Training Dataset Helpers.**
   Create `channel_heads/training/datasets.py` for manifest loading, valid-row
   filters, CNN CV pool selection, deterministic validation split, feature
   constants, and artifact path helpers. Repoint CNN scripts only after identity
   tests prove same row selection/splits.

3. **CNN Training Wrappers.**
   Extend `channel_heads/training/cnn.py` with high-level baseline/regime and
   multi-seed training functions. Keep `train_cnn` unchanged. Reduce
   `train_cnn_baseline.py`, `train_cnn_regime.py`, and
   `train_cnn_multiseed.py` to wrappers.

4. **Earth XGBoost Training Package.**
   Create `channel_heads/training/xgboost.py` for strict CNN feature extraction,
   baseline phase6B variant training, regime geom+emb training, threshold
   tuning, metrics, and artifact writes. Reduce
   `train_combined_xgb_phase6b.py` and `train_combined_xgb_regime.py` to
   wrappers.

5. **Regime Earth Dataset + Patch Generation.**
   Create `channel_heads/training/regime.py` for `process_basin`,
   master-dataset assembly, regime stream loader, patch manifest generation,
   and the Step 2→5 orchestrator. Keep dependencies on `geometric_analysis.py`
   and `rasterizer.py` stable; do not refactor those modules in this slice.

6. **Regime Mars Inference Package.**
   Move `channel_heads/inference/regime.py` to `channel_heads/models/regime.py`
   with a shim, and add `run_mars_regime_inference(...)` to own current
   `run_mars_combined_regime.py` behavior.

7. **LOBO Evaluation Package.**
   Add `channel_heads/eval/lobo.py` for the XGBoost LOBO report and reduce
   `scripts/eval_lobo_cv.py` to a writer wrapper.

8. **Pipeline/CLI Integration.**
   Update `channel_heads/pipelines/earth.py` and add regime pipeline entry
   points once package functions exist. Update `scripts/run_regime_pipeline.sh`
   or replace it with a Python CLI wrapper. Only then consider archiving old
   transitional scripts.

## Risks

1. **Scientific drift in feature rows.** Regime feature generation is sensitive
   to DEM masking, threshold-cell conversion, pruning order, outlet filtering,
   and per-outlet prefilter distances.
2. **Node-ID consistency.** Regime patch generation must rebuild the exact same
   pruned stream used to generate `master_dataset_<regime>.csv`; otherwise
   `head_1` / `head_2` / `confluence` IDs will not resolve to the same geometry.
3. **Strict vs lenient CNN loads.** Generic Earth embeddings are lenient; regime
   and combined training/inference are strict. Merging helpers can silently
   change artifact compatibility behavior.
4. **Embedding overwrite behavior.** Regime Mars inference intentionally
   replaces baseline Mars `emb_*` columns with regime CNN embeddings and drops
   pairs without patches.
5. **Feature order.** XGBoost feature order is part of the model contract; text
   files and DataFrame selection must match exactly.
6. **Threshold policy ambiguity.** Training threshold (`max precision at recall
   >= 0.50`) and retune threshold (`F1_optimal_retune`) are different workflows.
   Do not collapse them into one generic threshold function without explicit
   call-site policy.
7. **Artifact paths.** Production artifacts are frozen; baseline research
   variants and regime variants must keep explicit suffixes and avoid
   overwriting production files unless the user explicitly requests a rebuild.
8. **Pipeline shell coupling.** `run_regime_pipeline.sh` currently supports only
   regA/regB despite docs mentioning regC; changing that is a behavior/CLI
   change and should be handled deliberately.
9. **Hardcoded path depth.** Several scripts compute roots with
   `Path(__file__).resolve().parents[1]`; moving scripts into subdirectories or
   wrappers can break paths unless package path helpers replace them.
10. **Generated-output writes.** Future extraction must be careful not to
    generate or mutate data/models during tests; use temp paths and mocks.

## Tests Needed For Future Moves

| Future move | Tests to add or expand |
|-------------|------------------------|
| `training/datasets.py` | Synthetic manifest tests for valid-row filtering, optional `raster_status`, Taiwan exclusion, deterministic val split indices, `val_size` floor, feature constants/order. |
| CNN wrappers | Temp-path tests that monkeypatch `train_cnn`; assert baseline/regime artifact paths, history CSV schema, model path suffixes, CLI defaults, and no production overwrite from regime paths. |
| Multi-seed CNN | Monkeypatch `train_cnn` to return controlled val losses; assert seeds used, best-state selection, output path, and metrics CSV rows. |
| XGBoost training | Synthetic DataFrame tests for `scale_pos_weight`, GroupShuffleSplit grouping, feature order files, threshold tie/fallback behavior, metrics keys/source strings, strict embedding/logit extraction shapes. |
| Regime feature generation | Unit tests for `stratified_subsample_negatives`, `_resolve_basins`, default min/max outlet options, stats schema; mocked TopoToolbox tests for threshold/pruning call order if feasible. |
| Regime patch generation | Mock `precompute_raster_dataset` and stream loader; assert output root, manifest path, target size default, regime threshold/pruning use, missing DEM behavior. |
| `models/regime.py` | Expand `test_inference_regime.py` for strict load, patch filtering/drop, absolute/relative path resolution, embedding overwrite, finite checks, output schema. |
| Mars regime inference | Mock model/feature loaders and `attach_regime_embeddings`; assert feature-missing failure, NaN logging behavior, prediction columns, high-confidence counts, per-network CSV schema, GPKG layer behavior. |
| LOBO | Synthetic multi-basin data with a single-class fold; assert OOF alignment, fold-AUC skip rule, threshold policy, output metrics schema. |
| Pipeline integration | Update `tests/test_pipelines.py`: Earth/regime package stages should no longer be marked `TRANSITIONAL` after extraction and should not call `run_script`. |

## Stop Conditions For Future Refactor Slices

Stop and ask before changing any of these:

- Regime preset values or names.
- Feature order or number of features.
- Patch class count, target size, raster values, or augmentation/inference mode.
- CNN load strictness at any call site.
- Threshold policy or threshold file values.
- Model artifact paths or production/regime suffixes.
- Mars regime inference dropping/overwriting behavior.
- LOBO fold grouping or fold-AUC inclusion rules.
- Any `data/`, root `models/`, notebooks, DEMs, shapefiles, GeoPackages, CSV/
  parquet outputs, or figures.
