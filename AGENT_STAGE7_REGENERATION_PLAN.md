# AGENT_STAGE7_REGENERATION_PLAN.md

Report-only plan for Stage 7 Earth model-input regeneration.

Date: 2026-06-03
Branch: `refactor/package-first-architecture`
Type: planning only - no data, model, notebook, raster, embedding, prediction,
or training artifact was regenerated.

## 1. Purpose

Stage 7 regenerates Earth model inputs after the Stage 4 Earth-Mars regime
selection and the Stage 5 Earth network QA gate. Its job is to refresh the
model-ready Earth inputs that depend on the direct-final-grid rasterizer:
regime CNN patches, patch manifests, and any downstream CNN-derived tables that
must be rebuilt before model training.

This plan does not run the regeneration. It records the required inputs,
outputs, commands, risk controls, and validation gates for a later execution
slice.

## 2. Current Readiness Status

- Stage 4 regime rationale exists:
  - `docs/REGIME_SELECTION.md`
  - `notebooks/regime/00_calibration_overview.ipynb`
  - frozen presets in `channel_heads/regimes.py`
- Stage 5 QA passed:
  - QA report: `data/results/stage5_earth_network_qa_report.csv`
  - hard flags: 0
  - soft warnings: 6
- Soft warnings are documented and non-blocking:
  - `regA/yoro`: 9 pairs
  - `regB/yoro`: 8 pairs
  - `regC/finisterre`: touching ratio 0.03
  - `regC/luliang`: touching ratio 0.05
  - `regC/sierramadre`: touching ratio 0.03
  - `regC/vallefertil`: touching ratio 0.03

Input gap noted during planning: the request named `docs/STAGE_ASSET_MAP.md`,
but the file present in this checkout is `STAGE_ASSET_MAP.md` at repo root.
This plan uses the root file and marks that path mismatch as documentation
cleanup, not a data blocker.

## 3. Inputs Required For Stage 7

Earth regime pair / feature data:

- `data/results/master_dataset_regA.csv`
- `data/results/master_dataset_regB.csv`
- `data/results/master_dataset_regC.csv`
- `data/results/build_earth_features_regA_stats.csv`
- `data/results/build_earth_features_regB_stats.csv`
- `data/results/build_earth_features_regC_stats.csv`
- `data/results/<basin>/full_features_regA.csv`
- `data/results/<basin>/full_features_regB.csv`
- `data/results/<basin>/full_features_regC.csv`

Branch paths, labels, and geometric features:

- The `master_dataset_reg{A,B,C}.csv` files must include the Stage 6 columns
  used by rasterization and training:
  - node/pair columns: `outlet`, `confluence`, `head_1`, `head_2`, `basin`
  - labels: `touching`, `contact_px`, `y`
  - branch/path-derived columns: `L_1`, `L_2`, `delta_L`
  - geometric columns: `orientation_diff_deg`, `headhead_dist_m`,
    `headhead_dist_norm`, `apex_angle_deg`, `strahler_order_diff`,
    `proximity_mean_m`, `proximity_max_m`, `proximity_profile_norm`
  - QA/support columns: `size1_px`, `size2_px`, `skipped_prefilter`, `qc_flags`

Raster patch generation:

- Raw/source DEM inputs, read-only:
  - `data/cropped_DEMs/*.tif`
  - canonical map in `channel_heads/io/paths.py::EXAMPLE_DEMS`
- Regime definitions:
  - `channel_heads/regimes.py::REGIMES`
- Package helpers:
  - `channel_heads.training.regime.build_regime_patch_dataset`
  - `channel_heads.training.regime.regime_patch_paths`
  - `channel_heads.rasterization.earth_batch.precompute_raster_dataset`
  - `channel_heads.rasterization.earth_patches`

Patch manifests:

- Existing stale manifests to be refreshed:
  - `data/results/raster_manifest_regA.csv`
  - `data/results/raster_manifest_regB.csv`
  - `data/results/raster_manifest_regC.csv`
- Required manifest columns after regeneration include:
  - all master dataset columns
  - `raster_path`
  - `raster_debug_path`
  - `raster_status`
  - `raster_error`
  - `has_branch_a`
  - `has_branch_b`
  - `has_confluence`
  - `branch_a_connected`
  - `branch_b_connected`
  - `branches_connected`

Downstream CNN embedding extraction inputs:

- Stage 7 produces the patch manifests and `.npy` patch paths consumed by
  Stage 8 training scripts:
  - `scripts/cli/train_cnn_regime.py`
  - `scripts/cli/train_combined_xgb_regime.py`
- Downstream embedding extraction requires regenerated regime CNN models later:
  - `models/cnn_outlet_regA.pt`
  - `models/cnn_outlet_regB.pt`
  - `models/cnn_outlet_regC.pt`
- Do not touch those model artifacts in Stage 7.

## 4. Outputs To Regenerate

Primary Stage 7 outputs:

- Earth CNN raster patches for each regime:
  - `data/results/_rasters_regA/`
  - `data/results/_rasters_regB/`
  - `data/results/_rasters_regC/`
- Patch manifests:
  - `data/results/raster_manifest_regA.csv`
  - `data/results/raster_manifest_regB.csv`
  - `data/results/raster_manifest_regC.csv`

Feature tables:

- `data/results/master_dataset_reg{A,B,C}.csv` already passed Stage 5 and can be
  kept as the Stage 7 input.
- Regenerate feature tables only if a later execution slice decides the Stage 6
  caches must be refreshed with `--force`.
- If refreshed, the same paths are reused:
  - `data/results/<basin>/full_features_reg{A,B,C}.csv`
  - `data/results/master_dataset_reg{A,B,C}.csv`
  - `data/results/build_earth_features_reg{A,B,C}_stats.csv`

Combined geometric + CNN-input tables:

- `data/results/master_dataset_regA_with_emb.csv`
- `data/results/master_dataset_regB_with_emb.csv`
- `data/results/master_dataset_regC_with_emb.csv`

These are CNN-derived and stale. They should be regenerated during Stage 8
after new regime CNN models are trained, not during this Stage 7 raster-only
slice, unless an execution slice explicitly chooses a frozen CNN for embedding
refresh.

QA / contact-sheet outputs:

- No dedicated Earth regime contact-sheet CLI is confirmed in this checkout.
- Use existing diagnostics notebooks for visual QA if needed:
  - `notebooks/diagnostics/rasterization_diagnostics.ipynb`
  - `notebooks/training/03_feature_engineering.ipynb`
- Treat any new contact sheets as optional report artifacts until a canonical
  Earth patch QA command is added.

## 5. Existing Stale / Generated Outputs

Classification based on `docs/DATA_STATUS.md` and
`AGENT_DATA_CLEANUP_DRYRUN.md`.

Keep for now:

- `data/cropped_DEMs/*.tif` - `RAW_KEEP`, source input.
- `data/results/master_dataset_regA.csv`
- `data/results/master_dataset_regB.csv`
- `data/results/master_dataset_regC.csv`
- `data/results/build_earth_features_regA_stats.csv`
- `data/results/build_earth_features_regB_stats.csv`
- `data/results/build_earth_features_regC_stats.csv`
- `data/results/stage5_earth_network_qa_report.csv`
- `data/_rebuild_backup_20260531/` - backup snapshot; keep off-repo.

Can be overwritten by the later regeneration commands:

- `data/results/raster_manifest_regA.csv`
- `data/results/raster_manifest_regB.csv`
- `data/results/raster_manifest_regC.csv`

Should be archived before regeneration if preservation is desired:

- `data/results/_rasters_regA/`
- `data/results/_rasters_regB/`
- `data/results/_rasters_regC/`
- `data/results/master_dataset_regA_with_emb.csv`
- `data/results/master_dataset_regB_with_emb.csv`
- `data/results/master_dataset_regC_with_emb.csv`

Unclear, do not touch in Stage 7:

- `data/results/<basin>/rasters/` - stale baseline per-basin rasters, but not
  required for regime patch regeneration.
- `data/results/raster_manifest.csv` - baseline manifest, not a regime Stage 7
  output.
- `data/results/master_dataset_v4_cnn_full.csv` - baseline CNN-derived table;
  stale but belongs to baseline training/embedding refresh, not this regime
  Stage 7 slice.
- `models/*` - do not touch model artifacts until Stage 8.
- `data/Mars/model_inputs/*` and `data/Mars/model_outputs/*` - Mars stages are
  downstream and out of scope for Earth Stage 7.
- `data/results/figures_models/`, `data/exports/*.pdf`, and Mars figure folders
  - report artifacts; regenerate only after model/inference outputs are current.

## 6. Proposed Command Sequence

Do not run these commands until a later execution slice.

Preflight checks:

```bash
test -f docs/REGIME_SELECTION.md
test -f STAGE_ASSET_MAP.md
test -f data/results/stage5_earth_network_qa_report.csv
test -f data/results/master_dataset_regA.csv
test -f data/results/master_dataset_regB.csv
test -f data/results/master_dataset_regC.csv
```

Optional Earth regime feature refresh only if the Stage 6 caches need to be
rebuilt. Omit `--force` to preserve cache behavior; add `--force` only after
archiving any existing generated outputs that must be preserved.

```bash
python scripts/cli/build_earth_features_regime.py --regime regA -v
python scripts/cli/build_earth_features_regime.py --regime regB -v
python scripts/cli/build_earth_features_regime.py --regime regC -v
```

Regime CNN patch regeneration:

```bash
python scripts/cli/build_cnn_patches_regime.py --regime regA -v
python scripts/cli/build_cnn_patches_regime.py --regime regB -v
python scripts/cli/build_cnn_patches_regime.py --regime regC -v
```

Patch manifest validation:

```bash
python - <<'PY'
from pathlib import Path

import pandas as pd

for regime in ["regA", "regB", "regC"]:
    manifest = Path(f"data/results/raster_manifest_{regime}.csv")
    df = pd.read_csv(manifest)
    assert len(df) > 0, manifest
    assert "raster_status" in df and "raster_path" in df, manifest
    ok = df[df["raster_status"] == "ok"]
    assert len(ok) > 0, manifest
    missing = [path for path in ok["raster_path"] if not Path(path).exists()]
    assert not missing, (manifest, len(missing))
    print(regime, len(df), len(ok))
PY
```

Optional QA / contact-sheet review:

```bash
jupyter notebook notebooks/diagnostics/rasterization_diagnostics.ipynb
jupyter notebook notebooks/training/03_feature_engineering.ipynb
```

Smoke tests after regeneration:

```bash
conda run -n ch-heads python -m pytest tests/test_training_regime.py tests/test_rasterizer.py tests/test_cnn_consolidation.py
conda run -n ch-heads python -m pytest
```

Do not use `scripts/run_regime_pipeline.sh` for Stage 7 alone because it
continues into Stage 8 training and Stage 11 Mars inference.

## 7. Risk Controls

- No deletion before backup or archive.
- Do not touch raw/source/manual data:
  - `data/raw/`
  - `data/cropped_DEMs/`
  - `data/final_valleys/`
  - raw DEMs, shapefiles, GeoPackages, parquet/csv source inputs
- Do not touch `models/` in Stage 7.
- Do not retrain in Stage 7.
- Do not modify notebooks in Stage 7.
- Do not run `scripts/run_regime_pipeline.sh` for this slice because it trains
  models and runs Mars inference.
- Stop if any expected input is missing:
  - Stage 5 QA report
  - `master_dataset_reg{A,B,C}.csv`
  - DEMs for the Stage 5 basins
  - `channel_heads.regimes.REGIMES` entries for `regA`, `regB`, `regC`
- Preserve regime preset values, threshold-to-cells conversion, DEM
  z-threshold masking, `CONNECTIVITY=8`, pruning order, and the 5-class
  `128x128 uint8` patch contract.

## 8. Validation Plan

After regeneration, check:

- Expected file counts:
  - each `data/results/_rasters_reg{A,B,C}/` directory exists
  - each `raster_manifest_reg{A,B,C}.csv` has rows
  - each manifest has at least one `raster_status == "ok"` row
- No missing patches:
  - every `raster_path` for `raster_status == "ok"` exists on disk
- Raster class values are valid:
  - sampled `.npy` patches contain only the frozen classes `0, 1, 2, 3, 4`
  - dtype is `uint8`
  - shape is `128x128`
- Patch manifests match feature rows:
  - manifest row count equals the corresponding `master_dataset_reg*.csv` row
    count unless an explicit skip policy is documented
  - `basin`, `outlet`, `confluence`, `head_1`, `head_2`, and `y` remain present
- QA flags are sane:
  - inspect `has_branch_a`, `has_branch_b`, `has_confluence`,
    `branch_a_connected`, `branch_b_connected`, `branches_connected`
  - investigate any basin with zero ok rasters
- Smoke tests pass:
  - `tests/test_training_regime.py`
  - `tests/test_rasterizer.py`
  - `tests/test_cnn_consolidation.py`
  - full pytest
- Visual QA:
  - contact sheets or notebook visual checks exist for sampled patches before
    proceeding to training
- Documentation gate:
  - `STAGE_ASSET_MAP.md` can mark Stage 7 regeneration complete only after the
    regenerated manifests and patch directories validate.

## 9. Decision Gate

Proceed to Stage 8 only after Stage 7 regeneration passes validation.

Stage 8:

- train/retrain Earth CNNs
- train/retrain combined XGBoost variants
- refresh threshold and metrics outputs

Stage 9:

- run validation, LOBO diagnostics, threshold tuning, and model comparison

Do not proceed to Stage 8 or Stage 9 using stale `STALE_AFTER_RASTER_FIX`
patches or manifests.

## 10. Recommended Next Action

Recommended next action: archive stale generated Earth model-input artifacts
before running regeneration commands.

Minimum archive candidates:

- `data/results/_rasters_regA/`
- `data/results/_rasters_regB/`
- `data/results/_rasters_regC/`
- `data/results/raster_manifest_regA.csv`
- `data/results/raster_manifest_regB.csv`
- `data/results/raster_manifest_regC.csv`
- `data/results/master_dataset_regA_with_emb.csv`
- `data/results/master_dataset_regB_with_emb.csv`
- `data/results/master_dataset_regC_with_emb.csv`

If disk space or time is a stronger constraint, the alternative is to run the
three patch regeneration commands directly and let manifests be overwritten,
but only after confirming no stale outputs need to be preserved.
