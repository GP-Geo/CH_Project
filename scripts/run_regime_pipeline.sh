#!/usr/bin/env bash
# Mars-calibration pipeline for one regime (regA, regB, or regC).
#
# Usage:
#   scripts/run_regime_pipeline.sh <regime> [mode]
#
#   <regime>  regA | regB | regC
#   [mode]    full     (default) Steps 2-6: rebuild Earth features + CNN patches,
#                       retrain CNN + combined XGBoost, then Mars inference.
#             retrain  Steps 4-5 ONLY: retrain CNN + combined XGBoost on the
#                       EXISTING reconciled Stage-7 rasters/manifests. Skips the
#                       feature rebuild (2), patch rebuild (3), and Mars inference
#                       (6). Use this when Stage 7 is already reconciled and you
#                       only want fresh regime models (Stage 8).
#
# Examples:
#   scripts/run_regime_pipeline.sh regA            2>&1 | tee /tmp/pipeline_regA.log
#   scripts/run_regime_pipeline.sh regC retrain    2>&1 | tee /tmp/pipeline_regC.log
#
# Each step writes its own log to /tmp/regime_<regime>/<step>.log so progress is
# visible without buffering through pipes.
set -euo pipefail

REGIME="${1:?usage: $0 regA|regB|regC [full|retrain]}"
MODE="${2:-full}"
case "$REGIME" in
  regA|regB|regC) ;;
  *) echo "regime must be regA, regB, or regC"; exit 2 ;;
esac
case "$MODE" in
  full|retrain) ;;
  *) echo "mode must be 'full' or 'retrain'"; exit 2 ;;
esac

cd "$(dirname "$0")/.."
LOG_DIR="/tmp/regime_${REGIME}"
mkdir -p "$LOG_DIR"

echo "=== run_regime_pipeline: regime=$REGIME mode=$MODE ==="

if [ "$MODE" = "full" ]; then
  echo "=== Step 2: Earth features (regime=$REGIME) ==="
  python scripts/cli/build_earth_features_regime.py \
      --regime "$REGIME" -v \
      > "$LOG_DIR/step2_features.log" 2>&1
  tail -20 "$LOG_DIR/step2_features.log"

  echo "=== Step 3: CNN patches (regime=$REGIME) ==="
  python scripts/cli/build_cnn_patches_regime.py \
      --regime "$REGIME" -v \
      > "$LOG_DIR/step3_patches.log" 2>&1
  tail -20 "$LOG_DIR/step3_patches.log"
else
  echo "=== retrain mode: skipping Step 2 (features) + Step 3 (patches);"
  echo "    consuming EXISTING data/results/raster_manifest_${REGIME}.csv ==="
fi

echo "=== Step 4: Train CNN (regime=$REGIME) ==="
python scripts/cli/train_cnn_regime.py \
    --regime "$REGIME" -v \
    > "$LOG_DIR/step4_cnn.log" 2>&1
tail -30 "$LOG_DIR/step4_cnn.log"

echo "=== Step 5: Train combined XGBoost (regime=$REGIME) ==="
python scripts/cli/train_combined_xgb_regime.py \
    --regime "$REGIME" -v \
    > "$LOG_DIR/step5_xgb.log" 2>&1
tail -30 "$LOG_DIR/step5_xgb.log"

if [ "$MODE" = "full" ]; then
  echo "=== Step 6: Mars inference (regime=$REGIME) ==="
  python scripts/cli/run_mars_combined_regime.py \
      --regime "$REGIME" -v \
      > "$LOG_DIR/step6_mars.log" 2>&1
  tail -30 "$LOG_DIR/step6_mars.log"
else
  echo "=== retrain mode: skipping Step 6 (Mars inference) ==="
fi

echo "=== Pipeline regime=$REGIME mode=$MODE complete; logs in $LOG_DIR/ ==="
