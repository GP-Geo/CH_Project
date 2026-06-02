#!/usr/bin/env bash
# Full Mars-calibration pipeline for one regime (regA or regB).
#
# Usage:
#   scripts/run_regime_pipeline.sh regA   2>&1 | tee /tmp/pipeline_regA.log
#   scripts/run_regime_pipeline.sh regB   2>&1 | tee /tmp/pipeline_regB.log
#
# Each step writes its own log to /tmp/<regime>_<step>.log so progress is
# visible without buffering through pipes.
set -euo pipefail

REGIME="${1:?usage: $0 regA|regB}"
case "$REGIME" in
  regA|regB) ;;
  *) echo "regime must be regA or regB"; exit 2 ;;
esac

cd "$(dirname "$0")/.."
LOG_DIR="/tmp/regime_${REGIME}"
mkdir -p "$LOG_DIR"

echo "=== Step 2: Earth features (regime=$REGIME) ==="
python scripts/build_earth_features_regime.py \
    --regime "$REGIME" -v \
    > "$LOG_DIR/step2_features.log" 2>&1
tail -20 "$LOG_DIR/step2_features.log"

echo "=== Step 3: CNN patches (regime=$REGIME) ==="
python scripts/build_cnn_patches_regime.py \
    --regime "$REGIME" -v \
    > "$LOG_DIR/step3_patches.log" 2>&1
tail -20 "$LOG_DIR/step3_patches.log"

echo "=== Step 4: Train CNN (regime=$REGIME) ==="
python scripts/train_cnn_regime.py \
    --regime "$REGIME" -v \
    > "$LOG_DIR/step4_cnn.log" 2>&1
tail -30 "$LOG_DIR/step4_cnn.log"

echo "=== Step 5: Train combined XGBoost (regime=$REGIME) ==="
python scripts/train_combined_xgb_regime.py \
    --regime "$REGIME" -v \
    > "$LOG_DIR/step5_xgb.log" 2>&1
tail -30 "$LOG_DIR/step5_xgb.log"

echo "=== Step 6: Mars inference (regime=$REGIME) ==="
python scripts/run_mars_combined_regime.py \
    --regime "$REGIME" -v \
    > "$LOG_DIR/step6_mars.log" 2>&1
tail -30 "$LOG_DIR/step6_mars.log"

echo "=== Pipeline regime=$REGIME complete; logs in $LOG_DIR/ ==="
