#!/usr/bin/env bash
# End-to-end model rebuild on the fixed direct-rasterization patches.
#
# Runs unattended. Trains the production baseline + regA/regB/regC, re-extracts
# Earth + Mars embeddings, and reruns Mars combined inference for every config.
#
# Pure-geometry artifacts (geom-only tabular model + Mars 5feat tabular
# predictions) are raster-independent, so they are restored from the pre-rebuild
# backup instead of retrained.
#
# Usage:
#   conda run -n ch-heads bash scripts/run_full_rebuild.sh 2>&1 | tee /tmp/rebuild/run.log
set -uo pipefail

cd "$(dirname "$0")/.."
LOG_DIR="/tmp/rebuild"
mkdir -p "$LOG_DIR"
BK="data/_rebuild_backup_20260531"

step() {  # step <name> <logfile> -- <cmd...>
  local name="$1" logf="$2"; shift 3
  echo ">>> [$(date +%H:%M:%S)] START $name"
  if "$@" > "$logf" 2>&1; then
    echo ">>> [$(date +%H:%M:%S)] OK    $name"
    tail -4 "$logf" | sed 's/^/      /'
  else
    echo "!!! [$(date +%H:%M:%S)] FAIL  $name  (see $logf)"
    tail -25 "$logf" | sed 's/^/      /'
    exit 1
  fi
}

echo "=================== FULL REBUILD START ==================="

# --- 0. Restore raster-independent Mars tabular predictions ----------------
echo ">>> Restoring raster-independent Mars 5feat tabular predictions"
for ext in parquet csv gpkg; do
  if [ -f "$BK/mars_model_outputs/mars_xgb_predictions_5feat.$ext" ]; then
    cp "$BK/mars_model_outputs/mars_xgb_predictions_5feat.$ext" \
       "data/Mars/model_outputs/mars_xgb_predictions_5feat.$ext"
  fi
done

# ========================= BASELINE (production) ==========================
step "baseline:train_cnn"        "$LOG_DIR/b1_cnn.log"        -- python -m channel_heads train-cnn-baseline -v
step "baseline:combined_xgb"     "$LOG_DIR/b2_xgb.log"        -- python -m channel_heads train-combined-xgb-phase6b
step "baseline:mars_embeddings"  "$LOG_DIR/b3_mars_emb.log"   -- python -m channel_heads run-mars-pipeline --stage embeddings
step "baseline:mars_combined"    "$LOG_DIR/b4_mars_comb.log"  -- python -m channel_heads run-mars-pipeline --stage combined

# ============================= REGIMES ====================================
for R in regA regB regC; do
  MASTER="data/results/master_dataset_${R}.csv"
  if [ ! -f "$MASTER" ]; then
    echo "!!! Missing $MASTER — skipping $R (run build_earth_features_regime.py --regime $R)"
    continue
  fi
  step "$R:patches"        "$LOG_DIR/${R}_1_patches.log"  -- python -m channel_heads build-cnn-patches --regime "$R" -v
  step "$R:train_cnn"      "$LOG_DIR/${R}_2_cnn.log"      -- python -m channel_heads train-cnn-regime --regime "$R" -v
  step "$R:combined_xgb"   "$LOG_DIR/${R}_3_xgb.log"      -- python -m channel_heads train-combined-xgb-regime --regime "$R" -v
  step "$R:mars_combined"  "$LOG_DIR/${R}_4_mars.log"     -- python -m channel_heads run-mars-combined-regime --regime "$R" -v
done

echo "=================== FULL REBUILD COMPLETE ==================="
