#!/usr/bin/env bash
# scripts/auto_extract.sh
# ─────────────────────────────────────────────────────────────────────────────
# Waits for the data pipeline to finish, then runs feature extraction.
# Run this in a terminal and leave it overnight — everything will be ready
# for Colab in the morning.
#
# Usage:
#   bash scripts/auto_extract.sh [pipeline_pid]
#
# If no PID given, watches for outputs/dataset.csv to appear/grow.
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

PIPELINE_PID="${1:-9238}"
DATASET_CSV="outputs/dataset.csv"
FEATURES_NPZ="outputs/features.npz"
SCRIPTS_ZIP="outputs/vci_estimator_scripts.zip"
LOG="outputs/extract_features.log"

echo "================================================================"
echo "  VCI Feature Extraction — Auto Launcher"
echo "  Watching pipeline PID: $PIPELINE_PID"
echo "  Will extract to: $FEATURES_NPZ"
echo "================================================================"

# ── Wait for pipeline to finish ─────────────────────────────────────────────
echo ""
echo "Waiting for pipeline (PID $PIPELINE_PID) to finish..."
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
    # Show brief progress every 2 minutes
    LATEST=$(tail -1 "$DATASET_CSV.in_progress" 2>/dev/null \
             || grep "GPS progress" outputs/prepare_data.log 2>/dev/null | tail -1 \
             || echo "  (pipeline running)")
    echo "  [$(date '+%H:%M')] $LATEST"
    sleep 120
done

echo ""
echo "[$(date '+%H:%M')] Pipeline finished."

# ── Verify dataset.csv exists and has data ───────────────────────────────────
if [[ ! -f "$DATASET_CSV" ]]; then
    echo "ERROR: $DATASET_CSV not found after pipeline exit. Check prepare_data.log"
    exit 1
fi
N_ROWS=$(wc -l < "$DATASET_CSV")
echo "  dataset.csv: $((N_ROWS - 1)) rows"

# ── Run feature extraction ───────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M')] Starting feature extraction..."
echo "  Log: $LOG"
echo "  This will take ~2 hours on CPU."

source venv/bin/activate

python scripts/extract_features.py \
    --dataset    "$DATASET_CSV" \
    --output     "$FEATURES_NPZ" \
    --batch-size 32 \
    --num-workers 4 \
    --resume \
    2>&1 | tee "$LOG"

# ── Package scripts for Colab ────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M')] Packaging project scripts → $SCRIPTS_ZIP"
rm -f "$SCRIPTS_ZIP"
zip -q -r "$SCRIPTS_ZIP" \
    src/ \
    scripts/train_features.py \
    scripts/pretrain_pci.py \
    scripts/extract_features.py \
    configs/ \
    -x "**/__pycache__/*" "**/*.pyc"

SIZE_F=$(du -h "$FEATURES_NPZ" | cut -f1)
SIZE_D=$(du -h "$DATASET_CSV"  | cut -f1)
SIZE_Z=$(du -h "$SCRIPTS_ZIP"  | cut -f1)

echo ""
echo "================================================================"
echo "  ALL DONE — Ready for Colab!"
echo ""
echo "  Upload these 3 files to Google Drive:"
echo "    MyDrive/vci_estimator/"
echo "    ├── features.npz    ($SIZE_F)"
echo "    ├── dataset.csv     ($SIZE_D)"
echo "    └── vci_estimator_scripts.zip  ($SIZE_Z)"
echo ""
echo "  Then open colab_train.ipynb in Google Colab (GPU → T4)"
echo "================================================================"
