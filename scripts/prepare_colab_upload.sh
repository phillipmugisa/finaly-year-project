#!/usr/bin/env bash
# scripts/prepare_colab_upload.sh
# ─────────────────────────────────────────────────────────────────────────────
# Packages everything needed for Colab GPU training into uploadable files.
#
# Run this AFTER the pipeline produces dataset.csv (or use dataset_2122.csv
# for a quick validation run).
#
# Usage:
#   bash scripts/prepare_colab_upload.sh [dataset.csv] [features_output.npz]
#
# Defaults:
#   dataset    = outputs/dataset.csv
#   features   = outputs/features.npz
#
# What it does:
#   1. Creates outputs/vci_estimator_scripts.zip  (project code for Colab)
#   2. Runs extract_features.py on the dataset (can take 30–60 min)
#   3. Prints upload instructions
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."

DATASET="${1:-outputs/dataset.csv}"
FEATURES="${2:-outputs/features.npz}"
ZIP_OUT="outputs/vci_estimator_scripts.zip"

echo "================================================================"
echo "  VCI Estimator — Colab Upload Prep"
echo "  Dataset  : $DATASET"
echo "  Features : $FEATURES"
echo "================================================================"

# ── 1. Package project scripts ──────────────────────────────────────────────
echo ""
echo "[1/2] Packaging project scripts → $ZIP_OUT"
rm -f "$ZIP_OUT"
zip -q -r "$ZIP_OUT" \
    src/ \
    scripts/train_features.py \
    scripts/pretrain_pci.py \
    scripts/extract_features.py \
    configs/ \
    -x "**/__pycache__/*" \
    -x "**/*.pyc" \
    -x "**/.DS_Store"

SIZE_ZIP=$(du -h "$ZIP_OUT" | cut -f1)
echo "  Created $ZIP_OUT  ($SIZE_ZIP)"

# ── 2. Extract features ──────────────────────────────────────────────────────
echo ""
echo "[2/2] Extracting backbone features → $FEATURES"
echo "  (This runs EfficientNet-B3 over every image — may take 30–90 min)"
echo ""

source venv/bin/activate 2>/dev/null || true

python scripts/extract_features.py \
    --dataset    "$DATASET" \
    --output     "$FEATURES" \
    --batch-size 32 \
    --num-workers 4 \
    --resume

# ── 3. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Done! Upload these files to Google Drive:"
echo "    MyDrive/vci_estimator/"
echo "    ├── $(basename $FEATURES)"
echo "    ├── $(basename $DATASET)"
echo "    └── $(basename $ZIP_OUT)"
echo ""
echo "  Then open colab_train.ipynb in Google Colab (GPU runtime)"
echo "================================================================"
