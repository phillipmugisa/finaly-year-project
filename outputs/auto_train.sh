#!/bin/bash
# Waits for prepare_data.py (PID $1) to finish, then starts training.
set -e
cd /home/mugisa/Desktop/fyp
source venv/bin/activate

PIPELINE_PID=${1:-11951}
echo "[$(date)] Waiting for pipeline (PID $PIPELINE_PID) to finish..."
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
    sleep 60
done

ROWS=$(tail -n +2 outputs/dataset.csv 2>/dev/null | wc -l)
echo "[$(date)] Pipeline done. dataset.csv has $ROWS image rows."

if [ "$ROWS" -lt 500 ]; then
    echo "[$(date)] ERROR: dataset too small ($ROWS rows). Check prepare_data.log."
    exit 1
fi

echo "[$(date)] Starting CPU training (batch 8, 40 epochs)..."
python scripts/train_model.py \
    --dataset    outputs/dataset.csv \
    --device     cpu \
    --epochs     40 \
    --batch-size 8 \
    >> outputs/training.log 2>&1

echo "[$(date)] Training complete." >> outputs/training.log

echo "[$(date)] Starting model export..."
python scripts/export_model.py \
    --checkpoint outputs/checkpoints/best.pt \
    >> outputs/training.log 2>&1

echo "[$(date)] Export complete." >> outputs/training.log
