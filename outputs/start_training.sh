#!/bin/bash
set -e
cd /home/mugisa/Desktop/fyp
source venv/bin/activate

echo "[$(date)] Polling for prepare_data.py (PID 9560) to finish..."
while kill -0 9560 2>/dev/null; do
  sleep 30
done

ROWS=$(tail -n +2 outputs/dataset.csv 2>/dev/null | wc -l)
echo "[$(date)] Pipeline done. dataset.csv has $ROWS image rows."

if [ "$ROWS" -lt 100 ]; then
  echo "[$(date)] ERROR: dataset too small ($ROWS rows). Aborting training."
  exit 1
fi

echo "[$(date)] Starting training..."
python scripts/train_model.py \
  --dataset    outputs/dataset.csv \
  --device     cuda \
  --epochs     40 \
  --batch-size 32

echo "[$(date)] Training complete."
