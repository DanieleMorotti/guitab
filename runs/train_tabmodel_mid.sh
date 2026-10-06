#!/bin/bash
# Train (or resume) the mid-size CQT tab model with a memory cap; optional STOP_AT=HH:MM.
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python tools/train_tabmodel.py --out runs/tabmodel_mid --size mid \
  --batch 16 --steps 12000 --eval_every 1000 --resume ${STOP_AT:+--stop_at $STOP_AT}
