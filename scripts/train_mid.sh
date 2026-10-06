#!/bin/bash
# E8: mid CQT tab model on GuitarSet only; resumes from last.pt; optional STOP_AT=HH:MM.
cd "$(dirname "$0")/.."  # repository root
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_mid --size mid \
  --batch 16 --steps 12000 --eval_every 1000 --resume ${STOP_AT:+--stop_at $STOP_AT}
