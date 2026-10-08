#!/bin/bash
# E14: fine-tune E12 with the time-sharp low band added to the input (new input channels start at zero),
# same recipe as E12 (GAPS half the crops, 2 augmented copies, clean + degraded validation); resumes from last.pt.
cd "$(dirname "$0")/.."  # repository root
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_mid_gaps_sharp --size mid \
  --batch 16 --steps 6000 --lr 1e-4 --warmup 300 --eval_every 1000 --gaps_share 0.5 --gaps_tab_weight 0.5 \
  --init runs/tabmodel_mid_gaps_aug/best.pt --aug_copies 2 --aug_share 0.5 --eval_degraded --sharp \
  --resume ${STOP_AT:+--stop_at $STOP_AT}
