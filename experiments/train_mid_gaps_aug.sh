#!/bin/bash
# E12: fine-tune E10 with audio augmentation (phone / room renderings, 2 copies per track, half the crops);
# validates on clean and held-out degraded copies; resumes from last.pt; optional STOP_AT=HH:MM.
cd "$(dirname "$0")/.."  # repository root
export OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_mid_gaps_aug --size mid --device ${DEVICE:-auto} --max_memory 0.35 \
  --batch 16 --steps 6000 --lr 1e-4 --warmup 300 --eval_every 1000 --gaps_share 0.5 --gaps_tab_weight 0.5 \
  --init runs/tabmodel_mid_gaps/best.pt --aug_copies 2 --aug_share 0.5 --eval_degraded \
  --resume ${STOP_AT:+--stop_at $STOP_AT}
