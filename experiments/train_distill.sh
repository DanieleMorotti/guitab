#!/bin/bash
# E24 / E25: student distilled from the E18 + E19 average (scripts/train.py --teacher), E19's recipe otherwise.
# SIZE=half (E24, 2.9M) | small (E25, 6.2M) -> runs/tabmodel_<size>_distill; resumes from last.pt; optional STOP_AT=HH:MM.
# Usage: SIZE=small experiments/train_distill.sh > runs/tabmodel_small_distill/train.out 2>&1
cd "$(dirname "$0")/.."  # repository root
export OMP_NUM_THREADS=2
SIZE=${SIZE:-half}
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_${SIZE}_distill --size $SIZE --device ${DEVICE:-auto} --max_memory 0.35 \
  --batch 16 --steps 24000 --lr 3e-4 --warmup 1000 --eval_every 1000 \
  --gaps_share 0.25 --gaps_tab_weight 0.5 --synth_share 0.15 --synth_tab_weight 0.3 --ss_share 0.35 --ss_tab_weight 0.5 \
  --aug_copies 3 --aug_share 0.6 --eval_degraded \
  --teacher runs/tabmodel_large_all/best.pt --teacher runs/tabmodel_mid_all/best.pt --distill 0.5 \
  --resume ${STOP_AT:+--stop_at $STOP_AT}
