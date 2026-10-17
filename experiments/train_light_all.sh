#!/bin/bash
# E20-E23: lighter models with E19's recipe (from scratch, all data, 24k steps): SIZE=small | gru | tcn | tiny
# (guitab/model.py) -> runs/tabmodel_<size>_all; default memory cap; resumes from last.pt; optional STOP_AT=HH:MM.
cd "$(dirname "$0")/.."  # repository root
export OMP_NUM_THREADS=2
SIZE=${SIZE:?set SIZE=small, gru, tcn or tiny}
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_${SIZE}_all --size $SIZE --device ${DEVICE:-auto} --max_memory 0.35 \
  --batch 16 --steps 24000 --lr 3e-4 --warmup 1000 --eval_every 1000 \
  --gaps_share 0.25 --gaps_tab_weight 0.5 --synth_share 0.15 --synth_tab_weight 0.3 --ss_share 0.35 --ss_tab_weight 0.5 \
  --aug_copies 3 --aug_share 0.6 --eval_degraded --resume ${STOP_AT:+--stop_at $STOP_AT}
