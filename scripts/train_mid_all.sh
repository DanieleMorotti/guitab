#!/bin/bash
# E19: mid model (24M) from scratch on all data with the enlarged custom dataset (~350 songs): GuitarSet .25, GAPS .25
# (strings .5), SynthTab .15 (strings .3), custom-dataset .35 (strings .5); 3 augmented copies on 60% of crops; clean +
# degraded validation; default memory cap; resumes from last.pt; optional STOP_AT=HH:MM.
cd "$(dirname "$0")/.."  # repository root
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_mid_all --size mid \
  --batch 16 --steps 24000 --lr 3e-4 --warmup 1000 --eval_every 1000 \
  --gaps_share 0.25 --gaps_tab_weight 0.5 --synth_share 0.15 --synth_tab_weight 0.3 --ss_share 0.35 --ss_tab_weight 0.5 \
  --aug_copies 3 --aug_share 0.6 --eval_degraded --resume ${STOP_AT:+--stop_at $STOP_AT}
