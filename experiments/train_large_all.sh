#!/bin/bash
# E18: "large" model (66M) from scratch on all data mixed per crop: GuitarSet .25, GAPS .3 (strings .5), SynthTab .2
# (strings .3), custom-dataset .25 (strings .5); 3 augmented copies on 60% of crops; clean + degraded validation.
# 15 GB memory cap, batch 16 in 2 chunks; resumes from last.pt; optional STOP_AT=HH:MM.
# When the laptop is otherwise idle: HIGH=0.8 LOW=0.7 ACCUM=1 (~22 GB cap, whole batch at once, faster).
cd "$(dirname "$0")/.."  # repository root
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${HIGH:-0.53} PYTORCH_MPS_LOW_WATERMARK_RATIO=${LOW:-0.45} OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_large_all --size large \
  --batch 16 --accum ${ACCUM:-2} --steps 30000 --lr 3e-4 --warmup 1000 --eval_every 1000 \
  --gaps_share 0.3 --gaps_tab_weight 0.5 --synth_share 0.2 --synth_tab_weight 0.3 --ss_share 0.25 --ss_tab_weight 0.5 \
  --aug_copies 3 --aug_share 0.6 --eval_degraded --resume ${STOP_AT:+--stop_at $STOP_AT}
