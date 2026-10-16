#!/bin/bash
# E17: fine-tune E15 with real fingerstyle recordings labelled by their reference tab (30% of crops, string labels at 0.5),
# plus GAPS / SynthTab / GuitarSet and the 3 augmented copies; clean + degraded validation; resumes from last.pt.
cd "$(dirname "$0")/.."  # repository root
export OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_mid_custom --size mid --device ${DEVICE:-auto} --max_memory 0.35 \
  --batch 16 --steps 6000 --lr 1e-4 --warmup 300 --eval_every 1000 --gaps_share 0.3 --gaps_tab_weight 0.5 \
  --synth_share 0.2 --synth_tab_weight 0.3 --ss_share 0.3 --ss_tab_weight 0.5 --init runs/tabmodel_mid_close/best.pt \
  --aug_copies 3 --aug_share 0.6 --eval_degraded --resume ${STOP_AT:+--stop_at $STOP_AT}
