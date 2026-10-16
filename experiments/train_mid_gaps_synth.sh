#!/bin/bash
# E13: fine-tune E12 on GuitarSet + GAPS + a 20 h SynthTab fingerstyle subset (rendered acoustic guitars, exact labels),
# with audio augmentation; validates on clean + degraded real audio (selection) and reports SynthTab validation.
cd "$(dirname "$0")/.."  # repository root
export OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_mid_gaps_synth --size mid --device ${DEVICE:-auto} --max_memory 0.35 \
  --batch 16 --steps 8000 --lr 1e-4 --warmup 300 --eval_every 1000 --gaps_share 0.35 --gaps_tab_weight 0.5 \
  --synth_share 0.3 --init runs/tabmodel_mid_gaps_aug/best.pt --aug_copies 2 --aug_share 0.5 --eval_degraded \
  --resume ${STOP_AT:+--stop_at $STOP_AT}
