#!/bin/bash
# E15: fine-tune E12 for close-mic phone recordings with pinch-rich fingerstyle data: SynthTab (string labels at weight
# 0.3) + augmented copies incl. aug2 (close-mic chain); clean + degraded validation; resumes from last.pt.
cd "$(dirname "$0")/.."  # repository root
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_mid_close --size mid \
  --batch 16 --steps 8000 --lr 1e-4 --warmup 300 --eval_every 1000 --gaps_share 0.35 --gaps_tab_weight 0.5 \
  --synth_share 0.3 --synth_tab_weight 0.3 --init runs/tabmodel_mid_gaps_aug/best.pt --aug_copies 3 --aug_share 0.6 \
  --eval_degraded --resume ${STOP_AT:+--stop_at $STOP_AT}
