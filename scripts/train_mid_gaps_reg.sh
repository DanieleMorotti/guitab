#!/bin/bash
# E11: E10 (mid, GuitarSet + GAPS) with Kong-style attack targets and a release head; resumes from last.pt; optional STOP_AT=HH:MM.
cd "$(dirname "$0")/.."  # repository root
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2
exec nice -n 10 .venv/bin/python scripts/train.py --out runs/tabmodel_mid_gaps_reg --size mid \
  --batch 16 --steps 16000 --eval_every 1000 --gaps_share 0.5 --gaps_tab_weight 0.5 \
  --onset_target regression --eval_params '{"onset": 0.4, "subframe": true}' --resume ${STOP_AT:+--stop_at $STOP_AT}
