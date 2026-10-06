"""Evaluate tab-model checkpoints on GAPS (classical / fingerstyle, real YouTube recordings).

Per split (default gaps_test): pitch F1 (onset +-50 ms) against all fine-aligned MIDI notes, and
string agreement = share of pitch-correct notes (with a score string label) on the score's string.
Score fingerings are an editor's suggestion, so agreement is a noisy proxy for tab accuracy.

    python tools/eval_gaps.py --checkpoints runs/tabmodel_mid/best.pt runs/tabmodel_mid_gaps/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabdata import load_split  # noqa: E402
from tabmodel_infer import load_model  # noqa: E402
from train_tabmodel import evaluate_gaps, gaps_ref  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--split", default="gaps_test")
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = load_split(args.split)
    refs = {tr["name"]: gaps_ref(tr["labels"]) for tr in data}
    print(f"{args.split}: {len(data)} pieces, {sum(len(r) for r in refs.values())} notes")
    for ckpt in args.checkpoints:
        model = load_model(ckpt, device)
        pitch_f1, agree = evaluate_gaps(model, data, device, refs)
        print(f"{ckpt}: pitch F1 {pitch_f1:.4f}, string agreement {agree:.4f}", flush=True)
        del model
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
