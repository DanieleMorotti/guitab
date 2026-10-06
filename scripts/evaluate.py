"""Evaluate a tab-model checkpoint on the fixed GuitarSet 15 s chunks (protocol: guitab/evaluation.py).

    python scripts/evaluate.py --checkpoint runs/tabmodel_mid_gaps/best.pt --split test
    python scripts/evaluate.py --checkpoint ... --split validation --tune   # threshold search (validation only!)
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.evaluation import chunk_text, load_chunks, score  # noqa: E402
from guitab.fingering import refinger_text  # noqa: E402
from guitab.inference import DEFAULT_PARAMS, extract_notes, load_model, predict  # noqa: E402
from guitab.paths import CACHE  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--params", default=None, help="JSON note-extraction params")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    refs, meta = load_chunks(args.split)
    probs = {}
    for path in sorted({m[0] for m in meta}):
        name = Path(path).stem.removesuffix("_mic")
        feat = np.load(CACHE / f"{name}.npz")["feat"]
        probs[path] = predict(model, feat, device)

    def run(params):
        notes = {p: extract_notes(*probs[p], params) for p in probs}
        preds = [chunk_text(r, notes[p], s) for r, (p, s) in zip(refs, meta)]
        return preds, score(refs, preds)

    params = {**DEFAULT_PARAMS, **(json.loads(args.params) if args.params else {})}
    if args.tune:
        results = []
        if next(iter(probs.values()))[2] is None:
            grid = [{"onset": on, "frame": fr, "orphan": orphan, "legato": legato} for on, fr, orphan, legato
                    in itertools.product([0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6], [True, False], [3, 5, 8])]
        else:  # release head (E11): also the release threshold (1.01 = releases unused) and sub-frame attack times
            grid = [{"onset": on, "frame": fr, "orphan": False, "legato": legato, "offset": off, "subframe": sub}
                    for on, fr, legato, off, sub in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6], [5, 8],
                                                                      [0.3, 0.5, 0.7, 1.01], [False, True])]
        for g in grid:
            prm = {**params, **g}
            _, s = run(prm)
            results.append((s["tick_string_fret_f1"] + s["time_string_fret_f1"] + s["time_pitch_f1"], prm, s))
        results.sort(key=lambda r: -r[0])
        for _, prm, s in results[:5]:
            print(json.dumps(prm), s)
        params = results[0][1]
        print("BEST", json.dumps(params))

    preds, s = run(params)
    print(f"{args.split} model strings      :", s)
    for name, w in [("auto", {"position": 0.05, "move": 0.35, "model": 2.0}), ("strong", {"position": 0.05, "move": 0.2, "model": 4.0})]:
        print(f"{args.split} + fingering {name:6}:", score(refs, [refinger_text(p, w) for p in preds]))


if __name__ == "__main__":
    main()
