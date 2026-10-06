"""Evaluate the CQT tab model on the fixed GuitarSet 15 s chunks (data/eval_chunks.json) with the note metric.

Each chunk's prediction is the full-track note list cut to [start, start + 15 s), written as
token text at the reference tempo with exact waits, so the repo metric
(compute_note_detection_metrics, 50 ms / 120 ticks) compares like with like.

    python tools/eval_tabmodel.py --checkpoint runs/tabmodel_mid/best.pt --split test
    python tools/eval_tabmodel.py --checkpoint ... --split validation --tune   # threshold search
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from fingering import refinger_text  # noqa: E402
from tabdata import CACHE  # noqa: E402
from note_metrics import compute_note_detection_metrics  # noqa: E402
from tabmodel_infer import DEFAULT_PARAMS, extract_notes, load_model, predict  # noqa: E402

KEYS = ["time_pitch_f1", "time_string_fret_f1", "tick_pitch_f1", "tick_string_fret_f1"]


def load_chunks(split):
    """(refs, meta[(audio_path, start_s)]) of the fixed GuitarSet chunks.

    data/eval_chunks.json was generated once with the Audio2Tab repo's evaluation schedule (DadaGPChunkDataset,
    15 s chunks, hop 15 s) on GuitarSet players 04/05; it is a frozen artifact - do not regenerate."""
    chunks = json.loads((ROOT / "data" / "eval_chunks.json").read_text())[split]
    return [c["text"] for c in chunks], [(c["audio_path"], c["start"]) for c in chunks]


def score(refs, preds):
    m = compute_note_detection_metrics(refs, preds, 50, 120)
    return {k: round(m[k], 4) for k in KEYS}


def chunk_text(ref, notes, start, dur=15.0):
    tempo = int(next(t for t in ref.split() if t.startswith("tempo:")).split(":")[1])
    tps = tempo / 60 * 960
    out, cur = [f"tempo:{tempo}", "start"], start
    for n in sorted((n for n in notes if start <= n["onset"] < start + dur), key=lambda n: (n["onset"], n["string"])):
        w = int(round((n["onset"] - cur) * tps))
        if w > 0:
            out.append(f"wait:{w}")
            cur += w / tps
        out.append(f"note:s{n['string']}:f{n['fret']}")
    return " ".join(out)


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
        for on, fr, orphan, legato in itertools.product([0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6], [True, False], [3, 5, 8]):
            prm = {**params, "onset": on, "frame": fr, "orphan": orphan, "legato": legato}
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
