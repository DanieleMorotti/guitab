"""Evaluate the chord model on the held-out data (data/chords/, scripts/build_chord_data.py).

    python scripts/evaluate_chords.py [--weights runs/chords/best.safetensors] [--split test]

Sets: GuitarSet (comp and solo takes; played and instructed chords) and the custom-dataset songs of the chosen half
("test", or "val" for the checkpoint-selection half). Scores (guitab.evaluation.chord_scores): majmin = right root and
major / minor family over the reference chords' time, root, exact chord, and the share of the time with a chord shown;
"all" shows every decoded chord, "confident" only those with mean probability >= guitab.chords.CONFIDENCE.
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.chord_data import CHORD_DATA  # noqa: E402
from guitab.chords import CONFIDENCE, chord_probs, chord_spans, chord_weights, load_chord_model  # noqa: E402
from guitab.evaluation import chord_scores  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", default=None)
    p.add_argument("--split", default="test", choices=["test", "validation"])
    args = p.parse_args()
    torch.set_num_threads(4)
    model = load_chord_model(args.weights or chord_weights())
    gs = pickle.load(open(CHORD_DATA / "guitarset.pkl", "rb"))[args.split]
    custom = [s for s in pickle.load(open(CHORD_DATA / "custom_eval.pkl", "rb"))
              if s["half"] == ("test" if args.split == "test" else "val")]
    sets = [("GuitarSet comp, played", [s for s in gs if s["take"] == "comp"], "chords"),
            ("GuitarSet comp, instructed", [s for s in gs if s["take"] == "comp"], "instructed"),
            ("GuitarSet solo, played", [s for s in gs if s["take"] == "solo"], "chords"),
            (f"custom songs ({len(custom)})", custom, "chords")]
    print(f"{'':28s} {'majmin':>7s} {'root':>6s} {'exact':>6s} {'shown':>6s} | confident: {'majmin':>7s} {'shown':>6s}")
    for name, songs, key in sets:
        acc = {}
        for song in songs:
            bounds, probs = chord_probs(model, song["notes"], song["beats"], song["downbeats"])
            for mode, conf in (("all", 0.0), ("confident", CONFIDENCE)):
                s = chord_scores(song[key], chord_spans(bounds, probs, confidence=conf))
                acc[mode] = {k: acc.get(mode, {}).get(k, 0) + v for k, v in s.items()}
        a, c = acc["all"], acc["confident"]
        print(f"{name:28s} {a['majmin'] / a['majmin_time']:7.3f} {a['root'] / a['time']:6.3f} {a['exact'] / a['time']:6.3f} "
              f"{a['shown'] / a['time']:6.2f} | {'':10s} {c['majmin'] / c['majmin_time']:7.3f} {c['shown'] / c['time']:6.2f}")


if __name__ == "__main__":
    main()
