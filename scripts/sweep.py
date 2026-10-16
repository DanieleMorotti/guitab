"""Precision/recall sweep of note-extraction and fingering settings, plus standard GuitarSet metrics.

1. Sweep (chunk protocol, same as scripts/evaluate.py): onset threshold, onset-less notes ("orphan"),
   fingering preset -> precision / recall / F1 for pitch and string+fret, on validation AND test.
   Choose settings on validation; test is only reported.
2. Benchmark metrics on full test tracks (comparable in definition to TabCNN-style papers):
   frame-level multi-pitch P/R/F and tablature P/R/F (per frame, per string+fret), and note-level
   onset-only (50 ms) F1 by pitch and by string+fret against the JAMS annotations.

    python scripts/sweep.py --checkpoint runs/tabmodel_mid_gaps/best.pt --out runs/sweep_tabmodel_mid_gaps.json
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.device import DEVICES, pick_device  # noqa: E402
from guitab.evaluation import chunk_text, load_chunks, match_f1, ref_notes, token_metrics  # noqa: E402
from guitab.fingering import FINGERING_PRESETS, refinger_text  # noqa: E402
from guitab.guitar import OPEN_PITCH, STRINGS  # noqa: E402
from guitab.guitarset import jams_path  # noqa: E402
from guitab.inference import DEFAULT_PARAMS, extract_notes, load_model, predict  # noqa: E402
from guitab.paths import CACHE  # noqa: E402


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return round(p, 4), round(r, 4), round(2 * p * r / max(p + r, 1e-9), 4)


def frame_metrics(tab_probs, labels):
    """Frame-level multi-pitch and tablature P/R/F (TabCNN definition) over [T, 6] class labels."""
    pred = tab_probs.argmax(-1)  # [T, 6] class 0 = silent
    valid = (labels >= 0).all(1)
    pred, labels = pred[valid], labels[valid]
    opens = np.array([OPEN_PITCH[s] for s in STRINGS])
    tab_tp = int(((pred == labels) & (labels > 0)).sum())
    tab_fp = int(((pred > 0) & (pred != labels)).sum())
    tab_fn = int(((labels > 0) & (pred != labels)).sum())
    mp_tp = mp_fp = mp_fn = 0
    for p_row, l_row in zip(pred, labels):
        ps = {opens[i] + p_row[i] - 1 for i in range(6) if p_row[i] > 0}
        ls = {opens[i] + l_row[i] - 1 for i in range(6) if l_row[i] > 0}
        mp_tp += len(ps & ls)
        mp_fp += len(ps - ls)
        mp_fn += len(ls - ps)
    return np.array([tab_tp, tab_fp, tab_fn, mp_tp, mp_fp, mp_fn])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=DEVICES, help="auto = cuda, else mps, else cpu")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)

    splits = {}
    for split in ("validation", "test"):
        refs, meta = load_chunks(split)
        probs, labels = {}, {}
        for path in sorted({m[0] for m in meta}):
            name = Path(path).stem.removesuffix("_mic")
            z = np.load(CACHE / f"{name}.npz")
            probs[path] = predict(model, z["feat"], device)
            labels[path] = (name, z["tab"].astype(np.int64))
        splits[split] = (refs, meta, probs, labels)
        print(f"cached {split}", flush=True)

    results = {"sweep": [], "benchmark": {}}
    configs = list(itertools.product([0.2, 0.3, 0.4, 0.5, 0.6], [False, True], ["model", "auto", "low"]))
    for onset, orphan, fing in configs:
        params = {**DEFAULT_PARAMS, "onset": onset, "orphan": orphan, "frame": 0.5 if orphan else DEFAULT_PARAMS["frame"]}
        row = {"onset": onset, "orphan": orphan, "fingering": fing}
        for split, (refs, meta, probs, _) in splits.items():
            notes = {p: extract_notes(*probs[p], params) for p in probs}
            preds = [chunk_text(r, notes[p], s) for r, (p, s) in zip(refs, meta)]
            if fing != "model":
                preds = [refinger_text(t, FINGERING_PRESETS[fing]) for t in preds]
            m = token_metrics(refs, preds)
            for k in ("time_pitch", "time_string_fret"):
                row[f"{split}_{k}"] = [round(m[f"{k}_precision"], 4), round(m[f"{k}_recall"], 4), round(m[f"{k}_f1"], 4)]
            row[f"{split}_notes_per_chunk"] = round(sum(len([n for n in notes[p] if s <= n["onset"] < s + 15]) for p, s in meta) / len(meta), 1)
        results["sweep"].append(row)
        print(json.dumps(row), flush=True)

    # benchmark-style metrics on full test tracks
    _, _, probs, labels = splits["test"]
    counts = np.zeros(6, dtype=np.int64)
    note_sf, note_p = [], []
    for path, (tab, onset) in probs.items():
        name, lab = labels[path]
        n = min(len(lab), len(tab))
        counts += frame_metrics(tab[:n], lab[:n])
        ref = ref_notes(jams_path(name))
        est = extract_notes(tab, onset)
        note_sf.append(match_f1(ref, est, key=("string", "fret")))
        note_p.append(match_f1(ref, est, key=("pitch",)))
    results["benchmark"] = {
        "frame_tab_PRF": prf(*counts[:3]),
        "frame_multipitch_PRF": prf(*counts[3:]),
        "note_onset50ms_pitch_F1_track_mean": round(float(np.mean(note_p)), 4),
        "note_onset50ms_string_fret_F1_track_mean": round(float(np.mean(note_sf)), 4),
    }
    print("BENCHMARK", json.dumps(results["benchmark"]))
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
