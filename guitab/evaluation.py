"""Evaluation helpers.

GuitarSet chunk protocol (scripts/evaluate.py, scripts/sweep.py): the fixed 15 s chunks of
data/eval_chunks.json (146 test / 147 validation, GuitarSet players 05 / 04). Each prediction is the
full-track note list cut to the chunk and written as token text at the reference tempo, then scored
with metrics.compute_note_detection_metrics (onset +-50 ms or +-120 ticks; pitch or string/fret).

Full-track checks used during training: match_f1 on GuitarSet (string/fret, pitch) and GAPS
(pitch F1 + string agreement with the score fingering).
"""

import json
from pathlib import Path

import numpy as np

from .guitar import OPEN_PITCH, STRINGS
from .guitarset import string_notes
from .inference import extract_notes, predict
from .metrics import compute_note_detection_metrics
from .paths import EVAL_CHUNKS

KEYS = ["time_pitch_f1", "time_string_fret_f1", "tick_pitch_f1", "tick_string_fret_f1"]


def load_chunks(split):
    """(refs, meta[(audio_path, start_s)]) of the fixed GuitarSet chunks.

    data/eval_chunks.json was generated once with the Audio2Tab repo's evaluation schedule (DadaGPChunkDataset,
    15 s chunks, hop 15 s) on GuitarSet players 04/05; it is a frozen artifact - do not regenerate."""
    chunks = json.loads(EVAL_CHUNKS.read_text())[split]
    return [c["text"] for c in chunks], [(c["audio_path"], c["start"]) for c in chunks]


def score(refs, preds):
    m = compute_note_detection_metrics(refs, preds, 50, 120)
    return {k: round(m[k], 4) for k in KEYS}


def chunk_text(ref, notes, start, dur=15.0):
    """Notes in [start, start + dur) as token text at the reference tempo with exact waits."""
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


def match_f1(ref, est, tol=0.05, key=("string", "fret")):
    """Greedy note F1 by onset (+-tol s) and identity (`key` fields)."""
    used = set()
    tp = 0
    by_id = {}
    for i, n in enumerate(est):
        by_id.setdefault(tuple(n[k] for k in key), []).append(i)
    for r in ref:
        cands = [i for i in by_id.get(tuple(r[k] for k in key), []) if i not in used and abs(est[i]["onset"] - r["onset"]) <= tol]
        if cands:
            used.add(min(cands, key=lambda i: abs(est[i]["onset"] - r["onset"])))
            tp += 1
    prec = tp / max(len(est), 1)
    rec = tp / max(len(ref), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def ref_notes(jams_path, shift=0):
    """GuitarSet reference notes as dicts (onset, offset, string, fret, pitch)."""
    out = []
    for on, off, col, fret in string_notes(jams_path, shift):
        s = STRINGS[col]
        out.append({"onset": on, "offset": off, "string": s, "fret": fret, "pitch": OPEN_PITCH[s] + fret})
    return out


def evaluate_guitarset(model, val, device, ref_cache):
    """Mean per-track note F1 (onset +-50 ms) by string/fret and by pitch on full GuitarSet tracks."""
    model.eval()
    sf, pf = [], []
    for tr in val:
        tab, onset = predict(model, tr["feat"], device)
        est = extract_notes(tab, onset)
        ref = ref_cache[tr["name"]]
        sf.append(match_f1(ref, est, key=("string", "fret")))
        pf.append(match_f1(ref, est, key=("pitch",)))
    model.train()
    return float(np.mean(sf)), float(np.mean(pf))


def evaluate_gaps(model, val, device, ref_cache):
    """Pitch F1 against all GAPS MIDI notes, and string agreement on pitch-correct notes with known strings."""
    model.eval()
    pf, agree, total = [], 0, 0
    for tr in val:
        tab, onset = predict(model, tr["feat"], device)
        est = extract_notes(tab, onset)
        ref = ref_cache[tr["name"]]
        pf.append(match_f1(ref, est, key=("pitch",)))
        known = [r for r in ref if r["string"]]
        sf_hits = match_f1(known, est, key=("string", "fret"))
        p_hits = match_f1(known, est, key=("pitch",))
        if p_hits > 0:
            agree += sf_hits / p_hits * len(known)
            total += len(known)
    model.train()
    return float(np.mean(pf)), agree / max(total, 1)


def gaps_ref(label_path):
    """GAPS reference notes as dicts; string 0 / fret None = string unknown."""
    out = []
    for on, off, s, f in json.loads(Path(label_path).read_text())["notes"]:
        pitch = OPEN_PITCH[s] + f if s else f
        out.append({"onset": on, "offset": off, "string": s, "fret": f if s else None, "pitch": pitch})
    return out
