"""Evaluation helpers.

GuitarSet chunk protocol (scripts/evaluate.py, scripts/sweep.py): the fixed 15 s chunks of
data/eval_chunks.json (146 test / 147 validation, GuitarSet players 05 / 04). Each prediction is the
full-track note list cut to the chunk and written as token text at the reference tempo, then scored
with token_metrics (onset +-50 ms or +-120 ticks; pitch or string/fret): notes are matched one-to-one,
closest onsets first, over the whole set of chunks' counts.

Full-track checks used during training: match_f1 on GuitarSet (string/fret, pitch) and GAPS
(pitch F1 + string agreement with the score fingering).
"""

import json
from pathlib import Path

import numpy as np

from .guitar import OPEN_PITCH, STRINGS
from .guitarset import string_notes
from .inference import extract_notes, predict
from .paths import EVAL_CHUNKS

KEYS = ["time_pitch_f1", "time_string_fret_f1", "tick_pitch_f1", "tick_string_fret_f1"]


def load_chunks(split):
    """(refs, meta[(audio_path, start_s)]) of the fixed GuitarSet chunks.

    data/eval_chunks.json was generated once with the Audio2Tab repo's evaluation schedule (DadaGPChunkDataset,
    15 s chunks, hop 15 s) on GuitarSet players 04/05; it is a frozen artifact - do not regenerate."""
    chunks = json.loads(EVAL_CHUNKS.read_text())[split]
    return [c["text"] for c in chunks], [(c["audio_path"], c["start"]) for c in chunks]


def token_notes(text):
    """[(onset s, onset ticks, string, fret)] of a token text (960 ticks per quarter at its first tempo)."""
    tokens = text.split()
    tempo = next((int(t.split(":")[1]) for t in tokens if t.startswith("tempo:")), 120)
    seconds, ticks, out = 0.0, 0, []
    for t in tokens:
        if t.startswith("wait:"):
            w = int(t.split(":")[1])
            ticks += w
            seconds += w / (960.0 * tempo / 60.0)  # summed wait by wait: the recorded scores depend on this rounding
        elif t.startswith("note:") or ":note:" in t:
            s, f = t.split(":")[-2:]
            out.append((seconds, ticks, int(s[1:]), int(f[1:])))
    return out


def _matches(ref, est, onset, ident, tol):
    """One-to-one matches with equal identity and |onset difference| <= tol, closest pairs first."""
    pairs = sorted((abs(r[onset] - e[onset]), i, j) for i, r in enumerate(ref) for j, e in enumerate(est)
                   if ident(r) == ident(e) and abs(r[onset] - e[onset]) <= tol)
    used_r, used_e = set(), set()
    for _, i, j in pairs:
        if i not in used_r and j not in used_e:
            used_r.add(i)
            used_e.add(j)
    return len(used_r)


def token_metrics(refs, preds, time_tol=0.05, tick_tol=120):
    """Precision / recall / F1 over all (reference, prediction) token texts, for time_ / tick_ onsets and _pitch /
    _string_fret identities."""
    pitch = lambda n: OPEN_PITCH[n[2]] + n[3]  # noqa: E731
    string_fret = lambda n: (n[2], n[3])  # noqa: E731
    configs = {"time_pitch": (0, pitch, time_tol), "time_string_fret": (0, string_fret, time_tol),
               "tick_pitch": (1, pitch, tick_tol), "tick_string_fret": (1, string_fret, tick_tol)}
    counts = {k: [0, 0, 0] for k in configs}  # matched, references, predictions
    for ref_text, pred_text in zip(refs, preds):
        ref, est = token_notes(ref_text), token_notes(pred_text)
        for k, (onset, ident, tol) in configs.items():
            c = counts[k]
            c[0] += _matches(ref, est, onset, ident, tol)
            c[1] += len(ref)
            c[2] += len(est)
    out = {}
    for k, (tp, n_ref, n_est) in counts.items():
        p, r = (tp / n_est if n_est else 0.0), (tp / n_ref if n_ref else 0.0)
        out.update({f"{k}_precision": p, f"{k}_recall": r, f"{k}_f1": 2 * p * r / (p + r) if p + r else 0.0})
    return out


def score(refs, preds):
    m = token_metrics(refs, preds)
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


def match(ref, est, key=("string", "fret"), tol=0.05):
    """Greedy one-to-one matching of note dicts: each reference note in turn takes the unused estimate with the same
    `key` fields and the closest onset within +-tol s. Returns [(ref index, est index)]."""
    by_id, used, pairs = {}, set(), []
    for i, n in enumerate(est):
        by_id.setdefault(tuple(n[k] for k in key), []).append(i)
    for j, r in enumerate(ref):
        cands = [i for i in by_id.get(tuple(r[k] for k in key), []) if i not in used and abs(est[i]["onset"] - r["onset"]) <= tol]
        if cands:
            i = min(cands, key=lambda i: abs(est[i]["onset"] - r["onset"]))
            used.add(i)
            pairs.append((j, i))
    return pairs


def match_f1(ref, est, tol=0.05, key=("string", "fret")):
    """Note F1 of the greedy matching (match)."""
    tp = len(match(ref, est, key, tol))
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
        est = extract_notes(tab, onset, getattr(model, "decoding", None))
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
        est = extract_notes(tab, onset, getattr(model, "decoding", None))
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


def _beat_span(ref, est):
    """est beats inside the stretches where the reference has beats (gaps over twice its median interval split them):
    GAPS references leave out passages the score alignment cannot place."""
    if len(ref) < 2:
        return est
    ibi = np.median(np.diff(ref))
    cut = np.where(np.diff(ref) > 2 * ibi)[0]
    keep = np.zeros(len(est), dtype=bool)
    for a, b in zip(np.r_[ref[0], ref[cut + 1]], np.r_[ref[cut], ref[-1]]):
        keep |= (est >= a - ibi / 2) & (est <= b + ibi / 2)
    return est[keep]


def beat_fmeasure(ref, est, tol=0.07):
    """F-measure of beat (or downbeat) times, one-to-one within +-tol s, over the reference's span (_beat_span)."""
    ref, est = np.asarray(ref, dtype=float), np.asarray(est, dtype=float)
    est = _beat_span(ref, est)
    if not len(ref) or not len(est):
        return 0.0
    used, hit = np.zeros(len(est), dtype=bool), 0
    for r in ref:
        d = np.abs(est - r)
        d[used] = np.inf
        k = int(np.argmin(d))
        if d[k] <= tol:
            used[k] = True
            hit += 1
    return 2 * hit / (len(ref) + len(est))


def rhythm_accuracy(ref, est, onsets):
    """Share of consecutive note-onset intervals that get the same 16th-grid length (in beats) with the estimated beats
    as with the reference beats, under the best global factor 1, 2 or 1/2 (a piece written in eighths instead of
    quarters is still right). Only onsets between two reference beats at most 1.5 median intervals apart count."""
    ref, est, on = (np.asarray(a, dtype=float) for a in (ref, est, onsets))
    if len(ref) < 2 or len(est) < 2:
        return np.nan
    k = np.searchsorted(ref, on)
    ok = (k > 0) & (k < len(ref))
    gap = np.full(len(on), np.inf)
    gap[ok] = ref[k[ok]] - ref[k[ok] - 1]
    on = on[gap <= 1.5 * np.median(np.diff(ref))]
    on = on[(on >= max(ref[0], est[0])) & (on <= min(ref[-1], est[-1]))]
    if len(on) < 3:
        return np.nan

    def grid(beats):
        return np.diff(np.round(np.interp(on, beats, np.arange(len(beats))) * 4) / 4)

    r, e = grid(ref), grid(est)
    pos = r > 0
    return max(float(np.mean(np.abs(e[pos] - f * r[pos]) < 1e-6)) for f in (1, 2, 0.5))
