"""Inference for the CQT tab model: sliding-window prediction and note extraction.

Notes are extracted per string:
  - a note starts at an onset peak (local max of the onset probability above `onset`), or where
    the string becomes active (P(not silent) > `frame`) for >= `min_frames` frames without one
    and `orphan` is enabled;
  - its fret is the argmax of the fret posterior averaged over its first frames;
  - it ends when the string goes silent, at the next onset on that string, or when the frame
    fret changes for `legato` consecutive frames (hammer-on / pull-off / slide -> new note);
  - the same pitch started on two strings within `dedup` frames is one note played once: only the
    note whose fret posterior (mean over its frames) is higher is kept (0 = off);
  - a note that starts exactly where the previous note on the same string and fret ends (an onset peak that split a
    ringing note) is a real re-attack only if its onset probability >= `reattack`; otherwise the two are merged (0 = off).
With a capo (apply_capo, D6), notes below the capo move to another string that can play their pitch.
"""

import numpy as np
import torch

from .audio import FPS, compute_cqt, load_audio
from .guitar import OPEN_PITCH, STRINGS
from .model import TabModel

# tuned on GuitarSet validation (player 04) with scripts/evaluate.py --tune
DEFAULT_PARAMS = {"onset": 0.6, "frame": 0.6, "min_frames": 3, "legato": 8, "orphan": False, "fret_frames": 3, "dedup": 4,
                  "reattack": 0.75}


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TabModel(ckpt["size"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, feat, device, win=432, hop=288):
    """feat [T, 288] -> (tab_probs [T, 6, 21], onset_probs [T, 6]) with overlapping windows."""
    t_total = len(feat)
    tab = np.zeros((t_total, 6, model.n_classes), dtype=np.float32)
    onset = np.zeros((t_total, 6), dtype=np.float32)
    weight = np.zeros((t_total, 1), dtype=np.float32)
    starts = list(range(0, max(t_total - win, 0) + 1, hop))
    if starts[-1] + win < t_total:
        starts.append(t_total - win)
    taper = np.hanning(win + 2)[1:-1].astype(np.float32)[:, None] + 1e-3
    for s in starts:
        x = torch.from_numpy(np.asarray(feat[s:s + win], dtype=np.float32))[None].to(device)
        out = model(x)
        n = x.shape[1]
        w = taper[:n]
        tab[s:s + n] += torch.softmax(out["tab"][0].float(), -1).cpu().numpy() * w[:, :, None]
        onset[s:s + n] += torch.sigmoid(out["onset"][0].float()).cpu().numpy() * w
        weight[s:s + n] += w
    return tab / weight[:, :, None], onset / weight


def extract_notes(tab, onset, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    t_total = len(tab)
    notes = []
    for col in range(6):
        string = STRINGS[col]
        active = 1.0 - tab[:, col, 0]
        frets = tab[:, col, 1:].argmax(-1)
        o = onset[:, col]
        peaks = [t for t in range(t_total) if o[t] >= p["onset"] and o[t] >= o[max(t - 1, 0)] and o[t] > o[min(t + 1, t_total - 1)]]
        peak_set = set(peaks)
        t = 0
        while t < t_total:
            start = None
            if t in peak_set:
                start = t
            elif p["orphan"] and active[t] > p["frame"] and (t == 0 or active[t - 1] <= p["frame"]):
                run = active[t:t + p["min_frames"]]
                if len(run) == p["min_frames"] and (run > p["frame"]).all():
                    start = t
            if start is None:
                t += 1
                continue
            head = tab[start:start + p["fret_frames"], col, 1:].mean(0)
            fret = int(head.argmax())
            end, changed = start + 1, 0
            while end < t_total:
                if end in peak_set or active[end] <= p["frame"]:
                    break
                changed = changed + 1 if frets[end] != fret else 0
                if changed >= p["legato"]:
                    end -= changed - 1
                    break
                end += 1
            if end - start >= 2 or start in peak_set:
                score = float(max(o[start], active[start]))
                notes.append({"onset": start / FPS, "offset": end / FPS, "string": string, "fret": fret,
                              "pitch": OPEN_PITCH[string] + fret, "score": score,
                              "fret_conf": float(tab[start:max(end, start + 1), col, fret + 1].mean())})
            # legato change: the next note starts where the fret changed (no onset needed)
            if end < t_total and end not in peak_set and active[end] > p["frame"]:
                peak_set.add(end)
            t = max(end, start + 1)
    notes = sorted(notes, key=lambda n: (n["onset"], n["string"]))
    if p["dedup"]:
        notes = drop_duplicates(notes, p["dedup"] / FPS)
    for n in notes:
        del n["fret_conf"]
    if p["reattack"]:
        notes = merge_reattacks(notes, onset, p["reattack"])
    return notes


def merge_reattacks(notes, onset, thr):
    """Merge a note into the previous note on the same string and fret when it starts where that note ends and its
    onset probability is below thr (the onset head split one ringing note in two)."""
    out, last = [], {}
    for n in sorted(notes, key=lambda n: n["onset"]):
        prev = last.get(n["string"])
        t, col = int(round(n["onset"] * FPS)), STRINGS.index(n["string"])
        if prev is not None and prev["fret"] == n["fret"] and abs(prev["offset"] - n["onset"]) <= 1.5 / FPS and onset[t, col] < thr:
            prev["offset"] = n["offset"]
            continue
        out.append(n)
        last[n["string"]] = n
    return sorted(out, key=lambda n: (n["onset"], n["string"]))


def apply_capo(notes, capo, max_fret=20):
    """Notes (absolute frets) playable with a capo at `capo`: a note below the capo keeps its pitch and moves to the
    string with the lowest playable fret (>= capo) that is not sounding at its onset; dropped if there is none (a pitch
    below the capo'd low E). Frets stay absolute (fret 0 of the capo'd tab = fret `capo`)."""
    if not capo:
        return notes
    out = sorted((dict(n) for n in notes if n["fret"] >= capo), key=lambda n: n["onset"])
    for n in sorted((n for n in notes if n["fret"] < capo), key=lambda n: n["onset"]):
        busy = {m["string"] for m in out if m["onset"] - 0.03 <= n["onset"] < m["offset"]}
        options = [s for s in STRINGS if s not in busy and capo <= n["pitch"] - OPEN_PITCH[s] <= max_fret]
        if options:
            s = min(options, key=lambda s: n["pitch"] - OPEN_PITCH[s])
            out.append({**n, "string": s, "fret": n["pitch"] - OPEN_PITCH[s]})
    return sorted(out, key=lambda n: (n["onset"], n["string"]))


def drop_duplicates(notes, window):
    """Same pitch starting on two strings within `window` s: keep the note with the higher fret posterior."""
    keep = [True] * len(notes)
    for i, a in enumerate(notes):
        for j in range(i + 1, len(notes)):
            b = notes[j]
            if b["onset"] - a["onset"] > window:
                break
            if keep[i] and keep[j] and a["pitch"] == b["pitch"] and a["string"] != b["string"]:
                if a["fret_conf"] >= b["fret_conf"]:
                    keep[j] = False
                else:
                    keep[i] = False
    return [n for n, k in zip(notes, keep) if k]


def transcribe_file(checkpoint, audio_path, device, params=None, music_lm=None):
    """music_lm: None = default note extraction; a dict of decoding settings (guitab.tablm.DEFAULT_DECODING, optionally
    with "path" to the LM checkpoint) = joint decoding with the tab language model (fuller chords, opt-in)."""
    model = load_model(checkpoint, device)
    feat = compute_cqt(load_audio(audio_path))
    tab, onset = predict(model, feat, device)
    if music_lm is None:
        return extract_notes(tab, onset, params)
    from .paths import TABLM_CHECKPOINT
    from .tablm import DEFAULT_DECODING, decode_with_lm, load_tablm

    cfg = {**DEFAULT_DECODING, **music_lm}
    lm = load_tablm(cfg.pop("path", TABLM_CHECKPOINT), device)
    return decode_with_lm(tab, onset, lm, device, **cfg)
