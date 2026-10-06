"""Inference for the CQT tab model: sliding-window prediction and note extraction.

Notes are extracted per string:
  - a note starts at an onset peak (local max of the onset probability above `onset`), or where
    the string becomes active (P(not silent) > `frame`) for >= `min_frames` frames without one
    and `orphan` is enabled;
  - its fret is the argmax of the fret posterior averaged over its first frames;
  - it ends when the string goes silent, at the next onset on that string, or when the frame
    fret changes for `legato` consecutive frames (hammer-on / pull-off / slide -> new note);
  - the same pitch started on two strings within `dedup` frames is one note played once: only the
    note whose fret posterior (mean over its frames) is higher is kept (0 = off).
Models with a release head (E11, onset_target "regression"):
  - `subframe`: the attack time is refined by a parabola through the onset peak and its neighbours;
  - a note also ends at a release peak (local max of the offset probability >= `offset`).
"""

import numpy as np
import torch

from .audio import FPS, compute_cqt, load_audio
from .guitar import OPEN_PITCH, STRINGS
from .model import TabModel

# tuned on GuitarSet validation (player 04) with scripts/evaluate.py --tune
DEFAULT_PARAMS = {"onset": 0.6, "frame": 0.6, "min_frames": 3, "legato": 8, "orphan": False, "fret_frames": 3, "dedup": 4,
                  "offset": 0.5, "subframe": False}


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TabModel(ckpt["size"], offset_head=ckpt.get("arch", {}).get("offset_head", False))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, feat, device, win=432, hop=288):
    """feat [T, 288] -> (tab_probs [T, 6, 21], onset_probs [T, 6], offset_probs [T, 6] or None) with overlapping windows."""
    t_total = len(feat)
    tab = np.zeros((t_total, 6, model.n_classes), dtype=np.float32)
    onset = np.zeros((t_total, 6), dtype=np.float32)
    offset = np.zeros((t_total, 6), dtype=np.float32) if model.offset_head is not None else None
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
        if offset is not None:
            offset[s:s + n] += torch.sigmoid(out["offset"][0].float()).cpu().numpy() * w
        weight[s:s + n] += w
    return tab / weight[:, :, None], onset / weight, None if offset is None else offset / weight


def peaks_of(x, threshold):
    """Frames where x is a local maximum (plateaus count once, at their start) and >= threshold."""
    n = len(x)
    return [t for t in range(n) if x[t] >= threshold and x[t] >= x[max(t - 1, 0)] and x[t] > x[min(t + 1, n - 1)]]


def extract_notes(tab, onset, offset=None, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    t_total = len(tab)
    notes = []
    for col in range(6):
        string = STRINGS[col]
        active = 1.0 - tab[:, col, 0]
        frets = tab[:, col, 1:].argmax(-1)
        o = onset[:, col]
        peak_set = set(peaks_of(o, p["onset"]))
        releases = set(peaks_of(offset[:, col], p["offset"])) if offset is not None else set()
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
            end, changed, released = start + 1, 0, False
            while end < t_total:
                if end in peak_set or active[end] <= p["frame"]:
                    break
                if end in releases:
                    released = True
                    break
                changed = changed + 1 if frets[end] != fret else 0
                if changed >= p["legato"]:
                    end -= changed - 1
                    break
                end += 1
            if end - start >= 2 or start in peak_set:
                score = float(max(o[start], active[start]))
                t_on = start
                if p["subframe"] and start in peak_set and 0 < start < t_total - 1:
                    a, b, c = o[start - 1], o[start], o[start + 1]
                    curv = a - 2 * b + c
                    t_on = start + (float(np.clip(0.5 * (a - c) / curv, -0.5, 0.5)) if curv < 0 else 0.0)
                notes.append({"onset": t_on / FPS, "offset": end / FPS, "string": string, "fret": fret,
                              "pitch": OPEN_PITCH[string] + fret, "score": score,
                              "fret_conf": float(tab[start:max(end, start + 1), col, fret + 1].mean())})
            # legato change: the next note starts where the fret changed (no onset needed); not after a release
            if end < t_total and not released and end not in peak_set and active[end] > p["frame"]:
                peak_set.add(end)
            t = max(end, start + 1)
    notes = sorted(notes, key=lambda n: (n["onset"], n["string"]))
    if p["dedup"]:
        notes = drop_duplicates(notes, p["dedup"] / FPS)
    for n in notes:
        del n["fret_conf"]
    return notes


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


def transcribe_file(checkpoint, audio_path, device, params=None):
    model = load_model(checkpoint, device)
    feat = compute_cqt(load_audio(audio_path))
    tab, onset, offset = predict(model, feat, device)
    return extract_notes(tab, onset, offset, params)
