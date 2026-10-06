"""Training/evaluation data for the tab model: frame labels, track lists, splits and the feature cache.

Labels per frame (11.6 ms) and string (column order STRINGS, low E .. high E):
  tab[t, s]   0 = string silent, 1 + fret for fret 0..19, -100 = ignore (fret > 19 / masked)
  onset[t, s] 1 on the frame closest to a note attack (-1 = ignore)
  unk[t]      1 while a note whose string is unknown sounds (GAPS)
With load_split(..., targets="regression") (E11), onset is replaced by a soft attack target and an
offset (release) target is added, both from the exact note times (regression_targets).
Tracks: GuitarSet (data/guitarset.csv), its pitch-shifted copies (data/guitarset_shift.csv) and the
GAPS selection (gaps_selection). Splits: train / validation / test (GuitarSet players 00-03 / 04 / 05),
gaps_train / gaps_validation / gaps_test.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .audio import FPS, compute_cqt, load_audio
from .gaps import GAPS_SPLITS, audio_path, label_path, safe
from .guitar import N_FRETS, STRINGS
from .guitarset import jams_path, string_notes
from .paths import CACHE, DATA, GAPS, resolve


def gaps_notes(label_path):
    """[(onset_s, offset_s, column or None, fret)] from scripts/gaps_labels.py output; None = unknown string."""
    out = []
    for on, off, string, fret in json.loads(Path(label_path).read_text())["notes"]:
        out.append((on, off, STRINGS.index(string) if string else None, fret))
    return out


def frame_labels(notes, n_frames):
    """tab [T, 6] (-100 ignore), onset [T, 6] (-1 ignore), unk [T] (1 = a note of unknown string sounds).

    Notes with column None (string unknown, GAPS) are masked: strings not otherwise labelled active
    are ignored by the tab loss during the note, and onsets are ignored around its attack."""
    tab = np.zeros((n_frames, 6), dtype=np.int8)
    onset = np.zeros((n_frames, 6), dtype=np.int8)
    unk = np.zeros(n_frames, dtype=np.uint8)
    unknown = []
    for on, off, col, fret in sorted(notes, key=lambda n: n[0]):
        a = int(round(on * FPS))
        b = max(a + 1, int(round(off * FPS)))
        if a >= n_frames:
            continue
        if col is None:
            unknown.append((a, b))
            continue
        tab[a:b, col] = fret + 1 if 0 <= fret < N_FRETS else -100
        onset[a, col] = 1
    for a, b in unknown:
        unk[a:b] = 1
        seg = tab[a:b]
        seg[seg == 0] = -100
        win = onset[max(a - 1, 0):a + 2]
        win[win == 0] = -1
    return tab, onset, unk


def regression_targets(notes, n_frames, width=4, offsets=True):
    """Kong et al.-style targets: onset/offset [T, 6] float32 = max(0, 1 - |t - event| / width) around each exact
    attack / release time (in frames), -1 = ignore.

    Notes with column None (string unknown, GAPS) set strings not otherwise targeted to ignore around their events.
    offsets=False ignores every release (GAPS: MIDI durations come from score note values, not from the string)."""
    onset = np.zeros((n_frames, 6), dtype=np.float32)
    offset = np.zeros((n_frames, 6), dtype=np.float32) if offsets else np.full((n_frames, 6), -1.0, dtype=np.float32)
    unknown = []

    def bump(target, col, centre):
        lo, hi = max(int(np.ceil(centre - width)), 0), min(int(np.floor(centre + width)), n_frames - 1)
        for t in range(lo, hi + 1):
            target[t, col] = max(target[t, col], 1.0 - abs(t - centre) / width)

    for on, off, col, fret in notes:
        if on * FPS >= n_frames:
            continue
        if col is None:
            unknown.append((on, off))
            continue
        bump(onset, col, on * FPS)
        if offsets:
            bump(offset, col, off * FPS)
    w = int(np.ceil(width))
    for on, off in unknown:
        for target, centre, used in ((onset, on * FPS, True), (offset, off * FPS, offsets)):
            if not used:
                continue
            a, b = max(int(round(centre)) - w, 0), min(int(round(centre)) + w + 1, n_frames)
            seg = target[a:b]
            seg[seg == 0] = -1.0
    return onset, offset


def gaps_selection(min_match=0.5, min_content=0.9):
    """GAPS pieces usable for training/eval: standard tuning, MIDI matches its score, >= min_match of
    notes with a string label, audio downloaded. GAPS test stays test; every 10th train piece
    (sorted by hash) is held out as gaps_validation."""
    summary = GAPS / "labels_summary.csv"
    if not summary.exists():
        return []
    split = {r["filename"][:-4]: r["split"] for r in csv.DictReader(open(GAPS_SPLITS))}
    rows = [r for r in csv.DictReader(open(summary)) if r["status"] == "ok" and r["standard_tuning"] == "True"
            and float(r["match_rate"]) >= min_match and float(r["content_sim"]) >= min_content
            and audio_path(r["hash"]).exists()]
    out = []
    train_hashes = sorted(r["hash"] for r in rows if split.get(r["hash"]) == "train")
    for r in rows:
        h = r["hash"]
        if split.get(h) == "test":
            sp = "gaps_test"
        elif split.get(h) == "train":
            sp = "gaps_validation" if train_hashes.index(h) % 10 == 0 else "gaps_train"
        else:
            continue
        out.append((f"gaps_{safe(h)}", str(audio_path(h)), str(label_path(h)), 0, sp))
    return out


def tracks():
    """(name, audio_path, label_path, shift, split): GuitarSet, its pitch-shifted copies, GAPS selection."""
    out = []
    for row in csv.DictReader(open(DATA / "guitarset.csv")):
        out.append((row["sample_id"], str(resolve(row["audio_path"])), str(jams_path(row["sample_id"])), 0, row["split"]))
    for row in csv.DictReader(open(DATA / "guitarset_shift.csv")):
        base, k = row["sample_id"].rsplit("_up", 1)
        out.append((row["sample_id"], str(resolve(row["audio_path"])), str(jams_path(base)), int(k), row["split"]))
    return out + gaps_selection()


def compute_track(item):
    """(feat, tab, onset, unk) for one track of tracks()."""
    name, audio, labels, shift, split = item
    feat = compute_cqt(load_audio(audio))
    notes = gaps_notes(labels) if labels.endswith(".json") else string_notes(labels, shift)
    tab, onset, unk = frame_labels(notes, len(feat))
    return feat, tab, onset, unk


def cache_one(item):
    name = item[0]
    path = CACHE / f"{name}.npz"
    if path.exists():
        return name
    feat, tab, onset, unk = compute_track(item)
    np.savez(path, feat=feat, tab=tab, onset=onset, unk=unk)
    return name


def load_split(split, include_shift=True, targets="frame"):
    """targets: "frame" (cached onset frames) or "regression" (regression_targets onset + offset, built from the labels)."""
    items = [t for t in tracks() if t[4] == split and (include_shift or t[3] == 0)]
    data = []
    for name, audio, labels, shift, _ in items:
        z = np.load(CACHE / f"{name}.npz")
        onset = z["onset"].astype(np.int8)
        unk = z["unk"] if "unk" in z.files else np.zeros(len(z["feat"]), dtype=np.uint8)
        tr = {"name": name, "audio": audio, "labels": labels, "shift": shift, "feat": z["feat"], "tab": z["tab"],
              "onset": onset, "unk": unk, "source": "gaps" if name.startswith("gaps_") else "guitarset"}
        if targets == "regression":
            gaps = labels.endswith(".json")
            notes = gaps_notes(labels) if gaps else string_notes(labels, shift)
            tr["onset"], tr["offset"] = regression_targets(notes, len(z["feat"]), offsets=not gaps)
        data.append(tr)
    return data
