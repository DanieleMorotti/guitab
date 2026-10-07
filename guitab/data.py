"""Training/evaluation data for the tab model: frame labels, track lists, splits and the feature cache.

Labels per frame (11.6 ms) and string (column order STRINGS, low E .. high E):
  tab[t, s]   0 = string silent, 1 + fret for fret 0..19, -100 = ignore (fret > 19 / masked)
  onset[t, s] 1 on the frame closest to a note attack (-1 = ignore)
  unk[t]      1 while a note whose string is unknown sounds (GAPS)
Tracks: GuitarSet (data/guitarset.csv), its pitch-shifted copies (data/guitarset_shift.csv), the
GAPS selection (gaps_selection) and the SynthTab subset (guitab/synthtab.py).
Splits: train / validation / test (GuitarSet players 00-03 / 04 / 05),
gaps_train / gaps_validation / gaps_test, synth_train / synth_validation.
Augmented audio (guitab/augment.py, scripts/build_aug_cache.py) only changes the features: AUG_CACHE holds
<name>__aug<k>.npy (training copies) and <name>__deg.npy (held-out degraded test copies), with the labels
of the original track.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .audio import FPS, compute_cqt, load_audio
from .gaps import GAPS_SPLITS, audio_path, label_path, safe
from .guitar import N_FRETS, STRINGS
from .guitarset import jams_path, string_notes
from .paths import AUG_CACHE, CACHE, DATA, GAPS, resolve
from . import synthtab


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
    synth = [(f"synth_{r['sid']}", str(synthtab.audio_path(r["sid"])), str(synthtab.label_path(r["sid"])), 0, r["split"])
             for r in synthtab.selection() if synthtab.audio_path(r["sid"]).exists()]
    return out + gaps_selection() + synth


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


def aug_path(name, variant):
    """Features of an augmented copy: variant "aug<k>" (training) or "deg" (held-out degraded test copy)."""
    return AUG_CACHE / f"{name}__{variant}.npy"


def load_split(split, include_shift=True, aug_copies=0, variant=None):
    """aug_copies: attach up to that many augmented feature copies per track as tr["aug"] (memory-mapped, existing ones);
    variant "deg": replace the features with the degraded copy (robustness evaluation)."""
    items = [t for t in tracks() if t[4] == split and (include_shift or t[3] == 0)]
    data = []
    for name, audio, labels, shift, _ in items:
        z = np.load(CACHE / f"{name}.npz")
        onset = z["onset"].astype(np.int8)
        unk = z["unk"] if "unk" in z.files else np.zeros(len(z["feat"]), dtype=np.uint8)
        tr = {"name": name, "audio": audio, "labels": labels, "shift": shift, "feat": z["feat"], "tab": z["tab"],
              "onset": onset, "unk": unk, "source": name.split("_")[0] if name.startswith(("gaps_", "synth_")) else "guitarset"}
        if variant:
            tr["feat"] = np.load(aug_path(name, variant))
        if aug_copies:
            tr["aug"] = [np.load(aug_path(name, f"aug{k}"), mmap_mode="r") for k in range(aug_copies)
                         if aug_path(name, f"aug{k}").exists()]
        data.append(tr)
    return data
