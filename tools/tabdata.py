"""Features and frame-level labels for the CQT tab model.

Features: log-magnitude CQT, 36 bins/octave from E1 (288 bins, up to ~10.5 kHz), 22.05 kHz,
hop 256 (11.6 ms). The model builds the harmonic stack itself, so one CQT per track is cached.

Labels (from GuitarSet JAMS, one note_midi annotation per string):
  tab[t, s]   0 = string silent, 1 + fret for fret 0..19, -100 = ignore (fret > 19)
  onset[t, s] 1 on the frame closest to a note attack
Pitch-shifted copies (data/guitarset_shift, +1/+2 semitones) reuse the JAMS with fret + k.
GAPS (classical/fingerstyle, data/gaps, labels from tools/gaps_labels.py) adds splits gaps_train /
gaps_validation / gaps_test; notes whose string is unknown are masked (see frame_labels).

Usage:
    python tools/tabdata.py --workers 4      # cache everything to data/tabmodel_cache/
"""

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "tabmodel_cache"
SR = 22050
HOP = 256
FPS = SR / HOP
N_BINS = 288
BINS_PER_OCTAVE = 36
FMIN_NOTE = "E1"
N_FRETS = 20  # frets 0..19 -> classes 1..20
N_CLASSES = N_FRETS + 1
OPEN_PITCH = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}  # string number -> MIDI
STRINGS = [6, 5, 4, 3, 2, 1]  # label column order: low E .. high E


def compute_cqt(y):
    import librosa

    c = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz(FMIN_NOTE), n_bins=N_BINS,
                           bins_per_octave=BINS_PER_OCTAVE))
    db = librosa.amplitude_to_db(c, ref=np.max(c) + 1e-9, top_db=80.0)
    return ((db + 80.0) / 80.0).T.astype(np.float16)  # [T, F] in [0, 1]


def load_audio(path):
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def string_notes(jams_path, shift=0):
    """[(onset_s, offset_s, column, fret)] from a GuitarSet JAMS file."""
    jams = json.loads(Path(jams_path).read_text())
    notes = []
    for ann in jams["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        string = 6 - int(ann["annotation_metadata"]["data_source"])
        col = STRINGS.index(string)
        for n in ann["data"]:
            fret = int(round(n["value"])) - OPEN_PITCH[string] + shift
            notes.append((n["time"], n["time"] + n["duration"], col, fret))
    return notes


def gaps_notes(label_path):
    """[(onset_s, offset_s, column or None, fret)] from tools/gaps_labels.py output; None = unknown string."""
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
    from gaps_common import audio_path, label_path, safe

    summary = ROOT / "data" / "gaps" / "labels_summary.csv"
    if not summary.exists():
        return []
    split = {r["filename"][:-4]: r["split"] for r in csv.DictReader(open(ROOT / "data" / "gaps" / "gaps_v1" / "gaps_v1_splits.csv"))}
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
    for row in csv.DictReader(open(ROOT / "data" / "guitarset.csv")):
        jams = ROOT / "data" / "guitarset" / "annotation" / f"{row['sample_id']}.jams"
        out.append((row["sample_id"], row["audio_path"], str(jams), 0, row["split"]))
    for row in csv.DictReader(open(ROOT / "data" / "guitarset_shift.csv")):
        base, k = row["sample_id"].rsplit("_up", 1)
        jams = ROOT / "data" / "guitarset" / "annotation" / f"{base}.jams"
        out.append((row["sample_id"], row["audio_path"], str(jams), int(k), row["split"]))
    return out + gaps_selection()


def cache_one(item):
    name, audio, labels, shift, split = item
    path = CACHE / f"{name}.npz"
    if path.exists():
        return name
    feat = compute_cqt(load_audio(audio))
    notes = gaps_notes(labels) if labels.endswith(".json") else string_notes(labels, shift)
    tab, onset, unk = frame_labels(notes, len(feat))
    np.savez(path, feat=feat, tab=tab, onset=onset, unk=unk)
    return name


def load_split(split, include_shift=True):
    items = [t for t in tracks() if t[4] == split and (include_shift or t[3] == 0)]
    data = []
    for name, audio, labels, shift, _ in items:
        z = np.load(CACHE / f"{name}.npz")
        onset = z["onset"].astype(np.int8)
        unk = z["unk"] if "unk" in z.files else np.zeros(len(z["feat"]), dtype=np.uint8)
        data.append({"name": name, "audio": audio, "labels": labels, "shift": shift, "feat": z["feat"], "tab": z["tab"],
                     "onset": onset, "unk": unk, "source": "gaps" if name.startswith("gaps_") else "guitarset"})
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    items = tracks()
    with Pool(args.workers) as pool:
        for i, name in enumerate(pool.imap_unordered(cache_one, items), 1):
            if i % 20 == 0 or i == len(items):
                print(f"{i}/{len(items)} {name}", flush=True)


if __name__ == "__main__":
    main()
