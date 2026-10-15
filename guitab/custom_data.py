"""custom fingerstyle dataset: real recordings (the tab's synced YouTube video) with the tab as labels.

Personal-use research data (scripts/custom_fetch.py); nothing here is redistributed. Files:
  data/custom/selection.csv        chosen songs (songId, revision, videoId, artist, title, split, ...) - versioned
  data/custom/tabs/<songId>.json   custom-dataset track JSON + the video's measure start times
  data/custom/audio/<songId>.flac  the video's audio (22.05 kHz mono)
  data/custom/labels/<songId>.json {"notes": [[onset_s, offset_s, string, fret], ...]} (string 1 = high E)
Timing (scripts/custom_fetch.py label):
  1. coarse: the dataset's video sync gives the start time of every measure; notes are placed linearly inside a measure;
  2. refined: DTW (band +-0.5 s) between the coarse score pitch roll and the mean pitch roll of four models on the audio
     (Sungha Jung's "Remember Me": E15 pitch F1 vs labels at +-100 ms .45 coarse -> .68 refined);
  3. snapped: each onset moves to the nearest audio onset (librosa spectral flux) within 50 ms.
Pitches and strings always come from the tab. Frets include the capo. Dead notes and harmonics are skipped; tied notes
extend the note they continue.
"""

import csv
from fractions import Fraction

import numpy as np

from .paths import DATA

CUSTOM_DATA = DATA / "custom"
SELECTION = CUSTOM_DATA / "selection.csv"
STANDARD = [64, 59, 55, 50, 45, 40]  # custom-dataset string order: high E .. low E
SNAP = 0.05


def tab_path(song_id):
    return CUSTOM_DATA / "tabs" / f"{song_id}.json"


def audio_path(song_id):
    return CUSTOM_DATA / "audio" / f"{song_id}.flac"


def label_path(song_id):
    return CUSTOM_DATA / "labels" / f"{song_id}.json"


def selection():
    if not SELECTION.exists():
        return []
    return list(csv.DictReader(open(SELECTION)))


def score_notes(track):
    """Notes in score time: [{measure, pos (quarters from measure start), dur (quarters), string 1..6, fret}] and
    the measure lengths in quarters. Ties extend the previous note on the string."""
    notes, lengths, last, sig = [], [], {}, [4, 4]
    for mi, m in enumerate(track["measures"]):
        sig = m.get("signature", sig)  # a signature holds until the next change
        lengths.append(Fraction(4 * sig[0], sig[1]))
        for v in m["voices"]:
            pos = Fraction(0)
            for b in v["beats"]:
                dur = Fraction(b["duration"][0], b["duration"][1]) * 4
                for n in b["notes"]:
                    if n.get("rest") or n.get("dead") or n.get("harmonic"):
                        continue
                    s = n["string"] + 1
                    if n.get("tie"):
                        if s in last:
                            last[s]["end"] = (mi, pos + dur)
                        continue
                    note = {"measure": mi, "pos": pos, "end": (mi, pos + dur), "string": s,
                            "fret": n["fret"] + (track.get("capo") or 0)}
                    notes.append(note)
                    last[s] = note
                pos += dur
    return notes, lengths


def measure_times(points, lengths):
    """Start and end time (s) of every measure from the video's measure start times."""
    starts = list(points[: len(lengths)])
    ends = starts[1:] + [starts[-1] + float(lengths[-1]) * (starts[-1] - starts[-2]) / float(lengths[-2])]
    return starts, ends


def to_seconds(measure, pos, lengths, starts, ends):
    measure = min(measure, len(lengths) - 1)
    return starts[measure] + float(pos / lengths[measure]) * (ends[measure] - starts[measure])


def coarse_notes(tab):
    """[(onset, offset, string, fret)] on the sync-point timeline (step 1)."""
    notes, lengths = score_notes(tab["track"])
    starts, ends = measure_times(tab["points"], lengths)
    out = [(to_seconds(n["measure"], n["pos"], lengths, starts, ends), to_seconds(*n["end"], lengths, starts, ends),
            n["string"], n["fret"]) for n in notes]
    return sorted((on, max(off, on + 0.03), s, f) for on, off, s, f in out)


def snap(t, onsets):
    near = onsets[np.abs(onsets - t) <= SNAP]
    return float(near[np.argmin(np.abs(near - t))]) if len(near) else t
