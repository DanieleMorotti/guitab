"""SynthTab (Zang et al., ICASSP 2024; CC BY-NC 4.0): DadaGP tabs rendered with sampled guitars.

We use a small, selected subset (scripts/synthtab_fetch.py): fingerpicked acoustic timbres, standard tuning,
polyphonic fingerstyle-like songs. Files:
  data/synthtab/selection.csv          chosen songs (sid, song, timbre, split, Box file id, zip member, ...) - versioned
  data/synthtab/audio/<sid>.flac       rendered audio (44.1 kHz mono, "nonoise_mono_body" microphone)
  data/synthtab/labels/<sid>.json      {"notes": [[onset_s, offset_s, string, fret], ...]} (string 1 = high E)
Labels come from the per-string MIDI the audio was rendered from (fret = MIDI pitch - open-string pitch),
shifted by the timbre's rendering latency (LATENCY_MS): the sampled instruments sound later than the MIDI.
"""

import csv
import io
import json

from .guitar import OPEN_PITCH
from .paths import DATA

SYNTHTAB = DATA / "synthtab"
SELECTION = SYNTHTAB / "selection.csv"
LABEL_ZIP = SYNTHTAB / "all_jams_midi.zip"
STANDARD = [OPEN_PITCH[s] for s in range(1, 7)]  # open pitch of strings 1..6
# Rendering latency per timbre: median over 12 songs each of (E12-predicted attack - MIDI attack), same pitch,
# within 150 ms. E12's own bias on real audio is ~0 (GuitarSet validation -0.7 ms, GAPS validation +0.6 ms).
LATENCY_MS = {"taylor_finger": 57.5, "martin_finger": 24.1, "gibson_thumb": 21.4, "luthier_finger": 16.7}


def audio_path(sid):
    return SYNTHTAB / "audio" / f"{sid}.flac"


def label_path(sid):
    return SYNTHTAB / "labels" / f"{sid}.json"


def selection():
    if not SELECTION.exists():
        return []
    return list(csv.DictReader(open(SELECTION)))


def midi_notes(zf, song):
    """[(onset_s, offset_s, string, fret)] from the per-string MIDI files of `song` in the label zip."""
    import pretty_midi

    folder = f"outall/{song}/"
    notes = []
    for string in range(1, 7):
        name = f"{folder}string_{string}.mid"
        if name not in zf.NameToInfo:  # no file for a string that is never played
            continue
        midi = pretty_midi.PrettyMIDI(io.BytesIO(zf.read(name)))
        for inst in midi.instruments:
            for n in inst.notes:
                fret = n.pitch - OPEN_PITCH[string]
                if 0 <= fret <= 24:  # skips the pitch-24 marker note each file starts with
                    notes.append((round(n.start, 4), round(n.end, 4), string, fret))
    return sorted(notes)


def write_labels(zf, row):
    lat = LATENCY_MS[row["timbre"]] / 1000
    notes = [(round(on + lat, 4), round(off + lat, 4), s, f) for on, off, s, f in midi_notes(zf, row["song"])]
    label_path(row["sid"]).write_text(json.dumps({"song": row["song"], "timbre": row["timbre"], "notes": notes}))
    return len(notes)
