"""GuitarSet (acoustic guitar, hexaphonic per-string annotations): notes, splits, manifests.

Split by player: 00-03 train, 04 validation, 05 test. Pitch-shifted copies of the train players
(+k semitones, scripts/guitarset_shift.py) reuse the same JAMS with fret + k.
"""

import json
from pathlib import Path

from .guitar import OPEN_PITCH, STRINGS
from .paths import GUITARSET

MAX_FRET = 24  # notes above this (or below fret 0) are not playable / not annotated as tab


def jams_path(sample_id):
    return GUITARSET / "annotation" / f"{sample_id}.jams"


def split_for_player(player):
    return {"04": "validation", "05": "test"}.get(player, "train")


def _note_midi(jams):
    """Yield (string, note dict) for every note_midi annotation (one annotation per string)."""
    for ann in jams["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        string = 6 - int(ann["annotation_metadata"]["data_source"])
        for n in ann["data"]:
            yield string, n


def string_notes(path, shift=0):
    """[(onset_s, offset_s, column, fret)] from a GuitarSet JAMS file (column = index in STRINGS)."""
    jams = json.loads(Path(path).read_text())
    notes = []
    for string, n in _note_midi(jams):
        fret = int(round(n["value"])) - OPEN_PITCH[string] + shift
        notes.append((n["time"], n["time"] + n["duration"], STRINGS.index(string), fret))
    return notes


def playable_frets(path):
    """Frets of the notes on frets 0..MAX_FRET (the notes kept as tab)."""
    jams = json.loads(Path(path).read_text())
    frets = []
    for string, n in _note_midi(jams):
        fret = int(round(n["value"])) - OPEN_PITCH[string]
        if 0 <= fret <= MAX_FRET:
            frets.append(fret)
    return frets
