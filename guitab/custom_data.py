"""Custom fingerstyle dataset (not distributed): real solo-guitar recordings with their tab as labels, aligned to the audio.

Used by E17-E19 (EXPERIMENTS.md); the tools that built it are not published. Training reads, if present:
  data/custom/selection.csv          songs (songId, split = ss_train / ss_validation, ...)
  data/custom/labels_summary.csv     alignment check per song (status ok / dropped)
  data/custom/audio/<songId>.flac    22.05 kHz mono audio
  data/custom/labels/<songId>.json   {"notes": [[onset_s, offset_s, string, fret], ...]} (string 1 = high E, frets incl. capo)
"""

import csv

from .paths import DATA

CUSTOM_DATA = DATA / "custom"
SELECTION = CUSTOM_DATA / "selection.csv"


def audio_path(song_id):
    return CUSTOM_DATA / "audio" / f"{song_id}.flac"


def label_path(song_id):
    return CUSTOM_DATA / "labels" / f"{song_id}.json"


def selection():
    if not SELECTION.exists():
        return []
    return list(csv.DictReader(open(SELECTION)))
