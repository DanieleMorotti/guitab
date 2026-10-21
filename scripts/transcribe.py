"""Transcribe a guitar recording of any length to a Guitar Pro (.gp5) tab.

The CQT frame/onset tab model (default checkpoint: E25 if trained, else E15; guitab/paths.py) predicts notes
(seconds, string, fret); they are snapped to a grid that follows the beats detected by Beat This!
(or a fixed --tempo), optionally re-fingered, and written as tab tokens and .gp5.

Usage:
    python scripts/transcribe.py --audio my_song.wav --out_dir out/my_song [--onset 0.4] [--fingering low]
    python scripts/transcribe.py --audio my_song.wav --out_dir out/my_song --music_lm   # fuller chords (tab language model)
    python scripts/transcribe.py --audio my_song.wav --out_dir out/my_song --capo 3     # played with a capo at fret 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
