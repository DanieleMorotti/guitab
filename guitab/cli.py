"""Transcribe a guitar recording of any length to a Guitar Pro (.gp5) tab.

The CQT frame/onset tab model (default checkpoint runs/tabmodel_mid_all/best.pt, E19) predicts notes
(seconds, string, fret); they are snapped to a grid that follows the beats detected by Beat This!
(or a fixed --tempo), optionally re-fingered, and written as tab tokens and .gp5.

Usage:
    guitab --audio my_song.wav --out_dir out/my_song [--onset 0.4] [--fingering low]
    guitab --audio my_song.wav --out_dir out/my_song --music_lm   # fuller chords (tab language model)
    guitab --audio my_song.wav --out_dir out/my_song --capo 3     # played with a capo at fret 3
"""

import argparse
import json
from pathlib import Path

import torch

from .fingering import FINGERING_PRESETS
from .gp5 import write_gp5
from .inference import apply_capo, transcribe_file
from .paths import DEFAULT_CHECKPOINT
from .tab import BeatGrid, detect_beats, notes_to_tokens


def resolve_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="tab model checkpoint (.pt).")
    parser.add_argument("--tempo", default="auto",
                        help="'auto' = follow the beats detected by Beat This!; or a fixed tempo in BPM.")
    parser.add_argument("--fingering", choices=["model", "auto", "low"], default="model",
                        help="'model' keeps the tab model's strings; 'auto' re-optimises cautiously; 'low' prefers first position.")
    parser.add_argument("--onset", type=float, default=None,
                        help="onset confidence threshold (default 0.6, tuned on GuitarSet validation; lower = more notes).")
    parser.add_argument("--orphan", action="store_true",
                        help="also keep notes whose string sounds without a detected attack (more notes).")
    parser.add_argument("--params", default=None,
                        help='JSON note-extraction settings, e.g. the tuned ones of a checkpoint (see EXPERIMENTS.md); '
                             '--onset / --orphan are applied on top.')
    parser.add_argument("--music_lm", action="store_true",
                        help="decode jointly with the tab language model (runs/tablm/tablm.pt): fuller chords, ~10-20 s slower; "
                             "ignores --onset / --orphan / --params (EXPERIMENTS.md, A1).")
    parser.add_argument("--music_lm_weight", type=float, default=0.3, help="weight of the tab LM (0.3 tuned; >= 0.6 overrides the audio).")
    parser.add_argument("--music_lm_bonus", type=float, default=1.0, help="per-note bonus with --music_lm (higher = fuller chords, more false notes).")
    parser.add_argument("--capo", type=int, default=0,
                        help="capo fret: no notes below it, tab frets written relative to it, capo set in the .gp5 (D6).")
    parser.add_argument("--grid", type=int, default=240, help="Quantization grid in ticks (960 = quarter; 240 = 16th).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default=None, help="Output file stem (default: audio file stem).")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    device = resolve_device(args.device)
    params = json.loads(args.params) if args.params else {}
    if args.onset is not None:
        params["onset"] = args.onset
    if args.orphan:
        params.update(orphan=True, frame=0.5)
    music_lm = {"lam": args.music_lm_weight, "beta": args.music_lm_bonus} if args.music_lm else None
    notes = apply_capo(transcribe_file(args.checkpoint, audio_path, device, params, music_lm), args.capo)
    print(f"{len(notes)} notes")
    if args.tempo != "auto":
        grid = BeatGrid([], tempo=int(args.tempo))
    else:
        beats, downbeats = detect_beats(audio_path, device)
        grid = BeatGrid(beats, downbeats)
    print(f"tempo {grid.tempo} bpm, {grid.num_beats}/4")

    weights = None if args.fingering == "model" else FINGERING_PRESETS[args.fingering]
    stem = args.name or audio_path.stem
    tokens = notes_to_tokens(notes, grid, stem, args.grid, weights, capo=args.capo)
    if tokens is None:
        raise SystemExit("No notes were transcribed.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    gp5_path = out_dir / f"{stem}.gp5"
    write_gp5(tokens, gp5_path, args.capo)
    print(f"-> {gp5_path}")


if __name__ == "__main__":
    main()
