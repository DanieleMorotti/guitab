"""Transcribe a guitar recording of any length to a Guitar Pro (.gp5) tab.

Engines:
  tabmodel     our CQT frame/onset tab model (default: runs/tabmodel_mid_gaps/best.pt)
  basic_pitch  Spotify Basic Pitch notes (run in .venv_bp) + the fingering optimiser
Notes (seconds, string, fret) are snapped to a grid that follows the beats detected by Beat This!
(or a fixed --tempo), optionally re-fingered, and written as DadaGP tokens and .gp5.

Usage:
    python tools/transcribe.py --audio my_song.wav --out_dir out/my_song [--onset 0.4] [--fingering low]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "runs" / "tabmodel_mid_gaps" / "best.pt"
FINGERING_PRESETS = {
    "auto": {"position": 0.05, "move": 0.35, "model": 2.0},           # cautious: keeps model choices unless clearly awkward
    "low": {"position": 0.6, "move": 0.35, "model": 0.3, "high": 2.0},  # strongly prefer open/first positions
}


def resolve_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def run_basic_pitch(audio_path, amp_min):
    """Basic Pitch notes from the separate .venv_bp (cached in data/basic_pitch/)."""
    subprocess.run([str(ROOT / ".venv_bp" / "bin" / "python"), str(ROOT / "tools" / "run_basic_pitch.py"), "--audio", str(audio_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cache = ROOT / "data" / "basic_pitch" / f"{audio_path.parent.name}__{audio_path.stem}.json"
    return [{"onset": on, "offset": off, "pitch": p, "score": a} for on, off, p, a in json.loads(cache.read_text()) if a >= amp_min]


def decode_to_gp5(tokens, gp5_path):
    from dadagp_utils import DadaGP

    restored = [f"clean0:{t}" if t.startswith("note:") or t == "rest" else t for t in tokens]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write("\n".join(restored) + "\n")
        name = handle.name
    DadaGP.decode(name, str(gp5_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--engine", choices=["tabmodel", "basic_pitch"], default="tabmodel",
                        help="'tabmodel' = our CQT tab model; 'basic_pitch' = Basic Pitch notes + fingering optimiser.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="tabmodel checkpoint (.pt).")
    parser.add_argument("--tempo", default="auto",
                        help="'auto' = follow the beats detected by Beat This!; or a fixed tempo in BPM.")
    parser.add_argument("--fingering", choices=["model", "auto", "low"], default="model",
                        help="'model' keeps the tab model's strings; 'auto' re-optimises cautiously; 'low' prefers first position.")
    parser.add_argument("--onset", type=float, default=None,
                        help="tabmodel: onset confidence threshold (default 0.6, tuned on GuitarSet validation; lower = more notes).")
    parser.add_argument("--orphan", action="store_true",
                        help="tabmodel: also keep notes whose string sounds without a detected attack (more notes).")
    parser.add_argument("--grid", type=int, default=240, help="Quantization grid in ticks (960 = quarter; 240 = 16th).")
    parser.add_argument("--bp_amp", type=float, default=0.4, help="basic_pitch: minimum note amplitude.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default=None, help="Output file stem (default: audio file stem).")
    args = parser.parse_args()

    from notes_to_tab import BeatGrid, detect_beats, notes_to_tokens

    audio_path = Path(args.audio).resolve()
    device = resolve_device(args.device)
    if args.engine == "basic_pitch":
        notes = run_basic_pitch(audio_path, args.bp_amp)
    else:
        from tabmodel_infer import transcribe_file

        params = {}
        if args.onset is not None:
            params["onset"] = args.onset
        if args.orphan:
            params.update(orphan=True, frame=0.5)
        notes = transcribe_file(args.checkpoint, audio_path, device, params)
    print(f"{len(notes)} notes from {args.engine}")
    if args.tempo != "auto":
        grid = BeatGrid([], tempo=int(args.tempo))
    else:
        beats, downbeats = detect_beats(audio_path, device)
        grid = BeatGrid(beats, downbeats)
    print(f"tempo {grid.tempo} bpm, {grid.num_beats}/4")

    weights = None if args.fingering == "model" else FINGERING_PRESETS[args.fingering]
    if args.engine == "basic_pitch" and weights is None:
        weights = FINGERING_PRESETS["low"]  # Basic Pitch has no string information
    stem = args.name or audio_path.stem
    tokens = notes_to_tokens(notes, grid, stem, args.grid, weights)
    if tokens is None:
        raise SystemExit("No notes were transcribed.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    gp5_path = out_dir / f"{stem}.gp5"
    decode_to_gp5(tokens, gp5_path)
    print(f"-> {gp5_path}")


if __name__ == "__main__":
    main()
