"""Run Spotify Basic Pitch (polyphonic note detector, ONNX build) on audio files and cache note events.

Usage (in the separate .venv_bp environment):
    .venv_bp/bin/python tools/run_basic_pitch.py --manifest data/guitarset.csv --splits validation test
    .venv_bp/bin/python tools/run_basic_pitch.py --audio out/my_video/my_video.wav
Output: data/basic_pitch/<audio stem>.json with [onset_s, offset_s, midi_pitch, amplitude] per note.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "basic_pitch"


def cache_path(audio_path):
    p = Path(audio_path)
    return OUT / f"{p.parent.name}__{p.stem}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--splits", nargs="*", default=["validation", "test"])
    parser.add_argument("--audio", nargs="*", default=[])
    args = parser.parse_args()

    import basic_pitch
    from basic_pitch.inference import Model, predict

    model = Model(str(Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"))
    paths = list(args.audio)
    if args.manifest:
        paths += sorted({r["audio_path"] for r in csv.DictReader(open(args.manifest)) if r["split"] in args.splits})
    OUT.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        out = cache_path(path)
        if out.exists():
            continue
        # Guitar range (E2..~E6); low thresholds keep weak chord tones, fusion decides what to trust.
        _, _, events = predict(path, model, onset_threshold=0.4, frame_threshold=0.25,
                               minimum_note_length=50, minimum_frequency=75, maximum_frequency=1400)
        notes = sorted([round(float(s), 4), round(float(e), 4), int(p), round(float(a), 4)] for s, e, p, a, _ in events)
        out.write_text(json.dumps(notes))
        print(f"{i + 1}/{len(paths)} {Path(path).name}: {len(notes)} notes", flush=True)


if __name__ == "__main__":
    main()
