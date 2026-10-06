"""Make pitch-shifted copies (+1/+2 semitones) of the GuitarSet training recordings.

Shifting up by k semitones keeps every note playable on the same string (fret + k), whereas
shifting down would push open strings below fret 0; copies whose highest fret would exceed 24 are
skipped. Audio is shifted offline with librosa (16 kHz mono, as used for all existing copies) and
listed in data/guitarset_shift.csv; labels are taken from the original JAMS with fret + k.

    python scripts/guitarset_shift.py [--workers 4]
"""

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.guitarset import MAX_FRET, jams_path, playable_frets  # noqa: E402
from guitab.paths import DATA, GUITARSET_SHIFT, relative, resolve  # noqa: E402

SAMPLE_RATE = 16000


def job(args):
    row, k = args
    if max(playable_frets(jams_path(row["sample_id"]))) + k > MAX_FRET:
        return None
    import librosa
    import soundfile as sf

    name = f"{row['sample_id']}_up{k}"
    audio = GUITARSET_SHIFT / "audio" / f"{name}.wav"
    if not audio.exists():
        y, _ = librosa.load(str(resolve(row["audio_path"])), sr=SAMPLE_RATE, mono=True)
        sf.write(str(audio), librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=k), SAMPLE_RATE, subtype="PCM_16")
    return {"sample_id": name, "audio_path": relative(audio), "split": "train"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    (GUITARSET_SHIFT / "audio").mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(DATA / "guitarset.csv")) if r["split"] == "train"]
    jobs = [(r, k) for r in rows for k in args.shifts]
    with Pool(args.workers) as pool:
        results = [r for r in pool.map(job, jobs) if r]
    with open(DATA / "guitarset_shift.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["sample_id", "audio_path", "split"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {len(results)} shifted copies ({len(jobs) - len(results)} skipped for frets > {MAX_FRET})")


if __name__ == "__main__":
    main()
