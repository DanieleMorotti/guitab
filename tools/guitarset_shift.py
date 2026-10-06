"""Make pitch-shifted copies of GuitarSet training recordings with matching tabs.

Shifting up by k semitones keeps every note playable on the same string (fret + k), whereas
shifting down would push open strings below fret 0. Copies whose highest fret would exceed 24
are skipped. Audio is shifted offline with librosa so training stays fast.
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
MAX_FRET = 24


def shift_tokens(tokens, k):
    out = []
    for t in tokens:
        if t.startswith("note:"):
            _, s, f = t.split(":")
            fret = int(f[1:]) + k
            if fret > MAX_FRET:
                return None
            out.append(f"note:{s}:f{fret}")
        else:
            out.append(t)
    return out


def job(args):
    row, k, out_dir = args
    name = f"{row['sample_id']}_up{k}"
    tokens = Path(row["tab_path"]).read_text().split()
    shifted = shift_tokens(tokens, k)
    if shifted is None:
        return None
    tab_path = out_dir / "tokens" / f"{name}.txt"
    audio_path = out_dir / "audio" / f"{name}.wav"
    if not audio_path.exists():
        y, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        sf.write(str(audio_path), librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=k), SAMPLE_RATE, subtype="PCM_16")
    shifted[0] = name
    tab_path.write_text("\n".join(shifted) + "\n", encoding="utf-8")
    return {"sample_id": name, "dataset_name": "GuitarSetShift", "audio_path": str(audio_path), "tab_path": str(tab_path), "split": "train"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/guitarset.csv")
    parser.add_argument("--out_dir", default="data/guitarset_shift")
    parser.add_argument("--out_manifest", default="data/guitarset_shift.csv")
    parser.add_argument("--shifts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    (out_dir / "tokens").mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ROOT / args.manifest)) if r["split"] == "train"]
    jobs = [(r, k, out_dir) for r in rows for k in args.shifts]
    with Pool(args.workers) as pool:
        results = [r for r in pool.map(job, jobs) if r]
    with open(ROOT / args.out_manifest, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["sample_id", "dataset_name", "audio_path", "tab_path", "split"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {len(results)} shifted copies ({len(jobs) - len(results)} skipped for frets > {MAX_FRET})")


if __name__ == "__main__":
    main()
