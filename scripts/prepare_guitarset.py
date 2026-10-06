"""Write the GuitarSet track list data/guitarset.csv (sample_id, audio_path, split).

Download GuitarSet (CC-BY, https://zenodo.org/records/3371780): annotation.zip and
audio_mono-mic.zip, unzipped into data/guitarset/annotation/ and data/guitarset/audio_mic/.
A track is kept when its mic audio exists and it has at least one note on frets 0..24.
Split by player: 00-03 train, 04 validation, 05 test.

    python scripts/prepare_guitarset.py
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.guitarset import playable_frets, split_for_player  # noqa: E402
from guitab.paths import DATA, GUITARSET, relative  # noqa: E402


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows = []
    for jams in sorted((GUITARSET / "annotation").glob("*.jams")):
        name = jams.stem
        audio = GUITARSET / "audio_mic" / f"{name}_mic.wav"
        if not audio.exists() or not playable_frets(jams):
            continue
        rows.append({"sample_id": name, "audio_path": relative(audio), "split": split_for_player(name[:2])})
    with open(DATA / "guitarset.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "audio_path", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {DATA / 'guitarset.csv'}")


if __name__ == "__main__":
    main()
