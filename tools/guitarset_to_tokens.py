"""Convert GuitarSet JAMS annotations to DadaGP token text files (reference tabs for evaluation).

Onsets are quantized to a 32nd-note grid at the excerpt's annotated tempo (GuitarSet was
recorded to a click, with beat annotations starting at t=0). Notes still sounding at a
barline are re-emitted with nfx:tie, and silent gaps of at least a quarter become rests.
"""

import argparse
import csv
import json
from pathlib import Path

TICKS_PER_BEAT = 960  # DadaGP convention (src/tab.py ticks_per_second)
GRID = 120  # 32nd note
MIN_REST_TICKS = 960
# Note values DadaGP can encode on this grid (subset of dadagp.supported_times).
SUPPORTED_WAITS = [3840, 2880, 1920, 1440, 960, 720, 480, 360, 240, 120]
OPEN_PITCH = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
MAX_FRET = 24


def load_notes(jams):
    notes = []  # (onset_tick, offset_tick, string, fret)
    tempo = None
    num_beats = 4
    for ann in jams["annotations"]:
        if ann["namespace"] == "tempo":
            tempo = float(ann["data"][0]["value"])
        if ann["namespace"] == "beat_position" and ann["data"]:
            num_beats = int(ann["data"][0]["value"]["num_beats"])
    ticks_per_second = tempo / 60.0 * TICKS_PER_BEAT
    for ann in jams["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        string_value = 6 - int(ann["annotation_metadata"]["data_source"])
        for note in ann["data"]:
            fret = int(round(note["value"])) - OPEN_PITCH[string_value]
            if fret < 0 or fret > MAX_FRET:
                continue
            onset = int(round(note["time"] * ticks_per_second / GRID)) * GRID
            offset = max(onset + GRID, int(round((note["time"] + note["duration"]) * ticks_per_second / GRID)) * GRID)
            notes.append((onset, offset, string_value, fret))
    return notes, int(round(tempo)), num_beats


def build_tokens(name, notes, tempo, num_beats):
    measure_ticks = num_beats * TICKS_PER_BEAT
    by_onset = {}
    for onset, offset, string_value, fret in notes:
        group = by_onset.setdefault(onset, {})
        if string_value not in group or offset > group[string_value][1]:
            group[string_value] = (fret, offset)
    onsets = sorted(by_onset)

    tokens = [name, "downtune:0", f"tempo:{tempo}", "start", "new_measure"]
    sounding = {}  # string -> (fret, offset)
    cursor = 0

    def advance(target):
        """Emit waits from cursor to target as encodable note values, splitting at barlines.

        Notes still sounding across a split are re-emitted with nfx:tie, as Guitar Pro does.
        """
        nonlocal cursor
        while cursor < target:
            next_bar = (cursor // measure_ticks + 1) * measure_ticks
            remaining = min(target, next_bar) - cursor
            step = next(w for w in SUPPORTED_WAITS if w <= remaining)
            tokens.append(f"wait:{step}")
            cursor += step
            for s in [s for s, (_, off) in sounding.items() if off <= cursor]:
                del sounding[s]
            if cursor == next_bar:
                tokens.append("new_measure")
            if cursor < target:
                if sounding:
                    for s in sorted(sounding):
                        tokens.append(f"note:s{s}:f{sounding[s][0]}")
                        tokens.append("nfx:tie")
                else:
                    tokens.append("rest")

    if onsets and onsets[0] > 0:
        tokens.append("rest")
        advance(onsets[0])

    for index, onset in enumerate(onsets):
        if onset > cursor:
            advance(onset)
        group = by_onset[onset]
        for s in sorted(group):
            fret, offset = group[s]
            tokens.append(f"note:s{s}:f{fret}")
            sounding[s] = (fret, offset)
        next_onset = onsets[index + 1] if index + 1 < len(onsets) else None
        if next_onset is None:
            last_off = max(off for _, off in sounding.values()) if sounding else onset + GRID
            advance(max(last_off, onset + GRID))
            break
        last_off = max(off for _, off in sounding.values())
        if next_onset - last_off >= MIN_REST_TICKS:
            advance(max(last_off, onset + GRID))
            tokens.append("rest")
    if tokens[-1] == "new_measure":
        tokens.pop()
    tokens.append("end")
    return tokens


def split_for_player(player):
    return {"04": "validation", "05": "test"}.get(player, "train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", default="data/guitarset/annotation")
    parser.add_argument("--audio_dir", default="data/guitarset/audio_mic")
    parser.add_argument("--tokens_dir", default="data/guitarset/tokens")
    parser.add_argument("--manifest", default="data/guitarset.csv")
    args = parser.parse_args()

    root = Path.cwd()
    tokens_dir = root / args.tokens_dir
    tokens_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for jams_path in sorted((root / args.annotation_dir).glob("*.jams")):
        name = jams_path.stem
        audio_path = root / args.audio_dir / f"{name}_mic.wav"
        if not audio_path.exists():
            continue
        jams = json.loads(jams_path.read_text())
        notes, tempo, num_beats = load_notes(jams)
        if not notes:
            continue
        tokens = build_tokens(name, notes, tempo, num_beats)
        token_path = tokens_dir / f"{name}.txt"
        token_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
        rows.append(
            {
                "sample_id": name,
                "dataset_name": "GuitarSet",
                "audio_path": str(audio_path),
                "tab_path": str(token_path),
                "split": split_for_player(name[:2]),
            }
        )

    with open(root / args.manifest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "dataset_name", "audio_path", "tab_path", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
