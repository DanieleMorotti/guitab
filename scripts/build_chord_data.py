"""Build the chord model's data in data/chords/ (not versioned) and guitab/chord_bigram.json.

    python scripts/build_chord_data.py [--chordonomicon data/chordonomicon/chordonomicon_v2.csv] [--device cpu]

- progressions.npz: the chord classes of every Chordonomicon song (Kyriakou et al. 2024, CC BY-NC 4.0; download
  chordonomicon_v2.csv from huggingface.co/datasets/ailsntua/Chordonomicon), and guitab/chord_bigram.json, its
  chord-to-chord counts relative to the previous root (used when decoding).
- voicings.json: the guitar voicing library (guitab.chord_data.voicing_library).
- custom_tabs.pkl: the custom-dataset tabs with chord names, as songs (training); the evaluation songs are left out.
- guitarset.pkl: GuitarSet (train / validation / test) with the tab model's notes and the tracked beats (as the pipeline
  computes them) and the played and instructed chords.
- custom_eval.pkl: the custom-dataset recordings whose tab has chord names and a good alignment, with the tab model's
  notes, the tracked beats and the tab's chords timed through the alignment (evaluation only; "val" half for choosing
  the checkpoint, "test" half reported once).

With --pseudo WEIGHTS, only pseudo.pkl (self-training): real recordings with exact notes (the aligned custom-dataset
songs outside the evaluation, EGDB's test clips through three amplifiers), each with the tab model's notes and the
tracked beats, labelled by the chord model WEIGHTS reading the exact notes (its confident chords; the rest unlabelled).
"""

import argparse
import csv
import difflib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.chord_data import (CHORD_DATA, OPEN, bigram_counts, custom_tab_songs, load_voicings,  # noqa: E402
                               read_progressions)
from guitab.chords import (IGNORE, NO_CHORD, NQ, QUALITIES, TONES, chord_probs, chord_spans, load_chord_model,  # noqa: E402
                           parse_chord)
from guitab.data import tracks  # noqa: E402
from guitab.inference import extract_notes, load_model, predict  # noqa: E402
from guitab.paths import CACHE, DATA, DEFAULT_CHECKPOINT, ROOT  # noqa: E402
from guitab.tab import detect_beats  # noqa: E402

CUSTOM = DATA / "custom"


def transcribe(model, name, audio, device, feat=None):
    """The tab model's notes (as the guitab command extracts them) and the tracked beats / downbeats; feat: the CQT
    (default: the cached one of track name). Four windows at a time (~1 GB instead of ~3.7 GB)."""
    feat = np.load(CACHE / f"{name}.npz")["feat"] if feat is None else feat
    tab, onset = predict(model, feat, device, batch=4)
    notes = [{"onset": n["onset"], "offset": n["offset"], "pitch": n["pitch"], "fret": n["fret"]}
             for n in extract_notes(tab, onset, model.decoding)]
    beats, downs = detect_beats(audio, str(device), onset)
    return notes, np.asarray(beats), np.asarray(downs)


def jams_chords(path):
    """[[(start, end, class)] played, [(start, end, class)] instructed] from a GuitarSet JAMS file."""
    d = json.loads(Path(path).read_text())
    anns = [a for a in d["annotations"] if a["namespace"] == "chord"]
    return [[(x["time"], x["time"] + x["duration"], parse_chord(x["value"])) for x in a["data"]] for a in anns]


def aligned_tab_chords(song_id):
    """The tab's chords timed through the aligned notes ([(start, end, class)]), or None. Tab notes and aligned notes are
    matched in order (difflib on string and fret); a chord takes the time of the first matched note of its beat. Chord
    names written as capo shapes are moved to sounding pitch (whichever fits the notes better)."""
    tab = json.loads((CUSTOM / "tabs" / f"{song_id}.json").read_text())
    capo = tab["track"].get("capo", 0) or 0
    lab = sorted(json.loads((CUSTOM / "labels" / f"{song_id}.json").read_text())["notes"], key=lambda x: (x[0], x[2]))
    seq, names, beat_id = [], {}, 0
    for measure in tab["track"]["measures"]:
        for beat in measure["voices"][0]["beats"]:
            beat_id += 1
            if beat.get("chord", {}).get("text"):
                names[beat_id] = beat["chord"]["text"]
            for n in sorted([n for n in beat.get("notes", []) if "string" in n], key=lambda n: n["string"]):
                if not n.get("tie") and not n.get("rest"):
                    seq.append((beat_id, n["string"] + 1, n["fret"]))
    match = difflib.SequenceMatcher(None, [(s, f + capo) for _, s, f in seq], [(x[2], x[3]) for x in lab], autojunk=False)
    time_of = {}
    for a, b, size in match.get_matching_blocks():
        for i in range(size):
            time_of.setdefault(a + i, lab[b + i][0])
    beat_time = {}
    for i, (b, _, _) in enumerate(seq):
        if i in time_of and b not in beat_time:
            beat_time[b] = time_of[i]
    fit = {0: 0, capo: 0}
    for b, text in names.items():
        k = parse_chord(text)
        pcs = {(OPEN[6 - s] + f) % 12 for b2, s, f in seq if b2 == b}
        if not 0 <= k < NO_CHORD or not pcs:
            continue
        for shift in fit:
            tones = {(k // NQ + j) % 12 for j in TONES[QUALITIES[k % NQ]]}
            fit[shift] += len({(p + shift) % 12 for p in pcs} & tones) - len({(p + shift) % 12 for p in pcs} - tones)
    shift = capo if capo and fit[0] >= fit[capo] else 0  # names fit the shapes: add the capo
    timed = []
    for b in sorted(names):
        t = beat_time.get(b, beat_time.get(next((x for x in sorted(beat_time) if x > b), None)))
        if t is not None:
            k = parse_chord(names[b])
            timed.append([t, k if k in (NO_CHORD, IGNORE) else (k // NQ + shift) % 12 * NQ + k % NQ])
    timed.sort()
    end = max(x[1] for x in lab)
    out = [(a[0], b[0], a[1]) for a, b in zip(timed, timed[1:] + [[end, None]]) if b[0] > a[0]]
    return out or None, len(time_of) / max(len(seq), 1)


def teacher_chords(teacher, clean, beats, downs, capo=0):
    """The chord model's confident chords on exact notes: [(start, end, class)]."""
    bounds, probs = chord_probs(teacher, clean, beats, downs, capo)
    return chord_spans(bounds, probs)


def build_pseudo(weights, device):
    import pretty_midi

    from guitab.audio import compute_cqt, load_audio
    from guitab.guitar import OPEN_PITCH

    teacher = load_chord_model(weights)
    model = load_model(DEFAULT_CHECKPOINT, device)
    evaluation = {r["name"] for r in pickle.load(open(CHORD_DATA / "custom_eval.pkl", "rb"))}
    status = {r["songId"]: r["status"] for r in csv.DictReader(open(CUSTOM / "labels_summary.csv"))}
    rows = []
    for r in csv.DictReader(open(CUSTOM / "selection.csv")):
        sid = r["songId"]
        if sid in evaluation or status.get(sid) != "ok" or not (CACHE / f"ss_{sid}.npz").exists():
            continue
        tab = json.loads((CUSTOM / "tabs" / f"{sid}.json").read_text())
        capo = tab["track"].get("capo", 0) or 0
        clean = [{"onset": a, "offset": b, "pitch": OPEN_PITCH[s] + f, "fret": f}
                 for a, b, s, f in json.loads((CUSTOM / "labels" / f"{sid}.json").read_text())["notes"]]
        notes, beats, downs = transcribe(model, f"ss_{sid}", CUSTOM / "audio" / f"{sid}.flac", device)
        rows.append({"name": sid, "notes": notes, "beats": beats, "downbeats": downs, "capo": 0, "outside": IGNORE,
                     "chords": teacher_chords(teacher, clean, beats, downs, capo)})
        print(f"  custom {sid}: {len(rows[-1]['chords'])} chords", flush=True)
    opens = set(OPEN_PITCH.values())
    for mid in sorted((DATA / "egdb" / "audio_label").glob("*.midi")):
        pm = pretty_midi.PrettyMIDI(str(mid))
        clean = [{"onset": n.start, "offset": n.end, "pitch": n.pitch, "fret": 0 if n.pitch in opens else None}
                 for inst in pm.instruments for n in inst.notes]
        for amp in ("audio_DI", "audio_JCjazz", "audio_Plexi"):
            wav = DATA / "egdb" / amp / f"{mid.stem}.wav"
            if not wav.exists():
                continue
            notes, beats, downs = transcribe(model, None, wav, device, compute_cqt(load_audio(wav)))
            rows.append({"name": f"egdb_{amp}_{mid.stem}", "notes": notes, "beats": beats, "downbeats": downs,
                         "capo": 0, "outside": IGNORE, "chords": teacher_chords(teacher, clean, beats, downs)})
        print(f"  EGDB {mid.stem}: {len(rows[-1]['chords'])} chords", flush=True)
    pickle.dump(rows, open(CHORD_DATA / "pseudo.pkl", "wb"))
    print(f"pseudo-labelled recordings: {len(rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chordonomicon", default=str(DATA / "chordonomicon" / "chordonomicon_v2.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pseudo", default=None, help="chord model weights: build only pseudo.pkl")
    args = parser.parse_args()
    CHORD_DATA.mkdir(parents=True, exist_ok=True)
    if args.pseudo:
        return build_pseudo(args.pseudo, torch.device(args.device))

    songs = read_progressions(args.chordonomicon)
    lengths = np.array([len(s) for s in songs])
    np.savez_compressed(CHORD_DATA / "progressions.npz", chords=np.concatenate(songs), lengths=lengths)
    (ROOT / "guitab" / "chord_bigram.json").write_text(json.dumps(
        {"qualities": QUALITIES, "source": "Chordonomicon (CC BY-NC 4.0): consecutive different chords, "
                                           "count[from quality][root interval][to quality]",
         "count": bigram_counts(songs).tolist()}, separators=(",", ":")))
    print(f"progressions: {len(songs)} songs, {lengths.sum()} chords", flush=True)
    load_voicings()

    # custom-dataset songs whose tab has chord names and whose alignment is good: evaluation
    status = {r["songId"]: r["status"] for r in csv.DictReader(open(CUSTOM / "labels_summary.csv"))}
    split = {r["songId"]: r["split"] for r in csv.DictReader(open(CUSTOM / "selection.csv"))}
    evaluation = []
    for path in sorted((CUSTOM / "tabs").glob("*.json")):
        sid = path.stem
        tab = json.loads(path.read_text())
        named = sum(bool(b.get("chord", {}).get("text")) for m in tab["track"]["measures"] for b in m["voices"][0]["beats"])
        if named >= 10 and status.get(sid) == "ok":
            evaluation.append(sid)
    tabs = custom_tab_songs(exclude=set(evaluation))
    pickle.dump(tabs, open(CHORD_DATA / "custom_tabs.pkl", "wb"))
    print(f"custom tabs with chord names (training): {len(tabs)} songs", flush=True)

    device = torch.device(args.device)
    model = load_model(DEFAULT_CHECKPOINT, device)
    rows = []
    for name in evaluation:  # validation-split songs first in the "val" half, then alternate
        chords, matched = aligned_tab_chords(name)
        notes, beats, downs = transcribe(model, f"ss_{name}", CUSTOM / "audio" / f"{name}.flac", device)
        rows.append({"name": name, "custom_split": split.get(name), "notes": notes, "beats": beats, "downbeats": downs,
                     "chords": chords, "matched": matched})
        print(f"  custom {name}: {len(chords)} chords, {matched:.2f} of the tab notes aligned", flush=True)
    rows.sort(key=lambda r: (r["custom_split"] != "ss_validation", r["name"]))
    for i, r in enumerate(rows):
        r["half"] = "val" if r["custom_split"] == "ss_validation" or i % 2 == 0 else "test"
    pickle.dump(rows, open(CHORD_DATA / "custom_eval.pkl", "wb"))

    gs = {"train": [], "validation": [], "test": []}
    for name, audio, labels, shift, sp in tracks():
        if shift != 0 or sp not in gs:
            continue
        notes, beats, downs = transcribe(model, name, audio, device)
        played, instructed = jams_chords(labels)
        gs[sp].append({"name": name, "take": name.rsplit("_", 1)[-1], "notes": notes, "beats": beats,
                       "downbeats": downs, "chords": played, "instructed": instructed})
    pickle.dump(gs, open(CHORD_DATA / "guitarset.pkl", "wb"))
    print({k: len(v) for k, v in gs.items()}, "GuitarSet tracks", flush=True)


if __name__ == "__main__":
    main()
