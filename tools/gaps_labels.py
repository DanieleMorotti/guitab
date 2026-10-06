"""Build per-string note labels for GAPS (classical / fingerstyle guitar) recordings.

GAPS gives, per piece:
  midi/<h>-fine-aligned.mid    note onsets/offsets aligned to the YouTube audio (timing truth)
  musicxml/<h>.xml             the score, usually with a tab staff carrying <string>/<fret>
  syncpoints/<h>-syncpoints.json  [measure, time] or [measure, time, tick] anchors (480 ticks/quarter);
                               measure = index in the score with repeats unfolded
The fine-aligned MIDI is the score aligned to the audio, so its notes come in score order: chord
sequences (score onset groups vs MIDI onset groups) are aligned by order with an LCS-style DP on
pitch overlap, and string/fret is copied by pitch inside matched chords. (Interpolating the
syncpoints proved unreliable for many pieces.) Notes without a match keep string 0 (unknown);
training masks the tab/onset loss for those.
Pieces are flagged when the score's strings imply a non-standard tuning (e.g. drop D) or when the
match rate is low; those are excluded from training by tabdata.py.

    python tools/gaps_labels.py --workers 4
Output: data/gaps/labels/<safe(h)>.json and data/gaps/labels_summary.csv (files use the case-safe
names of tools/gaps_common.py; run `python tools/gaps_common.py` first to extract the zip).
"""

import argparse
import csv
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gaps_common import label_path, midi_path, xml_path

ROOT = Path(__file__).resolve().parents[1]
GAPS = ROOT / "data" / "gaps"
OPEN_PITCH = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
TICKS = 480


def score_notes(xml_path):
    import music21 as m21

    score = m21.converter.parse(str(xml_path))

    def n_strings(part):
        return sum(1 for n in part.recurse().notes for x in (n.notes if n.isChord else [n])
                   if any(isinstance(a, m21.articulations.StringIndication) for a in x.articulations))

    part = max(score.parts, key=n_strings)
    try:  # syncpoints index measures as played, i.e. with repeats / voltas unfolded
        part = part.expandRepeats()
    except Exception:
        pass
    measures = list(part.getElementsByClass("Measure"))
    m_offsets = [float(m.offset) for m in measures]
    m_offsets.append(m_offsets[-1] + float(measures[-1].duration.quarterLength))
    notes = []  # (score_quarter, midi_pitch, string, fret)
    for m in measures:
        for el in m.recurse().notes:
            q = float(m.offset) + float(el.getOffsetInHierarchy(m))
            for x in (el.notes if el.isChord else [el]):
                if x.tie is not None and x.tie.type in ("stop", "continue"):
                    continue
                s = next((a.number for a in x.articulations if isinstance(a, m21.articulations.StringIndication)), None)
                f = next((a.number for a in x.articulations if isinstance(a, m21.articulations.FretIndication)), None)
                notes.append((q, int(x.pitch.midi), s, f))
    return notes, m_offsets


def group_onsets(items, key, gap):
    """Group items sorted by key into onset groups (consecutive keys closer than gap)."""
    groups = []
    for it in sorted(items, key=key):
        if groups and key(it) - key(groups[-1][-1]) < gap:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def align_groups(a, b):
    """Monotonic alignment of two chord sequences (pitch sets) maximising pitch overlap (LCS-style DP)."""
    n, m = len(a), len(b)
    sim = np.zeros((n, m), dtype=np.float32)
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            inter = len(pa & pb)
            if inter:
                sim[i, j] = inter / max(len(pa), len(pb))
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    for i in range(1, n + 1):
        t = np.maximum(dp[i - 1, 1:], dp[i - 1, :-1] + sim[i - 1])
        dp[i, 1:] = np.maximum.accumulate(t)
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if dp[i, j] == dp[i, j - 1]:
            j -= 1
        elif dp[i, j] == dp[i - 1, j]:
            i -= 1
        else:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
    return pairs[::-1]


def build(h):
    import pretty_midi

    xml, mid = xml_path(h), midi_path(h)
    if not (xml.exists() and mid.exists()):
        return {"hash": h, "status": "missing files"}
    try:
        snotes, _ = score_notes(xml)
    except Exception as exc:  # malformed score
        return {"hash": h, "status": f"parse error {exc!r}"[:120]}
    pm = pretty_midi.PrettyMIDI(str(mid))
    mnotes = sorted((n for i in pm.instruments for n in i.notes), key=lambda n: n.start)

    # some GAPS MIDI files hold a different piece than their score: compare pitch histograms
    hs, hm = np.zeros(128), np.zeros(128)
    for n in snotes:
        hs[n[1]] += 1
    for n in mnotes:
        hm[n.pitch] += 1
    content_sim = float(hs @ hm / (np.linalg.norm(hs) * np.linalg.norm(hm) + 1e-9))

    # tuning implied by the score: open pitch per string = pitch - fret (mode)
    implied = {}
    for _, p, s, f in snotes:
        if s and f is not None and 1 <= s <= 6:
            implied.setdefault(s, Counter())[p - f] += 1
    tuning = {s: c.most_common(1)[0][0] for s, c in implied.items()}
    standard = all(tuning.get(s, OPEN_PITCH[s]) == OPEN_PITCH[s] for s in OPEN_PITCH)

    # The fine-aligned MIDI is the score aligned to the audio, so notes come in score order:
    # align chord sequences by order and copy string/fret by pitch inside matched chords.
    sgroups = group_onsets(snotes, key=lambda n: n[0], gap=1e-6)
    mgroups = group_onsets(mnotes, key=lambda n: n.start, gap=0.035)
    pairs = align_groups([{n[1] for n in g} for g in sgroups], [{n.pitch for n in g} for g in mgroups])
    fingering = {}  # id(midi note) -> (string, fret)
    for i, j in pairs:
        by_pitch = {}
        for _, p, s, f in sgroups[i]:
            if s and f is not None and 1 <= s <= 6 and p - f == OPEN_PITCH[s] and 0 <= f <= 24:
                by_pitch.setdefault(p, (s, f))
        for n in mgroups[j]:
            if n.pitch in by_pitch:
                fingering[id(n)] = by_pitch[n.pitch]
    out = []
    for n in mnotes:
        s, f = fingering.get(id(n), (0, int(n.pitch)))  # unknown string: keep the pitch in the fret slot
        out.append([round(n.start, 4), round(n.end, 4), s, f])
    rate = len(fingering) / max(len(out), 1)
    label_path(h).parent.mkdir(exist_ok=True)
    label_path(h).write_text(json.dumps(
        {"notes": out, "standard_tuning": standard, "match_rate": rate, "content_sim": content_sim}))
    return {"hash": h, "status": "ok", "notes": len(out), "match_rate": round(rate, 3), "content_sim": round(content_sim, 3),
            "standard_tuning": standard, "tuning": json.dumps(tuning)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    hashes = args.only or sorted({r["filename"][:-4] for r in csv.DictReader(open(GAPS / "gaps_v1" / "gaps_v1_splits.csv"))
                                  if r["filename"].endswith(".wav")})
    with Pool(args.workers) as pool:
        rows = pool.map(build, hashes)
    keys = ["hash", "status", "notes", "match_rate", "content_sim", "standard_tuning", "tuning"]
    with open(GAPS / "labels_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(ok)}/{len(rows)} built; standard tuning {sum(r['standard_tuning'] for r in ok)}; "
          f"median match rate {np.median([r['match_rate'] for r in ok]):.3f}")


if __name__ == "__main__":
    main()
