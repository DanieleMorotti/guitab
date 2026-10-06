"""GAPS (classical / fingerstyle guitar, Zenodo 13962272): case-safe files and per-string labels.

Case-safe names: GAPS ids (scorehash, e.g. "GM1wc" vs "gM1wc") are case-sensitive; unzipping the
dataset on macOS silently overwrote 278 files whose names differ only by case. Everything GAPS-related
is stored under safe(h) = h + "_" + md5(h)[:6], extracted straight from the zip by extract().

Labels: per piece GAPS gives
  midi/<h>-fine-aligned.mid    note onsets/offsets aligned to the YouTube audio (timing truth)
  musicxml/<h>.xml             the score, usually with a tab staff carrying <string>/<fret>
The fine-aligned MIDI is the score aligned to the audio, so its notes come in score order: chord
sequences (score onset groups vs MIDI onset groups) are aligned by order with an LCS-style DP on
pitch overlap, and string/fret is copied by pitch inside matched chords. (Interpolating the
syncpoints proved unreliable for many pieces.) Notes without a match keep string 0 (unknown);
training masks the tab/onset loss for those. Pieces are flagged when the score's strings imply a
non-standard tuning (e.g. drop D) or when the match rate is low (see data.gaps_selection).
"""

import hashlib
import json
import re
import zipfile
from collections import Counter

import numpy as np

from .guitar import OPEN_PITCH
from .paths import GAPS

SAFE = GAPS / "safe"  # midi/, musicxml/, syncpoints/ with safe names; csv files copied as-is
GAPS_SPLITS = GAPS / "gaps_v1" / "gaps_v1_splits.csv"
GAPS_METADATA = GAPS / "gaps_v1" / "gaps_v1_metadata.csv"
TICKS = 480


def safe(h):
    return f"{h}_{hashlib.md5(h.encode()).hexdigest()[:6]}"


def midi_path(h):
    return SAFE / "midi" / f"{safe(h)}-fine-aligned.mid"


def xml_path(h):
    return SAFE / "musicxml" / f"{safe(h)}.xml"


def audio_path(h):
    return GAPS / "audio" / f"{safe(h)}.wav"


def label_path(h):
    return GAPS / "labels" / f"{safe(h)}.json"


def extract(out_dir=SAFE):
    z = zipfile.ZipFile(GAPS / "gaps_v1_no_audio.zip")
    n = 0
    for name in z.namelist():
        base = name.split("/")[-1]
        if not base:
            continue
        sub = name.split("/")[-2]
        m = re.match(r"^(.{5})(-fine-aligned\.mid|-syncpoints\.json|\.xml)$", base)
        out = out_dir / sub / (f"{safe(m.group(1))}{m.group(2)}" if m else base)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(name))
        n += 1
    return n


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


def build_labels(h):
    """(summary row, label payload or None) for GAPS piece h; payload["notes"] = [onset, offset, string, fret],
    string 0 = unknown (fret then holds the MIDI pitch)."""
    import pretty_midi

    xml, mid = xml_path(h), midi_path(h)
    if not (xml.exists() and mid.exists()):
        return {"hash": h, "status": "missing files"}, None
    try:
        snotes, _ = score_notes(xml)
    except Exception as exc:  # malformed score
        return {"hash": h, "status": f"parse error {exc!r}"[:120]}, None
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
    payload = {"notes": out, "standard_tuning": standard, "match_rate": rate, "content_sim": content_sim}
    return {"hash": h, "status": "ok", "notes": len(out), "match_rate": round(rate, 3), "content_sim": round(content_sim, 3),
            "standard_tuning": standard, "tuning": json.dumps(tuning)}, payload
