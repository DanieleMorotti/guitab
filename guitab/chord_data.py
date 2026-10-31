"""Training data for the chord model (guitab/chords.py): synthetic guitar parts, tabs with chord names, real recordings.

Synthetic songs (synth_song): a stretch of a real chord progression (Chordonomicon), played on a guitar the way guitarists
do: chord shapes from a voicing library (voicing_library: open, barre and movable shapes, inversions, capo), ringing
open strings, anticipated changes, and a texture (strummed, arpeggiated, Travis picking, melody over a bass, or melody
alone); the beat grid has the tracker's typical errors. degrade then gives the notes the tab model's typical errors
(missed notes in dense chords, octave notes, wrong frets, shorter notes). The label is the chord being played, also
under a melody alone. Tabs with chord names (custom_tab_songs) give real guitar parts with their authors' chords;
GuitarSet and the custom-dataset recordings give the tab model's real notes on real audio (scripts/build_chord_data.py).
"""

import csv
import itertools
import json

import numpy as np

from .chords import IGNORE, NO_CHORD, NQ, QUALITIES, TONES, features, labels_at, parse_chord, segment_bounds
from .paths import DATA

CHORD_DATA = DATA / "chords"
OPEN = [40, 45, 50, 55, 59, 64]  # strings low E .. high E (index 0 = string 6)
TEXTURES = {"strum": 0.3, "arpeggio": 0.25, "travis": 0.15, "melody_bass": 0.2, "melody": 0.1}


# ---------- Chordonomicon ----------

def read_progressions(path):
    """Chord classes per song from the Chordonomicon CSV (section markers dropped, unknown chords skipped)."""
    csv.field_size_limit(10 ** 9)
    songs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            seq = [parse_chord(t) for t in row["chords"].split() if not t.startswith("<")]
            seq = [k for k in seq if 0 <= k < NO_CHORD]
            if len(seq) >= 4:
                songs.append(np.array(seq, dtype=np.int16))
    return songs


class Progressions:
    """The Chordonomicon songs saved by scripts/build_chord_data.py (progressions.npz), read once into one array: an
    item is a view of one song's chord classes."""

    def __init__(self, path=None):
        with np.load(path or CHORD_DATA / "progressions.npz") as f:  # each f[key] decompresses the whole array again
            self.chords, lengths = f["chords"], f["lengths"]
        self.starts = np.r_[0, np.cumsum(lengths)]

    def __len__(self):
        return len(self.starts) - 1

    def __getitem__(self, i):
        return self.chords[self.starts[i]:self.starts[i + 1]]


def bigram_counts(songs):
    """count[from quality, root interval, to quality] over consecutive different chords."""
    count = np.zeros((NQ, 12, NQ), dtype=np.int64)
    for seq in songs:
        a, b = seq[:-1].astype(int), seq[1:].astype(int)
        change = a != b
        np.add.at(count, (a[change] % NQ, (b[change] // NQ - a[change] // NQ) % 12, b[change] % NQ), 1)
    return count


# ---------- voicings ----------

def voicing_library(max_base=10):
    """{chord class: [(frets per string low E .. high E, -1 = muted), score, inversion]} for standard tuning: every
    shape within a 4-fret hand position (plus open strings) whose notes are exactly the chord's tones (a 7th chord may
    omit the fifth), sounding strings contiguous (at most one muted inside), playable with four fingers or a barre."""
    lib = {}
    for k in range(NO_CHORD):
        root, q = k // NQ, QUALITIES[k % NQ]
        tones = {(root + i) % 12 for i in TONES[q]}
        optional = {(root + 7) % 12} if len(tones) >= 4 else set()
        found = {}
        for base in range(0, max_base + 1):
            frets = [f for f in range(max(base, 1), base + 4)]
            cands = [[-1] + [f for f in [0] + frets if (OPEN[s] + f) % 12 in tones] for s in range(6)]
            for combo in itertools.product(*cands):
                sounded = [s for s in range(6) if combo[s] >= 0]
                if len(sounded) < (2 if q == "5" else 3) or (q == "5" and len(sounded) > 3):
                    continue
                if sounded[-1] - sounded[0] + 1 - len(sounded) > 1:
                    continue
                pcs = {(OPEN[s] + combo[s]) % 12 for s in sounded}
                if not (tones - optional) <= pcs:
                    continue
                fretted = [combo[s] for s in sounded if combo[s] > 0]
                barre = False
                if fretted:
                    if min(fretted) != base and base > 0 or max(fretted) - min(fretted) > 3:
                        continue
                    if len(fretted) > 4:
                        low = min(fretted)
                        if fretted.count(low) < 2 or len(fretted) - fretted.count(low) > 3:
                            continue
                        barre = True
                inversion = (OPEN[sounded[0]] + combo[sounded[0]]) % 12 != root
                if inversion and q == "5":
                    continue
                span = max(fretted) - min(fretted) if fretted else 0
                open_shape = 0 in [combo[s] for s in sounded] and max(fretted, default=0) <= 3
                score = 0.3 * len(sounded) + 1.5 * open_shape - 0.12 * base - 0.25 * span - 0.4 * barre - 1.2 * inversion
                if combo not in found or score > found[combo][0]:
                    found[combo] = (score, inversion)
        ranked = sorted(found.items(), key=lambda x: -x[1][0])
        roots = [(list(c), sc, False) for c, (sc, inv) in ranked if not inv]
        invs = [(list(c), sc, True) for c, (sc, inv) in ranked if inv]
        lib[k] = roots[:12] + invs[:4]
    return lib


def load_voicings():
    path = CHORD_DATA / "voicings.json"
    if not path.exists():
        CHORD_DATA.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(k): v for k, v in voicing_library().items()}))
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


# ---------- synthetic songs ----------

def _pick_voicing(rng, options, prev_pos, capo):
    pos = np.array([min([f for f in v[0] if f > 0], default=0) for v in options], dtype=float)
    score = np.array([v[1] for v in options]) - (0.15 * np.abs(pos - prev_pos) if prev_pos is not None else 0)
    score -= 0.8 * np.array([v[2] for v in options])
    if capo:  # with a capo, players use open shapes
        score += 1.0 * (pos <= 3)
    p = np.exp((score - score.max()) / 0.6)
    return options[rng.choice(len(options), p=p / p.sum())]


def _scale(tonic, minor):
    steps = (0, 2, 3, 5, 7, 8, 10) if minor else (0, 2, 4, 5, 7, 9, 11)
    return {(tonic + s) % 12 for s in steps}


def _place(pitch, near, capo, rng):
    """(string index, absolute fret) for a pitch, preferring frets near the hand position."""
    opts = [(s, pitch - OPEN[s]) for s in range(6) if capo <= pitch - OPEN[s] <= 17]
    if not opts:
        return None
    opts.sort(key=lambda o: abs(o[1] - near) + 0.01 * rng.random())
    return opts[0]


def synth_song(rng, progressions, voicings):
    """A synthetic guitar part from a song of progressions (Progressions): {"notes", "beats", "downbeats", "chords"
    [(start, end, class)], "capo"}."""
    seq = progressions[rng.integers(len(progressions))]
    n = int(rng.integers(12, 41))
    start = int(rng.integers(0, max(len(seq) - n, 0) + 1))
    shift = int(rng.integers(-5, 7))
    seq = [(int(k) // NQ + shift) % 12 * NQ + int(k) % NQ for k in seq[start:start + n]]
    meter = 3 if rng.random() < 0.15 else 4
    period = 60.0 / rng.uniform(60, 150)
    capo = 0 if rng.random() < 0.75 else int(rng.integers(1, 6))
    style = rng.choice(["bar", "half", "two", "mixed"], p=[0.5, 0.15, 0.15, 0.2])
    lengths = {"bar": [meter], "half": [meter // 2 if meter == 4 else meter], "two": [2 * meter]}.get(str(style))
    lead = rng.uniform(0, 2.0)
    # chord spans in beats; a change is anticipated by an eighth now and then
    push = rng.uniform(0, 0.35)
    spans, t = [], 0.0
    for k in seq:
        beats = lengths[0] if lengths else int(rng.choice([meter, 2 * meter, max(meter // 2, 1)], p=[0.6, 0.15, 0.25]))
        spans.append([t, t + beats, k])
        t += beats
    for a, b in zip(spans, spans[1:]):
        if a[2] != b[2] and rng.random() < push and b[0] - a[0] > 1:
            a[1] = b[0] = b[0] - 0.5
    total_beats = t
    # key for melodies: the most common root, minor if its chord is
    roots = [k // NQ for k in seq]
    tonic = max(set(roots), key=roots.count)
    minor = QUALITIES[next(k for k in seq if k // NQ == tonic) % NQ] in ("min", "min7")
    scale = _scale(tonic, minor)
    texture = rng.choice(list(TEXTURES), p=list(TEXTURES.values()))
    drones = rng.random() < 0.3
    # strum pattern over a bar of eighths: (position, down)
    pattern = [(e, e % 2 == 0) for e in range(2 * meter) if rng.random() < (0.9 if e % 2 == 0 else 0.45)] or [(0, True)]
    arp_unit = 0.5 if rng.random() < 0.7 else 0.25
    arp_order = rng.choice(["up", "updown", "bass_up", "random"])
    ring = rng.random() < 0.6
    notes, prev_pos, chosen = [], None, {}
    melody_pitch = 64 + int(rng.integers(0, 10))
    for i, (b0, b1, k) in enumerate(spans):
        if i % 8 == 0 and i and rng.random() < 0.25:
            texture = rng.choice(list(TEXTURES), p=list(TEXTURES.values()))
        shape = ((k // NQ - capo) % 12) * NQ + k % NQ
        if shape not in chosen:
            chosen[shape] = _pick_voicing(rng, voicings[shape], prev_pos, capo)
        frets = [f + capo if f >= 0 else -1 for f in chosen[shape][0]]
        if drones:  # ringing open treble strings over open-position shapes
            for s in (5, 4):
                if frets[s] < 0 and max(frets) <= capo + 4 and rng.random() < 0.7:
                    frets[s] = capo
        sounded = [s for s in range(6) if frets[s] >= 0]
        fretted = [f for f in frets if f > capo]
        prev_pos = min(fretted) - capo if fretted else 0
        t0, t1 = lead + b0 * period, lead + b1 * period
        tones = {(k // NQ + j) % 12 for j in TONES[QUALITIES[k % NQ]]}

        def add(s, f, on, off):
            if off - on > 0.03:
                notes.append({"onset": on, "offset": off, "pitch": OPEN[s] + f, "fret": f, "string": s})

        if texture == "strum":
            hits = [t0]
            for bar in range(int(b0 // meter), int(np.ceil(b1 / meter))):
                hits += [lead + (bar * meter + e * 0.5) * period for e, _ in pattern if b0 <= bar * meter + e * 0.5 < b1]
            hits = sorted(set(round(h, 6) for h in hits))
            for j, h in enumerate(hits):
                end = hits[j + 1] if j + 1 < len(hits) else t1
                down = round((h - lead) / period * 2) % 2 == 0
                strings = sounded if down else sounded[-int(rng.integers(2, 5)):][::-1]
                for m, s in enumerate(strings):
                    add(s, frets[s], h + m * rng.uniform(0.006, 0.02), end + 0.02)
        elif texture == "arpeggio":
            order = {"up": sounded, "updown": sounded + sounded[-2:0:-1], "bass_up": [sounded[0]] + sounded[2:] + sounded[1:2],
                     "random": list(rng.permutation(sounded))}[str(arp_order)]
            steps = np.arange(t0, t1 - 1e-6, arp_unit * period)
            for j, h in enumerate(steps):
                s = order[j % len(order)]
                add(s, frets[s], h, t1 + 0.02 if ring else h + arp_unit * period * rng.uniform(1, 3))
        elif texture == "travis":
            bass = sounded[:3] if len(sounded) >= 3 else sounded
            treble = [s for s in sounded if s >= 3] or sounded[-2:]
            beat = b0
            while beat < b1 - 1e-6:
                h = lead + beat * period
                s = bass[0] if int(beat) % 2 == 0 else bass[min(len(bass) - 1, 1 + int(rng.integers(0, 2)))]
                add(s, frets[s], h, h + period * 1.5)
                if beat + 0.5 < b1:
                    for s in rng.choice(treble, size=min(len(treble), int(rng.integers(1, 3))), replace=False):
                        add(int(s), frets[int(s)], h + 0.5 * period, h + period * 1.5)
                beat += 1
        else:  # melody over a bass, or a melody alone
            if texture == "melody_bass":
                s = sounded[0]
                add(s, frets[s], t0, t1 + 0.02)
                if b1 - b0 >= 4 and rng.random() < 0.6:
                    s2 = sounded[min(1, len(sounded) - 1)]
                    add(s2, frets[s2], t0 + 2 * period, t1 + 0.02)
            beat = b0
            while beat < b1 - 1e-6:
                strong = abs(beat - round(beat)) < 1e-6
                pool = tones if rng.random() < (0.75 if strong else 0.4) else scale
                cands = [p for p in range(melody_pitch - 4, melody_pitch + 5) if p % 12 in pool and 57 <= p <= 84]
                if cands:
                    melody_pitch = int(rng.choice(cands))
                step = 0.5 if rng.random() < 0.65 else 1.0
                place = _place(melody_pitch, (prev_pos or 0) + capo + 2, capo, rng)
                if place and rng.random() < 0.9:
                    h = lead + beat * period
                    add(place[0], place[1], h, h + step * period * rng.uniform(0.8, 1.6))
                    if rng.random() < 0.2:  # a chord tone under the melody
                        below = [p for p in range(melody_pitch - 9, melody_pitch - 2) if p % 12 in tones]
                        if below:
                            pl = _place(int(rng.choice(below)), place[1], capo, rng)
                            if pl and pl[0] != place[0]:
                                add(pl[0], pl[1], h, h + step * period * rng.uniform(0.8, 1.6))
                beat += step
    # the tracker's beats: jitter, sometimes half or double tempo; downbeats sometimes shifted or missing
    beats = lead + np.arange(0, total_beats + 1e-6) * period
    beats = beats + rng.normal(0, 0.01, len(beats))
    r = rng.random()
    if r < 0.1:
        beats = beats[::2]
    elif r < 0.2:
        beats = np.sort(np.r_[beats, (beats[:-1] + beats[1:]) / 2])
    downs = lead + np.arange(0, total_beats + 1e-6, meter) * period
    r = rng.random()
    if r < 0.15:
        downs = downs + period
    elif r < 0.25:
        downs = downs[:0]
    chords = [(lead + a * period, lead + b * period, k) for a, b, k in spans]
    return {"notes": sorted(notes, key=lambda x: x["onset"]), "beats": beats, "downbeats": downs, "chords": chords,
            "capo": capo}


def degrade(notes, rng, strength=1.0):
    """The tab model's typical errors, at song-level rates around those measured on unseen recordings (EXPERIMENTS.md,
    C1: on fingerstyle covers it finds 81% of single notes but 50% of notes in chords of 4+, and 40% of its notes are
    extra: a ringing note struck again, unrelated pitches, a semitone or two off, notes in silence): missed notes (more
    in dense chords), split notes, extra notes, wrong pitches, note lengths from very short to doubled, onset jitter.
    strength scales the error rates (0: none)."""
    if not notes or strength <= 0:
        return list(notes)
    u = lambda lo, hi: rng.uniform(lo, hi) * strength  # noqa: E731
    miss, dense, split = u(0.05, 0.2), u(0.03, 0.12), u(0.03, 0.15)
    extra, wrong, octave, stray = u(0.02, 0.12), u(0.01, 0.07), u(0, 0.03), u(0, 0.08)
    onsets = np.array([n["onset"] for n in notes])
    out = []
    for n in notes:
        together = int(np.sum(np.abs(onsets - n["onset"]) < 0.03))
        if rng.random() < min(miss + dense * (together - 1), 0.75):
            continue
        on = n["onset"] + rng.normal(0, 0.015)
        r = rng.random()
        ratio = rng.uniform(0.03, 0.25) if r < 0.15 * strength else rng.uniform(1.05, 2.0) if r > 1 - 0.15 * strength \
            else rng.uniform(0.6, 1.05)
        off = on + max((n["offset"] - n["onset"]) * ratio, 0.04)
        m = dict(n, onset=on, offset=off)
        if rng.random() < wrong:
            m["pitch"] += int(rng.choice([-2, -1, 1, 2]))
            m["fret"] = None
        out.append(m)
        if rng.random() < split and off - on > 0.2:  # a ringing note struck again
            cut = on + (off - on) * rng.uniform(0.2, 0.8)
            m["offset"] = cut
            out.append(dict(m, onset=cut, offset=off))
        if rng.random() < extra:
            p = n["pitch"] + int(rng.integers(-12, 13))
            out.append({"onset": on + rng.normal(0, 0.03), "offset": on + (off - on) * rng.uniform(0.1, 0.8), "pitch": p,
                        "fret": None})
        if rng.random() < octave:
            out.append({"onset": on, "offset": on + (off - on) * rng.uniform(0.2, 0.8),
                        "pitch": n["pitch"] + int(rng.choice([12, -12, 7, -7, 19])), "fret": None})
    if stray > 0 and len(notes) > 1:  # notes where nothing sounds
        span = (min(onsets), max(n["offset"] for n in notes))
        for _ in range(rng.poisson(stray * len(notes))):
            t = rng.uniform(*span)
            out.append({"onset": t, "offset": t + rng.uniform(0.05, 0.5), "pitch": int(rng.integers(40, 80)), "fret": None})
    return sorted(out, key=lambda x: x["onset"])


# ---------- tabs with chord names (custom dataset) ----------

def tab_song(tab, seconds_per_quarter=0.6):
    """{"notes", "beats", "downbeats", "chords", "capo"} from a custom-dataset tab (JSON): notes in score time (ties
    joined; a note rings until the next note on its string, at most two quarters past its written length), chords from
    the chord names written in the tab (as capo shapes or as sounding chords, whichever fits the notes better)."""
    track = tab["track"]
    capo = track.get("capo", 0) or 0
    notes, names, downs, last_on = [], [], [], {}
    t = 0.0
    for measure in track["measures"]:
        downs.append(t)
        length = 0.0
        for v, voice in enumerate(measure["voices"]):
            pos = t
            for beat in voice["beats"]:
                dur = 4.0 * beat["duration"][0] / beat["duration"][1] if "duration" in beat else 1.0
                if v == 0 and beat.get("chord", {}).get("text"):
                    names.append((pos, beat["chord"]["text"]))
                for note in beat.get("notes", []):
                    if "string" not in note or note.get("rest") or note.get("dead"):
                        continue
                    s = 5 - note["string"]  # tab string 0 = high E
                    if note.get("tie") and s in last_on:
                        last_on[s]["end"] = pos + dur
                        continue
                    n = {"onset": pos, "end": pos + dur, "pitch": OPEN[s] + note["fret"] + capo, "fret": note["fret"] + capo,
                         "string": s}
                    notes.append(n)
                    last_on[s] = n
                pos += dur
            if v == 0:
                length = pos - t
        t += length or 4.0
    by_string = {}
    for n in sorted(notes, key=lambda n: n["onset"]):
        by_string.setdefault(n["string"], []).append(n)
    for seq in by_string.values():
        for a, b in zip(seq, seq[1:] + [None]):
            limit = a["end"] + 2.0
            a["offset"] = min(b["onset"], limit) if b else limit
    # chord names as capo shapes or as sounding chords
    parsed = [(p, parse_chord(x)) for p, x in names]
    fit = {0: 0, capo: 0}
    for p, k in parsed:
        if not 0 <= k < NO_CHORD:
            continue
        sounding = {n["pitch"] % 12 for n in notes if abs(n["onset"] - p) < 1e-6}
        for shift in fit:
            tones = {(k // NQ + shift + j) % 12 for j in TONES[QUALITIES[k % NQ]]}
            fit[shift] += len(sounding & tones) - len(sounding - tones)
    shift = capo if capo and fit[capo] > fit[0] else 0
    chords = []
    for (p, k), nxt in zip(parsed, parsed[1:] + [(t, None)]):
        if nxt[0] <= p:
            continue
        k = k if k in (NO_CHORD, IGNORE) else (k // NQ + shift) % 12 * NQ + k % NQ
        chords.append((p * seconds_per_quarter, nxt[0] * seconds_per_quarter, k))
    sec = seconds_per_quarter
    return {"notes": [{"onset": n["onset"] * sec, "offset": n["offset"] * sec, "pitch": n["pitch"], "fret": n["fret"]}
                      for n in notes],
            "beats": np.arange(0, t + 1e-6) * sec, "downbeats": np.array(downs) * sec, "chords": chords, "capo": capo,
            "outside": IGNORE}  # no chord name yet: unknown, not "no chord"


def custom_tab_songs(exclude=(), min_chords=10):
    """Songs from the custom-dataset tabs with at least min_chords chord names (ids in exclude left out)."""
    out = []
    for path in sorted((DATA / "custom" / "tabs").glob("*.json")):
        if path.stem in exclude:
            continue
        tab = json.loads(path.read_text())
        named = sum(bool(b.get("chord", {}).get("text")) for m in tab["track"]["measures"] for v in m["voices"][:1]
                    for b in v["beats"])
        if named < min_chords:
            continue
        song = tab_song(tab)
        if song["chords"] and song["notes"]:
            song["name"] = path.stem
            out.append(song)
    return out


# ---------- examples ----------

def example(song, rng=None, strength=0.0):
    """(pc features, global features, labels) for a song, after degrade(strength) when rng is given."""
    notes = degrade(song["notes"], rng, strength) if rng is not None and strength > 0 else song["notes"]
    if not notes:
        notes = song["notes"]
    end = max([n["offset"] for n in notes] + [c[1] for c in song["chords"]])
    bounds = segment_bounds(song["beats"], end)
    pcf, glob = features(notes, bounds, song["beats"], song["downbeats"], song.get("capo", 0))
    return pcf, glob, labels_at(bounds, song["chords"], song.get("outside", NO_CHORD))
