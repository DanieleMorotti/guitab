"""Re-choose string/fret for predicted notes so the tab is playable the way a guitarist would play it.

The model's pitches are kept; only the fingering changes. Each onset group (single note or chord)
gets candidate fingerings (distinct strings, limited fret span, open strings allowed), and a
Viterbi pass picks the sequence minimising:
    position cost  - prefer low frets / open strings
    span cost      - prefer compact shapes
    movement cost  - prefer small hand shifts between consecutive groups
    model bonus    - small preference for the fingering the model itself predicted
Tied notes keep the string of the note they continue.
"""

import itertools
import re

OPEN_PITCH = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
NOTE = re.compile(r"^note:s([1-6]):f(\d+)$")

DEFAULT_WEIGHTS = {
    "position": 0.25,   # per fret of the hand position (lowest fretted fret)
    "high": 1.0,        # extra per note above fret 12
    "span": 0.6,        # per fret of stretch inside a shape
    "move": 0.35,       # per fret of hand shift between groups
    "model": 0.8,       # bonus per note matching the model's own string choice
    "max_fret": 20,
    "max_span": 4,
}


def parse_groups(tokens):
    """Split tokens into items; each note group is a dict with note positions and their indices."""
    groups, current = [], None
    for i, t in enumerate(tokens):
        m = NOTE.match(t)
        if m:
            if current is None:
                current = {"notes": []}
                groups.append(current)
            tied = i + 1 < len(tokens) and tokens[i + 1] == "nfx:tie"
            s, f = int(m.group(1)), int(m.group(2))
            current["notes"].append({"idx": i, "string": s, "fret": f, "pitch": OPEN_PITCH[s] + f, "tie": tied})
        elif t.startswith(("wait:", "rest", "new_measure")):
            current = None
    return groups


def options_for_pitch(pitch, max_fret):
    return [(s, pitch - p) for s, p in OPEN_PITCH.items() if 0 <= pitch - p <= max_fret]


def hand_position(fingering):
    fretted = [f for _, f in fingering if f > 0]
    return min(fretted) if fretted else None


def candidates(notes, w, limit=40):
    """Playable fingerings for the non-tied notes of a group, cheapest (static cost) first."""
    per_note = [options_for_pitch(n["pitch"], w["max_fret"]) for n in notes]
    if any(not o for o in per_note):
        return [tuple((n["string"], n["fret"]) for n in notes)]
    combos = []
    for combo in itertools.product(*per_note):
        strings = [s for s, _ in combo]
        if len(set(strings)) != len(strings):
            continue
        fretted = [f for _, f in combo if f > 0]
        span = (max(fretted) - min(fretted)) if fretted else 0
        if span > w["max_span"]:
            continue
        combos.append(combo)
    if not combos:  # unplayable as predicted (e.g. 7 notes); keep the model's fingering
        return [tuple((n["string"], n["fret"]) for n in notes)]
    combos.sort(key=lambda c: static_cost(c, notes, w))
    return combos[:limit]


def static_cost(combo, notes, w):
    fretted = [f for _, f in combo if f > 0]
    pos = min(fretted) if fretted else 0
    span = (max(fretted) - min(fretted)) if fretted else 0
    high = sum(1 for f in fretted if f > 12)
    match = sum(1 for (s, _), n in zip(combo, notes) if s == n["string"])
    return w["position"] * pos + w["span"] * span + w["high"] * high - w["model"] * match


def refinger_tokens(tokens, weights=None):
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    tokens = list(tokens)
    groups = parse_groups(tokens)
    # Non-tied notes are optimised; tied notes follow the string their pitch was last played on.
    free = [[n for n in g["notes"] if not n["tie"]] for g in groups]
    steps = [(g, nf, candidates(nf, w)) for g, nf in zip(groups, free) if nf]
    if not steps:
        return tokens

    # Viterbi over candidate fingerings.
    prev_cost = [static_cost(c, steps[0][1], w) for c in steps[0][2]]
    back = []
    last_pos = [hand_position(c) for c in steps[0][2]]
    for _, nf, cands in steps[1:]:
        costs, ptrs, poss = [], [], []
        for c in cands:
            pos = hand_position(c)
            best, arg = None, 0
            for j, pc in enumerate(prev_cost):
                move = 0.0 if pos is None or last_pos[j] is None else w["move"] * abs(pos - last_pos[j])
                total = pc + move
                if best is None or total < best:
                    best, arg = total, j
            costs.append(best + static_cost(c, nf, w))
            ptrs.append(arg)
            poss.append(pos if pos is not None else last_pos[arg])
        back.append(ptrs)
        prev_cost, last_pos = costs, poss

    choice = [min(range(len(prev_cost)), key=prev_cost.__getitem__)]
    for ptrs in reversed(back):
        choice.append(ptrs[choice[-1]])
    choice.reverse()

    chosen = {}
    for (g, nf, cands), k in zip(steps, choice):
        for n, (s, f) in zip(nf, cands[k]):
            chosen[n["idx"]] = (s, f)

    last_string_for_pitch = {}
    for g in groups:
        used = set()
        for n in g["notes"]:
            if n["idx"] in chosen:
                s, f = chosen[n["idx"]]
            else:  # tied note: continue on the string this pitch was last assigned to
                s = last_string_for_pitch.get(n["pitch"], n["string"])
                f = n["pitch"] - OPEN_PITCH[s]
                if s in used or not 0 <= f <= 24:
                    s, f = n["string"], n["fret"]
            used.add(s)
            last_string_for_pitch[n["pitch"]] = s
            tokens[n["idx"]] = f"note:s{s}:f{f}"
    return tokens


def refinger_text(text, weights=None):
    return " ".join(refinger_tokens(text.split(), weights))
