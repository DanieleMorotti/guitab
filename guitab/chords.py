"""Chord suggestions from the transcribed notes: a small learned model on the beat grid.

The tab model already hears the notes, so the chord model only maps notes to harmony. Per eighth note of the beat grid
(segments), the notes give pitch-class features: sounding time, onsets, bass, open strings, low register and top note
(features). The model (ChordNet) looks at them relative to each of the 12 roots with shared weights, so it learns
harmony once for all keys: per root, a small MLP and a bidirectional GRU over the segments score the chord qualities
(QUALITIES), and a no-chord score comes from all roots together. Decoding (decode) is a Viterbi over the segments with a
cost per change, whose moves are scored by Chordonomicon's chord-to-chord statistics (chord_bigram.json); a chord is
shown when the model is confident enough (CONFIDENCE). Training: scripts/train_chords.py, data: guitab/chord_data.py.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

QUALITIES = ["maj", "min", "7", "maj7", "min7", "hdim7", "dim", "sus2", "sus4", "5"]
TONES = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
         "hdim7": (0, 3, 6, 10), "dim": (0, 3, 6), "sus2": (0, 2, 7), "sus4": (0, 5, 7), "5": (0, 7)}
SUFFIX = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "hdim7": "m7b5", "dim": "dim", "sus2": "sus2",
          "sus4": "sus4", "5": "5"}
NQ = len(QUALITIES)
N_CHORDS = 12 * NQ  # class = root * NQ + quality; class N_CHORDS = no chord
NO_CHORD = N_CHORDS
IGNORE = -100  # a label outside the vocabulary (aug, unparsable): no loss, not scored
ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]  # the usual guitar spellings
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11, "H": 11}  # H: German B
SUBDIVIDE = 2  # segments per beat (eighth notes)
N_PC_FEATURES = 6
N_GLOBAL = 6
LOW_PITCH = 55  # below G3: bass register
CHANGE = 2.0  # decoding: cost of a chord change (log-probability units)
PROGRESSION = 0.25  # weight of the Chordonomicon move statistics
CONFIDENCE = 0.5  # a chord is shown when its family's mean probability over its span is at least this (chord_spans)
WEIGHTS_NAME = "guitab_chords.safetensors"


def chord_class(root, quality):
    return root * NQ + QUALITIES.index(quality)


def chord_name(index):
    return "N.C." if index == NO_CHORD else ROOT_NAMES[index // NQ] + SUFFIX[QUALITIES[index % NQ]]


def _quality(q):
    """Our quality for the text after the root (chord-sheet or Chordonomicon spelling), None if outside."""
    q = q.strip().replace("(", "").replace(")", "")
    if re.match(r"^(no3|5$)", q):
        return "5"
    if re.search(r"(m7b5|m7-5|min7b5|ø|hdim)", q):
        return "hdim7"
    if re.match(r"^(dim|°|o$|o7)", q):
        return "dim"
    if re.match(r"^(aug|\+)", q):
        return None
    if re.match(r"^(maj7|maj9|maj11|maj13|M7|M9|7M|Δ|ma7)", q):
        return "maj7"
    if re.match(r"^(m|min|-)(?!aj)", q):
        return "min7" if re.match(r"^(m|min|-)(7|9|11|13)", q) else "min"
    if re.match(r"^(7sus|9sus|sus4|sus$|4$)", q):
        return "sus4"
    if re.match(r"^(sus2|2$)", q):
        return "sus2"
    if re.match(r"^(7|9|11|13)", q):
        return "7"
    if q == "" or re.match(r"^(maj|M|6|add|2|4|b|#)", q):
        return "maj"
    return None


def parse_chord(text):
    """Chord class of a chord name, NO_CHORD for N / N.C., IGNORE if outside the vocabulary. Accepts Harte labels
    ("C:maj7", "D:(1,5)", "A:sus2(b7,*5)"), chord-sheet names ("F#m7(b5)", "H7", "F7M", "Dsus4/F#") and Chordonomicon
    tokens ("Fs7", "Amin", "Cno3d")."""
    text = text.strip()
    if text in ("N", "N.C.", "NC", "X", ""):
        return NO_CHORD
    if ":" in text:  # Harte
        root, rest = text.split(":", 1)
        base, _, extra = rest.split("/")[0].partition("(")
        if "*3" in extra and base in ("", "maj", "min", "7", "min7", "maj7"):
            base = "5"
        harte = {"maj": "maj", "min": "min", "7": "7", "maj7": "maj7", "min7": "min7", "hdim7": "hdim7", "dim": "dim",
                 "dim7": "dim", "sus2": "sus2", "sus4": "sus4", "maj6": "maj", "min6": "min", "9": "7", "maj9": "maj7",
                 "min9": "min7", "min11": "min7", "11": "7", "13": "7", "minmaj7": "min", "5": "5"}
        q = harte.get(base, "5" if base == "" and extra else None)
    else:
        m = re.match(r"^([A-H])(#|b|s(?!us))?(.*)$", text)
        if not m:
            return IGNORE
        root = m.group(1) + {"s": "#"}.get(m.group(2), m.group(2) or "")
        q = _quality(m.group(3).split("/")[0])
    m = re.match(r"^([A-H])([#b]*)$", root)
    if not m or q is None:
        return IGNORE
    return chord_class((PC[m.group(1)] + m.group(2).count("#") - m.group(2).count("b")) % 12, q)


def triad(index):
    """'maj' / 'min' family of a chord class (majmin scoring), 'other' for the other qualities, None for no chord."""
    if index == NO_CHORD:
        return None
    q = QUALITIES[index % NQ]
    return "maj" if q in ("maj", "7", "maj7") else "min" if q in ("min", "min7") else "other"


def segment_bounds(beats, end, subdivide=SUBDIVIDE):
    """Segment edges: the beats split into `subdivide` equal parts, the grid extended with the median beat to cover
    [0, end]."""
    beats = np.asarray(beats, dtype=np.float64)
    beats = beats[beats < end]
    if len(beats) < 2:
        beats = np.arange(0, end + 0.5, 0.5)
    period = float(np.median(np.diff(beats)))
    before = np.arange(beats[0] - period, -period, -period)[::-1]
    after = np.arange(beats[-1] + period, end + period, period)
    grid = np.r_[before, beats, after]
    steps = np.arange(subdivide) / subdivide
    edges = np.r_[(grid[:-1, None] + np.diff(grid)[:, None] * steps).ravel(), grid[-1]]
    edges = edges[(edges > 0) & (edges < end)]
    return np.r_[0.0, edges, end]


def features(notes, bounds, beats=(), downbeats=(), capo=0):
    """Per segment: pitch-class features [S, N_PC_FEATURES, 12] and global features [S, N_GLOBAL] (float32).
    pc channels: sounding share, onsets, bass (lowest note sounding >= 30% of the segment), open-string sounding share,
    low-register sounding share, top note. Global: notes sounding, onsets, starts on a beat, starts on a downbeat, has
    notes, segment length. notes: dicts with onset, offset, pitch and optionally fret (absolute; open = fret == capo)."""
    n = len(bounds) - 1
    pcf = np.zeros((n, N_PC_FEATURES, 12), dtype=np.float32)
    glob = np.zeros((n, N_GLOBAL), dtype=np.float32)
    low, high = np.full(n, 999), np.full(n, -1)
    length = np.diff(bounds)
    for note in notes:
        on, off, pitch = note["onset"], note["offset"], note["pitch"]
        pc = pitch % 12
        first = max(int(np.searchsorted(bounds, on, "right")) - 1, 0)
        last = min(int(np.searchsorted(bounds, off, "left")), n)
        if first < n and bounds[first] <= on < bounds[first + 1]:
            pcf[first, 1, pc] += 1
            glob[first, 1] += 1
        is_open = note.get("fret") == capo
        for i in range(first, min(max(last, first + 1), n)):
            share = (min(off, bounds[i + 1]) - max(on, bounds[i])) / length[i]
            if share <= 0:
                continue
            pcf[i, 0, pc] += share
            glob[i, 0] += share
            if is_open:
                pcf[i, 3, pc] += share
            if pitch < LOW_PITCH:
                pcf[i, 4, pc] += share
            if share >= 0.3:
                low[i], high[i] = min(low[i], pitch), max(high[i], pitch)
    has = low < 999
    rows = np.flatnonzero(has)
    pcf[rows, 2, low[has] % 12] = 1
    pcf[rows, 5, high[has] % 12] = 1
    for c in (0, 1, 3, 4):
        pcf[:, c] = np.minimum(pcf[:, c], 2) / 2
    starts = bounds[:-1]
    for col, times in ((2, beats), (3, downbeats)):
        times = np.asarray(times, dtype=np.float64)
        if len(times) > 1:
            j = np.clip(np.searchsorted(times, starts), 1, len(times) - 1)
            glob[:, col] = np.minimum(np.abs(times[j] - starts), np.abs(times[j - 1] - starts)) < 0.03
    glob[:, 0] = np.minimum(glob[:, 0], 6) / 6
    glob[:, 1] = np.minimum(glob[:, 1], 6) / 6
    glob[:, 4] = has
    glob[:, 5] = np.minimum(length / 0.5, 2)
    return pcf, glob


def labels_at(bounds, chords, outside=NO_CHORD):
    """Chord class per segment from [(start s, end s, class)]: the chord at the segment's middle (outside if none)."""
    mid = (bounds[:-1] + bounds[1:]) / 2
    out = np.full(len(mid), outside, dtype=np.int64)
    for start, end, k in chords:
        out[(mid >= start) & (mid < end)] = k
    return out


class ChordNet(nn.Module):
    """Root-relative chord scorer: for each root, the features rotated to that root go through a shared MLP and a
    bidirectional GRU over the segments, then a linear layer scores the qualities; a no-chord logit comes from the
    mean over the roots. Output: logits [B, S, N_CHORDS + 1] (class root * NQ + quality, then no chord)."""

    def __init__(self, width=64, layers=2, dropout=0.1):
        super().__init__()
        self.config = {"width": width, "layers": layers}
        self.inp = nn.Sequential(nn.Linear(N_PC_FEATURES * 12 + N_GLOBAL, width), nn.GELU(), nn.Linear(width, width),
                                 nn.GELU())
        self.gru = nn.GRU(width, width, layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.quality = nn.Linear(2 * width, NQ)
        self.none = nn.Linear(2 * width, 1)
        self.register_buffer("rotation", (torch.arange(12)[:, None] + torch.arange(12)[None]) % 12, persistent=False)

    def forward(self, pcf, glob):
        b, s = pcf.shape[:2]
        x = pcf[:, :, :, self.rotation]  # [B, S, C, root, 12]: the pitch classes relative to each root
        x = x.permute(0, 1, 3, 2, 4).reshape(b, s, 12, N_PC_FEATURES * 12)
        x = torch.cat([x, glob[:, :, None].expand(b, s, 12, N_GLOBAL)], -1)
        h = self.inp(x).permute(0, 2, 1, 3).reshape(b * 12, s, -1)
        h, _ = self.gru(h)
        h = h.reshape(b, 12, s, -1).permute(0, 2, 1, 3)  # [B, S, root, 2W]
        return torch.cat([self.quality(h).reshape(b, s, N_CHORDS), self.none(h.mean(2))], -1)


def load_chord_model(path, device="cpu"):
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), "pt") as f:
        config = json.loads((f.metadata() or {}).get("config", "{}"))
    model = ChordNet(**config)
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval()


def _progression_scores():
    """[C+1, C+1] log(P(next | previous) * N_CHORDS) from Chordonomicon (0: as likely as uniform); no chord: 0."""
    data = json.loads((Path(__file__).parent / "chord_bigram.json").read_text())
    used = [data["qualities"].index(q) for q in QUALITIES]
    count = np.asarray(data["count"], dtype=np.float64)[np.ix_(used, range(12), used)] + 1
    p = count / count.sum((1, 2), keepdims=True)
    root, q = np.arange(N_CHORDS) // NQ, np.arange(N_CHORDS) % NQ
    out = np.zeros((N_CHORDS + 1, N_CHORDS + 1))
    out[:N_CHORDS, :N_CHORDS] = np.log(p[q[:, None], (root[None, :] - root[:, None]) % 12, q[None, :]] * N_CHORDS)
    return out


def decode(log_probs, change=CHANGE, progression=PROGRESSION):
    """Chord class per segment: Viterbi over log_probs [S, C+1] with a cost per change and Chordonomicon move scores."""
    n, c = log_probs.shape
    move = progression * _progression_scores() - change
    np.fill_diagonal(move, -np.inf)
    total, back = log_probs[0].copy(), np.zeros((n, c), dtype=np.int64)
    for i in range(1, n):
        cand = total[:, None] + move
        best_from = np.argmax(cand, 0)
        best = cand[best_from, np.arange(c)]
        stay = total >= best
        back[i] = np.where(stay, np.arange(c), best_from)
        total = np.where(stay, total, best) + log_probs[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(total))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


@torch.no_grad()
def chord_probs(model, notes, beats, downbeats=(), capo=0):
    """(segment bounds, probabilities [S, C+1]) for a song."""
    bounds = segment_bounds(beats, max(n["offset"] for n in notes))
    pcf, glob = features(notes, bounds, beats, downbeats, capo)
    device = next(model.parameters()).device
    logits = model(torch.from_numpy(pcf)[None].to(device), torch.from_numpy(glob)[None].to(device))[0]
    return bounds, torch.softmax(logits.float(), -1).cpu().numpy()


def _families():
    """Family of each class (same root and major / minor / other family, e.g. Am and Am7); no chord: its own."""
    fam = {"maj": 0, "min": 1, "other": 2}
    return np.array([k // NQ * 3 + fam[triad(k)] for k in range(N_CHORDS)] + [36])


def chord_spans(bounds, probs, change=CHANGE, progression=PROGRESSION, confidence=CONFIDENCE, merge=True):
    """[(start s, end s, class)] of the shown chords. The decoded path is cut where the chord changes; with merge, only
    where the family changes (root, major / minor), and a span is named by its family's most probable chord. A span is
    shown when its mean probability (of the whole family with merge) is >= confidence."""
    path = decode(np.log(np.maximum(probs, 1e-9)), change, progression)
    family = _families()
    key = family[path] if merge else path
    members = np.equal.outer(family, family) if merge else np.eye(len(family), dtype=bool)
    out, start = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or key[i] != key[start]:
            k = int(path[start])
            if k != NO_CHORD:
                same = members[k]
                if probs[start:i][:, same].sum(1).mean() >= confidence:
                    k = int(np.flatnonzero(same)[np.argmax(probs[start:i][:, same].sum(0))])
                    out.append((float(bounds[start]), float(bounds[i]), k))
            start = i
    return out


def chord_weights():
    """The chord model's weights: runs/chords/best.safetensors when trained here, else weights/; None if absent."""
    from .paths import RUNS, WEIGHTS

    return next((p for p in (RUNS / "chords" / "best.safetensors", WEIGHTS / WEIGHTS_NAME) if p.exists()), None)


def suggest_chords(notes, beats, downbeats=(), capo=0, model=None):
    """[(start s, end s, name)] for the parts where the chord model is confident; [] without a trained model.
    notes: dicts with onset, offset (s), pitch and fret (absolute)."""
    if not notes or len(beats) < 2:
        return []
    if model is None:
        path = chord_weights()
        if path is None:
            return []
        model = load_chord_model(path)
    bounds, probs = chord_probs(model, notes, beats, downbeats, capo)
    return [(s, e, chord_name(k)) for s, e, k in chord_spans(bounds, probs)]
