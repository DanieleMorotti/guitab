"""Turn note events in seconds into a quantized Guitar Pro tab, following the detected beats.

Back end of the transcriber:
  1. beat tracking (Beat This!) gives beat and downbeat times;
  2. every time is mapped to a fractional beat index by interpolating between beats, so the
     tab follows tempo drift instead of a single fixed tempo; tick 0 is put on a downbeat;
  3. onsets/offsets are snapped to a 16th (or 32nd) grid;
  4. notes without a string get one (distinct strings per chord), then the Viterbi
     fingering optimiser picks playable positions;
  5. tokens are written with guitarset_to_tokens.build_tokens and decoded to .gp5.
"""

import numpy as np

from fingering import OPEN_PITCH, refinger_tokens
from guitarset_to_tokens import TICKS_PER_BEAT, build_tokens


def detect_beats(audio_path, device="cpu"):
    """(beats, downbeats) in seconds from Beat This!."""
    from pathlib import Path

    from beat_this.inference import File2Beats

    ckpt = Path(__file__).resolve().parents[1] / "data" / "models" / "beat_this-final0.ckpt"
    file2beats = File2Beats(checkpoint_path=str(ckpt) if ckpt.exists() else "final0", device=device, dbn=False)
    beats, downbeats = file2beats(str(audio_path))
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def beats_per_measure(beats, downbeats):
    if len(downbeats) < 2:
        return 4
    idx = np.searchsorted(beats, downbeats - 1e-3)
    counts = np.bincount(np.diff(idx), minlength=8)[:8]
    mode = int(np.argmax(counts))
    # trust the detected meter only when one bar length clearly dominates
    return mode if 2 <= mode <= 7 and counts[mode] >= 0.6 * counts.sum() else 4


class BeatGrid:
    """Maps seconds to ticks through the beat sequence (linear extrapolation outside it)."""

    def __init__(self, beats, downbeats=None, tempo=None):
        beats = np.asarray(beats, dtype=float)
        if len(beats) < 2:  # constant tempo grid
            period = 60.0 / (tempo or 120)
            beats = np.arange(0, 3600, period)
        self.beats = beats
        self.period = float(np.median(np.diff(beats)))
        self.tempo = int(round(60.0 / self.period))
        self.num_beats = beats_per_measure(beats, downbeats) if downbeats is not None else 4
        # beat index of the first downbeat, so tick 0 falls on a barline
        first_down = 0
        if downbeats is not None and len(downbeats):
            first_down = int(np.argmin(np.abs(beats - downbeats[0])))
        self.origin = first_down % self.num_beats

    def beat_index(self, t):
        b = self.beats
        idx = np.interp(t, b, np.arange(len(b)))
        if t < b[0]:
            idx = (t - b[0]) / self.period
        elif t > b[-1]:
            idx = len(b) - 1 + (t - b[-1]) / self.period
        return idx - self.origin

    def ticks(self, t, grid):
        return int(round(self.beat_index(t) * TICKS_PER_BEAT / grid)) * grid


def assign_strings(group):
    """Give notes without a string a provisional distinct string (lowest playable first)."""
    used = {n["string"] for n in group if n.get("string")}
    for n in sorted(group, key=lambda n: n["pitch"]):
        if n.get("string"):
            continue
        free = [s for s in sorted(OPEN_PITCH, reverse=True) if s not in used and 0 <= n["pitch"] - OPEN_PITCH[s] <= 20]
        if not free:
            n["string"] = None
            continue
        n["string"] = min(free, key=lambda s: n["pitch"] - OPEN_PITCH[s])
        used.add(n["string"])
    return [n for n in group if n["string"]]


def notes_to_tokens(notes, grid_obj, name="song", grid=240, fingering=None, max_notes=6):
    """notes: dicts with onset, offset (s), pitch, optional string, optional score (for pruning)."""
    groups = {}
    for n in notes:
        on = grid_obj.ticks(n["onset"], grid)
        off = max(on + grid, grid_obj.ticks(n["offset"], grid))
        groups.setdefault(on, []).append({**n, "on": on, "off": off})
    events = []
    for on in sorted(groups):
        group = groups[on]
        # duplicate pitches at one onset: keep the stronger
        best = {}
        for n in group:
            if n["pitch"] not in best or n.get("score", 0) > best[n["pitch"]].get("score", 0):
                best[n["pitch"]] = n
        group = sorted(best.values(), key=lambda n: -n.get("score", 0))[:max_notes]
        # a string can sound once per onset
        seen, kept = set(), []
        for n in group:
            if n.get("string") in seen:
                n = {**n, "string": None}
            if n.get("string"):
                seen.add(n["string"])
            kept.append(n)
        for n in assign_strings(kept):
            events.append(n)
    if not events:
        return None
    shift = min(n["on"] for n in events)
    measure = grid_obj.num_beats * TICKS_PER_BEAT
    shift = (shift // measure) * measure  # keep barlines, drop leading empty measures
    tab_notes = [(n["on"] - shift, n["off"] - shift, n["string"], n["pitch"] - OPEN_PITCH[n["string"]]) for n in events]
    tokens = build_tokens(name, tab_notes, grid_obj.tempo, grid_obj.num_beats)
    if fingering is not None:
        tokens = refinger_tokens(tokens, fingering)
    return tokens
