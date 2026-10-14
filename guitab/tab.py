"""Turn note events in seconds into a quantized Guitar Pro tab, following the detected beats.

Back end of the transcriber:
  1. beat tracking (Beat This!) gives beat and downbeat times;
  2. every time is mapped to a fractional beat index by interpolating between beats, so the
     tab follows tempo drift instead of a single fixed tempo; tick 0 is put on a downbeat;
  3. onsets/offsets are snapped to a 16th (or 32nd) grid;
  4. notes without a string get one (distinct strings per chord), then the Viterbi
     fingering optimiser picks playable positions;
  5. tokens are written in the DadaGP format (build_tokens) and converted to .gp5 (decode_to_gp5).
With a capo (D6), tokens hold frets relative to the capo and the .gp5 track has its capo set; notes must already be
playable with it (inference.apply_capo).

Tab tokens: 960 ticks per quarter; notes still sounding at a barline are re-emitted with nfx:tie, and
silent gaps of at least a quarter become rests.
"""

import tempfile

import numpy as np

from .fingering import refinger_tokens
from .guitar import OPEN_PITCH
from .paths import BEAT_THIS_CKPT

TICKS_PER_BEAT = 960  # DadaGP convention
GRID = 120  # 32nd note: finest grid the token writer supports
MIN_REST_TICKS = 960
# Note values DadaGP can encode on this grid (subset of dadagp.supported_times).
SUPPORTED_WAITS = [3840, 2880, 1920, 1440, 960, 720, 480, 360, 240, 120]


def detect_beats(audio_path, device="cpu"):
    """(beats, downbeats) in seconds from Beat This! (local checkpoint if present, else downloaded)."""
    from beat_this.inference import File2Beats

    ckpt = BEAT_THIS_CKPT
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


def assign_strings(group, capo=0):
    """Give notes without a string a provisional distinct string (lowest playable first)."""
    used = {n["string"] for n in group if n.get("string")}
    for n in sorted(group, key=lambda n: n["pitch"]):
        if n.get("string"):
            continue
        free = [s for s in sorted(OPEN_PITCH, reverse=True) if s not in used and capo <= n["pitch"] - OPEN_PITCH[s] <= 20]
        if not free:
            n["string"] = None
            continue
        n["string"] = min(free, key=lambda s: n["pitch"] - OPEN_PITCH[s])
        used.add(n["string"])
    return [n for n in group if n["string"]]


def notes_to_tokens(notes, grid_obj, name="song", grid=240, fingering=None, max_notes=6, capo=0):
    """notes: dicts with onset, offset (s), pitch, optional string, optional score (for pruning).
    capo: tokens get frets relative to it (refingering then works in the capo'd tuning)."""
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
        for n in assign_strings(kept, capo):
            events.append(n)
    if not events:
        return None
    shift = min(n["on"] for n in events)
    measure = grid_obj.num_beats * TICKS_PER_BEAT
    shift = (shift // measure) * measure  # keep barlines, drop leading empty measures
    tab_notes = [(n["on"] - shift, n["off"] - shift, n["string"], n["pitch"] - OPEN_PITCH[n["string"]] - capo) for n in events]
    tokens = build_tokens(name, tab_notes, grid_obj.tempo, grid_obj.num_beats)
    if fingering is not None:
        tokens = refinger_tokens(tokens, fingering)
    return tokens


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


def decode_to_gp5(tokens, gp5_path, capo=0):
    from .dadagp_utils import DadaGP

    restored = [f"clean0:{t}" if t.startswith("note:") or t == "rest" else t for t in tokens]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write("\n".join(restored) + "\n")
        name = handle.name
    DadaGP.decode(name, str(gp5_path))
    if capo:
        import guitarpro

        song = guitarpro.parse(str(gp5_path))
        for track in song.tracks:
            track.offset = capo  # the Guitar Pro capo
        guitarpro.write(song, str(gp5_path))
