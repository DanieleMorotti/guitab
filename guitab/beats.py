"""Beats and downbeats for the tab grid (B1).

The Beat This! beat activation plus the tab model's summed onset probabilities is tracked by dynamic programming (Ellis
2007, the algorithm of librosa.beat.beat_track, reimplemented so that the browser version, docs/js/beats.js, finds the
same beats): the beats stay regular but follow the player when the tempo changes. Downbeats: the bar length (2, 3 or 4
beats) and phase whose beats have the highest mean downbeat activation, one metre per song as the tab writes it.
"""

import numpy as np

from .audio import FPS, HOP, SR

BEAT_FPS = 50  # Beat This! frame rate
TEMPO_WIN = 689  # 8 s tempo windows (as librosa.feature.tempo at 22.05 kHz, hop 256)
TEMPO_HOP = 16  # one window every 16 frames (librosa: every frame)
TINY = np.finfo(np.float64).tiny


def _seq_sum(x):
    """Sum in index order (numpy's own sums are pairwise; the browser adds in order)."""
    return float(np.add.accumulate(np.asarray(x, dtype=np.float64))[-1]) if len(x) else 0.0


def _round(x, step):
    return np.floor(np.asarray(x, dtype=np.float64) / step + 0.5) * step


def activation(logits, n):
    """Beat This! logits (50 fps, already rounded to 0.01) -> probabilities at the tab model's n frames: linear
    interpolation, rounded to 1e-6 (the browser's exp can differ in the last bit)."""
    p = _round(1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64))), 1e-6)
    pos = np.arange(n) * HOP / SR * BEAT_FPS
    j = np.minimum(np.floor(pos).astype(int), len(p) - 1)
    frac = pos - j
    nxt = p[np.minimum(j + 1, len(p) - 1)]
    return _round(p[j] * (1 - frac) + nxt * frac, 1e-6)


def envelope(beat_act, onset=None):
    """Beat activation plus the summed onset probabilities, each scaled to a maximum of 1, rounded to 0.01 (the
    browser's tab probabilities differ from Python's by ~1e-6)."""
    env = np.asarray(beat_act, dtype=np.float64) / (np.max(beat_act) + 1e-9)
    if onset is not None:
        o = np.asarray(onset, dtype=np.float64)
        total = o[:, 0].copy()
        for c in range(1, o.shape[1]):
            total = total + o[:, c]
        env = env + total / (total.max() + 1e-9)
    return _round(env, 0.01)


def beat_period(env):
    """Beat period in frames: mean normalised autocorrelation of Hann-windowed 8 s windows (every TEMPO_HOP frames)
    with a log-normal prior around 120 BPM (std 1 octave), tempi up to 320 BPM (librosa.feature.tempo)."""
    n, w = len(env), TEMPO_WIN
    pad = w // 2
    padded = np.concatenate([env[0] * np.arange(pad) / pad, env, env[-1] * np.arange(pad, 0, -1) / pad])
    win = _round(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(w) / w), 1e-12)
    frames = np.stack([padded[t:t + w] * win for t in range(0, n, TEMPO_HOP)])
    spec = np.abs(np.fft.rfft(frames, n=2 * w, axis=1)) ** 2
    ac = np.fft.irfft(spec, n=2 * w, axis=1)[:, :w]
    ac = np.where(ac[:, :1] > TINY, ac / np.maximum(ac[:, :1], TINY), 0.0)
    tg = _round(ac.mean(0), 1e-6)
    lag = np.arange(1, w)
    bpm = 60 * FPS / lag
    prior = _round(-0.5 * (np.log2(bpm) - np.log2(120.0)) ** 2, 1e-9)
    score = _round(np.log1p(1e6 * tg[1:]) + prior, 1e-9)
    score[bpm >= 320] = -np.inf
    return int(lag[np.argmax(score)])


def dp_beats(env, period, tightness=100):
    """Beat frames by dynamic programming (Ellis 2007, as librosa.beat.beat_track with trim=False)."""
    env = np.asarray(env, dtype=np.float64)
    n = len(env)
    mean = _seq_sum(env) / n
    std = np.sqrt(_seq_sum((env - mean) ** 2) / (n - 1))
    onsets = env / (std + TINY)
    k = period
    window = _round(np.exp(-0.5 * (np.arange(-k, k + 1) * 32.0 / k) ** 2), 1e-12)
    K, half = 2 * k + 1, k
    local = np.zeros(n)
    for j in range(K):  # local[i] = sum_j window[j] * onsets[i + half - j], j in index order
        lo, hi = max(0, j - half), min(n, n + j - half)
        local[lo:hi] += window[j] * onsets[lo + half - j:hi + half - j]
    first, last_d = int(np.floor(k / 2 + 0.5)) if k % 2 else k // 2, 2 * k
    penalty = _round(tightness * (np.log(np.arange(1, last_d + 1)) - np.log(k)) ** 2, 1e-9)
    thresh = 0.01 * local.max()
    cum, back = np.zeros(n), np.full(n, -1)
    first_beat = True
    for i in range(n):
        lo = max(i - last_d, 0)
        hi = i - first  # candidates loc = hi, hi - 1, ..., lo (the first best in that order wins)
        best_loc = -1
        if hi >= 0 and hi >= lo:
            locs = np.arange(hi, lo - 1, -1)
            scores = cum[locs] - penalty[i - locs - 1]
            b = int(np.argmax(scores))
            best_loc = int(locs[b])
            cum[i] = local[i] + scores[b]
        else:
            cum[i] = local[i]
        if first_beat and local[i] < thresh:
            back[i] = -1
        else:
            back[i] = best_loc
            first_beat = False
    padded = np.concatenate([cum[:1], cum, cum[-1:]])
    peak = (cum > padded[:-2]) & (cum >= padded[2:])
    vals = np.sort(cum[peak])
    m = len(vals)
    median = (vals[(m - 1) // 2] + vals[m // 2]) / 2 if m else 0.0
    tail = n - 1
    for t in range(n - 1, -1, -1):
        if peak[t] and cum[t] >= 0.5 * median:
            tail = t
            break
    beats = np.zeros(n, dtype=bool)
    t = tail
    while t >= 0:
        beats[t] = True
        t = back[t]
    t = 0
    while t < n and local[t] <= 0:
        beats[t] = False
        t += 1
    t = n - 1
    while t >= 0 and local[t] <= 0:
        beats[t] = False
        t -= 1
    return np.flatnonzero(beats)


def track_beats(beat_act, onset=None, tightness=100):
    """Beat times (s): dynamic programming (dp_beats) on envelope(beat_act, onset), at the period of beat_period."""
    env = envelope(beat_act, onset)
    if not env.any():
        return np.array([])
    return dp_beats(env, beat_period(env), tightness) * HOP / SR


def pick_downbeats(beats, down_act, lengths=(2, 3, 4)):
    """Downbeats among the beats: the bar length and phase with the highest mean downbeat activation (one metre per
    song, as the tab's BeatGrid writes it; the first best in order of length, then phase)."""
    if len(beats) < 2:
        return beats[:1]
    frames = np.clip(np.round(np.asarray(beats) * FPS).astype(int), 0, len(down_act) - 1)
    act = np.asarray(down_act, dtype=np.float64)[frames]
    best, choice = -np.inf, (lengths[0], 0)
    for k in lengths:
        for p in range(min(k, len(act))):
            mean = _seq_sum(act[p::k]) / len(act[p::k])
            if mean > best:
                best, choice = mean, (k, p)
    return np.asarray(beats)[choice[1]::choice[0]]
