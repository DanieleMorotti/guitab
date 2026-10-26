"""Beats from the tab model itself (B1): a beat / downbeat head on the frozen tab model, and the beat decoding.

The tab model is not changed: its trunk runs as usual on 5 s windows (432 frames, hop 288), and in the same pass gives
features [T, 512] = the output of transformer layer 2 and the final normalised state (the tab / onset heads' input),
overlap-added like the tab predictions (predict_features). With a BeatBranch (B2), the first half is replaced by a
trainable copy of the tab model's last transformer layers, started from the state after layer 2. BeatHead, a dilated temporal convolutional network (kernel 3,
dilations 1..512, ±11.9 s of context) over the whole track's features, predicts beat and downbeat logits; it is trained
on Beat This! activations and real beat labels (scripts/train_beat_head.py).

Decoding: dynamic-programming beat tracking (Ellis 2007, librosa.beat.beat_track) on the beat activation plus the summed
onset probabilities, which gives regular beats that still follow tempo changes; downbeats: the bar length (2, 3 or 4
beats) and phase whose beats have the highest mean downbeat activation.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from .audio import FPS, HOP, SR
from .model import DilatedConvs

FEATURE_LAYERS = (1, -1)  # transformer layer 2 and the last one (normalised): 2 x d_model features per frame


class BeatBranch(nn.Module):
    """A trainable copy of the tab model's transformer layers after FEATURE_LAYERS[0] and of its final norm (B2): from
    the frozen state after layer 2 it computes rhythm-oriented features, while the tab model itself stays unchanged."""

    def __init__(self, model):
        super().__init__()
        self.layers = copy.deepcopy(model.encoder.layers[FEATURE_LAYERS[0] + 1:])
        self.norm = copy.deepcopy(model.norm)

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


def forward_features(model, x, branch=None, grad=False):
    """The tab model's forward pass (TabModel with a transformer) that also returns the head features [B, T, 2d]:
    [state after layer 2 (or the branch's output), final normalised state]. The tab model runs without gradients;
    grad=True lets them flow through the branch."""
    b, t, _ = x.shape
    with torch.no_grad():
        h = model.cnn(model.harmonic_stack(x))
        h = model.proj(h.permute(0, 2, 1, 3).reshape(b, t, -1)) + model.pos[:, :t]
        mid = None
        for i, layer in enumerate(model.encoder.layers):
            h = layer(h)
            if i == FEATURE_LAYERS[0]:
                mid = h
        h = model.norm(h)
    if branch is not None:
        with torch.set_grad_enabled(grad):
            mid = branch(mid)
    return {"tab": model.tab_head(h).view(b, t, 6, model.n_classes), "onset": model.onset_head(h),
            "features": torch.cat([mid, h], -1)}


@torch.no_grad()
def predict_features(model, feat, device, win=432, hop=288, batch=16, branch=None):
    """As inference.predict (same windows and taper, same tab / onset probabilities) plus the head features [T, 2d]."""
    t_total = len(feat)
    tab = np.zeros((t_total, 6, model.n_classes), dtype=np.float32)
    onset = np.zeros((t_total, 6), dtype=np.float32)
    feats = None
    weight = np.zeros((t_total, 1), dtype=np.float32)
    starts = list(range(0, max(t_total - win, 0) + 1, hop))
    if starts[-1] + win < t_total:
        starts.append(t_total - win)
    taper = np.hanning(win + 2)[1:-1].astype(np.float32)[:, None] + 1e-3
    for b in range(0, len(starts), batch):
        group = starts[b:b + batch]
        x = torch.from_numpy(np.stack([np.asarray(feat[s:s + win], dtype=np.float32) for s in group])).to(device)
        out = forward_features(model, x, branch)
        tab_b = torch.softmax(out["tab"].float(), -1).cpu().numpy()
        onset_b = torch.sigmoid(out["onset"].float()).cpu().numpy()
        f_b = out["features"].float().cpu().numpy()
        if feats is None:
            feats = np.zeros((t_total, f_b.shape[-1]), dtype=np.float32)
        n = x.shape[1]
        w = taper[:n]
        for i, s in enumerate(group):
            tab[s:s + n] += tab_b[i] * w[:, :, None]
            onset[s:s + n] += onset_b[i] * w
            feats[s:s + n] += f_b[i] * w
            weight[s:s + n] += w
    return tab / weight[:, :, None], onset / weight, feats / weight


class BeatHead(nn.Module):
    """[B, T, d_in] frozen tab-model features -> [B, T, 2] beat and downbeat logits."""

    def __init__(self, d_in=512, d=128, layers=10, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.tcn = DilatedConvs(d, layers, dropout)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2)

    def forward(self, f):
        return self.out(self.norm(self.tcn(self.inp(f))))


@torch.no_grad()
def beat_activations(head, feats, device):
    """(beat, downbeat) probabilities [T] for a whole track's features [T, d_in]."""
    out = torch.sigmoid(head(torch.from_numpy(np.asarray(feats, dtype=np.float32))[None].to(device)))[0].cpu().numpy()
    return out[:, 0], out[:, 1]


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
