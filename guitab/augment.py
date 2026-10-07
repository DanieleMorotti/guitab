"""Audio-domain augmentation: make clean training audio sound like phone / camera recordings in a room.

Two recipes, deliberately built from different effects so the test is not a copy of training:
  train_chain  random per track copy (pedalboard): algorithmic room reverb, phone-like band-limiting,
               random EQ peaks, compression, background noise, MP3 encoding.
  degrade      held-out recipe for the robustness test sets: convolution with a synthetic room impulse
               response, zero-phase Butterworth band-pass, pink noise at 20 dB SNR, AAC 64 kbit/s (ffmpeg).
All effects keep the timing (checked: 0 ms shift) and the length, so the original labels stay valid.
"""

import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

from .audio import SR


def seed_for(name, salt):
    return zlib.crc32(f"{name}|{salt}".encode())


def coloured_noise(n, rng, exponent):
    """Noise with power spectrum ~ 1/f^exponent (0 white, 1 pink, 2 brown), unit RMS."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / f ** (exponent / 2), n)
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12)).astype(np.float32)


def add_noise(y, rng, snr_db, exponent):
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    return y + coloured_noise(len(y), rng, exponent) * rms / 10 ** (snr_db / 20)


def peak_normalise(y, peak=0.9):
    m = np.max(np.abs(y))
    return (y * (peak / m)).astype(np.float32) if m > 0 else y.astype(np.float32)


def train_chain(y, rng):
    """One random phone/room rendering of y (float32 mono at SR). Returns (audio, description)."""
    import pedalboard as pb

    u = rng.uniform
    fx, desc = [], []
    if u() < 0.5:
        room, wet = u(0.1, 0.7), u(0.05, 0.35)
        fx.append(pb.Reverb(room_size=room, damping=u(0.2, 0.8), wet_level=wet, dry_level=1.0 - 0.5 * wet, width=0.0))
        desc.append(f"reverb {room:.2f}/{wet:.2f}")
    if u() < 0.8:
        hp, lp = float(np.exp(u(np.log(60), np.log(300)))), float(np.exp(u(np.log(3500), np.log(10500))))
        fx += [pb.HighpassFilter(hp), pb.LowpassFilter(lp)]
        desc.append(f"band {hp:.0f}-{lp:.0f}")
    if u() < 0.7:
        for _ in range(rng.integers(1, 4)):
            f, g = float(np.exp(u(np.log(150), np.log(6000)))), u(-8, 8)
            fx.append(pb.PeakFilter(f, g, u(0.5, 2.0)))
            desc.append(f"eq {f:.0f}:{g:+.0f}")
    if u() < 0.6:
        fx.append(pb.Compressor(threshold_db=u(-35, -10), ratio=u(2, 8), attack_ms=u(1, 20), release_ms=u(50, 300)))
        desc.append("comp")
    out = pb.Pedalboard(fx)(peak_normalise(y), SR) if fx else peak_normalise(y)
    if u() < 0.5:
        snr, exp = u(15, 40), float(rng.choice([0.0, 1.0, 2.0]))
        out = add_noise(out, rng, snr, exp)
        desc.append(f"noise {snr:.0f}dB/{exp:.0f}")
    out = peak_normalise(out)
    if u() < 0.5:
        q = u(2, 9)
        out = pb.Pedalboard([pb.MP3Compressor(vbr_quality=q)])(out, SR)
        desc.append(f"mp3 q{q:.1f}")
    return peak_normalise(out), ", ".join(desc) or "clean"


def room_ir(rng, rt60):
    """Synthetic room impulse response: direct path + exponentially decaying band-limited noise tail."""
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)  # -60 dB at rt60
    tail = np.convolve(tail, np.ones(4) / 4, mode="same")  # soften the very top end
    tail[: int(0.003 * SR)] = 0.0  # first reflections after ~3 ms
    ir = 0.5 * tail / np.sqrt(np.sum(tail ** 2))
    ir[0] = 1.0
    return ir.astype(np.float32)


def degrade(y, name):
    """Held-out phone/room recipe for the robustness test sets; deterministic per track name."""
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    rng = np.random.default_rng(seed_for(name, "degrade"))
    y = peak_normalise(y)
    out = fftconvolve(y, room_ir(rng, rng.uniform(0.4, 0.9)))[: len(y)]
    out = sosfiltfilt(butter(4, [250, 7000], btype="bandpass", fs=SR, output="sos"), out)
    out = add_noise(peak_normalise(out), rng, 20.0, 1.0)
    return aac_roundtrip(peak_normalise(out), 64)


def aac_roundtrip(y, kbps):
    """Encode to AAC at kbps with ffmpeg and decode back (same length, SR)."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        src, enc, dec = Path(tmp) / "a.wav", Path(tmp) / "a.m4a", Path(tmp) / "b.wav"
        sf.write(src, y, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:a", "aac", "-b:a", f"{kbps}k", str(enc)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc), "-ar", str(SR), "-ac", "1", str(dec)], check=True)
        out, _ = sf.read(dec, dtype="float32")
    out = out[: len(y)]
    return np.pad(out, (0, len(y) - len(out))).astype(np.float32)
