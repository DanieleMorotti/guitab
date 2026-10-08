"""Audio loading and the model input feature.

Feature: log-magnitude CQT, 36 bins/octave from E1 (288 bins, up to ~10.5 kHz), 22.05 kHz, hop 256
(11.6 ms), normalised per track to [0, 1] (0 = 80 dB below the track maximum). The model builds the
harmonic stack itself, so one CQT per track is computed/cached.

Time-sharp low band (E14): the same bins with 4x shorter filters (filter_scale 0.25) for the 5 octaves from E1
(180 bins). At 36 bins/oct a bin's window is ~51 periods (0.62 s at E2), which smears bass attacks (median
+2.5 dB in the note's own bin at a fresh attack, vs +6.7 dB with the sharp filters); with model.sharp_bins
the model input is [CQT 288 | sharp 180].
"""

import numpy as np

SR = 22050
HOP = 256
FPS = SR / HOP
N_BINS = 288
BINS_PER_OCTAVE = 36
FMIN_NOTE = "E1"
SHARP_BINS = 180
SHARP_FILTER_SCALE = 0.25


def compute_cqt(y, n_bins=N_BINS, filter_scale=1.0):
    import librosa

    c = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz(FMIN_NOTE), n_bins=n_bins,
                           bins_per_octave=BINS_PER_OCTAVE, filter_scale=filter_scale))
    db = librosa.amplitude_to_db(c, ref=np.max(c) + 1e-9, top_db=80.0)
    return ((db + 80.0) / 80.0).T.astype(np.float16)  # [T, F] in [0, 1]


def compute_sharp(y):
    """Time-sharp low band [T, SHARP_BINS] in [0, 1], frame-aligned with compute_cqt(y) (both centred on t * HOP;
    librosa's shorter filters can add one trailing frame, trimmed here)."""
    return compute_cqt(y, SHARP_BINS, SHARP_FILTER_SCALE)[: 1 + len(y) // HOP]


def model_features(y, sharp):
    """Model input for audio y: the CQT, plus the sharp low band when the model uses it."""
    feat = compute_cqt(y)
    return np.concatenate([feat, compute_sharp(y)], 1) if sharp else feat


def load_audio(path):
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y
