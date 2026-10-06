"""Audio loading and the model input feature.

Feature: log-magnitude CQT, 36 bins/octave from E1 (288 bins, up to ~10.5 kHz), 22.05 kHz, hop 256
(11.6 ms), normalised per track to [0, 1] (0 = 80 dB below the track maximum). The model builds the
harmonic stack itself, so one CQT per track is computed/cached.
"""

import numpy as np

SR = 22050
HOP = 256
FPS = SR / HOP
N_BINS = 288
BINS_PER_OCTAVE = 36
FMIN_NOTE = "E1"


def compute_cqt(y):
    import librosa

    c = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz(FMIN_NOTE), n_bins=N_BINS,
                           bins_per_octave=BINS_PER_OCTAVE))
    db = librosa.amplitude_to_db(c, ref=np.max(c) + 1e-9, top_db=80.0)
    return ((db + 80.0) / 80.0).T.astype(np.float16)  # [T, F] in [0, 1]


def load_audio(path):
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y
