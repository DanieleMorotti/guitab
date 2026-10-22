"""Audio loading and the model input feature.

Feature: log-magnitude CQT, 36 bins/octave from E1 (288 bins, up to ~10.5 kHz), 22.05 kHz, hop 256
(11.6 ms), normalised per track to [0, 1] (0 = 80 dB below the track maximum). The model builds the
harmonic stack itself, so one CQT per track is computed/cached.
"""

from pathlib import Path

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
    """Mono audio at SR. Formats libsndfile reads (wav, flac, ogg, mp3, aiff, ...) need nothing else; anything else
    (video, m4a / aac, webm) is decoded with ffmpeg, which is then required."""
    import librosa
    import soundfile

    try:
        soundfile.info(str(path))
    except soundfile.LibsndfileError:
        return _load_with_ffmpeg(path)
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def _load_with_ffmpeg(path):
    import shutil
    import subprocess
    import tempfile

    import soundfile

    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"cannot read {Path(path).name}: this format (video, m4a, ...) needs ffmpeg, which is not "
                           "installed. Install it (e.g. brew install ffmpeg, apt install ffmpeg) or convert the file to "
                           "wav, mp3 or flac.")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), str(wav)],
                           capture_output=True)
        if r.returncode:
            raise RuntimeError(f"ffmpeg could not read {Path(path).name}: " + r.stderr.decode(errors="replace")[-300:])
        y, _ = soundfile.read(str(wav), dtype="float32")
    return y
