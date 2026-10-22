"""Export what the browser app (docs/) needs: the tab models and the beat tracker as ONNX, plus the DSP constants.

    python scripts/export_web.py                       # every trained model in paths.MODELS + Beat This! small0
    python scripts/export_web.py --models E20 --out docs

Writes <out>/models/<id>.onnx (not versioned: host them where the page can fetch them, see docs/models.json),
<out>/models.json (the models the page offers) and <out>/assets/dsp.json (constants, versioned):
  - the CQT: librosa's sparse FFT filter basis of one octave (every octave uses the same filters at half the sample
    rate), the filter lengths, and an FIR filter matching the soxr_hq halving librosa uses between octaves;
  - the Beat This! log-mel spectrogram: its mel filterbank;
  - the constant parts of the .gp5 files guitab.gp5 writes (pyguitarpro's defaults).
Each ONNX model is checked against PyTorch with onnxruntime.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.audio import BINS_PER_OCTAVE, FMIN_NOTE, HOP, N_BINS, SR  # noqa: E402
from guitab.inference import load_model  # noqa: E402
from guitab.paths import DATA, MODELS  # noqa: E402

HALVING_TAPS = 2047
TAB_WINDOW = 432  # frames per model window (inference.predict)
BEAT_CHUNK = 1500  # Beat This! frames per chunk (30 s at 50 fps)


class TabProbs(torch.nn.Module):
    """[1, T, 288] CQT feature -> fret posteriors [1, T, 6, 21] and onset probabilities [1, T, 6]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.softmax(out["tab"].float(), -1), torch.sigmoid(out["onset"].float())


class BeatLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["beat"], out["downbeat"]


def export(module, example, path, names):
    """Fixed input shape (the exporters bake the sequence length into the attention reshapes); the page pads shorter
    inputs with silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, (example,), str(path), input_names=["x"], output_names=names, opset_version=17,
                      dynamo=False)


def check(module, path, inputs, names):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for x in inputs:
        with torch.no_grad():
            ref = [r.numpy() for r in module(x)]
        got = sess.run(names, {"x": x.numpy()})
        worst = max(worst, *(float(np.abs(a - b).max()) for a, b in zip(ref, got)))
    return worst


def dsp_constants():
    import librosa
    from librosa import filters
    from librosa.core import constantq
    from torchaudio.transforms import MelSpectrogram

    freqs = librosa.cqt_frequencies(N_BINS, fmin=librosa.note_to_hz(FMIN_NOTE), bins_per_octave=BINS_PER_OCTAVE)
    alpha = filters._relative_bandwidth(freqs=freqs)
    lengths, _ = filters.wavelet_lengths(freqs=freqs, sr=SR, window="hann", filter_scale=1, gamma=0, alpha=alpha)
    vqt_filter_fft = getattr(constantq, "__vqt_filter_fft")
    basis, n_fft, _ = vqt_filter_fft(SR, freqs[-BINS_PER_OCTAVE:], 1, 1, 0.01, window="hann", gamma=0,
                                     dtype=np.complex64, alpha=alpha[-BINS_PER_OCTAVE:])
    basis = basis.tocoo()

    # soxr_hq 2:1 halving (librosa's resampler between octaves) as a zero-phase FIR filter: its magnitude response,
    # measured with cosines on 0 .. 0.25 cycles/sample (flat below ~0.23, zero above 0.25), Kaiser-windowed
    n = np.arange(1 << 15)
    grid = np.linspace(0, 0.25, 1025)
    response = []
    for f in grid:
        y = librosa.resample(np.cos(2 * np.pi * f * n), orig_sr=2, target_sr=1, res_type="soxr_hq", scale=False)
        mid = slice(len(y) // 4, 3 * len(y) // 4)
        response.append(np.sqrt(2 * np.mean(y[mid] ** 2)) if f > 0 else 1.0)
    big = 1 << 15
    f = np.arange(big // 2 + 1) / big
    taps = np.roll(np.fft.irfft(np.where(f <= 0.25, np.interp(f, grid, response), 0.0), n=big), HALVING_TAPS // 2)
    taps = taps[:HALVING_TAPS] * np.kaiser(HALVING_TAPS, 10)
    taps /= taps.sum()

    mel = MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=441, f_min=30, f_max=11000, n_mels=128,
                         mel_scale="slaney", normalized="frame_length", power=1)
    fb = mel.mel_scale.fb.numpy()  # [513, 128]
    return {
        "sr": SR, "hop": HOP, "n_bins": N_BINS, "bins_per_octave": BINS_PER_OCTAVE,
        "cqt": {"n_fft": int(n_fft), "rows": basis.row.tolist(), "cols": basis.col.tolist(),
                "re": np.round(basis.data.real, 9).tolist(), "im": np.round(basis.data.imag, 9).tolist(),
                "lengths": np.round(lengths, 6).tolist()},
        "halving": {"taps": taps.tolist()},
        "mel": {"n_fft": 1024, "hop": 441, "n_mels": 128, "log_multiplier": 1000,
                "fb": [[[int(k), round(float(fb[k, m]), 9)] for k in np.nonzero(fb[:, m])[0]] for m in range(fb.shape[1])]},
    }


def gp5_template():
    """The constant byte runs of the .gp5 files guitab.gp5.write_gp5 writes, from pyguitarpro itself: the page writes
    the title, tempo, capo, measure headers and beats between them (docs/js/gp5.js)."""
    import tempfile

    from guitarpro.gp5 import GP5File

    from guitab.gp5 import write_gp5

    marks = {}
    originals = {}

    def mark(name):
        def wrapper(self, *args):
            marks[name] = self.data.tell()
            originals[name](self, *args)
            marks[name + "_end"] = self.data.tell()
        return wrapper

    tokens = ["T", "downtune:0", "tempo:{tempo}", "start", "new_measure", "note:s1:f3", "wait:3840", "end"]

    def song_bytes(tempo, capo):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.gp5"
            write_gp5([t.format(tempo=tempo) for t in tokens], path, capo)
            return path.read_bytes()

    for name in ("writeInfo", "writeMeasureHeaders", "writeTracks", "writeMeasures"):
        originals[name] = getattr(GP5File, name)
        setattr(GP5File, name, mark(name))
    try:
        base = song_bytes(120, 0)
    finally:
        for name, fn in originals.items():
            setattr(GP5File, name, fn)

    def diff_at(other):
        idx = [i for i in range(len(base)) if base[i] != other[i]]
        return idx[0]

    tempo_at = diff_at(song_bytes(121, 0))
    capo_at = diff_at(song_bytes(120, 3))
    head2 = (marks["writeInfo_end"], marks["writeMeasureHeaders"] - 8)  # the measure and track counts are written by the page
    track = (marks["writeTracks"], marks["writeTracks_end"])
    assert head2[0] < tempo_at < head2[1] and track[0] < capo_at < track[1] and marks["writeMeasures_end"] == len(base)
    return {"head1": base[: marks["writeInfo"]].hex(), "head2": base[head2[0]: head2[1]].hex(),
            "tempo_at": tempo_at - head2[0], "track": base[track[0]: track[1]].hex(), "capo_at": capo_at - track[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs")
    parser.add_argument("--models", nargs="*", default=None, help="ids from paths.MODELS (default: every trained one)")
    parser.add_argument("--beat", default="small0", help="Beat This! checkpoint name in data/models (small0 / final0)")
    args = parser.parse_args()
    out = Path(args.out)
    torch.manual_seed(0)

    (out / "assets").mkdir(parents=True, exist_ok=True)
    dsp = dsp_constants()
    dsp["gp5"] = gp5_template()
    (out / "assets" / "dsp.json").write_text(json.dumps(dsp, separators=(",", ":")))
    print("assets/dsp.json written", flush=True)

    manifest = {"tab_models": [], "beat_model": None}
    ids = args.models or [k for k, (p, _) in MODELS.items() if p.exists()]
    for mid in ids:
        ckpt, label = MODELS[mid]
        module = TabProbs(load_model(ckpt, "cpu")).eval()
        path = out / "models" / f"{mid}.onnx"
        export(module, torch.rand(1, TAB_WINDOW, N_BINS), path, ["tab", "onset"])
        err = check(module, path, [torch.rand(1, TAB_WINDOW, N_BINS) for _ in range(3)], ["tab", "onset"])
        size = path.stat().st_size / 2**20
        print(f"{mid}: {path} {size:.1f} MB, max |onnx - torch| {err:.2e}", flush=True)
        manifest["tab_models"].append({"id": mid, "label": label, "url": f"models/{mid}.onnx", "mb": round(size, 1),
                                       "window": TAB_WINDOW})

    from beat_this.inference import load_model as load_beat_model

    beat = BeatLogits(load_beat_model(str(DATA / "models" / f"beat_this-{args.beat}.ckpt"), "cpu")).eval()
    path = out / "models" / f"beat_this_{args.beat}.onnx"
    export(beat, torch.rand(1, BEAT_CHUNK, 128), path, ["beat", "downbeat"])
    err = check(beat, path, [torch.rand(1, BEAT_CHUNK, 128) * 5 for _ in range(3)], ["beat", "downbeat"])
    size = path.stat().st_size / 2**20
    print(f"Beat This! {args.beat}: {path} {size:.1f} MB, max |onnx - torch| {err:.2e}", flush=True)
    manifest["beat_model"] = {"id": args.beat, "url": path.relative_to(out).as_posix(), "mb": round(size, 1),
                              "chunk": BEAT_CHUNK, "border": 6}
    (out / "models.json").write_text(json.dumps(manifest, indent=1))
    print(f"{out / 'models.json'} written", flush=True)


if __name__ == "__main__":
    main()
