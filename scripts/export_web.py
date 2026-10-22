"""Export what the browser app (docs/) needs: the tab models and the beat tracker as ONNX, plus the DSP constants.

    python scripts/export_web.py      # E25, E19, E24 + Beat This! final0 (int8 weights) into docs/models/

Writes <out>/models/guitab_<id>.onnx (not versioned: the published page fetches them from --base, the Hugging Face
model repo, where they sit next to the .pt weights), <out>/models.json (the models the page offers, with their download
URL, size and own decoding settings) and <out>/assets/dsp.json (constants, versioned):
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
WEB_MODELS = ["E25", "E19", "E24"]  # offered by the page, first = preselected


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


def int8_weights(src, dst, min_size=4096):
    """Store the large MatMul / Conv / Gemm weights as int8 with a per-output-channel scale and a DequantizeLinear in
    front of them: a quarter of the download, while the arithmetic stays float32 (onnxruntime folds the dequantisation
    into the initializers when it loads the model). Unlike dynamic quantisation, no activation is rounded to 8 bits, so
    the browser and Python, which differ by ~1e-6 in their float arithmetic, find the same beats."""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(src))
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    axis_of = {}  # weight name -> output-channel axis
    for node in graph.node:
        if node.op_type in ("MatMul", "Gemm") and len(node.input) > 1 and node.input[1] in inits:
            axis_of[node.input[1]] = 0 if node.op_type == "Gemm" and any(a.name == "transB" and a.i for a in node.attribute) else 1
        elif node.op_type == "Conv" and node.input[1] in inits:
            axis_of[node.input[1]] = 0
    new_nodes = []
    for name, axis in axis_of.items():
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        if w.ndim < 2 or w.size < min_size:
            continue
        other = tuple(i for i in range(w.ndim) if i != axis)
        scale = np.maximum(np.abs(w).max(axis=other), 1e-12) / 127.0
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        graph.initializer.remove(inits[name])
        graph.initializer.extend([numpy_helper.from_array(q, name + "_q"),
                                  numpy_helper.from_array(scale.astype(np.float32), name + "_scale")])
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale"], [name], axis=axis,
                                          name=name + "_dequantize"))
    nodes = list(graph.node)  # the dequantised weights feed the nodes that use them: define them first
    del graph.node[:]
    graph.node.extend(new_nodes + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


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
    parser.add_argument("--models", nargs="*", default=WEB_MODELS, help="ids from paths.MODELS (default: E25 E19 E24)")
    parser.add_argument("--base", default="https://huggingface.co/danmor/guitab/resolve/main/",
                        help="URL prefix the published page downloads the .onnx files from (a page served from localhost "
                             "reads them from docs/models/)")
    parser.add_argument("--beat", default="final0", help="Beat This! checkpoint name in data/models (final0 / small0)")
    parser.add_argument("--beat_fp32", action="store_true", help="keep the beat tracker's weights in float32 (default: "
                        "int8 weights with float32 arithmetic, final0 79 -> 21 MB)")
    args = parser.parse_args()
    out = Path(args.out)
    torch.manual_seed(0)

    (out / "assets").mkdir(parents=True, exist_ok=True)
    dsp = dsp_constants()
    dsp["gp5"] = gp5_template()
    (out / "assets" / "dsp.json").write_text(json.dumps(dsp, separators=(",", ":")))
    print("assets/dsp.json written", flush=True)

    manifest = {"tab_models": [], "beat_model": None}
    for mid in args.models:
        ckpt, label = MODELS[mid]
        module = TabProbs(load_model(ckpt, "cpu")).eval()
        path = out / "models" / f"guitab_{mid.lower()}.onnx"
        export(module, torch.rand(1, TAB_WINDOW, N_BINS), path, ["tab", "onset"])
        err = check(module, path, [torch.rand(1, TAB_WINDOW, N_BINS) for _ in range(3)], ["tab", "onset"])
        size = path.stat().st_size / 2**20
        print(f"{mid}: {path} {size:.1f} MB, max |onnx - torch| {err:.2e}", flush=True)
        decoding = module.model.decoding  # the checkpoint's own note-extraction settings (e.g. onset .45 for E25)
        manifest["tab_models"].append({"id": mid, "label": label, "url": args.base + path.name, "mb": round(size, 1),
                                       "window": TAB_WINDOW, "decoding": decoding})

    from beat_this.inference import load_model as load_beat_model

    beat = BeatLogits(load_beat_model(str(DATA / "models" / f"beat_this-{args.beat}.ckpt"), "cpu")).eval()
    path = out / "models" / f"beat_this_{args.beat}.onnx"
    export(beat, torch.rand(1, BEAT_CHUNK, 128), path, ["beat", "downbeat"])
    if not args.beat_fp32:  # int8 weights, float32 arithmetic: a quarter of the download
        int8 = path.with_name(path.stem + "_int8.onnx")
        int8_weights(path, int8)
        path.unlink()
        path = int8
    err = check(beat, path, [torch.rand(1, BEAT_CHUNK, 128) * 5 for _ in range(3)], ["beat", "downbeat"])
    size = path.stat().st_size / 2**20
    print(f"Beat This! {args.beat}: {path} {size:.1f} MB, max |onnx - torch| {err:.2e}", flush=True)
    manifest["beat_model"] = {"id": args.beat, "url": args.base + path.name, "mb": round(size, 1),
                              "chunk": BEAT_CHUNK, "border": 6}
    (out / "models.json").write_text(json.dumps(manifest, indent=1))
    print(f"{out / 'models.json'} written", flush=True)


if __name__ == "__main__":
    main()
