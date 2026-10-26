"""Compare beat sources for the tab grid (B1) on GuitarSet and GAPS, against their reference beats.

    python scripts/evaluate_beats.py [--splits test,gaps_test]

Candidates: Beat This! final0 as the pipeline used it before B1 (int8 ONNX, minimal post-processing, its downbeats);
dynamic-programming tracking (guitab.beats.track_beats, with the tab model's onsets) on the activation of Beat This!
small0 (the 1.1 default) and final0 (paths.BEAT_MODEL), from the int8 ONNX files the pipeline uses. Downbeats of the DP
candidates:
guitab.beats.pick_downbeats on the same model's downbeat activation. Metrics (guitab.evaluation): rhythm accuracy, beat
F-measure (+-70 ms), bar lines (the downbeats the tab draws: tab.BeatGrid's metre and first bar, F-measure against the
reference downbeats), tempo within 4% (median beat interval). GuitarSet references: the JAMS beat annotations; GAPS:
gaps.score_beats. Paired Wilcoxon tests of rhythm accuracy against the 1.1 default.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab import gaps  # noqa: E402
from guitab.beats import activation, pick_downbeats, track_beats  # noqa: E402
from guitab.data import gaps_notes, tracks  # noqa: E402
from guitab.evaluation import beat_fmeasure, rhythm_accuracy  # noqa: E402
from guitab.guitarset import string_notes  # noqa: E402
from guitab.inference import load_model, predict  # noqa: E402
from guitab.paths import CACHE, DEFAULT_CHECKPOINT, ROOT, WEIGHTS, fetch  # noqa: E402
from guitab.tab import BEAT_CHUNK, BeatGrid, _PaddedChunks  # noqa: E402


def references(item):
    name, audio, labels, shift, split = item
    if labels.endswith(".jams"):
        d = json.loads(Path(labels).read_text())
        pos = [x for a in d["annotations"] if a["namespace"] == "beat_position" for x in a["data"]]
        beats = np.array([x["time"] for x in pos])
        downs = np.array([x["time"] for x in pos if x["value"]["position"] == 1])
        onsets = [n[0] for n in string_notes(labels)]
    else:
        beats, downs = gaps.score_beats(Path(audio).stem.rsplit("_", 1)[0])
        onsets = [n[0] for n in gaps_notes(labels)]
    return beats, downs, np.unique(np.round(onsets, 4))


def beat_this_logits(net, audio):
    """(beat, downbeat) logits at 50 fps, computed as guitab.tab.detect_beats does (rounded to 0.01)."""
    import soxr
    from beat_this.inference import split_predict_aggregate
    from beat_this.preprocessing import LogMelSpect
    from beat_this.preprocessing import load_audio as beat_this_audio

    signal, sr = beat_this_audio(str(audio))
    if signal.ndim == 2:
        signal = signal.mean(1)
    if sr != 22050:
        signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
    with torch.inference_mode():
        pred = split_predict_aggregate(LogMelSpect(device="cpu")(torch.tensor(signal, dtype=torch.float32)), BEAT_CHUNK,
                                       6, "keep_first", net)
    return [np.floor(pred[k].double().numpy() * 100 + 0.5) / 100 for k in ("beat", "downbeat")]


def onnx_tracker(name):
    """Beat This! <name> int8 ONNX (docs/models/ or weights/), run as guitab.tab does."""
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    local = ROOT / "docs" / "models" / f"beat_this_{name}_int8.onnx"
    path = local if local.exists() else fetch(WEIGHTS / local.name)
    sess = onnxruntime.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return _PaddedChunks(lambda x: sess.run(["beat", "downbeat"], {"x": x}))


def bar_lines(beats, downs):
    if len(beats) < 2:
        return np.asarray(beats)
    grid = BeatGrid(beats, downs)
    return np.asarray(beats)[grid.origin::grid.num_beats]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", default="test,gaps_test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from beat_this.model.postprocessor import Postprocessor
    from scipy.stats import wilcoxon

    dev = torch.device(args.device)
    model = load_model(DEFAULT_CHECKPOINT, dev)
    final0, small0 = onnx_tracker("final0"), onnx_tracker("small0")
    names = ["small0 + DP (1.1)", "before B1 (final0)", "final0 + DP"]
    for split in args.splits.split(","):
        rows = {k: {"rhythm": [], "beat_f": [], "bars": [], "tempo": []} for k in names}
        for item in [t for t in tracks() if t[4] == split and t[3] == 0]:
            ref_b, ref_d, onsets = references(item)
            if len(ref_b) < 4:
                continue
            feat = np.load(CACHE / f"{item[0]}.npz")["feat"]
            _, onset = predict(model, feat, dev)
            n = len(onset)
            fb, fd = beat_this_logits(final0, item[1])
            sb, sd = beat_this_logits(small0, item[1])
            b0, d0 = Postprocessor(type="minimal")(torch.from_numpy(fb), torch.from_numpy(fd))
            cands = {"before B1 (final0)": (np.asarray(b0, dtype=float), np.asarray(d0, dtype=float))}
            for name, (b, d, on) in {"final0 + DP": (activation(fb, n), activation(fd, n), onset),
                                     "small0 + DP (1.1)": (activation(sb, n), activation(sd, n), onset)}.items():
                beats = track_beats(b, on)
                cands[name] = (beats, pick_downbeats(beats, d))
            for name, (beats, downs) in cands.items():
                r = rows[name]
                r["rhythm"].append(rhythm_accuracy(ref_b, beats, onsets))
                r["beat_f"].append(beat_fmeasure(ref_b, beats))
                if len(ref_d) >= 2:
                    r["bars"].append(beat_fmeasure(ref_d, bar_lines(beats, downs)))
                ratio = np.median(np.diff(beats)) / np.median(np.diff(ref_b)) if len(beats) > 2 else 0
                r["tempo"].append(abs(ratio - 1) < 0.04)
        base = np.array(rows[names[0]]["rhythm"])
        print(f"\n{split} ({len(base)} tracks)          rhythm  beat F  bar lines  tempo   vs 1.1 default (rhythm)")
        for name in names:
            r = rows[name]
            x = np.array(r["rhythm"])
            ok = ~np.isnan(x) & ~np.isnan(base)
            cmp = ""
            if name != names[0]:
                d = x[ok] - base[ok]
                cmp = f"{d.mean():+.3f}, better/worse {np.sum(d > 0)}/{np.sum(d < 0)}, p={wilcoxon(x[ok], base[ok]).pvalue:.4f}"
            print(f"  {name:20s} {np.nanmean(x):6.3f}  {np.mean(r['beat_f']):6.3f}  {np.mean(r['bars']):9.3f}  "
                  f"{np.mean(r['tempo']):5.2f}   {cmp}")


if __name__ == "__main__":
    main()
