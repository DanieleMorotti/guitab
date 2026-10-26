"""Compare beat sources for the tab grid (B1) on GuitarSet and GAPS, against their reference beats.

    python scripts/evaluate_beats.py [--splits test,gaps_test] [--head runs/beat_head/best.pt]

Candidates: Beat This! final0 as the pipeline used it before B1 (int8 ONNX, minimal post-processing, its downbeats),
then dynamic-programming tracking (guitab.beats.track_beats, with the tab model's onsets) on the activation of
Beat This! final0, Beat This! small0 (data/models/beat_this-small0.ckpt) and the beat head; "head only" leaves the
onsets out. Downbeats of the DP candidates: guitab.beats.pick_downbeats on the same model's downbeat activation.
Metrics (guitab.evaluation): rhythm accuracy, beat and downbeat F-measure (+-70 ms), tempo within 4% (median beat
interval). GuitarSet references: the JAMS beat annotations; GAPS: gaps.score_beats. Paired Wilcoxon test of each
candidate's rhythm accuracy against the pre-B1 pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab import gaps  # noqa: E402
from guitab.audio import FPS  # noqa: E402
from guitab.beats import BeatHead, beat_activations, pick_downbeats, predict_features, track_beats  # noqa: E402
from guitab.data import gaps_notes, tracks  # noqa: E402
from guitab.evaluation import beat_fmeasure, rhythm_accuracy  # noqa: E402
from guitab.guitarset import string_notes  # noqa: E402
from guitab.inference import load_model  # noqa: E402
from guitab.paths import CACHE, DATA, DEFAULT_CHECKPOINT, ROOT, RUNS, WEIGHTS, fetch  # noqa: E402
from guitab.tab import BEAT_CHUNK, _PaddedChunks  # noqa: E402


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


def at_fps(logits, n):
    p = 1 / (1 + np.exp(-np.asarray(logits, dtype=float)))
    return np.interp(np.arange(n) / FPS, np.arange(len(p)) / 50, p)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", default="test,gaps_test")
    parser.add_argument("--head", default=str(RUNS / "beat_head" / "best.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from beat_this.inference import load_model as beat_this_model
    from beat_this.model.postprocessor import Postprocessor
    from scipy.stats import wilcoxon

    dev = torch.device(args.device)
    model = load_model(DEFAULT_CHECKPOINT, dev)
    head = BeatHead().to(dev).eval()
    head.load_state_dict(torch.load(args.head, map_location=dev, weights_only=True)["model"])
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    local = ROOT / "docs" / "models" / "beat_this_final0_int8.onnx"
    path = local if local.exists() else fetch(WEIGHTS / "beat_this_final0_int8.onnx")
    sess = onnxruntime.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    final0 = _PaddedChunks(lambda x: sess.run(["beat", "downbeat"], {"x": x}))  # the pre-B1 tracker (int8 ONNX)
    small = beat_this_model(str(DATA / "models" / "beat_this-small0.ckpt"), torch.device("cpu"))

    def run_small(x):
        with torch.inference_mode():
            out = small(torch.from_numpy(x))
        return out["beat"].float().numpy(), out["downbeat"].float().numpy()

    small0 = _PaddedChunks(run_small)
    names = ["before B1 (final0)", "final0 + DP", "small0 + DP", "head + DP", "head only"]
    for split in args.splits.split(","):
        rows = {k: {"rhythm": [], "beat_f": [], "down_f": [], "tempo": []} for k in names}
        for item in [t for t in tracks() if t[4] == split and t[3] == 0]:
            ref_b, ref_d, onsets = references(item)
            if len(ref_b) < 4:
                continue
            feat = np.load(CACHE / f"{item[0]}.npz")["feat"]
            _, onset, f = predict_features(model, feat, dev)
            n = len(onset)
            hb, hd = beat_activations(head, f, dev)
            fb, fd = beat_this_logits(final0, item[1])
            sb, sd = beat_this_logits(small0, item[1])
            b0, d0 = Postprocessor(type="minimal")(torch.from_numpy(fb), torch.from_numpy(fd))
            cands = {"before B1 (final0)": (np.asarray(b0, dtype=float), np.asarray(d0, dtype=float))}
            for name, (b, d, on) in {"final0 + DP": (at_fps(fb, n), at_fps(fd, n), onset),
                                     "small0 + DP": (at_fps(sb, n), at_fps(sd, n), onset),
                                     "head + DP": (hb, hd, onset), "head only": (hb, hd, None)}.items():
                beats = track_beats(b, on)
                cands[name] = (beats, pick_downbeats(beats, d))
            for name, (beats, downs) in cands.items():
                r = rows[name]
                r["rhythm"].append(rhythm_accuracy(ref_b, beats, onsets))
                r["beat_f"].append(beat_fmeasure(ref_b, beats))
                if len(ref_d) >= 2:
                    r["down_f"].append(beat_fmeasure(ref_d, downs))
                ratio = np.median(np.diff(beats)) / np.median(np.diff(ref_b)) if len(beats) > 2 else 0
                r["tempo"].append(abs(ratio - 1) < 0.04)
        base = np.array(rows[names[0]]["rhythm"])
        print(f"\n{split} ({len(base)} tracks)          rhythm  beat F  downbeat F  tempo   vs before B1 (rhythm)")
        for name in names:
            r = rows[name]
            x = np.array(r["rhythm"])
            ok = ~np.isnan(x) & ~np.isnan(base)
            cmp = ""
            if name != names[0]:
                d = x[ok] - base[ok]
                cmp = f"{d.mean():+.3f}, better/worse {np.sum(d > 0)}/{np.sum(d < 0)}, p={wilcoxon(x[ok], base[ok]).pvalue:.4f}"
            print(f"  {name:20s} {np.nanmean(x):6.3f}  {np.mean(r['beat_f']):6.3f}  {np.mean(r['down_f']):10.3f}  "
                  f"{np.mean(r['tempo']):5.2f}   {cmp}")


if __name__ == "__main__":
    main()
