"""Beat This! final0 activations (the teacher of the beat head, B1) for every training / validation track.

    python scripts/beat_teacher.py [--splits train,gaps_train,...] [--device auto] [--max_memory 0.35]

Writes data/beat_teacher/<name>.npz: beat and downbeat logits at 50 fps (float16), computed like the pipeline
(guitab.tab.detect_beats: 30 s chunks padded with silence) on the clean audio. Pitch-shifted GuitarSet copies reuse
their original's file (shifting keeps the timing).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.data import tracks  # noqa: E402
from guitab.device import limit_memory, pick_device  # noqa: E402
from guitab.paths import BEAT_TEACHER, DATA  # noqa: E402
from guitab.tab import BEAT_CHUNK, _PaddedChunks  # noqa: E402

SPLITS = "train,gaps_train,synth_train,ss_train,validation,gaps_validation,synth_validation,ss_validation"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", default=SPLITS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_memory", type=float, default=None)
    args = parser.parse_args()
    device = pick_device(args.device)
    limit_memory(device, args.max_memory)

    import soxr
    from beat_this.inference import load_model, split_predict_aggregate
    from beat_this.preprocessing import LogMelSpect
    from beat_this.preprocessing import load_audio as beat_this_audio

    ckpt = DATA / "models" / "beat_this-final0.ckpt"  # the teacher is always the large model
    model = load_model(str(ckpt) if ckpt.exists() else "final0", device)

    def run(x):
        with torch.inference_mode():
            out = model(torch.from_numpy(x).to(device))
        return out["beat"].float().cpu().numpy(), out["downbeat"].float().cpu().numpy()

    net = _PaddedChunks(run)
    spect = LogMelSpect(device="cpu")
    BEAT_TEACHER.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tracks() if t[4] in args.splits.split(",") and t[3] == 0
            and not (BEAT_TEACHER / f"{t[0]}.npz").exists()]
    for i, (name, audio, *_) in enumerate(todo):
        signal, sr = beat_this_audio(audio)
        if signal.ndim == 2:
            signal = signal.mean(1)
        if sr != 22050:
            signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
        with torch.inference_mode():
            pred = split_predict_aggregate(spect(torch.tensor(signal, dtype=torch.float32)), BEAT_CHUNK, 6, "keep_first",
                                           net)
        np.savez(BEAT_TEACHER / f"{name}.npz", beat=pred["beat"].numpy().astype(np.float16),
                 downbeat=pred["downbeat"].numpy().astype(np.float16))
        if i % 50 == 0:
            print(f"{i + 1}/{len(todo)} {name}", flush=True)


if __name__ == "__main__":
    main()
