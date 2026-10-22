"""Write the published weights (the Hugging Face repository, flat) from the trained checkpoints, as .safetensors:

    python scripts/export_weights.py --out huggingface

E25, E24 and E19 become guitab_<id>.safetensors (weights, plus size, step and decoding settings as JSON metadata), and
the tab LM tablm.safetensors. Each file is checked to give the same predictions as its .pt checkpoint. The code reads
both formats (guitab.inference.read_checkpoint), so locally trained .pt files keep working.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.inference import load_model, read_checkpoint, save_safetensors  # noqa: E402
from guitab.paths import RUNS, TABLM_RUN  # noqa: E402
from guitab.tablm import load_tablm  # noqa: E402

PUBLISHED = {"e25": "tabmodel_small_distill", "e24": "tabmodel_half_distill", "e19": "tabmodel_mid_all"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="huggingface")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    x = torch.rand(2, 432, 288)
    for mid, run in PUBLISHED.items():
        src, dst = RUNS / run / "best.pt", out / f"guitab_{mid}.safetensors"
        ckpt = read_checkpoint(src)
        save_safetensors({k: v for k, v in ckpt.items() if k in ("model", "size", "step", "decoding")}, dst)
        a, b = load_model(src, "cpu"), load_model(dst, "cpu")
        with torch.no_grad():
            same = all(torch.equal(a(x)[k], b(x)[k]) for k in ("tab", "onset", "pitch")) and a.decoding == b.decoding
        print(f"{dst}: {dst.stat().st_size / 2**20:.1f} MB, decoding {b.decoding}, identical outputs {same}", flush=True)
        assert same
    dst = out / "tablm.safetensors"
    save_safetensors({"model": torch.load(TABLM_RUN, map_location="cpu", weights_only=True)}, dst)
    a, b = load_tablm(TABLM_RUN, "cpu"), load_tablm(dst, "cpu")
    tokens = torch.randint(0, 123, (2, 40))
    with torch.no_grad():
        same = torch.equal(a(tokens), b(tokens))
    print(f"{dst}: {dst.stat().st_size / 2**20:.1f} MB, identical outputs {same}", flush=True)
    assert same


if __name__ == "__main__":
    main()
