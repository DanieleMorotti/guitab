"""Train the chord model (guitab/chords.py) on synthetic guitar parts, tabs with chord names and GuitarSet.

    python scripts/train_chords.py --out runs/chords [--steps 20000] [--max_memory 0.35]

Training crops (--length segments, eighth notes) are made on the fly by --workers processes: a synthetic song
(guitab.chord_data.synth_song) with probability --synth_share, a custom-dataset tab with chord names with
--tab_share, a self-labelled real recording (data/chords/pseudo.pkl: the tab model's notes, chords from a chord model
reading the exact notes) with --pseudo_share, else a GuitarSet training track (the tab model's notes on the recording,
played chords); all pass through guitab.chord_data.degrade (strength --degrade; --real_degrade for GuitarSet and
--pseudo_degrade for the self-labelled recordings, whose notes the tab model is partly trained on).
Loss: cross-entropy over the chord classes and no chord (label smoothing --smoothing; labels outside the vocabulary
ignored). Every --eval_every steps: GuitarSet validation (played and instructed chords, comp and solo takes) and the
"val" half of the custom-dataset evaluation songs, scored with guitab.evaluation.chord_scores on the decoded chords
(all shown, and shown when confident); the best checkpoint (mean majmin accuracy of GuitarSet validation and the custom
songs, all shown) is saved as best.safetensors.
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.chord_data import CHORD_DATA, Progressions, example, load_voicings, synth_song  # noqa: E402
from guitab.chords import CONFIDENCE, IGNORE, N_CHORDS, ChordNet, chord_probs, chord_spans  # noqa: E402
from guitab.device import DEVICES, limit_memory, pick_device  # noqa: E402
from guitab.evaluation import chord_scores  # noqa: E402


class Crops(IterableDataset):
    def __init__(self, args):
        self.args = args

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((info.id if info else 0) + 1000 * self.args.seed + int(time.time()) % 100000)
        progressions = Progressions()
        voicings = load_voicings()
        tabs = pickle.load(open(CHORD_DATA / "custom_tabs.pkl", "rb"))
        real = pickle.load(open(CHORD_DATA / "guitarset.pkl", "rb"))["train"]
        pseudo = pickle.load(open(CHORD_DATA / "pseudo.pkl", "rb")) if self.args.pseudo_share > 0 else []
        a, length = self.args, self.args.length
        while True:
            r = rng.random()
            if r < a.synth_share:
                pcf, glob, lab = example(synth_song(rng, progressions, voicings), rng, a.degrade)
            elif r < a.synth_share + a.tab_share:
                pcf, glob, lab = example(tabs[rng.integers(len(tabs))], rng, a.degrade)
            elif r < a.synth_share + a.tab_share + a.pseudo_share:
                pcf, glob, lab = example(pseudo[rng.integers(len(pseudo))], rng, a.pseudo_degrade)
            else:
                pcf, glob, lab = example(real[rng.integers(len(real))], rng, a.real_degrade)
            n = len(lab)
            if n > length:
                s = int(rng.integers(0, n - length + 1))
                pcf, glob, lab = pcf[s:s + length], glob[s:s + length], lab[s:s + length]
            elif n < length:
                pad = length - n
                pcf = np.pad(pcf, ((0, pad), (0, 0), (0, 0)))
                glob = np.pad(glob, ((0, pad), (0, 0)))
                lab = np.pad(lab, (0, pad), constant_values=IGNORE)
            if (lab != IGNORE).any():
                yield torch.from_numpy(pcf), torch.from_numpy(glob), torch.from_numpy(lab)


def evaluate(model, sets, device):
    """{set: {"all": scores, "confident": scores}} (guitab.evaluation.chord_scores summed over the songs)."""
    model.eval()
    out = {}
    for name, songs, key in sets:
        acc = {"all": None, "confident": None}
        for song in songs:
            bounds, probs = chord_probs(model, song["notes"], song["beats"], song["downbeats"])
            for mode, conf in (("all", 0.0), ("confident", CONFIDENCE)):
                s = chord_scores(song[key], chord_spans(bounds, probs, confidence=conf))
                acc[mode] = s if acc[mode] is None else {k: acc[mode][k] + v for k, v in s.items()}
        out[name] = acc
    model.train()
    return out


def summary(scores):
    s = scores
    return {"majmin": s["majmin"] / max(s["majmin_time"], 1e-9), "root": s["root"] / max(s["time"], 1e-9),
            "exact": s["exact"] / max(s["time"], 1e-9), "shown": s["shown"] / max(s["time"], 1e-9)}


def save(model, path):
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path),
              metadata={"config": json.dumps(model.config)})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=DEVICES, help="the model is small: the CPU is enough")
    p.add_argument("--max_memory", type=float, default=None, help="cap of the accelerator memory share")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--length", type=int, default=96, help="segments (eighth notes) per crop")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--synth_share", type=float, default=0.45)
    p.add_argument("--tab_share", type=float, default=0.2)
    p.add_argument("--pseudo_share", type=float, default=0.25, help="self-labelled real recordings (pseudo.pkl)")
    p.add_argument("--degrade", type=float, default=1.0, help="strength of the simulated note errors")
    p.add_argument("--real_degrade", type=float, default=0.7, help="the same for GuitarSet training tracks")
    p.add_argument("--pseudo_degrade", type=float, default=0.4, help="the same for the self-labelled recordings")
    p.add_argument("--workers", type=int, default=2, help="processes making the training crops")
    p.add_argument("--threads", type=int, default=4, help="CPU threads of the training process")
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = pick_device(args.device)
    limit_memory(device, args.max_memory)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = ChordNet(args.width, args.layers).to(device)
    print(f"ChordNet: {sum(p.numel() for p in model.parameters())} parameters, device {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step, best = 0, -1.0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"resumed at step {step} (best {best:.4f})", flush=True)

    gs = pickle.load(open(CHORD_DATA / "guitarset.pkl", "rb"))
    custom = pickle.load(open(CHORD_DATA / "custom_eval.pkl", "rb"))
    sets = [("gs_val_comp", [s for s in gs["validation"] if s["take"] == "comp"], "chords"),
            ("gs_val_solo", [s for s in gs["validation"] if s["take"] == "solo"], "chords"),
            ("gs_val_comp_instructed", [s for s in gs["validation"] if s["take"] == "comp"], "instructed"),
            ("custom_val", [s for s in custom if s["half"] == "val"], "chords")]
    loader = iter(DataLoader(Crops(args), batch_size=args.batch, num_workers=args.workers, persistent_workers=True))
    log = open(out / "log.jsonl", "a")
    t0, losses = time.time(), []
    while step < args.steps:
        pcf, glob, lab = (x.to(device) for x in next(loader))
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step / args.steps, 1)))
        for g in opt.param_groups:
            g["lr"] = lr
        logits = model(pcf, glob)
        loss = F.cross_entropy(logits.reshape(-1, N_CHORDS + 1), lab.reshape(-1), ignore_index=IGNORE,
                               label_smoothing=args.smoothing)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step} loss {np.mean(losses):.4f} lr {lr:.2e} {(time.time() - t0) / 100:.3f} s/step", flush=True)
            losses, t0 = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(model, sets, device)
            table = {name: {mode: summary(s) for mode, s in r.items()} for name, r in res.items()}
            select = np.mean([table["gs_val_comp"]["all"]["majmin"], table["gs_val_solo"]["all"]["majmin"],
                              table["custom_val"]["all"]["majmin"]])
            line = "  ".join(f"{name} {t['all']['majmin']:.3f} (confident {t['confident']['majmin']:.3f}, shown "
                             f"{t['confident']['shown']:.2f})" for name, t in table.items())
            print(f"EVAL step {step}: select {select:.4f}  {line}", flush=True)
            log.write(json.dumps({"step": step, "select": select, "scores": table}) + "\n")
            log.flush()
            if select > best:
                best = select
                save(model, out / "best.safetensors")
                print(f"  new best ({best:.4f})", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best}, out / "last.pt")


if __name__ == "__main__":
    main()
