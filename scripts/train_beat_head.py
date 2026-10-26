"""Train the beat / downbeat head (guitab.beats.BeatHead, B1) on the frozen tab model.

    python scripts/train_beat_head.py [--steps 8000] [--out runs/beat_head] [--device auto] [--max_memory 0.35]
    python scripts/train_beat_head.py --branch --out runs/beat_branch   # B2: also a trainable copy of the last layers

The tab model (default checkpoint, E25) is frozen and only computes features, so its tabs do not change. Each step takes
`batch` random crops of `crop` frames (~32 s; sources drawn equally: GuitarSet, GAPS, SynthTab, custom dataset; tracks by
length), half of them from the phone / room augmented copies of the CQT (the timing is unchanged, so the clean targets
apply). Targets: Beat This! final0 probabilities (scripts/beat_teacher.py, soft binary cross-entropy) on every track,
plus the real beats and downbeats (widened to +-1 frame at 0.5) on GuitarSet and GAPS. Every `eval_every` steps:
rhythm accuracy, beat and downbeat F-measure on GuitarSet validation and GAPS validation with the full decoding
(guitab.beats.track_beats / pick_downbeats); the best mean rhythm accuracy is kept as best.pt, every evaluated
step as step<k>.pt. --branch (B2) also trains a guitab.beats.BeatBranch (copies of the tab model's last two transformer
layers, started from its weights, learning rate lr * branch_lr); the tab model itself stays frozen.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab import gaps  # noqa: E402
from guitab.audio import FPS  # noqa: E402
from guitab.beats import (BeatBranch, BeatHead, beat_activations, forward_features, pick_downbeats,  # noqa: E402
                          predict_features, track_beats)
from guitab.data import aug_path, gaps_notes, tracks  # noqa: E402
from guitab.guitarset import string_notes  # noqa: E402
from guitab.device import limit_memory, pick_device  # noqa: E402
from guitab.evaluation import beat_fmeasure, rhythm_accuracy  # noqa: E402
from guitab.paths import BEAT_TEACHER, CACHE, DEFAULT_CHECKPOINT, RUNS  # noqa: E402

TRAIN = ("train", "gaps_train", "synth_train", "ss_train")
VAL = ("validation", "gaps_validation")
WIN, HOP = 432, 288


def source(name):
    return name.split("_")[0] if name.startswith(("gaps_", "synth_", "ss_")) else "guitarset"


def base_name(name, shift):
    return name.rsplit("_up", 1)[0] if shift else name


def real_beats(item):
    """(beats, downbeats) in seconds from the annotations, or None (SynthTab, custom dataset)."""
    name, audio, labels, shift, split = item
    if labels.endswith(".jams"):
        d = json.loads(Path(labels).read_text())
        pos = [x for a in d["annotations"] if a["namespace"] == "beat_position" for x in a["data"]]
        return np.array([x["time"] for x in pos]), np.array([x["time"] for x in pos if x["value"]["position"] == 1])
    if source(name) == "gaps":
        return gaps.score_beats(Path(audio).stem.rsplit("_", 1)[0])
    return None


def real_cache(items):
    """Real beats per track (GAPS score parsing is slow: cached in data/beat_teacher/real_beats.pkl)."""
    path = BEAT_TEACHER / "real_beats.pkl"
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    for it in items:
        key = base_name(it[0], it[3])
        if key not in cache:
            try:
                cache[key] = real_beats(it)
            except Exception as e:  # unparsable score
                print(f"no reference beats for {key}: {e!r}"[:160])
                cache[key] = None
    path.write_bytes(pickle.dumps(cache))
    return cache


def frame_targets(times, n):
    t = np.zeros(n, dtype=np.float32)
    for s in times:
        k = int(round(s * FPS))
        for d, v in ((-1, 0.5), (1, 0.5), (0, 1.0)):
            if 0 <= k + d < n:
                t[k + d] = max(t[k + d], v)
    return t


def teacher(name, n):
    """Beat This! probabilities [n, 2] at the tab model's frame rate."""
    z = np.load(BEAT_TEACHER / f"{name}.npz")
    t50 = np.arange(len(z["beat"])) / 50
    t = np.arange(n) / FPS
    return np.stack([np.interp(t, t50, 1 / (1 + np.exp(-z[k].astype(np.float32)))) for k in ("beat", "downbeat")], -1)


def crop_features(model, x, taper, branch=None):
    """[B, L, 288] CQT crops -> [B, L, 512] features from 5 s windows (hop 288) overlap-added with the inference taper."""
    b, n, _ = x.shape
    starts = list(range(0, n - WIN + 1, HOP))
    win = torch.stack([x[:, s:s + WIN] for s in starts], 1).reshape(-1, WIN, x.shape[-1])
    f = forward_features(model, win, branch, grad=branch is not None and branch.training)["features"]
    f = f.float().reshape(b, len(starts), WIN, -1)
    out = torch.zeros(b, n, f.shape[-1], device=x.device)
    w = torch.zeros(1, n, 1, device=x.device)
    for j, s in enumerate(starts):
        out[:, s:s + WIN] += f[:, j] * taper
        w[:, s:s + WIN] += taper
    return out / w


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="the frozen tab model")
    parser.add_argument("--out", default=str(RUNS / "beat_head"))
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=1, help="split each batch into this many passes (less memory)")
    parser.add_argument("--crop", type=int, default=2736, help="frames per crop: 432 + k * 288 (2736 = 32 s, 9 windows)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--real_weight", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--branch", action="store_true", help="B2: also train a copy of the last transformer layers")
    parser.add_argument("--branch_lr", type=float, default=0.1, help="branch learning rate, as a fraction of --lr")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_memory", type=float, default=None)
    args = parser.parse_args()
    if (args.crop - WIN) % HOP:
        parser.error("--crop must be 432 + k * 288")
    device = pick_device(args.device)
    limit_memory(device, args.max_memory)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    from guitab.inference import load_model

    model = load_model(args.checkpoint, device)
    for p in model.parameters():
        p.requires_grad_(False)
    head = BeatHead().to(device)
    branch = BeatBranch(model).to(device) if args.branch else None
    if branch is not None:
        for p in branch.parameters():
            p.requires_grad_(True)
        say(f"branch: {sum(p.numel() for p in branch.parameters()) / 1e6:.2f}M parameters (copy of the last layers)")
    say(f"tab model {args.checkpoint} (frozen); head {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M parameters")

    items = [t for t in tracks() if t[4] in TRAIN + VAL and (BEAT_TEACHER / f"{base_name(t[0], t[3])}.npz").exists()]
    real = real_cache(items)
    train = [t for t in items if t[4] in TRAIN]
    by_source = {}
    for t in train:
        n = np.load(CACHE / f"{t[0]}.npz")["feat"].shape[0]
        if n < WIN:
            continue
        by_source.setdefault(source(t[0]), []).append((t, n))
    say("train tracks: " + ", ".join(f"{s} {len(v)} ({sum(n for _, n in v) / FPS / 3600:.1f} h)" for s, v in by_source.items()))
    sources = sorted(by_source)
    taper = torch.from_numpy(np.hanning(WIN + 2)[1:-1].astype(np.float32)[:, None] + 1e-3).to(device)

    # validation: features are fixed (frozen model), computed once
    val = []
    for t in items:
        if t[4] not in VAL or real.get(t[0]) is None:
            continue
        feat = np.load(CACHE / f"{t[0]}.npz")["feat"]
        _, onset, f = predict_features(model, feat, device)
        if branch is not None:
            f = None  # recomputed at every evaluation (the branch changes)
        notes = gaps_notes(t[2]) if t[2].endswith(".json") else string_notes(t[2])
        val.append({"name": t[0], "split": t[4], "feats": None if f is None else f.astype(np.float16), "onset": onset,
                    "cqt": feat if branch is not None else None, "ref": real[t[0]],
                    "note_onsets": np.unique(np.round([n[0] for n in notes], 4))})
    say(f"validation: {len(val)} tracks")

    def sample():
        s = sources[rng.integers(len(sources))]
        pool = by_source[s]
        lens = np.array([n for _, n in pool], dtype=float)
        (name, audio, labels, shift, split), n = pool[rng.choice(len(pool), p=lens / lens.sum())]
        L = min(args.crop, n)
        L -= (L - WIN) % HOP  # whole windows
        start = int(rng.integers(0, n - L + 1))
        feat = np.load(CACHE / f"{name}.npz")["feat"]
        if rng.random() < args.aug_prob:
            copies = [p for p in (aug_path(name, f"aug{k}") for k in range(3)) if p.exists()]
            if copies:
                feat = np.load(copies[rng.integers(len(copies))], mmap_mode="r")
        x = np.zeros((args.crop, feat.shape[1]), dtype=np.float32)
        x[:L] = feat[start:start + L]
        tgt = np.zeros((args.crop, 2), dtype=np.float32)
        tgt[:L] = teacher(base_name(name, shift), n)[start:start + L]
        r = real.get(base_name(name, shift))
        rt = np.zeros((args.crop, 2), dtype=np.float32)
        has_real = r is not None and len(r[0]) > 0
        if has_real:
            rt[:L, 0] = frame_targets(r[0], n)[start:start + L]
            rt[:L, 1] = frame_targets(r[1], n)[start:start + L]
        mask = np.zeros(args.crop, dtype=np.float32)
        mask[:L] = 1
        return x, tgt, rt, mask, np.float32(has_real)

    def evaluate():
        head.eval()
        if branch is not None:
            branch.eval()
        res = {}
        for split in VAL:
            m = {"rhythm": [], "rhythm_head_only": [], "beat_f": [], "down_f": []}
            for v in [v for v in val if v["split"] == split]:
                ref_b, ref_d = v["ref"]
                if len(ref_b) < 4:
                    continue
                feats = v["feats"] if branch is None else predict_features(model, v["cqt"], device, branch=branch)[2]
                b_act, d_act = beat_activations(head, feats, device)
                beats = track_beats(b_act, v["onset"])
                m["rhythm_head_only"].append(rhythm_accuracy(ref_b, track_beats(b_act), v["note_onsets"]))
                downs = pick_downbeats(beats, d_act)
                m["rhythm"].append(rhythm_accuracy(ref_b, beats, v["note_onsets"]))
                m["beat_f"].append(beat_fmeasure(ref_b, beats))
                if len(ref_d) >= 2:
                    m["down_f"].append(beat_fmeasure(ref_d, downs))
            res[split] = {k: float(np.nanmean(x)) for k, x in m.items()}
        head.train()
        if branch is not None:
            branch.train()
        return res

    groups = [{"params": list(head.parameters()), "lr": args.lr}]
    if branch is not None:
        groups.append({"params": list(branch.parameters()), "lr": args.lr * args.branch_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps,
                                                pct_start=0.05)
    best, t0, losses = -1.0, time.time(), []
    head.train()
    for step in range(1, args.steps + 1):
        batch = [sample() for _ in range(args.batch)]
        opt.zero_grad()
        total = 0.0
        for part in np.array_split(np.arange(args.batch), args.accum):
            x, tgt, rt, mask, has = map(np.stack, zip(*[batch[i] for i in part]))
            x, tgt, rt, mask, has = (torch.from_numpy(a).to(device) for a in (x, tgt, rt, mask, has))
            logits = head(crop_features(model, x, taper, branch))
            m = mask[..., None]
            loss = (F.binary_cross_entropy_with_logits(logits, tgt, reduction="none") * m).sum() / (2 * m.sum())
            mr = m * has[:, None, None]
            if mr.sum() > 0:
                loss = loss + args.real_weight * (F.binary_cross_entropy_with_logits(logits, rt, reduction="none") * mr).sum() / (2 * mr.sum())
            (loss / args.accum).backward()
            total += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        losses.append(total)
        if not np.isfinite(losses[-1]):
            say(f"step {step}: non-finite loss")
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate()
            score = np.mean([res[s]["rhythm"] for s in VAL])
            say(f"step {step} loss {np.mean(losses):.4f} ({(time.time() - t0) / 60:.0f} min) | "
                + " | ".join(f"{s}: rhythm {r['rhythm']:.3f} (head only {r['rhythm_head_only']:.3f}) beat F {r['beat_f']:.3f} downbeat F {r['down_f']:.3f}"
                             for s, r in res.items()) + f" | mean rhythm {score:.4f}" + (" *" if score > best else ""))
            losses = []
            ckpt = {"model": head.state_dict(), "branch": None if branch is None else branch.state_dict(), "step": step,
                    "tab_checkpoint": Path(args.checkpoint).parent.name, "val": res}
            torch.save(ckpt, out / "last.pt")
            torch.save(ckpt, out / f"step{step}.pt")
            if score > best:
                best = score
                torch.save(ckpt, out / "best.pt")


if __name__ == "__main__":
    main()
