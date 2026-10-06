"""Train the CQT frame/onset tab model on GuitarSet (players 00-03 + pitch-shifted copies).

Validation (player 04) every --eval_every steps: note-level F1 (onset +-50 ms) by string/fret and by
pitch against the JAMS annotations; the best string/fret F1 checkpoint is kept as best.pt.

With --gaps_share > 0, GAPS classical/fingerstyle pieces (tools/gaps_labels.py) are mixed in: each
crop comes from GAPS with that probability. GAPS notes with unknown strings are masked, and its
string labels (score fingerings, not necessarily what was played) get --gaps_tab_weight in the tab
loss. Validation then also reports GAPS (gaps_validation) pitch F1 and string agreement, and the best
checkpoint maximises the mean of GuitarSet string/fret F1 and GAPS pitch F1.

Augmentation on the CQT: pitch roll up 0-2 semitones (frets shifted, fret > 19 ignored), level /
noise floor / spectral tilt changes, short time masks.

Example (memory-capped for a laptop that is also used for other work):
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 \\
    python tools/train_tabmodel.py --out runs/tabmodel_mid --size mid --batch 16 --steps 20000
"""

import argparse
import datetime as dt
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabdata import N_CLASSES, N_FRETS, ROOT, load_split  # noqa: E402
from tabmodel import TabModel, count_params, pitch_targets  # noqa: E402
from tabmodel_infer import extract_notes, match_f1, predict  # noqa: E402


def sample_batch(pools, batch, length, augment, rng, tab_weights):
    """pools: [(share, tracks, length weights)]; returns x, tab, onset, unk, per-sample tab weight."""
    xs, tabs, onsets, unks, ws = [], [], [], [], []
    shares = [p[0] for p in pools]
    for _ in range(batch):
        _, tracks, weights = pools[rng.choices(range(len(pools)), shares)[0]]
        tr = tracks[rng.choices(range(len(tracks)), weights)[0]]
        n = len(tr["feat"])
        s = rng.randint(0, max(n - length, 0))
        x = np.asarray(tr["feat"][s:s + length], dtype=np.float32)
        tab = tr["tab"][s:s + length].astype(np.int64)
        on = tr["onset"][s:s + length].astype(np.float32)
        unk = tr["unk"][s:s + length].astype(np.float32)
        if len(x) < length:
            pad = length - len(x)
            x = np.pad(x, ((0, pad), (0, 0)))
            tab = np.pad(tab, ((0, pad), (0, 0)))
            on = np.pad(on, ((0, pad), (0, 0)))
            unk = np.pad(unk, (0, pad))
        if augment:
            k = rng.choice([0, 0, 1, 2])
            if k:
                x = np.concatenate([np.zeros((length, 3 * k), np.float32), x[:, :-3 * k]], axis=1)
                fretted = tab > 0
                tab = np.where(fretted, tab + k, tab)
                tab = np.where(tab > N_FRETS, -100, tab)
            x = x + rng.uniform(-0.15, 0.05)  # level
            x = x + rng.uniform(-0.1, 0.1) * np.linspace(-1, 1, x.shape[1], dtype=np.float32)  # tilt
            if rng.random() < 0.5:  # noise floor
                floor = rng.uniform(0.05, 0.35)
                x = np.maximum(x, floor + 0.03 * np.random.randn(*x.shape).astype(np.float32))
            for _ in range(rng.randint(0, 2)):  # time masks
                w = rng.randint(2, 12)
                t0 = rng.randint(0, length - w)
                x[t0:t0 + w] = 0.0
            x = np.clip(x, 0.0, 1.0)
        xs.append(x)
        tabs.append(tab)
        onsets.append(on)
        unks.append(unk)
        ws.append(tab_weights.get(tr["source"], 1.0))
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(tabs)), torch.from_numpy(np.stack(onsets)),
            torch.from_numpy(np.stack(unks)), torch.tensor(ws, dtype=torch.float32))


def loss_fn(out, tab, onset, unk, tab_w):
    """tab -100 = ignore; onset -1 = ignore; unk frames (unknown-string notes) skip the pitch loss."""
    b, t = tab.shape[:2]
    ce = F.cross_entropy(out["tab"].reshape(-1, N_CLASSES).float(), tab.reshape(-1), ignore_index=-100, reduction="none")
    valid = (tab.reshape(-1) != -100).float()
    w = tab_w[:, None, None].expand(b, t, 6).reshape(-1) * valid
    tab_loss = (ce * w).sum() / w.sum().clamp(min=1.0)
    # soft onset targets: +-1 frame neighbours at 0.5 tolerate annotation jitter
    on_mask = (onset >= 0).float()
    pos = onset.clamp(min=0)
    soft = torch.maximum(pos, 0.5 * (F.pad(pos, (0, 0, 1, 0))[:, :-1] + F.pad(pos, (0, 0, 0, 1))[:, 1:]).clamp(max=1))
    on_bce = F.binary_cross_entropy_with_logits(out["onset"].float(), soft, pos_weight=torch.tensor(5.0, device=soft.device),
                                                reduction="none")
    on_loss = (on_bce * on_mask).sum() / on_mask.sum().clamp(min=1.0)
    p_mask = (1.0 - unk)[:, :, None]
    p_bce = F.binary_cross_entropy_with_logits(out["pitch"].float(), pitch_targets(tab.clamp(min=0)), reduction="none")
    pitch_loss = (p_bce * p_mask).sum() / (p_mask.sum() * p_bce.shape[-1]).clamp(min=1.0)
    return tab_loss + on_loss + 0.5 * pitch_loss, {"tab": tab_loss.item(), "onset": on_loss.item(), "pitch": pitch_loss.item()}


def evaluate(model, val, device, ref_cache):
    model.eval()
    sf, pf = [], []
    for tr in val:
        tab, onset = predict(model, tr["feat"], device)
        est = extract_notes(tab, onset)
        ref = ref_cache[tr["name"]]
        sf.append(match_f1(ref, est, key=("string", "fret")))
        pf.append(match_f1(ref, est, key=("pitch",)))
    model.train()
    return float(np.mean(sf)), float(np.mean(pf))


def evaluate_gaps(model, val, device, ref_cache):
    """Pitch F1 against all GAPS MIDI notes, and string agreement on pitch-correct notes with known strings."""
    model.eval()
    pf, agree, total = [], 0, 0
    for tr in val:
        tab, onset = predict(model, tr["feat"], device)
        est = extract_notes(tab, onset)
        ref = ref_cache[tr["name"]]
        pf.append(match_f1(ref, est, key=("pitch",)))
        known = [r for r in ref if r["string"]]
        sf_hits = match_f1(known, est, key=("string", "fret"))
        p_hits = match_f1(known, est, key=("pitch",))
        if p_hits > 0:
            agree += sf_hits / p_hits * len(known)
            total += len(known)
    model.train()
    return float(np.mean(pf)), agree / max(total, 1)


def gaps_ref(label_path):
    from tabdata import OPEN_PITCH

    out = []
    for on, off, s, f in json.loads(Path(label_path).read_text())["notes"]:
        pitch = OPEN_PITCH[s] + f if s else f
        out.append({"onset": on, "offset": off, "string": s, "fret": f if s else None, "pitch": pitch})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="mid")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=432, help="frames per crop (432 = 5 s)")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--stop_at", default=None, help="HH:MM local time: save and exit at the first eval after it")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gaps_share", type=float, default=0.0, help="probability a crop comes from GAPS")
    parser.add_argument("--gaps_tab_weight", type=float, default=0.5, help="tab-loss weight of GAPS string labels")
    parser.add_argument("--max_steps_now", type=int, default=None, help="stop after this many steps (benchmarking)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)

    train = load_split("train")
    val = load_split("validation", include_shift=False)
    from tabmodel_infer import ref_notes

    ref_cache = {tr["name"]: ref_notes(ROOT / "data" / "guitarset" / "annotation" / f"{tr['name']}.jams") for tr in val}
    weights = [len(t["feat"]) for t in train]
    pools = [(1.0 - args.gaps_share, train, weights)]
    print(f"GuitarSet train tracks {len(train)} ({sum(weights) / 86.13 / 3600:.1f} h), val tracks {len(val)}", flush=True)
    gaps_val, gaps_ref_cache = [], {}
    if args.gaps_share > 0:
        gtrain = load_split("gaps_train")
        gaps_val = load_split("gaps_validation")
        gaps_ref_cache = {tr["name"]: gaps_ref(tr["labels"]) for tr in gaps_val}
        gw = [len(t["feat"]) for t in gtrain]
        pools.append((args.gaps_share, gtrain, gw))
        print(f"GAPS train pieces {len(gtrain)} ({sum(gw) / 86.13 / 3600:.1f} h), val pieces {len(gaps_val)}, "
              f"share {args.gaps_share}", flush=True)
    tab_weights = {"guitarset": 1.0, "gaps": args.gaps_tab_weight}

    model = TabModel(args.size).to(device)
    print(f"model {args.size}: {count_params(model) / 1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    step, best = 0, -1.0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        rng.seed(step)
        print(f"resumed at step {step}, best {best:.4f}", flush=True)

    def save(name):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "size": args.size},
                   out / name)

    log = open(out / "log.jsonl", "a")
    model.train()
    t0, run = time.time(), {}
    start_step = step
    while step < args.steps:
        x, tab, onset, unk, tab_w = sample_batch(pools, args.batch, args.length, True, rng, tab_weights)
        x, tab, onset, unk, tab_w = x.to(device), tab.to(device), onset.to(device), unk.to(device), tab_w.to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=args.bf16):
            pred = model(x)
        loss, parts = loss_fn(pred, tab, onset, unk, tab_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        for k, v in parts.items():
            run[k] = run.get(k, 0.0) + v
        if step % args.log_every == 0:
            el = time.time() - t0
            msg = {k: round(v / args.log_every, 4) for k, v in run.items()}
            print(f"step {step} {msg} lr {lr_at(step):.2e} {el / (step - start_step):.2f}s/step "
                  f"mem {torch.mps.current_allocated_memory() / 2**30:.1f}G", flush=True)
            log.write(json.dumps({"step": step, **msg}) + "\n")
            log.flush()
            run = {}
        if args.max_steps_now and step - start_step >= args.max_steps_now:
            break
        if step % args.eval_every == 0 or step == args.steps:
            sf, pf = evaluate(model, val, device, ref_cache)
            rec = {"step": step, "val_sf": sf, "val_pitch": pf}
            sel = sf
            if gaps_val:
                gp, ga = evaluate_gaps(model, gaps_val, device, gaps_ref_cache)
                rec.update(gaps_pitch=gp, gaps_string_agree=ga)
                sel = 0.5 * (sf + gp)
            if sel > best:
                best = sel
                save("best.pt")
            save("last.pt")
            extra = f", GAPS pitch F1 {rec['gaps_pitch']:.4f}, GAPS string agreement {rec['gaps_string_agree']:.4f}" if gaps_val else ""
            print(f"EVAL step {step}: val string/fret F1 {sf:.4f}, pitch F1 {pf:.4f}{extra} (best {best:.4f})", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.mps.empty_cache()
            if args.stop_at and dt.datetime.now().strftime("%H:%M") >= args.stop_at:
                print("stop_at reached, exiting", flush=True)
                return
    save("last.pt")


if __name__ == "__main__":
    main()
