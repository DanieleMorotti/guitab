"""Train the CQT frame/onset tab model on GuitarSet (players 00-03 + pitch-shifted copies).

Validation (player 04) every --eval_every steps: note-level F1 (onset +-50 ms) by string/fret and by
pitch against the JAMS annotations; the best string/fret F1 checkpoint is kept as best.pt.

With --gaps_share > 0, GAPS classical/fingerstyle pieces (scripts/gaps_labels.py) are mixed in: each
crop comes from GAPS with that probability. GAPS notes with unknown strings are masked, and its
string labels (score fingerings, not necessarily what was played) get --gaps_tab_weight in the tab
loss. Validation then also reports GAPS (gaps_validation) pitch F1 and string agreement, and the best
checkpoint maximises the mean of GuitarSet string/fret F1 and GAPS pitch F1.

Augmentation on the CQT: pitch roll up 0-2 semitones (frets shifted, fret > 19 ignored), level /
noise floor / spectral tilt changes, short time masks.

Audio augmentation (E12): with --aug_copies k, each training track gets k phone/room renderings
(scripts/build_aug_cache.py); a crop uses one of them with probability --aug_share. With --eval_degraded,
validation also scores the held-out degraded copies, and the best checkpoint maximises the mean of the
clean and degraded selection scores. --init starts from another checkpoint's weights (fine-tuning).

custom-dataset (E17): with --ss_share > 0, that share of crops comes from the custom dataset (real fingerstyle recordings
labelled with their tab, aligned; string labels at --ss_tab_weight). Validation reports custom-dataset string/fret and pitch F1
(ss_validation); the best checkpoint is still chosen on GuitarSet + GAPS.

SynthTab (E13): with --synth_share > 0, that share of crops comes from the selected SynthTab songs
(rendered fingerpicked acoustic guitars, exact pitches; tab-loss weight --synth_tab_weight, default 1, since its
string choices are the tab writer's and probably not audible). Validation reports
SynthTab string/fret and pitch F1 (synth_validation) but the best checkpoint is still chosen on real audio.

Distillation (E24): with --teacher (repeatable), the frozen teachers run on every batch and their averaged tab, onset
and pitch probabilities become soft targets; the loss is (1 - --distill) * label loss + --distill * soft loss.

Launch through scripts/train_mid*.sh (memory cap, low priority, resume), e.g.
    nohup experiments/train_mid_gaps.sh > runs/tabmodel_mid_gaps.log 2>&1 &
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.device import DEVICES, allocated_gb, empty_cache, limit_memory, pick_device  # noqa: E402
from guitab.data import load_split  # noqa: E402
from guitab.evaluation import evaluate_gaps, evaluate_guitarset, gaps_ref, ref_notes  # noqa: E402
from guitab.guitar import N_CLASSES, N_FRETS  # noqa: E402
from guitab.guitarset import jams_path  # noqa: E402
from guitab.inference import load_model  # noqa: E402
from guitab.model import SIZES, build_model, count_params, pitch_targets  # noqa: E402


def sample_batch(pools, batch, length, augment, rng, tab_weights, aug_share=0.0):
    """pools: [(share, tracks, length weights)]; returns x, tab, onset, unk, per-sample tab weight."""
    xs, tabs, onsets, unks, ws = [], [], [], [], []
    shares = [p[0] for p in pools]
    for _ in range(batch):
        _, tracks, weights = pools[rng.choices(range(len(pools)), shares)[0]]
        tr = tracks[rng.choices(range(len(tracks)), weights)[0]]
        n = len(tr["feat"])
        s = rng.randint(0, max(n - length, 0))
        feat = tr["feat"]
        if aug_share and tr.get("aug") and rng.random() < aug_share:
            feat = tr["aug"][rng.randrange(len(tr["aug"]))]
        x = np.asarray(feat[s:s + length], dtype=np.float32)
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


@torch.no_grad()
def teacher_targets(teachers, x):
    """Averaged teacher probabilities: tab [B, T, 6, 21], onset [B, T, 6], pitch [B, T, 48]."""
    outs = [t(x) for t in teachers]
    avg = lambda f: sum(f(o) for o in outs) / len(outs)  # noqa: E731
    return {"tab": avg(lambda o: torch.softmax(o["tab"].float(), -1)), "onset": avg(lambda o: torch.sigmoid(o["onset"].float())),
            "pitch": avg(lambda o: torch.sigmoid(o["pitch"].float()))}


def distill_loss(out, soft):
    """Soft cross-entropy on the tab and BCE on onsets and pitches against the teachers' probabilities (every frame)."""
    tab_loss = -(soft["tab"] * F.log_softmax(out["tab"].float(), -1)).sum(-1).mean()
    on_loss = F.binary_cross_entropy_with_logits(out["onset"].float(), soft["onset"])
    pitch_loss = F.binary_cross_entropy_with_logits(out["pitch"].float(), soft["pitch"])
    return tab_loss + on_loss + 0.5 * pitch_loss, tab_loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=DEVICES, help="auto = cuda, else mps, else cpu")
    parser.add_argument("--max_memory", type=float, default=None,
                        help="cap on this process's share of the GPU memory, e.g. 0.35 (keeps the machine usable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="mid", choices=SIZES)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=432, help="frames per crop (432 = 5 s)")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--accum", type=int, default=1, help="split each batch into this many chunks (gradient accumulation)")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--stop_at", default=None, help="HH:MM local time: save and exit at the first eval after it")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gaps_share", type=float, default=0.0, help="probability a crop comes from GAPS")
    parser.add_argument("--gaps_tab_weight", type=float, default=0.5, help="tab-loss weight of GAPS string labels")
    parser.add_argument("--max_steps_now", type=int, default=None, help="stop after this many steps (benchmarking)")
    parser.add_argument("--init", default=None, help="start from this checkpoint's weights (fresh optimizer, step 0)")
    parser.add_argument("--aug_copies", type=int, default=0, help="augmented audio copies per training track")
    parser.add_argument("--aug_share", type=float, default=0.0, help="probability a crop uses an augmented copy")
    parser.add_argument("--eval_degraded", action="store_true", help="also validate on the degraded copies")
    parser.add_argument("--synth_share", type=float, default=0.0, help="probability a crop comes from SynthTab")
    parser.add_argument("--synth_tab_weight", type=float, default=1.0, help="tab-loss weight of SynthTab string labels")
    parser.add_argument("--ss_share", type=float, default=0.0, help="probability a crop comes from the custom dataset")
    parser.add_argument("--ss_tab_weight", type=float, default=1.0, help="tab-loss weight of custom-dataset string labels")
    parser.add_argument("--teacher", action="append", default=[], help="teacher checkpoint for distillation (repeatable)")
    parser.add_argument("--distill", type=float, default=0.5, help="weight of the distillation loss (with --teacher)")
    args = parser.parse_args()

    device = pick_device(args.device)
    limit_memory(device, args.max_memory)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)

    train = load_split("train", aug_copies=args.aug_copies)
    val = load_split("validation", include_shift=False)
    ref_cache = {tr["name"]: ref_notes(jams_path(tr["name"])) for tr in val}
    weights = [len(t["feat"]) for t in train]
    pools = [(1.0 - args.gaps_share - args.synth_share - args.ss_share, train, weights)]
    print(f"GuitarSet train tracks {len(train)} ({sum(weights) / 86.13 / 3600:.1f} h), val tracks {len(val)}", flush=True)
    gaps_val, gaps_ref_cache = [], {}
    if args.gaps_share > 0:
        gtrain = load_split("gaps_train", aug_copies=args.aug_copies)
        gaps_val = load_split("gaps_validation")
        gaps_ref_cache = {tr["name"]: gaps_ref(tr["labels"]) for tr in gaps_val}
        gw = [len(t["feat"]) for t in gtrain]
        pools.append((args.gaps_share, gtrain, gw))
        print(f"GAPS train pieces {len(gtrain)} ({sum(gw) / 86.13 / 3600:.1f} h), val pieces {len(gaps_val)}, "
              f"share {args.gaps_share}", flush=True)
    tab_weights = {"guitarset": 1.0, "gaps": args.gaps_tab_weight, "synth": args.synth_tab_weight, "ss": args.ss_tab_weight}
    synth_val, synth_ref_cache = [], {}
    if args.synth_share > 0:
        strain = load_split("synth_train", aug_copies=args.aug_copies)
        synth_val = load_split("synth_validation")
        synth_ref_cache = {tr["name"]: gaps_ref(tr["labels"]) for tr in synth_val}
        sw = [len(t["feat"]) for t in strain]
        pools.append((args.synth_share, strain, sw))
        print(f"SynthTab train songs {len(strain)} ({sum(sw) / 86.13 / 3600:.1f} h), val songs {len(synth_val)}, "
              f"share {args.synth_share}", flush=True)
    ss_val, ss_ref_cache = [], {}
    if args.ss_share > 0:
        sstrain = load_split("ss_train", aug_copies=args.aug_copies)
        ss_val = load_split("ss_validation")
        ss_ref_cache = {tr["name"]: gaps_ref(tr["labels"]) for tr in ss_val}
        ssw = [len(t["feat"]) for t in sstrain]
        pools.append((args.ss_share, sstrain, ssw))
        print(f"custom-dataset train songs {len(sstrain)} ({sum(ssw) / 86.13 / 3600:.1f} h), val songs {len(ss_val)}, "
              f"share {args.ss_share}", flush=True)
    if args.eval_degraded:
        val_deg = load_split("validation", include_shift=False, variant="deg")
        gaps_val_deg = load_split("gaps_validation", variant="deg") if gaps_val else []

    model = build_model(args.size)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        print(f"initialised from {args.init}", flush=True)
    model = model.to(device)
    print(f"model {args.size}: {count_params(model) / 1e6:.1f}M params", flush=True)
    teachers = [load_model(p, device) for p in args.teacher]
    for p in args.teacher:
        print(f"teacher {p}, distillation weight {args.distill}", flush=True)
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
        x, tab, onset, unk, tab_w = sample_batch(pools, args.batch, args.length, True, rng, tab_weights, args.aug_share)
        x, tab, onset, unk, tab_w = x.to(device), tab.to(device), onset.to(device), unk.to(device), tab_w.to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)

        def batch_loss(pred, soft, tc, oc, uc, wc):
            loss, p = loss_fn(pred, tc, oc, uc, wc)
            if soft is not None:
                kd, kd_tab = distill_loss(pred, soft)
                loss = (1 - args.distill) * loss + args.distill * kd
                p["kd_tab"] = kd_tab
            return loss, p

        # teacher targets before the student's forward pass: its activations are not held while the teachers run
        if args.accum == 1:
            soft = teacher_targets(teachers, x) if teachers else None
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                pred = model(x)
            loss, parts = batch_loss(pred, soft, tab, onset, unk, tab_w)
            loss.backward()
        else:  # same batch, forward/backward in args.accum chunks (lower peak memory), gradients averaged
            parts = {}
            for xc, tc, oc, uc, wc in zip(*(v.chunk(args.accum) for v in (x, tab, onset, unk, tab_w))):
                soft = teacher_targets(teachers, xc) if teachers else None
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    pred = model(xc)
                loss, p = batch_loss(pred, soft, tc, oc, uc, wc)
                (loss / args.accum).backward()
                for k, v in p.items():
                    parts[k] = parts.get(k, 0.0) + v / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        for k, v in parts.items():
            run[k] = run.get(k, 0.0) + v
        if step % args.log_every == 0:
            el = time.time() - t0
            msg = {k: round(v / args.log_every, 4) for k, v in run.items()}
            print(f"step {step} {msg} lr {lr_at(step):.2e} {el / (step - start_step):.2f}s/step "
                  f"mem {allocated_gb(device):.1f}G", flush=True)
            log.write(json.dumps({"step": step, **msg}) + "\n")
            log.flush()
            run = {}
        if args.max_steps_now and step - start_step >= args.max_steps_now:
            break
        if step % args.eval_every == 0 or step == args.steps:
            sf, pf = evaluate_guitarset(model, val, device, ref_cache)
            rec = {"step": step, "val_sf": sf, "val_pitch": pf}
            sel = sf
            if gaps_val:
                gp, ga = evaluate_gaps(model, gaps_val, device, gaps_ref_cache)
                rec.update(gaps_pitch=gp, gaps_string_agree=ga)
                sel = 0.5 * (sf + gp)
            if args.eval_degraded:
                dsf, dpf = evaluate_guitarset(model, val_deg, device, ref_cache)
                rec.update(deg_val_sf=dsf, deg_val_pitch=dpf)
                deg_sel = dsf
                if gaps_val:
                    dgp, dga = evaluate_gaps(model, gaps_val_deg, device, gaps_ref_cache)
                    rec.update(deg_gaps_pitch=dgp, deg_gaps_string_agree=dga)
                    deg_sel = 0.5 * (dsf + dgp)
                sel = 0.5 * (sel + deg_sel)
            if synth_val:
                ssf, spf = evaluate_guitarset(model, synth_val, device, synth_ref_cache)
                rec.update(synth_sf=ssf, synth_pitch=spf)
            if ss_val:
                xsf, xpf = evaluate_guitarset(model, ss_val, device, ss_ref_cache)
                rec.update(ss_sf=xsf, ss_pitch=xpf)
            if sel > best:
                best = sel
                save("best.pt")
            save("last.pt")
            extra = f", GAPS pitch F1 {rec['gaps_pitch']:.4f}, GAPS string agreement {rec['gaps_string_agree']:.4f}" if gaps_val else ""
            if args.eval_degraded:
                extra += f" | degraded: val string/fret F1 {rec['deg_val_sf']:.4f}, pitch F1 {rec['deg_val_pitch']:.4f}"
                if gaps_val:
                    extra += f", GAPS pitch F1 {rec['deg_gaps_pitch']:.4f}, GAPS string agreement {rec['deg_gaps_string_agree']:.4f}"
            if synth_val:
                extra += f" | SynthTab val string/fret F1 {rec['synth_sf']:.4f}, pitch F1 {rec['synth_pitch']:.4f}"
            if ss_val:
                extra += f" | custom-dataset val string/fret F1 {rec['ss_sf']:.4f}, pitch F1 {rec['ss_pitch']:.4f}"
            print(f"EVAL step {step}: val string/fret F1 {sf:.4f}, pitch F1 {pf:.4f}{extra} (best {best:.4f})", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            empty_cache(device)
            if args.stop_at and dt.datetime.now().strftime("%H:%M") >= args.stop_at:
                print("stop_at reached, exiting", flush=True)
                return
    save("last.pt")


if __name__ == "__main__":
    main()
