"""Error breakdown of a checkpoint on validation (and train subsets, to measure the generalisation gap).

    python scripts/analyze_errors.py --checkpoint runs/tabmodel_mid_gaps/best.pt --out runs/errors_tabmodel_mid_gaps.json

Sets: GuitarSet validation (player 04), GAPS validation, every 4th GuitarSet train track (unshifted),
every 9th GAPS train piece. Notes are extracted with DEFAULT_PARAMS and matched greedily (onset +-50 ms).
Each missed reference note and each false note gets one category (first rule that applies, see analyse()).
Never run on test splits.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.audio import FPS  # noqa: E402
from guitab.data import load_split  # noqa: E402
from guitab.evaluation import gaps_ref, ref_notes  # noqa: E402
from guitab.guitar import OPEN_PITCH, STRINGS  # noqa: E402
from guitab.inference import extract_notes, load_model, predict  # noqa: E402
TOL = 0.05


def match(ref, est, key, tol=TOL):
    """Greedy matching like evaluation.match_f1; returns list of (ref_idx, est_idx)."""
    by_id, used, pairs = {}, set(), []
    for i, n in enumerate(est):
        by_id.setdefault(tuple(n[k] for k in key), []).append(i)
    for j, r in enumerate(ref):
        c = [i for i in by_id.get(tuple(r[k] for k in key), []) if i not in used and abs(est[i]["onset"] - r["onset"]) <= tol]
        if c:
            i = min(c, key=lambda i: abs(est[i]["onset"] - r["onset"]))
            used.add(i)
            pairs.append((j, i))
    return pairs


def prf(tp, n_ref, n_est):
    p, r = tp / max(n_est, 1), tp / max(n_ref, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def analyse(items):
    """items: [(track name, label path, shift, tab probs, onset probs)]."""
    files = items
    C = Counter()
    fn_cat, fp_cat = Counter(), Counter()
    dts, per_track = [], []
    fret_err = Counter()
    fn_ctx, all_ctx = Counter(), Counter()
    frame = Counter()
    for stem, labels, shift, tab, onset in items:
        ref = gaps_ref(labels) if labels.endswith(".json") else ref_notes(labels, shift)
        ref = [r for r in ref if r["pitch"] is not None and 40 <= r["pitch"] <= 40 + 19 + 24]
        est = extract_notes(tab, onset)
        pp = match(ref, est, ("pitch",))
        known = [j for j, r in enumerate(ref) if r["string"]]
        C["ref"] += len(ref)
        C["est"] += len(est)
        C["tp_pitch"] += len(pp)
        C["known"] += len(known)
        if len(known) == len(ref):  # string F1 only where all strings known
            sp = match(ref, est, ("string", "fret"))
            C["tp_sf"] += len(sp)
            C["ref_sf"] += len(ref)
            C["est_sf"] += len(est)
            per_track.append((stem, prf(len(sp), len(ref), len(est))[2], prf(len(pp), len(ref), len(est))[2]))
        else:
            per_track.append((stem, None, prf(len(pp), len(ref), len(est))[2]))
        # string accuracy on pitch-correct notes with a known string
        for j, i in pp:
            r, e = ref[j], est[i]
            dts.append(e["onset"] - r["onset"])
            if r["string"]:
                C["pc_known"] += 1
                if e["string"] == r["string"]:
                    C["pc_string_ok"] += 1
                else:
                    fret_err[(abs(STRINGS.index(e["string"]) - STRINGS.index(r["string"])))] += 1
                    C["wrong_str_highfret"] += int(max(e["fret"], r["fret"]) >= 5)
        matched_ref = {j for j, _ in pp}
        matched_est = {i for _, i in pp}
        T = len(tab)
        # ---- false negatives
        for j, r in enumerate(ref):
            t = int(round(r["onset"] * FPS))
            sounding = sum(1 for q in ref if q["onset"] - 0.03 <= r["onset"] < q["offset"])
            dur = r["offset"] - r["onset"]
            ctx = ("poly>=4" if sounding >= 4 else "poly2-3" if sounding >= 2 else "mono") + "|" + (
                "short<120ms" if dur < 0.12 else "long")
            all_ctx[ctx] += 1
            if j in matched_ref or t >= T:
                continue
            fn_ctx[ctx] += 1
            near = [e for i, e in enumerate(est) if i not in matched_est and abs(e["onset"] - r["onset"]) <= TOL]
            late = [e for i, e in enumerate(est) if e["pitch"] == r["pitch"] and TOL < abs(e["onset"] - r["onset"]) <= 0.15]
            # model evidence for the pitch: best string able to play it
            cand = [(c, r["pitch"] - OPEN_PITCH[s]) for c, s in enumerate(STRINGS) if 0 <= r["pitch"] - OPEN_PITCH[s] <= 19]
            a, b = max(t - 3, 0), min(t + 4, T)
            fr = max((tab[t:min(t + 6, T), c, fr + 1].mean() for c, fr in cand), default=0)
            on_any = onset[a:b].max()
            prev_same = any(q is not r and q["pitch"] == r["pitch"] and q["onset"] < r["onset"] and r["onset"] - q["offset"] < 0.06
                            for q in ref)
            if late:
                cat = "timing (same pitch 50-150ms off)"
            elif any(abs(e["pitch"] - r["pitch"]) in (12, 19, 24) for e in near):
                cat = "octave/harmonic confusion"
            elif any(abs(e["pitch"] - r["pitch"]) in (1, 2) for e in near):
                cat = "semitone/tone confusion"
            elif prev_same and fr > 0.5:
                cat = "re-attack of a ringing same pitch (merged)"
            elif fr > 0.5 and on_any < 0.6:
                cat = "heard (frame) but no onset >= 0.6"
            elif fr > 0.5:
                cat = "heard, onset on another note/string (chord collision)"
            elif fr > 0.2:
                cat = "weak frame evidence (0.2-0.5)"
            else:
                cat = "not heard at all"
            fn_cat[cat] += 1
        # ---- false positives
        for i, e in enumerate(est):
            if i in matched_est:
                continue
            sounding = [r for r in ref if r["onset"] - 0.05 <= e["onset"] < r["offset"]]
            if any(r["pitch"] == e["pitch"] and TOL < abs(r["onset"] - e["onset"]) <= 0.15 for r in ref):
                cat = "timing (same pitch 50-150ms off)"
            elif any(r["pitch"] == e["pitch"] for r in sounding):
                cat = "re-trigger of a ringing ref note"
            elif any(abs(r["pitch"] - e["pitch"]) in (12, 19, 24) for r in sounding):
                cat = "octave/harmonic of a ref note"
            elif any(abs(r["pitch"] - e["pitch"]) in (1, 2) for r in sounding):
                cat = "semitone/tone off a ref note"
            elif not sounding:
                cat = "in silence / after release"
            else:
                cat = "other pitch while notes sound"
            fp_cat[cat] += 1
        # frame-level multipitch
        ref_roll = np.zeros((T, 128), bool)
        for r in ref:
            ref_roll[int(round(r["onset"] * FPS)):int(round(r["offset"] * FPS)), r["pitch"]] = True
        est_roll = np.zeros((T, 128), bool)
        act = tab[:, :, 0] < 0.5
        fr_arg = tab[:, :, 1:].argmax(-1)
        for c, s in enumerate(STRINGS):
            tt = np.where(act[:, c])[0]
            est_roll[tt, OPEN_PITCH[s] + fr_arg[tt, c]] = True
        if len(known) == len(ref):
            frame["tp"] += (ref_roll & est_roll).sum()
            frame["ref"] += ref_roll.sum()
            frame["est"] += est_roll.sum()
    dts = np.abs(np.array(dts))
    out = {
        "n_tracks": len(files),
        "pitch_prf": [round(x, 3) for x in prf(C["tp_pitch"], C["ref"], C["est"])],
        "string_fret_prf": [round(x, 3) for x in prf(C["tp_sf"], C["ref_sf"], C["est_sf"])] if C["ref_sf"] else None,
        "string_acc_given_pitch": round(C["pc_string_ok"] / max(C["pc_known"], 1), 3),
        "wrong_string_by_distance": dict(sorted(fret_err.items())),
        "frame_multipitch_prf": [round(x, 3) for x in prf(frame["tp"], frame["ref"], frame["est"])] if frame["ref"] else None,
        "onset_abs_err_ms": {"median": round(1000 * float(np.median(dts)), 1), "p90": round(1000 * float(np.percentile(dts, 90)), 1),
                             "share<=25ms": round(float((dts <= 0.025).mean()), 3)},
        "n_ref": C["ref"], "n_est": C["est"], "n_fn": C["ref"] - C["tp_pitch"], "n_fp": C["est"] - C["tp_pitch"],
        "fn_categories": {k: [v, round(v / max(C['ref'] - C['tp_pitch'], 1), 3)] for k, v in fn_cat.most_common()},
        "fp_categories": {k: [v, round(v / max(C['est'] - C['tp_pitch'], 1), 3)] for k, v in fp_cat.most_common()},
        "miss_rate_by_context": {k: [all_ctx[k], round(fn_ctx[k] / all_ctx[k], 3)] for k in sorted(all_ctx)},
        "per_track_pitch_f1": {"min": round(min(p for _, _, p in per_track), 3), "median": round(float(np.median([p for _, _, p in per_track])), 3),
                               "max": round(max(p for _, _, p in per_track), 3)},
    }
    if all(not n.startswith("gaps_") for n, _, _ in per_track):
        for style in ("comp", "solo"):
            v = [(s, p) for n, s, p in per_track if n.endswith(style)]
            out[f"{style}_sf_pitch_f1"] = [round(float(np.mean([a for a, _ in v])), 3), round(float(np.mean([b for _, b in v])), 3)]
        genres = Counter()
        for n, s, p in per_track:
            g = n.split("_")[1].split("-")[0].rstrip("0123456789")
            genres[g] += 1
        out["sf_by_genre"] = {g: round(float(np.mean([s for n, s, _ in per_track if n.split("_")[1].split("-")[0].rstrip("0123456789") == g])), 3)
                              for g in genres}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None, help="write the breakdown as JSON")
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    sets = {
        "gs_val": load_split("validation", include_shift=False),
        "gaps_val": load_split("gaps_validation"),
        "gs_train": load_split("train", include_shift=False)[::4],
        "gaps_train": load_split("gaps_train")[::9],
    }
    res = {}
    for name, trs in sets.items():
        items = [(tr["name"], tr["labels"], tr["shift"], *predict(model, tr["feat"], device)) for tr in trs]
        res[name] = analyse(items)
        print(name, json.dumps(res[name]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
