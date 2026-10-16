"""Score checkpoints on the user's own recording against its aligned reference tab (see EXPERIMENTS.md, user-song benchmark).

    python scripts/evaluate_song.py --checkpoints runs/tabmodel_mid_close/best.pt [more ...]

Reference: data/my_video_reference.json (507 notes, aligned to out/my_video/my_video.wav). Matching: onset +-100 ms
(the alignment is approximate). Reports pitch and tab F1, bass recall (strings 4-6), and how completely groups of notes
struck together (attacks within 30 ms) are reproduced: single notes, 2-note groups, 3+ note chords.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.device import DEVICES, pick_device  # noqa: E402
from guitab.audio import compute_cqt, load_audio  # noqa: E402
from guitab.evaluation import match, match_f1  # noqa: E402
from guitab.inference import extract_notes, load_model, predict  # noqa: E402
from guitab.paths import DATA, ROOT  # noqa: E402

TOL = 0.1


def groups_of(ref):
    order = sorted(range(len(ref)), key=lambda j: ref[j]["onset"])
    out, used = [], set()
    for j in order:
        if j not in used:
            g = [k for k in order if k not in used and 0 <= ref[k]["onset"] - ref[j]["onset"] < 0.03]
            used.update(g)
            out.append(g)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=DEVICES, help="auto = cuda, else mps, else cpu")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--reference", default=str(DATA / "my_video_reference.json"))
    parser.add_argument("--audio", default=str(ROOT / "out" / "my_video" / "my_video.wav"))
    parser.add_argument("--music_lm", action="store_true", help="decode with the tab language model (guitab/tablm.py)")
    args = parser.parse_args()
    ref = json.loads(Path(args.reference).read_text())
    end = max(r["onset"] for r in ref) + 0.5
    groups = groups_of(ref)
    device = pick_device(args.device)
    feat = compute_cqt(load_audio(args.audio))
    print(f"reference: {len(ref)} notes; groups: {sum(len(g) == 1 for g in groups)} single, {sum(len(g) == 2 for g in groups)} "
          f"of 2, {sum(len(g) >= 3 for g in groups)} of 3+")
    for ck in args.checkpoints:
        tab, onset = predict(load_model(ck, device), feat, device)
        if args.music_lm:
            from guitab.paths import TABLM_CHECKPOINT
            from guitab.tablm import DEFAULT_DECODING, decode_with_lm, load_tablm

            notes = decode_with_lm(tab, onset, load_tablm(TABLM_CHECKPOINT, device), device, **DEFAULT_DECODING)
        else:
            notes = extract_notes(tab, onset)
        est = [e for e in notes if e["onset"] <= end]
        hit = {j for j, _ in match(ref, est, ("pitch",), tol=TOL)}
        parts = []
        for name, sel in (("single", lambda g: len(g) == 1), ("2-note", lambda g: len(g) == 2), ("3+", lambda g: len(g) >= 3)):
            gs = [g for g in groups if sel(g)]
            parts.append(f"{name} complete {np.mean([all(j in hit for j in g) for g in gs]):.2f} "
                         f"(notes {np.mean([j in hit for g in gs for j in g]):.2f})")
        bass = np.mean([j in hit for j, r in enumerate(ref) if r["string"] >= 4])
        print(f"{ck}: {len(est)} notes | pitch F1 {match_f1(ref, est, key=('pitch',), tol=TOL):.3f}, tab F1 "
              f"{match_f1(ref, est, tol=TOL):.3f}, bass recall {bass:.2f} | " + ", ".join(parts), flush=True)


if __name__ == "__main__":
    main()
