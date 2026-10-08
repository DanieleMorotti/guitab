"""Compute CQT features of augmented audio (guitab/augment.py) into data/tabmodel_cache_aug/.

    python scripts/build_aug_cache.py --copies 2 --workers 4   # training copies: <name>__aug<k>.npy for train, gaps_train, synth_train
    python scripts/build_aug_cache.py --degraded --workers 4   # held-out degraded copies: <name>__deg.npy for the
                                                                # validation / test splits (GuitarSet and GAPS)
    python scripts/build_aug_cache.py --sharp --workers 4      # time-sharp low band (E14) of the clean features and of
                                                                # every existing aug / deg copy -> data/tabmodel_cache_sharp/
Labels are the original track's (effects keep timing and length). Existing files are kept. Each copy's
random effect chain is seeded by (track name, copy), and listed in data/tabmodel_cache_aug/chains.tsv; --sharp
re-renders the same audio (deterministic, checked bit-identical CQT). SynthTab is skipped by --sharp (not in E14).
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.audio import compute_cqt, compute_sharp, load_audio  # noqa: E402
from guitab.augment import degrade, seed_for, train_chain  # noqa: E402
from guitab.data import aug_path, sharp_path, tracks  # noqa: E402
from guitab.paths import AUG_CACHE, SHARP_CACHE  # noqa: E402

TRAIN_SPLITS = ("train", "gaps_train", "synth_train")
EVAL_SPLITS = ("validation", "test", "gaps_validation", "gaps_test")


def render(y, name, variant):
    if variant is None:
        return y, "clean"
    if variant == "deg":
        return degrade(y, name), "degrade"
    return train_chain(y, np.random.default_rng(seed_for(name, variant)))


def build(job):
    name, audio, variant, sharp = job
    path = sharp_path(name, variant) if sharp else aug_path(name, variant)
    if path.exists():
        return name, variant, "kept"
    out, desc = render(load_audio(audio), name, variant)
    np.save(path, compute_sharp(out) if sharp else compute_cqt(out))
    return name, variant, "sharp" if sharp else desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=0, help="augmented training copies per track")
    parser.add_argument("--degraded", action="store_true", help="degraded copies of the validation / test splits")
    parser.add_argument("--sharp", action="store_true", help="sharp low band of the clean and existing aug / deg features")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    AUG_CACHE.mkdir(parents=True, exist_ok=True)
    SHARP_CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, audio, _, shift, split in tracks():
        if args.sharp:
            if not split.startswith("synth_"):
                variants = [None] + [v for v in ("aug0", "aug1", "aug2", "deg") if aug_path(name, v).exists()]
                jobs += [(name, audio, v, True) for v in variants]
            continue
        if split in TRAIN_SPLITS:
            jobs += [(name, audio, f"aug{k}", False) for k in range(args.copies)]
        if args.degraded and split in EVAL_SPLITS:
            jobs.append((name, audio, "deg", False))
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(args.workers) as pool, open(AUG_CACHE / "chains.tsv", "a") as log:
        for i, (name, variant, desc) in enumerate(pool.imap_unordered(build, jobs), 1):
            if desc not in ("kept", "sharp"):
                log.write(f"{name}\t{variant}\t{desc}\n")
                log.flush()
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {name} {variant}: {desc}", flush=True)


if __name__ == "__main__":
    main()
