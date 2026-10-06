"""Compute and cache CQT features + frame labels for every track (GuitarSet, shifted copies, GAPS selection).

    python scripts/build_cache.py --workers 4      # -> data/tabmodel_cache/<name>.npz (existing files are kept)
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.data import cache_one, tracks  # noqa: E402
from guitab.paths import CACHE  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    items = tracks()
    with Pool(args.workers) as pool:
        for i, name in enumerate(pool.imap_unordered(cache_one, items), 1):
            if i % 20 == 0 or i == len(items):
                print(f"{i}/{len(items)} {name}", flush=True)


if __name__ == "__main__":
    main()
