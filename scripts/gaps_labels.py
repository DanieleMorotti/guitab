"""Build per-string note labels for every GAPS piece (method: guitab/gaps.py).

    python scripts/gaps_labels.py --workers 4 [--only <hash> ...]
Output: data/gaps/labels/<safe(h)>.json and data/gaps/labels_summary.csv.
Run scripts/gaps_extract.py first.
"""

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from guitab.gaps import GAPS_SPLITS, build_labels, label_path  # noqa: E402
from guitab.paths import GAPS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    hashes = args.only or sorted({r["filename"][:-4] for r in csv.DictReader(open(GAPS_SPLITS)) if r["filename"].endswith(".wav")})
    with Pool(args.workers) as pool:
        results = pool.map(build_labels, hashes)
    rows = []
    for h, (row, payload) in zip(hashes, results):
        if payload is not None:
            label_path(h).parent.mkdir(exist_ok=True)
            label_path(h).write_text(json.dumps(payload))
        rows.append(row)
    keys = ["hash", "status", "notes", "match_rate", "content_sim", "standard_tuning", "tuning"]
    summary = GAPS / "labels_summary.csv"
    if args.only and summary.exists():  # update only the requested rows, keep the others
        rebuilt = {r["hash"]: r for r in rows}
        old = list(csv.DictReader(open(summary)))
        rows = [rebuilt.pop(r["hash"], r) for r in old] + list(rebuilt.values())
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(ok)}/{len(rows)} built; standard tuning {sum(str(r['standard_tuning']) == 'True' for r in ok)}; "
          f"median match rate {np.median([float(r['match_rate']) for r in ok]):.3f}")


if __name__ == "__main__":
    main()
