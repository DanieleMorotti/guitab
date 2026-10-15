"""Fetch a small, selected SynthTab subset without downloading the 2 TB dataset.

The full dataset (https://github.com/AirLabUR/SynthTab, UR Box) stores each timbre as zip parts < 50 GB with
uncompressed FLAC members, so single songs are read with HTTP range requests. Steps (run in order):

    python scripts/synthtab_fetch.py labels     # label archive all_jams_midi.zip (1.1 GB) -> data/synthtab/
    python scripts/synthtab_fetch.py select     # index the 4 fingerpicked timbres, score songs, write selection.csv
    python scripts/synthtab_fetch.py download   # range-download the selected FLACs (~5.7 GB) + write labels

`select` is deterministic (seed 0), and its output data/synthtab/selection.csv is versioned, so `download`
alone reproduces the subset. Selection: standard tuning, 60-400 s, < 15% silence, mean polyphony 2.2-5,
15-85% of attacks are chords, 3-14 notes/s, 20-65% of notes on strings 4-6, frets <= 19; one version per
song, <= 3 songs per artist; 20 h synth_train + ~40 min synth_validation; timbres assigned round-robin.
Behind a TLS-intercepting proxy: uses data/certs/macos_trusted.pem when present.
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import random
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.paths import CERTS  # noqa: E402
from guitab.synthtab import LABEL_ZIP, SELECTION, STANDARD, SYNTHTAB, audio_path, label_path, selection, write_labels  # noqa: E402

BOX = "https://rochester.app.box.com"
LABEL_FILE_ID = 1486897975091
TIMBRES = {"martin_finger": 257580635059, "taylor_finger": 257580872780, "luthier_finger": 257582454069,
           "gibson_thumb": 257781141038}
CTX = ssl.create_default_context(cafile=str(CERTS) if CERTS.exists() else None)


def file_url(file_id):
    return f"{BOX}/index.php?rm=box_download_shared_file&vanity_name=SynthTab-Full&file_id=f_{file_id}"


def folder_items(folder_id):
    page = subprocess.run(["curl", "-sL", f"{BOX}/v/SynthTab-Full/folder/{folder_id}"], capture_output=True, text=True).stdout
    data = json.loads(re.search(r"postStreamData = (\{.*?\});\s*</script>", page, re.S).group(1))
    return [it for v in data.values() if isinstance(v, dict) and "items" in v for it in v["items"]]


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (zipfile on a remote archive)."""

    def __init__(self, url):
        self.share_url = url
        self.resolve()
        self.pos = 0

    def resolve(self):
        """Follow the share link to a (temporary) download URL."""
        with urllib.request.urlopen(urllib.request.Request(self.share_url, method="HEAD"), context=CTX) as r:
            self.url = r.geturl()
            self.size = int(r.headers.get("X-Box-Original-Content-Length") or r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(6):  # a proxy can occasionally drop a TLS connection
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
                    data = r.read()
                if len(data) == n:
                    break
            except (OSError, ssl.SSLError):
                if attempt == 5:
                    raise
            time.sleep(2 * (attempt + 1))
            self.resolve()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_zip(file_id):
    return zipfile.ZipFile(io.BufferedReader(HttpFile(file_url(file_id)), buffer_size=1 << 20))


# ---------------------------------------------------------------- song statistics (for the selection)
def _init_stats():
    global ZF, JAMS
    ZF = zipfile.ZipFile(LABEL_ZIP)
    JAMS = {n.split("/")[1]: n for n in ZF.namelist() if n.endswith(".jams")}


def song_stats(song):
    path = JAMS.get(song)
    if path is None:
        return song, None
    jams = json.loads(ZF.read(path))
    tabs = [a for a in jams["annotations"] if a["namespace"] == "note_tab"]
    tempo = [a for a in jams["annotations"] if a["namespace"] == "tempo"][0]["data"][0]["value"]
    tuning = [a["sandbox"]["open_tuning"] for a in sorted(tabs, key=lambda a: a["sandbox"]["string_index"])]
    sec = 60.0 / tempo / 960.0
    notes = [(x["time"] * sec, (x["time"] + x["duration"]) * sec, a["sandbox"]["string_index"], x["value"]["fret"])
             for a in tabs for x in a["data"]]
    if not notes:
        return song, None
    on, off = np.array([n[0] for n in notes]), np.array([n[1] for n in notes])
    frets, strings = np.array([n[3] for n in notes]), np.array([n[2] for n in notes])
    dur = float(off.max())
    grid = np.arange(0, dur, 0.05)
    sounding = ((on[None, :] <= grid[:, None]) & (off[None, :] > grid[:, None])).sum(1)
    active = sounding[sounding > 0]
    _, counts = np.unique(np.round(on, 3), return_counts=True)
    return song, {"dur": dur, "standard": tuning == STANDARD and len(tabs) == 6, "nps": len(notes) / max(dur, 1e-6),
                  "poly_mean": float(active.mean()) if len(active) else 0.0, "silence": float((sounding == 0).mean()),
                  "chord_onsets": float((counts >= 2).mean()), "hi_fret_share": float((frets > 19).mean()),
                  "bass_share": float((strings >= 4).mean())}


def eligible(v):
    return (v is not None and v["standard"] and 60 <= v["dur"] <= 400 and v["silence"] < 0.15 and 2.2 <= v["poly_mean"] <= 5
            and 0.15 <= v["chord_onsets"] <= 0.85 and 3 <= v["nps"] <= 14 and 0.2 <= v["bass_share"] <= 0.65
            and v["hi_fret_share"] < 0.02)


def song_key(song):
    parts = song.split(" - ")
    title = re.sub(r"\s*\(\d+\)$", "", parts[1].strip().lower()) if len(parts) > 1 else ""
    return parts[0].strip().lower(), title


def cmd_labels(_):
    SYNTHTAB.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sL", "-o", str(LABEL_ZIP), file_url(LABEL_FILE_ID)], check=True)
    print(f"{LABEL_ZIP} {LABEL_ZIP.stat().st_size / 1e9:.2f} GB")


def cmd_select(args):
    index = {}
    for timbre, folder in TIMBRES.items():
        for it in folder_items(folder):
            if it["name"].endswith(".zip"):
                for info in remote_zip(it["id"]).infolist():
                    if info.filename.endswith(".flac"):
                        index.setdefault(info.filename.split("/")[0], {})[timbre] = (it["id"], info.filename, info.file_size)
        print(f"indexed {timbre}", flush=True)
    songs = sorted(s for s, t in index.items() if len(t) == len(TIMBRES))
    with Pool(args.workers, initializer=_init_stats) as pool:
        stats = dict(pool.map(song_stats, songs, chunksize=50))
    cands = sorted(s for s in songs if eligible(stats[s]))
    print(f"{len(songs)} songs, {len(cands)} eligible ({sum(stats[s]['dur'] for s in cands) / 3600:.0f} h)")
    random.Random(0).shuffle(cands)
    seen, per_artist, chosen = set(), collections.Counter(), []
    for s in cands:
        artist, title = song_key(s)
        if (artist, title) not in seen and per_artist[artist] < 3:
            seen.add((artist, title))
            per_artist[artist] += 1
            chosen.append(s)
    timbres = list(TIMBRES)
    budget = {"synth_train": args.train_hours * 3600, "synth_validation": args.val_hours * 3600}
    hours = {k: 0.0 for k in budget}
    rows = []
    for i, s in enumerate(chosen):
        split = "synth_validation" if hours["synth_validation"] < budget["synth_validation"] and i % 10 == 0 else "synth_train"
        if hours[split] >= budget[split]:
            if split == "synth_train":
                break
            continue
        timbre = timbres[len(rows) % len(timbres)]
        file_id, member, size = index[s][timbre]
        v = stats[s]
        rows.append({"sid": hashlib.md5(s.encode()).hexdigest()[:10], "song": s, "timbre": timbre, "split": split,
                     "box_file": file_id, "member": member, "size": size, "dur": round(v["dur"], 1),
                     "poly_mean": round(v["poly_mean"], 2), "nps": round(v["nps"], 2)})
        hours[split] += v["dur"]
    with open(SELECTION, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} songs -> {SELECTION}: " + ", ".join(f"{k} {v / 3600:.2f} h" for k, v in hours.items())
          + f", download {sum(r['size'] for r in rows) / 1e9:.2f} GB")


def cmd_download(args):
    (SYNTHTAB / "audio").mkdir(parents=True, exist_ok=True)
    (SYNTHTAB / "labels").mkdir(parents=True, exist_ok=True)
    rows = selection()
    by_file = collections.defaultdict(list)
    for r in rows:
        if not audio_path(r["sid"]).exists():
            by_file[r["box_file"]].append(r)

    def fetch(file_id):
        zf = remote_zip(int(file_id))
        for r in by_file[file_id]:
            tmp = audio_path(r["sid"]).with_suffix(".part")
            with zf.open(r["member"]) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            tmp.rename(audio_path(r["sid"]))
        return file_id, len(by_file[file_id])

    with ThreadPoolExecutor(args.workers) as ex:
        for file_id, n in ex.map(fetch, list(by_file)):
            print(f"zip {file_id}: {n} songs", flush=True)
    labels = zipfile.ZipFile(LABEL_ZIP)
    for r in rows:
        if not label_path(r["sid"]).exists():
            write_labels(labels, r)
    print(f"{len(rows)} songs: audio {sum(audio_path(r['sid']).stat().st_size for r in rows) / 1e9:.2f} GB, labels written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["labels", "select", "download"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train_hours", type=float, default=20.0)
    parser.add_argument("--val_hours", type=float, default=0.67)
    args = parser.parse_args()
    {"labels": cmd_labels, "select": cmd_select, "download": cmd_download}[args.step](args)


if __name__ == "__main__":
    main()
