"""Download GAPS audio (YouTube) for every piece in gaps_v1_splits.csv -> data/gaps/audio/<safe(h)>.wav.

safe(h) (guitab/gaps.py) keeps ids that differ only by case apart on macOS.

22.05 kHz mono WAV. The fine-aligned MIDI is in the timeline of the full YouTube audio (checked on
mvswc: best offset 0.02 s). On this network HTTPS is re-signed by a corporate proxy, so yt-dlp
uses the macOS trusted certificates exported to data/certs/macos_trusted.pem:
    (security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain;
     security find-certificate -a -p /Library/Keychains/System.keychain) > data/certs/macos_trusted.pem

    python scripts/gaps_download.py --workers 3   # run twice to retry transient 403 errors
"""

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.gaps import GAPS_METADATA, GAPS_SPLITS, audio_path  # noqa: E402
from guitab.paths import CERTS, GAPS, ROOT  # noqa: E402

YTDLP = ROOT / ".venv" / "bin" / "yt-dlp"


def entries():
    meta = {r["scorehash"]: r for r in csv.DictReader(open(GAPS_METADATA))}
    out, seen = [], set()
    for r in csv.DictReader(open(GAPS_SPLITS)):
        if not r["filename"].endswith(".wav"):
            continue
        h = r["filename"][:-4]
        if h in meta and h not in seen:
            seen.add(h)
            out.append((h, meta[h]["yt_id"], r["split"]))
    return out


def download(item):
    h, yt_id, _ = item
    out = audio_path(h)
    if out.exists():
        return h, "cached"
    env = {**os.environ, "SSL_CERT_FILE": str(CERTS)}
    cmd = [str(YTDLP), "-q", "--no-warnings", "--compat-options", "no-certifi", "-x", "--audio-format", "wav",
           "--postprocessor-args", "ffmpeg:-ar 22050 -ac 1", "-o", str(out.with_suffix(".%(ext)s")),
           f"https://www.youtube.com/watch?v={yt_id}"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return h, "ok" if out.exists() else f"FAILED {res.stderr.strip()[-200:]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    (GAPS / "audio").mkdir(parents=True, exist_ok=True)
    items = entries()
    print(f"{len(items)} entries", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (h, status) in enumerate(pool.map(download, items), 1):
            print(f"{i}/{len(items)} {h} {status}", flush=True)


if __name__ == "__main__":
    main()
