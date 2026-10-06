"""GAPS file naming that survives case-insensitive file systems (macOS).

GAPS ids (scorehash, e.g. "GM1wc" vs "gM1wc") are case-sensitive; unzipping the dataset on macOS
silently overwrote 278 files whose names differ only by case. Everything GAPS-related is stored
under safe(h) = h + "_" + md5(h)[:6] instead, extracted straight from the zip by extract().

    python tools/gaps_common.py     # (re)extract data/gaps/gaps_v1_no_audio.zip -> data/gaps/safe/
"""

import hashlib
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAPS = ROOT / "data" / "gaps"
SAFE = GAPS / "safe"  # midi/, musicxml/, syncpoints/ with safe names; csv files copied as-is


def safe(h):
    return f"{h}_{hashlib.md5(h.encode()).hexdigest()[:6]}"


def midi_path(h):
    return SAFE / "midi" / f"{safe(h)}-fine-aligned.mid"


def xml_path(h):
    return SAFE / "musicxml" / f"{safe(h)}.xml"


def audio_path(h):
    return GAPS / "audio" / f"{safe(h)}.wav"


def label_path(h):
    return GAPS / "labels" / f"{safe(h)}.json"


def extract():
    z = zipfile.ZipFile(GAPS / "gaps_v1_no_audio.zip")
    n = 0
    for name in z.namelist():
        base = name.split("/")[-1]
        if not base:
            continue
        sub = name.split("/")[-2]
        m = re.match(r"^(.{5})(-fine-aligned\.mid|-syncpoints\.json|\.xml)$", base)
        out = SAFE / sub / (f"{safe(m.group(1))}{m.group(2)}" if m else base)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(name))
        n += 1
    print(f"extracted {n} files to {SAFE}")


if __name__ == "__main__":
    extract()
