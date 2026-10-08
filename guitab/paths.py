"""Repository locations. Data, weights and outputs live outside version control (see .gitignore)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
CACHE = DATA / "tabmodel_cache"           # CQT features + frame labels (scripts/build_cache.py)
AUG_CACHE = DATA / "tabmodel_cache_aug"   # CQT of augmented / degraded audio (scripts/build_aug_cache.py)
SHARP_CACHE = DATA / "tabmodel_cache_sharp"  # time-sharp low band of every cached feature (scripts/build_aug_cache.py --sharp)
GUITARSET = DATA / "guitarset"            # annotation/ (JAMS), audio_mic/
GUITARSET_SHIFT = DATA / "guitarset_shift"
GAPS = DATA / "gaps"
EVAL_CHUNKS = DATA / "eval_chunks.json"   # frozen GuitarSet evaluation chunks (versioned)
BEAT_THIS_CKPT = DATA / "models" / "beat_this-final0.ckpt"
CERTS = DATA / "certs" / "macos_trusted.pem"
DEFAULT_CHECKPOINT = RUNS / "tabmodel_mid_gaps_aug" / "best.pt"


def resolve(path):
    """Absolute path; relative paths (as stored in the manifests) are taken from the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def relative(path):
    """Path relative to the repository root when inside it (for manifests), else unchanged."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
