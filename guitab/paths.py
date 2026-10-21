"""Repository locations. Data, weights and outputs live outside version control (see .gitignore)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
CACHE = DATA / "tabmodel_cache"           # CQT features + frame labels (scripts/build_cache.py)
AUG_CACHE = DATA / "tabmodel_cache_aug"   # CQT of augmented / degraded audio (scripts/build_aug_cache.py)
GUITARSET = DATA / "guitarset"            # annotation/ (JAMS), audio_mic/
GUITARSET_SHIFT = DATA / "guitarset_shift"
GAPS = DATA / "gaps"
EVAL_CHUNKS = DATA / "eval_chunks.json"   # frozen GuitarSet evaluation chunks (versioned)
BEAT_THIS_CKPT = DATA / "models" / "beat_this-final0.ckpt"
CERTS = DATA / "certs" / "macos_trusted.pem"
BEST_CHECKPOINT = RUNS / "tabmodel_small_distill" / "best.pt"  # E25, distilled from E18 + E19 (experiments/train_distill.sh)
PUBLIC_CHECKPOINT = RUNS / "tabmodel_mid_close" / "best.pt"  # E15, the best model trainable from public data only
# E25 when trained (it needs the custom dataset), else E15
DEFAULT_CHECKPOINT = BEST_CHECKPOINT if BEST_CHECKPOINT.exists() else PUBLIC_CHECKPOINT
TABLM_CHECKPOINT = RUNS / "tablm" / "tablm.pt"  # tab language model for --music_lm (scripts/train_tablm.py)
# checkpoints offered by the web app (guitab/app.py), first available = preselected
MODELS = {
    "E25": (RUNS / "tabmodel_small_distill" / "best.pt",
            "E25 · small model (6M) distilled from E18 + E19: E19 quality, 1.6x faster, best on phone/room audio"),
    "E19": (RUNS / "tabmodel_mid_all" / "best.pt", "E19 · mid model, all data (best on clean steel-string guitar)"),
    "E24": (RUNS / "tabmodel_half_distill" / "best.pt", "E24 · tiny distilled model (2.9M), E20 quality, 2.4x faster than E19"),
    "E15": (RUNS / "tabmodel_mid_close" / "best.pt", "E15 · mid model, public data only"),
    "E18": (RUNS / "tabmodel_large_all" / "best.pt", "E18 · large model, robust to phone/room audio, 2.5x slower"),
    "E20": (RUNS / "tabmodel_small_all" / "best.pt", "E20 · small model (6M), all data, ~1 point below E19 on clean audio"),
}


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
