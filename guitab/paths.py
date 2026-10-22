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
WEIGHTS = ROOT / "weights"  # the published weights (Hugging Face), one flat folder
# the browser version's beat tracker (Beat This! final0, int8 weights; scripts/export_web.py), published with the weights
BEAT_ONNX = next((p for p in (WEIGHTS, ROOT / "docs" / "models") if (p / "beat_this_final0_int8.onnx").exists()),
                 WEIGHTS) / "beat_this_final0_int8.onnx"


def checkpoint(run, published=None):
    """runs/<run>/best.pt when trained here, else the published file in weights/ (if this model is published)."""
    trained = RUNS / run / "best.pt"
    return WEIGHTS / published if published and not trained.exists() else trained


BEST_CHECKPOINT = checkpoint("tabmodel_small_distill", "guitab_e25.safetensors")  # E25, distilled from E18 + E19 (train_distill.sh)
PUBLIC_CHECKPOINT = checkpoint("tabmodel_mid_close")  # E15, the best model trainable from public data only
# E25 when trained or downloaded, else E15
DEFAULT_CHECKPOINT = BEST_CHECKPOINT if BEST_CHECKPOINT.exists() else PUBLIC_CHECKPOINT
TABLM_RUN = RUNS / "tablm" / "tablm.pt"  # where scripts/train_tablm.py writes the tab language model (--music_lm)
TABLM_CHECKPOINT = TABLM_RUN if TABLM_RUN.exists() else WEIGHTS / "tablm.safetensors"
# checkpoints offered by the web app (guitab/app.py), first available = preselected
MODELS = {
    "E25": (BEST_CHECKPOINT, "E25 · small distilled model (6M): best on phone recordings, 1.5x faster than E19"),
    "E19": (checkpoint("tabmodel_mid_all", "guitab_e19.safetensors"), "E19 · mid model, all data, E25's teacher"),
    "E24": (checkpoint("tabmodel_half_distill", "guitab_e24.safetensors"), "E24 · tiny distilled model (2.9M), 2.3x faster"),
    "E15": (checkpoint("tabmodel_mid_close"), "E15 · mid model, public data only"),
    "E18": (checkpoint("tabmodel_large_all"), "E18 · large model, robust to phone/room audio, 2.5x slower"),
    "E20": (checkpoint("tabmodel_small_all"), "E20 · small model (6M), all data, ~1 point below E19 on clean audio"),
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
