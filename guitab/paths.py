"""Repository locations. Data, weights and outputs live outside version control (see .gitignore); a published file
missing from weights/ is downloaded from the Hugging Face repository on first use (fetch)."""

import sys
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
# Beat This! model behind the beat grid (B1): "small0" (2M parameters, 3.6 MB) or "final0" (20M, 21 MB; ~3 points better
# rhythm on steady-tempo songs, the same on free-tempo ones; EXPERIMENTS.md, B1). Both versions use the same one.
BEAT_MODEL = "small0"
BEAT_THIS_CKPT = DATA / "models" / f"beat_this-{BEAT_MODEL}.ckpt"
CERTS = DATA / "certs" / "macos_trusted.pem"
WEIGHTS = ROOT / "weights"  # the published weights (Hugging Face), one flat folder
WEIGHTS_URL = "https://huggingface.co/danmor/guitab/resolve/main/"
# the browser version's beat tracker (BEAT_MODEL with int8 weights; scripts/export_web.py), published with the weights
BEAT_ONNX_NAME = f"beat_this_{BEAT_MODEL}_int8.onnx"
BEAT_ONNX = next((p for p in (WEIGHTS, ROOT / "docs" / "models") if (p / BEAT_ONNX_NAME).exists()), WEIGHTS) / BEAT_ONNX_NAME


def checkpoint(run, published=None):
    """runs/<run>/best.pt when trained here, else the published file in weights/ (if this model is published)."""
    trained = RUNS / run / "best.pt"
    return WEIGHTS / published if published and not trained.exists() else trained


BEST_CHECKPOINT = checkpoint("tabmodel_small_distill", "guitab_e25.safetensors")  # E25, distilled from E18 + E19 (train_distill.sh)
PUBLIC_CHECKPOINT = checkpoint("tabmodel_mid_close")  # E15, the best model trainable from public data only
# E25 (trained, downloaded, or downloaded on first use), unless only E15 is trained here
DEFAULT_CHECKPOINT = PUBLIC_CHECKPOINT if PUBLIC_CHECKPOINT.exists() and not BEST_CHECKPOINT.exists() else BEST_CHECKPOINT
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


def fetch(path):
    """path, after downloading it from WEIGHTS_URL if it is a published file missing from weights/."""
    path = Path(path)
    if path.exists() or path.resolve().parent != WEIGHTS.resolve():
        return path
    import shutil
    import urllib.request

    url = WEIGHTS_URL + path.name
    print(f"downloading {path.name} from {url} (once, to {path.parent})", file=sys.stderr, flush=True)
    part = path.with_name(path.name + ".part")
    try:
        WEIGHTS.mkdir(exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {url} ({e}); check the connection, or download the weights by hand "
                           "(README, Install)") from e
    part.rename(path)
    return path


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
