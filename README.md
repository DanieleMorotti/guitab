# guitab — guitar audio → Guitar Pro tab

Transcribes a guitar recording, with a focus on fingerstyle, into a tab: a `.gp5` file you can open in Guitar Pro, TuxGuitar,
MuseScore or any online gp5 player.

The model reads a harmonic constant-Q spectrogram and predicts, at every frame (11.6 ms), the fret played on each of the
6 strings plus note attacks. Notes are then snapped to the beats found by a beat tracker and written out as a tab.

Engines:

| Engine | What it is | GuitarSet test, string+fret F1 (±50 ms) | GAPS test, fingerstyle pitch F1 |
|---|---|---|---|
| `tabmodel` (default) | Our CQT frame/onset tab model, trained on GuitarSet + GAPS (`runs/tabmodel_mid_gaps/best.pt`) | **0.826** | **0.787** |
| `basic_pitch` | Spotify Basic Pitch notes + our fingering optimiser (no training) | 0.32 (pitch 0.71) | – |

The full history (including the original Whisper / Audio2Tab approach, which was replaced because it scored 0.13),
every setting and every result are in [`EXPERIMENTS.md`](EXPERIMENTS.md). Add a new entry there for every new experiment.

---

## 1. Quick start: transcribe your own recording

```bash
# 1) extract mono audio from a video or any audio file
ffmpeg -i my_video.mov -ac 1 -ar 22050 out/my_video/my_video.wav

# 2) transcribe (default engine and checkpoint = best model)
.venv/bin/python tools/transcribe.py --audio out/my_video/my_video.wav --out_dir out/my_video
```

The output is `out/my_video/my_video.gp5`, plus `my_video.tokens.txt` (the tab as text tokens).

Options of `tools/transcribe.py`:

| Option | Effect |
|---|---|
| `--onset 0.4` | Keeps less confident notes, so more notes and more false ones. The default 0.6 was tuned on GuitarSet validation; try 0.4–0.5 if notes are missing on your guitar. |
| `--orphan` | Also keeps notes without a detected attack. It was worse on GuitarSet, but may recover quiet or sustained notes. |
| `--fingering low` | Re-fingers everything towards first position (Viterbi optimiser, `tools/fingering.py`). The default `model` keeps the model's strings; `auto` re-fingers only when clearly awkward. |
| `--tempo 90` | Uses a fixed tempo. The default `auto` follows the beats detected by Beat This!. |
| `--grid 120` | Quantisation grid in ticks (960 = quarter): 240 = 16th notes (default), 120 = 32nd notes. |
| `--checkpoint PATH` | Another tab-model checkpoint, e.g. `runs/tabmodel_mid/best.pt` (GuitarSet only). |
| `--engine basic_pitch` | Uses Basic Pitch (runs in `.venv_bp`; strings are chosen by the fingering optimiser). |
| `--name X` | Output file stem. |

Inference needs about 1–2 GB of GPU memory and takes under a minute for a 4-minute song on an M-series Mac.

---

## 2. Repository layout

```
guitab/
├── README.md, EXPERIMENTS.md     this file / experiment log
├── tools/                        all scripts (section 5)
├── .venv/                        main Python environment (requirements.txt)
├── .venv_bp/                     separate environment for Basic Pitch (conflicting dependencies)
├── data/
│   ├── guitarset/                GuitarSet: annotation/ (JAMS), audio_mic/, tokens/ (reference tabs)
│   ├── guitarset.csv             track list + split (players 00-03 train, 04 validation, 05 test)
│   ├── guitarset_shift/ (+.csv)  train players pitch-shifted +1/+2 semitones
│   ├── gaps/                     GAPS: zip, safe/ (case-safe extraction), audio/, labels/, labels_summary.csv
│   ├── tabmodel_cache/           cached CQT features + frame labels (.npz) used for training/evaluation
│   ├── eval_chunks.json          frozen GuitarSet validation/test chunks used by every reported number
│   ├── basic_pitch/              cached Basic Pitch notes
│   ├── models/                   Beat This! beat-tracker checkpoint
│   └── certs/                    macOS trusted certificates (corporate HTTPS proxy, section 6)
├── runs/
│   ├── tabmodel_mid_gaps/        E10 best model (GuitarSet + GAPS): best.pt + log.jsonl
│   ├── tabmodel_mid/             E8 GuitarSet-only model: best.pt + log.jsonl
│   ├── train_tabmodel_*.sh       launch scripts of those runs
│   └── *.log, sweep_*.json       training logs and the threshold/fingering sweep
└── out/                          your transcriptions
```

---

## 3. Environments

All commands run from `guitab/` with Python 3.11.

```bash
# main environment: exact versions in requirements.txt (direct dependencies listed in requirements.in)
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt

# Basic Pitch environment (only for --engine basic_pitch; conflicting dependencies)
uv venv .venv_bp --python 3.11
uv pip install --python .venv_bp/bin/python "basic-pitch[onnx]" "setuptools<70"
```

- **Versions:** torch 2.10, librosa 0.11, beat-this (git commit pinned in `requirements.txt`), music21 10.5, pretty-midi 0.2.11,
  pyguitarpro 0.9.3, basic-pitch 0.4.
- **Old environment:** `env_freeze_audio2tab_old.txt` is the full freeze of the former Audio2Tab environment, kept for reference.
- **Hardware:** on a Mac, everything runs on MPS (Apple GPU) automatically.

---

## 4. Training and evaluation

### 4.1 Data preparation (one-off)

**GuitarSet** (CC-BY; [Zenodo 3371780](https://zenodo.org/records/3371780)): download `annotation.zip` and `audio_mono-mic.zip` into `data/guitarset/` and unzip them into `annotation/` and `audio_mic/`. Then:
```bash
.venv/bin/python tools/guitarset_to_tokens.py   # reference tabs (tokens/) + data/guitarset.csv
.venv/bin/python tools/guitarset_shift.py       # +1/+2 semitone copies of the train players
```

**GAPS** (classical / fingerstyle, CC BY-NC-SA; [Zenodo 13962272](https://zenodo.org/records/13962272)):
```bash
mkdir -p data/gaps && curl -L -o data/gaps/gaps_v1_no_audio.zip \
  "https://zenodo.org/records/13962272/files/gaps_v1_no_audio.zip?download=1"
(cd data/gaps && unzip -o gaps_v1_no_audio.zip "gaps_v1/*.csv")   # only the csv files are safe to unzip
.venv/bin/python tools/gaps_common.py                   # case-safe extraction of MIDI / MusicXML
.venv/bin/python tools/gaps_download.py --workers 3     # audio from YouTube; run twice to retry 403 errors
.venv/bin/python tools/gaps_labels.py --workers 4       # string/fret labels: score tab staff aligned to the MIDI
```
> GAPS ids differ only by upper/lower case (`GM1wc` ≠ `gM1wc`). A normal unzip on macOS silently overwrites files,
> so always use `tools/gaps_common.py`, which stores everything under case-safe names.

**Feature cache** (CQT + frame labels for GuitarSet, the shifted copies and the selected GAPS pieces; about 10 min, 2.6 GB):
```bash
.venv/bin/python tools/tabdata.py --workers 4
```
The GAPS selection (standard tuning, MIDI matches the score, ≥ 50% of notes with a string label) is in `tabdata.gaps_selection`. It gives 123 train, 14 validation and 19 test pieces.

### 4.2 Training

```bash
nohup runs/train_tabmodel_mid_gaps.sh > runs/tabmodel_mid_gaps.log 2>&1 &   # E10: GuitarSet + GAPS (best)
nohup runs/train_tabmodel_mid.sh      > runs/tabmodel_mid.log      2>&1 &   # E8: GuitarSet only
grep EVAL runs/tabmodel_mid_gaps.log                                         # validation every 1000 steps
```
- Both scripts call `tools/train_tabmodel.py`. Main options:
  - `--out` — output folder; use a new one per experiment.
  - `--size mid|large` — about 24M / 50M parameters.
  - `--steps`, `--batch`, `--lr`.
  - `--gaps_share` — the share of 5 s crops taken from GAPS.
  - `--gaps_tab_weight` — the loss weight of GAPS string labels.
  - `--eval_every`.
- **Memory:** the scripts cap GPU memory with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25`,
  about 8 GB for `mid` at batch 16, and run at low priority, so the laptop stays usable. Training runs at about 1.1 s per step;
  16k steps take about 5 h.
- **Resume / pause:** the scripts pass `--resume`, so re-running one continues from `last.pt`.
  `STOP_AT=18:30 runs/train_tabmodel_mid_gaps.sh` saves and stops at the first evaluation after 18:30.
- **Outputs:** `best.pt` (best validation score), `last.pt` (for resuming; can be deleted when finished) and `log.jsonl`.
- **Background jobs:** launch with `nohup`, so training survives the terminal or tool session that started it.

### 4.3 Evaluation

```bash
# GuitarSet test player 05: time/tick x pitch/string-fret F1 on the fixed 15 s chunks
.venv/bin/python tools/eval_tabmodel.py --checkpoint runs/tabmodel_mid_gaps/best.pt --split test
# tune note-extraction thresholds (ONLY on validation), then copy the best to DEFAULT_PARAMS in tools/tabmodel_infer.py
.venv/bin/python tools/eval_tabmodel.py --checkpoint runs/tabmodel_mid_gaps/best.pt --split validation --tune
# GAPS test (fingerstyle, unseen players / rooms): pitch F1 + string agreement with the score fingering
.venv/bin/python tools/eval_gaps.py --checkpoints runs/tabmodel_mid/best.pt runs/tabmodel_mid_gaps/best.pt
# precision/recall sweep of thresholds and fingering, plus benchmark-style frame/note metrics (~25 min)
.venv/bin/python tools/sweep_tabmodel.py --checkpoint runs/tabmodel_mid_gaps/best.pt --out runs/sweep.json
```
`data/eval_chunks.json` (146 test / 147 validation chunks of 15 s, with the reference tabs) was generated once with the
original Audio2Tab chunk schedule. It is a frozen artifact: keep it and don't regenerate it, so results stay comparable.

Rule: choose checkpoints and settings on validation (GuitarSet player 04, GAPS validation), and report test only once per model.

---

## 5. Script reference (`tools/`)

| Script | Purpose |
|---|---|
| **Inference** | |
| `transcribe.py` | Audio → `.gp5` (section 1). |
| `tabmodel_infer.py` | Tab model sliding-window prediction + note extraction (`DEFAULT_PARAMS`). |
| `notes_to_tab.py` | Notes in seconds → beat-following quantisation (Beat This!) → tab tokens. |
| `fingering.py` | Viterbi fingering optimiser (string/fret choice for given pitches). |
| `run_basic_pitch.py` | Runs Basic Pitch (in `.venv_bp`) and caches notes in `data/basic_pitch/`. |
| `dadagp_utils/` | Token ↔ Guitar Pro conversion (writes the `.gp5`), copied from Audio2Tab / DadaGP. |
| **Model and training** | |
| `tabmodel.py` | Model definition (harmonic-stacked CQT → CNN → transformer → tab/onset/pitch heads; sizes `mid`, `large`). |
| `tabdata.py` | Features, frame labels, splits (GuitarSet, shifted copies, GAPS); builds the cache. |
| `train_tabmodel.py` | Training loop (data mixing, masked losses, validation, best/last checkpoints). |
| **Evaluation** | |
| `eval_tabmodel.py`, `eval_gaps.py`, `sweep_tabmodel.py` | See section 4.3. |
| `note_metrics.py` | Note-level F1 (time/tick × pitch/string-fret), extracted from Audio2Tab so scores match every past result. |
| **Data** | |
| `guitarset_to_tokens.py`, `guitarset_shift.py` | GuitarSet → reference tabs (also provides the tab writer used by `notes_to_tab.py`); pitch-shifted copies. |
| `gaps_common.py`, `gaps_download.py`, `gaps_labels.py` | GAPS case-safe extraction, audio download, string labels. |

---

## 6. Network note

The office network re-signs HTTPS, so Python downloads (Hugging Face, Beat This!, yt-dlp) can fail with `CERTIFICATE_VERIFY_FAILED`.
- Download single files with `curl`, which uses the macOS keychain.
- For yt-dlp, `tools/gaps_download.py` sets `SSL_CERT_FILE=data/certs/macos_trusted.pem`. Regenerate that file with:
```bash
(security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain;
 security find-certificate -a -p /Library/Keychains/System.keychain) > data/certs/macos_trusted.pem
```
