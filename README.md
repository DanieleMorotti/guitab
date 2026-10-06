# guitab — guitar audio → Guitar Pro tab

Transcribes a guitar recording, with a focus on fingerstyle, into a tab: a `.gp5` file you can open in Guitar Pro, TuxGuitar,
MuseScore or any online gp5 player.

The model reads a harmonic constant-Q spectrogram and predicts, at every frame (11.6 ms), the fret played on each of the
6 strings plus note attacks. Notes are then snapped to the beats found by a beat tracker and written out as a tab.

| Test set | Metric | Best model (`runs/tabmodel_mid_gaps/best.pt`) |
|---|---|---|
| GuitarSet, player 05 | string + fret F1 (onset ±50 ms) | **0.828** |
| GuitarSet, player 05 | pitch F1 (onset ±50 ms) | **0.859** |
| GAPS test (fingerstyle, unseen players and rooms) | pitch F1 (onset ±50 ms) | **0.811** |

The full history (including the original Whisper / Audio2Tab approach and a Basic Pitch baseline, both replaced),
every setting and every result are in [`EXPERIMENTS.md`](EXPERIMENTS.md). Add a new entry there for every experiment,
and commit each change separately.

---

## 1. Quick start: transcribe your own recording

```bash
# 1) extract mono audio from a video or any audio file
ffmpeg -i my_video.mov -ac 1 -ar 22050 out/my_video/my_video.wav

# 2) transcribe (default checkpoint = best model)
.venv/bin/python scripts/transcribe.py --audio out/my_video/my_video.wav --out_dir out/my_video
```

The output is `out/my_video/my_video.gp5`, plus `my_video.tokens.txt` (the tab as text tokens).

| Option | Effect |
|---|---|
| `--onset 0.4` | Keeps less confident notes, so more notes and more false ones. The default 0.6 was tuned on GuitarSet validation; try 0.4–0.5 if notes are missing on your guitar. |
| `--orphan` | Also keeps notes without a detected attack. It was worse on GuitarSet, but may recover quiet or sustained notes. |
| `--fingering low` | Re-fingers everything towards first position (Viterbi optimiser). The default `model` keeps the model's strings; `auto` re-fingers only when clearly awkward. |
| `--tempo 90` | Uses a fixed tempo. The default `auto` follows the beats detected by Beat This!. |
| `--grid 120` | Quantisation grid in ticks (960 = quarter): 240 = 16th notes (default), 120 = 32nd notes. |
| `--checkpoint PATH` | Another checkpoint, e.g. `runs/tabmodel_mid/best.pt` (GuitarSet only). |
| `--name X` | Output file stem. |

Inference needs about 1–2 GB of GPU memory and takes under a minute for a 4-minute song on an M-series Mac.

---

## 2. Repository layout

```
guitab/
├── guitab/                 library (imported by the scripts)
│   ├── paths.py            all data / run locations
│   ├── guitar.py           tuning, string order, fret classes
│   ├── audio.py            audio loading + CQT feature
│   ├── model.py            TabModel (harmonic stack → CNN → transformer → tab/onset/pitch heads)
│   ├── inference.py        sliding-window prediction + note extraction (DEFAULT_PARAMS)
│   ├── tab.py              beat tracking, quantisation, tab tokens, .gp5 export
│   ├── fingering.py        Viterbi fingering optimiser (+ presets)
│   ├── data.py             frame labels, track lists, splits, feature cache
│   ├── guitarset.py        GuitarSet notes and splits
│   ├── gaps.py             GAPS case-safe files and string/fret labels
│   ├── evaluation.py       evaluation protocols (GuitarSet chunks, full tracks, GAPS)
│   ├── metrics.py          note-level F1 (from Audio2Tab, unchanged scores)
│   └── dadagp_utils/       token ↔ Guitar Pro conversion (from Audio2Tab / DadaGP)
├── scripts/                command-line entry points (section 5)
├── data/eval_chunks.json   frozen GuitarSet evaluation chunks (the only versioned data file)
├── requirements.txt        exact package versions
├── README.md, EXPERIMENTS.md
│
│   not versioned (.gitignore): rebuilt by the scripts or produced by training
├── .venv/                  Python environment
├── data/                   guitarset/, guitarset_shift/, gaps/, tabmodel_cache/, models/ (Beat This!), certs/,
│                           guitarset.csv, guitarset_shift.csv (track lists with repo-relative paths)
├── runs/                   checkpoints (best.pt), training logs, sweep results
└── out/                    your transcriptions
```

---

## 3. Environment

All commands run from the repository root with Python 3.11.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

- **Pinned versions:** `requirements.txt` pins every package; its header lists the direct dependencies.
  The main ones are torch 2.10, librosa 0.11, beat-this (pinned git commit), music21 10.5, pretty-midi 0.2.11 and pyguitarpro 0.9.3.
- **Hardware:** on a Mac, everything runs on MPS (Apple GPU) automatically.
- **Beat This! weights:** downloaded on first use, or put in `data/models/beat_this-final0.ckpt` (see section 6).

---

## 4. Training and evaluation

### 4.1 Data preparation (one-off)

**GuitarSet** (CC-BY; [Zenodo 3371780](https://zenodo.org/records/3371780)): download `annotation.zip` and `audio_mono-mic.zip`, and unzip them into `data/guitarset/annotation/` and `data/guitarset/audio_mic/`. Then:
```bash
.venv/bin/python scripts/prepare_guitarset.py     # data/guitarset.csv (players 00-03 train, 04 validation, 05 test)
.venv/bin/python scripts/guitarset_shift.py       # +1/+2 semitone copies of the train players (data/guitarset_shift.csv)
```

**GAPS** (classical / fingerstyle, CC BY-NC-SA; [Zenodo 13962272](https://zenodo.org/records/13962272)):
```bash
mkdir -p data/gaps && curl -L -o data/gaps/gaps_v1_no_audio.zip \
  "https://zenodo.org/records/13962272/files/gaps_v1_no_audio.zip?download=1"
(cd data/gaps && unzip -o gaps_v1_no_audio.zip "gaps_v1/*.csv")   # only the csv files are safe to unzip
.venv/bin/python scripts/gaps_extract.py          # MIDI / MusicXML with case-safe names (see note)
.venv/bin/python scripts/gaps_download.py         # audio from YouTube; run twice to retry 403 errors
.venv/bin/python scripts/gaps_labels.py           # string/fret labels: score tab staff aligned to the MIDI
```
> GAPS ids differ only by upper/lower case (`GM1wc` ≠ `gM1wc`). A normal unzip on macOS silently overwrites files,
> so always use `scripts/gaps_extract.py`.
> `gaps_labels.py --only <id> ...` rebuilds selected pieces and keeps the other rows of `labels_summary.csv`.

**Feature cache** (CQT + frame labels for all tracks; about 10 min, 2.6 GB):
```bash
.venv/bin/python scripts/build_cache.py --workers 4
```
The GAPS selection (standard tuning, MIDI matches the score, ≥ 50% of notes with a string label) is in `guitab/data.py`
(`gaps_selection`). It gives 123 train, 14 validation and 19 test pieces.

### 4.2 Training

```bash
nohup scripts/train_mid_gaps.sh > runs/tabmodel_mid_gaps.log 2>&1 &   # E10: GuitarSet + GAPS (best)
nohup scripts/train_mid.sh      > runs/tabmodel_mid.log      2>&1 &   # E8: GuitarSet only
grep EVAL runs/tabmodel_mid_gaps.log                                    # validation every 1000 steps
```
- **Options:** both launch scripts call `scripts/train.py`.
  - `--out` — output folder; use a new one per experiment.
  - `--size mid|large` — about 24M / 50M parameters.
  - `--steps`, `--batch`, `--lr`.
  - `--gaps_share` — the share of 5 s crops taken from GAPS.
  - `--gaps_tab_weight` — the loss weight of GAPS string labels.
  - `--eval_every`.
- **Memory:** the scripts cap GPU memory (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35`, `LOW=0.25`; about 8 GB for `mid` at batch 16)
  and run at low priority, so the laptop stays usable. Training runs at about 1.1 s per step; 16k steps take about 5 h.
- **Resume / pause:** the scripts pass `--resume`, so re-running one continues from `last.pt`.
  `STOP_AT=18:30 scripts/train_mid_gaps.sh` saves and stops at the first evaluation after 18:30.
- **Outputs (in `--out`):** `best.pt` (best validation score), `last.pt` (for resuming; delete when finished) and `log.jsonl`.

### 4.3 Evaluation

```bash
# GuitarSet test player 05: time/tick x pitch/string-fret F1 on the fixed 15 s chunks
.venv/bin/python scripts/evaluate.py --checkpoint runs/tabmodel_mid_gaps/best.pt --split test
# tune note-extraction thresholds (ONLY on validation), then copy the best to DEFAULT_PARAMS in guitab/inference.py
.venv/bin/python scripts/evaluate.py --checkpoint runs/tabmodel_mid_gaps/best.pt --split validation --tune
# GAPS test (fingerstyle, unseen players / rooms): pitch F1 + string agreement with the score fingering
.venv/bin/python scripts/evaluate_gaps.py --checkpoints runs/tabmodel_mid/best.pt runs/tabmodel_mid_gaps/best.pt
# precision/recall sweep of thresholds and fingering, plus benchmark-style frame/note metrics (~25 min)
.venv/bin/python scripts/sweep.py --checkpoint runs/tabmodel_mid_gaps/best.pt --out runs/sweep_tabmodel_mid_gaps.json
```
- **The rule:** choose checkpoints and settings on validation (GuitarSet player 04, GAPS validation), and report test only once per model.
- **`data/eval_chunks.json`** (146 test / 147 validation chunks of 15 s, with the reference tabs) was generated once with the original
  Audio2Tab chunk schedule. It is frozen: keep it unchanged, so results stay comparable.

---

## 5. Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `transcribe.py` | Audio → `.gp5` (section 1). |
| `prepare_guitarset.py`, `guitarset_shift.py` | GuitarSet track list; pitch-shifted copies. |
| `gaps_extract.py`, `gaps_download.py`, `gaps_labels.py` | GAPS case-safe extraction, audio download, string/fret labels. |
| `build_cache.py` | CQT features + frame labels for every track → `data/tabmodel_cache/`. |
| `train.py`, `train_mid.sh`, `train_mid_gaps.sh` | Training (the `.sh` files are the exact E8 / E10 runs). |
| `evaluate.py`, `evaluate_gaps.py`, `sweep.py` | Evaluation (section 4.3). |
| `analyze_errors.py` | Error breakdown on validation and train subsets: miss / false-note categories, string accuracy, generalisation gap. |

---

## 6. Network note

The office network re-signs HTTPS, so Python downloads (Beat This! weights, yt-dlp) can fail with `CERTIFICATE_VERIFY_FAILED`.
- Download single files with `curl`, which uses the macOS keychain. For example, the Beat This! checkpoint goes into `data/models/`.
- For yt-dlp, `scripts/gaps_download.py` sets `SSL_CERT_FILE=data/certs/macos_trusted.pem`. Regenerate that file with:
```bash
mkdir -p data/certs && (security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain;
 security find-certificate -a -p /Library/Keychains/System.keychain) > data/certs/macos_trusted.pem
```
