# guitab — guitar audio → Guitar Pro tab

Transcribes a solo guitar recording, with a focus on steel-string fingerstyle, into a tab: a `.gp5` file you can open in
Guitar Pro, TuxGuitar, MuseScore or any online gp5 player.

A convolutional + transformer model reads a constant-Q spectrogram of the audio and predicts, every 11.6 ms, which fret
is sounding on each of the six strings and where notes start. The notes are then cleaned up (duplicates, split notes),
optionally decoded with a small tab language model for fuller chords, snapped to the beats found by a beat tracker and
written out as a tab.

**What to expect.** On clean recordings it gets about 85% of notes right with the correct string and fret; dense
3–4 note chords, quiet inner voices and phone recordings in a room are much harder. Use the output as a first draft to
correct by ear, not as a finished tab.

| Test set | Metric (onset ±50 ms) | Default model (E19) |
|---|---|---|
| GuitarSet test (player 05, steel string) | string + fret F1 / pitch F1 | 0.843 / 0.866 |
| GuitarSet test, phone/room-degraded copy | string + fret F1 / pitch F1 | 0.640 / 0.721 |
| GAPS test (classical guitar, unseen players and rooms) | pitch F1 | 0.813 |
| Fingerstyle covers (19 held-out real recordings, private set) | string + fret F1 | 0.617 |

The whole research history (every experiment, setting and result, including approaches that did not work) is in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## 1. Install

Python 3.11, from the repository root:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt   # exact tested versions
uv pip install --python .venv/bin/python -e . --no-deps          # the `guitab` command
```
(`pip install -e ".[train]"` works too; `requirements.txt` pins the environment the results were produced with.)

On a Mac everything runs on the Apple GPU (MPS); CUDA and CPU work as well. The Beat This! beat-tracker weights are
downloaded on first use.

**Model weights are not distributed.** Train a model first (section 4); `guitab` uses `runs/tabmodel_mid_all/best.pt`
unless you pass `--checkpoint`.

## 2. Transcribe a recording

```bash
ffmpeg -i my_video.mov -ac 1 -ar 22050 out/my_song/my_song.wav    # any audio or video file -> mono wav
.venv/bin/guitab --audio out/my_song/my_song.wav --out_dir out/my_song
```

The output is `out/my_song/my_song.gp5`, plus `my_song.tokens.txt` (the same tab as text tokens). A 4-minute song takes
well under a minute and 1–2 GB of GPU memory.

| Option | Effect |
|---|---|
| `--capo 3` | The song is played with a capo at that fret: notes are kept on strings that can play them, frets are written relative to the capo, and the .gp5 has the capo set. |
| `--music_lm` | Fuller chords: notes are decoded jointly with a small tab language model trained on symbolic tabs. 3–4 note chords come out more complete; overall accuracy changes by about ±1 point; 10–20 s slower. |
| `--onset 0.4` | Keep less confident notes (default 0.6): more notes, more false ones. Try 0.4–0.5 if notes are missing. |
| `--orphan` | Also keep notes without a detected attack (quiet or sustained notes). |
| `--fingering low` | Re-finger towards first position (Viterbi optimiser). The default `model` keeps the model's strings; `auto` re-fingers only clearly awkward shapes. |
| `--tempo 90` | Fixed tempo instead of following the detected beats. |
| `--grid 120` | Quantisation grid in ticks (960 = quarter note): 240 = 16ths (default), 120 = 32nds. |
| `--checkpoint PATH` | Another trained model. |
| `--name X` | Output file name. |
| `--params JSON` | Note-extraction settings (`DEFAULT_PARAMS` in `guitab/inference.py`), e.g. `'{"legato": 5}'`. |
| `--music_lm_weight 0.3` / `--music_lm_bonus 1.0` | Language-model weight and per-note bonus (higher = fuller chords, more false notes). |

## 3. How it works

1. **Features** (`guitab/audio.py`): 22.05 kHz mono → constant-Q spectrogram, 36 bins per octave, 8 octaves, hop 256.
2. **Model** (`guitab/model.py`): harmonic stacking → CNN → transformer over time → per frame and string: fret
   (21 classes incl. silent), onset, plus a pitch head. Sizes `mid` (24M parameters, default) and `large` (66M).
3. **Note extraction** (`guitab/inference.py`): notes start at onset peaks, take the most likely fret, end at silence,
   a new attack or a fret change (legato); the same pitch on two strings at once is kept once; an attack that splits a
   ringing note is merged back; with a capo, notes below it move to a playable string.
4. **Optional chord decoding** (`guitab/tablm.py`, `--music_lm`): beam search over candidate chords scored by the
   acoustic model plus a 3.3M-parameter transformer trained on ~12.7k symbolic guitar tabs.
5. **Tab** (`guitab/tab.py`, `guitab/fingering.py`, `guitab/gp5.py`): beats and downbeats from Beat This!, quantisation
   to a grid that follows tempo drift, optional re-fingering, Guitar Pro 5 output.

## 4. Training

All commands run from the repository root.

### 4.1 Data (one-off)

**GuitarSet** (CC BY 4.0; [Zenodo 3371780](https://zenodo.org/records/3371780)): download `annotation.zip` and
`audio_mono-mic.zip` and unzip them into `data/guitarset/annotation/` and `data/guitarset/audio_mic/`. Then:
```bash
.venv/bin/python scripts/prepare_guitarset.py     # track list (players 00-03 train, 04 validation, 05 test)
.venv/bin/python scripts/guitarset_shift.py       # +1/+2 semitone copies of the training players
```

**GAPS** (classical guitar, CC BY-NC-SA 4.0; [Zenodo 13962272](https://zenodo.org/records/13962272)):
```bash
mkdir -p data/gaps && curl -L -o data/gaps/gaps_v1_no_audio.zip \
  "https://zenodo.org/records/13962272/files/gaps_v1_no_audio.zip?download=1"
(cd data/gaps && unzip -o gaps_v1_no_audio.zip "gaps_v1/*.csv")   # only the csv files: see the note below
.venv/bin/python scripts/gaps_extract.py          # MIDI / MusicXML with case-safe file names
.venv/bin/python scripts/gaps_download.py         # audio from the dataset's YouTube links; run twice to retry errors
.venv/bin/python scripts/gaps_labels.py           # string/fret labels from the score's tab staff, aligned to the MIDI
```
> GAPS ids differ only by letter case (`GM1wc` ≠ `gM1wc`); a plain unzip on a case-insensitive file system overwrites
> files, so always extract with `scripts/gaps_extract.py`.

**SynthTab subset** (tabs rendered with sampled guitars, CC BY-NC 4.0; [SynthTab](https://github.com/AirLabUR/SynthTab)):
428 selected fingerstyle songs (about 21 h; 5.3 GB audio + 1.1 GB labels), fetched with HTTP range requests from the
2 TB dataset:
```bash
.venv/bin/python scripts/synthtab_fetch.py labels     # label archive
.venv/bin/python scripts/synthtab_fetch.py download   # the songs in data/synthtab/selection.csv (versioned)
```

**custom dataset** (private: real fingerstyle recordings labelled with fan-made tabs, personal use only; not part of
the public release): `scripts/custom_fetch.py search|select|download|label`, see `guitab/custom_data.py`.

**Feature caches**:
```bash
.venv/bin/python scripts/build_cache.py --workers 4                    # CQT + frame labels (~10 min, 2.6 GB)
.venv/bin/python scripts/build_aug_cache.py --copies 2 --close --workers 4   # augmented training copies (phone/room, close mic)
.venv/bin/python scripts/build_aug_cache.py --degraded --workers 4     # held-out degraded copies for robustness tests
```

### 4.2 Train

The exact command lines of every training run are in `experiments/` (E8 … E19, described in EXPERIMENTS.md):
```bash
nohup experiments/train_mid_all.sh > runs/tabmodel_mid_all.log 2>&1 &   # E19, the default model (~8 h on an M-series Mac)
grep EVAL runs/tabmodel_mid_all.log                                      # validation every 1000 steps
.venv/bin/python scripts/train_tablm.py corpus && .venv/bin/python scripts/train_tablm.py train   # tab LM for --music_lm
```
- `scripts/train.py` options: `--size mid|large`, `--steps`, `--batch`, `--lr`, `--accum` (gradient accumulation);
  data shares `--gaps_share`, `--synth_share`, `--ss_share` and their string-label weights `--*_tab_weight`; audio
  augmentation `--aug_copies`, `--aug_share`; `--init` (fine-tune from a checkpoint); `--eval_degraded`.
- The launchers cap GPU memory (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`) and run at low priority so the machine stays usable.
  Re-running a launcher resumes from `last.pt`; `STOP_AT=18:30 experiments/...` stops at the first evaluation after 18:30.
- Outputs in `runs/<name>/`: `best.pt` (best validation score), `last.pt` (for resuming) and `log.jsonl`.

## 5. Evaluation

```bash
.venv/bin/python scripts/evaluate.py --checkpoint runs/tabmodel_mid_all/best.pt --split test             # GuitarSet test
.venv/bin/python scripts/evaluate.py --checkpoint runs/tabmodel_mid_all/best.pt --split test --variant deg   # degraded copy
.venv/bin/python scripts/evaluate.py --checkpoint runs/tabmodel_mid_all/best.pt --split validation --tune   # threshold search
.venv/bin/python scripts/evaluate_gaps.py --checkpoints runs/tabmodel_mid_all/best.pt                    # GAPS test
.venv/bin/python scripts/sweep.py --checkpoint runs/tabmodel_mid_all/best.pt --out runs/sweep.json       # P/R sweep + frame metrics
```
- Choose checkpoints and settings on validation only (GuitarSet player 04, GAPS validation); report test once per model.
- GuitarSet is scored on the fixed 15 s chunks of `data/eval_chunks.json` (frozen, so results stay comparable).
- `scripts/analyze_errors.py` breaks the errors down by category; `scripts/evaluate_song.py` is the private benchmark on
  the author's own recording.

## 6. Repository layout

```
guitab/            the library: audio, model, inference, tablm, tab, fingering, gp5, cli, data, augment,
                   guitarset, gaps, synthtab, evaluation, paths
scripts/           command-line tools: data preparation, caches, training, evaluation
experiments/       exact launchers of the training runs (see EXPERIMENTS.md)
data/              datasets and caches (not versioned except eval_chunks.json and the subset selections)
runs/              checkpoints and logs (not versioned)
out/               transcriptions (not versioned)
```

> **Behind a TLS-intercepting proxy** Python downloads can fail with `CERTIFICATE_VERIFY_FAILED`. On macOS, export the
> system certificates and point `SSL_CERT_FILE` at them (`scripts/gaps_download.py` uses `data/certs/macos_trusted.pem`):
> `mkdir -p data/certs && security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain /Library/Keychains/System.keychain > data/certs/macos_trusted.pem`

## Licence and credits

Code: MIT (see `LICENSE`). Datasets keep their own licences (GuitarSet CC BY 4.0, GAPS CC BY-NC-SA 4.0, SynthTab
CC BY-NC 4.0); a model trained on them inherits their non-commercial terms. Beat tracking: [Beat This!](https://github.com/CPJKU/beat_this).
The research started from [Audio2Tab](https://github.com/JackJamesLoth/Audio2Tab); the GuitarSet evaluation chunks
follow its schedule.
