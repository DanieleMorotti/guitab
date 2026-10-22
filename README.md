<p align="center"><img src="assets/logo.png" alt="Guitab logo" width="180"></p>

<h1 align="center">Guitab</h1>

<p align="center"><b>Guitar audio → Guitar Pro tab</b></p>

Guitab turns a recording of a solo guitar into a tab you can open in Guitar Pro, TuxGuitar, MuseScore or any online gp5 player. It is made for steel-string fingerstyle playing, works on any recording you can play (a phone video is fine), and runs on a laptop: a 4-minute song takes well under a minute.

You get a `.gp5` file with the notes on the right strings and frets, laid out on a grid that follows the beats of the recording, with an optional capo. A local web app lets you record or load a song, transcribe it and play the tab back in the browser.

**What to expect.** On clean recordings about 85% of the notes come out right, with the correct string and fret. Dense 3–4 note chords, quiet inner voices and phone recordings in a reverberant room are much harder. Treat the result as a first draft to correct by ear, not as a finished tab.

## Results

| Test set | Metric (onset ±50 ms) | E15 (public data) | E19 (+ custom dataset) | E25 (default) | E24 (tiny) |
|---|---|---|---|---|---|
| GuitarSet test (player 05, steel string) | string + fret F1 / pitch F1 | 0.836 / 0.866 | 0.843 / 0.866 | 0.838 / 0.861 | 0.833 / 0.856 |
| GuitarSet test, phone/room-degraded copy | string + fret F1 / pitch F1 | 0.650 / 0.735 | 0.640 / 0.721 | 0.685 / 0.757 | 0.655 / 0.737 |
| GAPS test (classical guitar, unseen players and rooms) | pitch F1 | 0.827 | 0.813 | 0.830 | 0.825 |
| Real fingerstyle recordings (19 held-out songs of the custom dataset) | string + fret F1 | 0.548 | 0.617 | 0.612 | 0.601 |
| EGDB electric guitar test (25 clips, clean DI, never seen in training) | string + fret F1 / pitch F1 | 0.636 / 0.833 | 0.712 / 0.850 | 0.721 / 0.849 | 0.691 / 0.830 |
| EGDB electric guitar test, through two amps (clean and crunch) | string + fret F1 / pitch F1 | 0.613 / 0.784 | 0.631 / 0.787 | 0.652 / 0.788 | 0.622 / 0.781 |

- **E25 (default)** is a 6M-parameter model, a quarter of E19's size, trained to imitate the average of E19 and the 66M-parameter E18 (distillation). It is within a point of E19 on clean steel-string audio, clearly better on phone recordings and classical guitar, and 1.5x faster.
- **E24 (tiny)** follows the same recipe at half the size: 2.9M parameters, an 11 MB file, 2.3x faster than E19 and about one point behind it on GuitarSet.
- **E19** is the strongest model trained on the labels alone. Besides the public datasets it used a custom dataset of real fingerstyle recordings with aligned tabs, which is not distributed.
- **E15** is trained on public datasets only, and anyone can reproduce it with the commands in section 4.

**Electric guitar.** The models never saw an electric guitar during training, but they also work on clean or lightly overdriven ones. On the [EGDB](https://ss12f32v.github.io/Guitar-Transcription/) test set (Chen et al., ICASSP 2022) the notes are found almost as well as on acoustic recordings, while the string choice is less reliable, so expect more fingering corrections. Heavy distortion was not tested.

The whole research history, with every experiment, setting and result (including the approaches that did not work), is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## 1. Install

From the repository root, with Python 3.11, on macOS, Linux or Windows:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt   # exact tested versions
uv pip install --python .venv/bin/python -e . --no-deps          # the `guitab` command
```

`pip install -e ".[train]"` works too; `requirements.txt` pins the environment the results were produced with. On Linux the default PyTorch wheel includes CUDA; for another CUDA version or a CPU-only build, install `torch` first as described on [pytorch.org](https://pytorch.org/get-started/locally/). On Windows, the commands below use `.venv\Scripts\` instead of `.venv/bin/`.

Every command takes `--device auto|cuda|mps|cpu`. The default, `auto`, uses an NVIDIA GPU if present, else the Apple GPU, else the CPU. The Beat This! beat-tracker weights are downloaded on first use.

**ffmpeg is optional.** WAV, FLAC, OGG, MP3 and AIFF files are read without it. The `guitab` command needs [ffmpeg](https://ffmpeg.org) only for other formats, such as videos or M4A files (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian or Ubuntu). The web app decodes those in the browser and needs no ffmpeg. Training needs ffmpeg for its audio augmentation and the GAPS download.

### Model weights

The weights are on Hugging Face at [`<hf-user>/guitab`](https://huggingface.co/<hf-user>/guitab):

| File | Model | Size |
|---|---|---|
| `guitab_e25.pt` | E25, the default | 24 MB |
| `guitab_e24.pt` | E24, the tiny model | 11 MB |
| `guitab_e19.pt` | E19 | 91 MB |
| `tablm.pt` | the tab language model behind `--music_lm` | 13 MB |

Put them in `weights/`, either from the Hugging Face page or with:

```bash
uv pip install --python .venv/bin/python huggingface_hub
.venv/bin/hf download <hf-user>/guitab --local-dir weights
```

`guitab` uses E25, and falls back to E15 (`runs/tabmodel_mid_close/best.pt`, section 4) when E25 is missing. Pass `--checkpoint` to use another model. A model you train yourself (`runs/<run>/best.pt`) takes precedence over the downloaded file.

## 2. Transcribe a recording

```bash
.venv/bin/guitab --audio my_song.mp3 --out_dir out/my_song      # wav, flac, ogg, mp3, aiff
.venv/bin/guitab --audio my_video.mov --out_dir out/my_video    # videos and other formats, with ffmpeg installed
```

The result is `out/my_song/my_song.gp5`, named after the audio file, plus `my_song.tokens.txt` with the same tab as text tokens. A 4-minute song takes well under a minute and 1–2 GB of GPU memory.

| Option | Effect |
|---|---|
| `--capo 3` | The song is played with a capo at that fret: notes stay on strings that can play them, frets are written relative to the capo, and the .gp5 has the capo set. |
| `--music_lm` | Fuller chords: the notes are decoded together with a small tab language model trained on symbolic tabs. 3–4 note chords come out more complete, overall accuracy changes by about ±1 point, and it is 10–20 s slower. |
| `--onset 0.4` | Keep less confident notes: more notes, and more false ones. The default is the model's own (0.45 for E25, 0.5 for E24, 0.6 for the others); try 0.1 below it if notes are missing. |
| `--orphan` | Also keep notes without a detected attack (quiet or sustained notes). |
| `--fingering low` | Re-finger towards first position with a Viterbi optimiser. The default, `model`, keeps the model's strings; `auto` re-fingers only clearly awkward shapes. |
| `--tempo 90` | A fixed tempo instead of following the detected beats. |
| `--grid 120` | Quantisation grid in ticks (960 = quarter note): 240 = 16ths (default), 120 = 32nds. |
| `--checkpoint PATH` | Another model. |
| `--name X` | The output file name. |
| `--params JSON` | Note-extraction settings (`DEFAULT_PARAMS` in `guitab/inference.py`), e.g. `'{"legato": 5}'`. |
| `--music_lm_weight 0.3` / `--music_lm_bonus 1.0` | Language-model weight and per-note bonus (higher = fuller chords, more false notes). |

### Web app

```bash
.venv/bin/guitab-app          # opens http://127.0.0.1:8765 in the browser
```

Load an audio or video file (the browser decodes it), or record from the microphone. Choose the model (any available checkpoint among E25, E19, E24, E15, E18 and E20), the capo, fuller chords, the note sensitivity (it starts at the model's own default), the fingering, the tempo and the grid, then press **Transcribe**.

The tab appears in an [alphaTab](https://www.alphatab.net) viewer with playback: play and pause (space), click a note to play from there, speed, metronome, count-in, loop, tab only or notation and tab, zoom, and download of the .gp5. Any existing .gp5 file can be opened too.

Every result is kept in `out/app/` and listed under *Tabs*. The server only listens on this computer; the viewer is loaded from a CDN, so the page needs internet.

## 3. How it works

1. **Features** (`guitab/audio.py`): 22.05 kHz mono audio becomes a constant-Q spectrogram with 36 bins per octave over 8 octaves, hop 256.
2. **Model** (`guitab/model.py`): harmonic stacking, a CNN and a transformer over time predict, for each frame and string, the fret (21 classes including silent) and the onset, plus a pitch head. Sizes: `half` (2.9M parameters), `small` (6.2M, E25), `mid` (24M) and `large` (66M).
3. **Note extraction** (`guitab/inference.py`): notes start at onset peaks, take the most likely fret, and end at silence, at a new attack or at a fret change (legato). The same pitch on two strings at once is kept once, an attack that splits a ringing note is merged back, and with a capo the notes below it move to a playable string.
4. **Optional chord decoding** (`guitab/tablm.py`, `--music_lm`): a beam search over candidate chords, scored by the acoustic model and by a 3.3M-parameter transformer trained on about 12.7k symbolic guitar tabs.
5. **Tab** (`guitab/tab.py`, `guitab/fingering.py`, `guitab/gp5.py`): beats and downbeats from Beat This!, quantisation to a grid that follows tempo drift, optional re-fingering, and Guitar Pro 5 output.

## 4. Training

All commands run from the repository root.

### 4.1 Data (one-off)

**GuitarSet** (CC BY 4.0, [Zenodo 3371780](https://zenodo.org/records/3371780)): download `annotation.zip` and `audio_mono-mic.zip`, unzip them into `data/guitarset/annotation/` and `data/guitarset/audio_mic/`, then run:

```bash
.venv/bin/python scripts/prepare_guitarset.py     # track list (players 00-03 train, 04 validation, 05 test)
.venv/bin/python scripts/guitarset_shift.py       # +1/+2 semitone copies of the training players
```

**GAPS** (classical guitar, CC BY-NC-SA 4.0, [Zenodo 13962272](https://zenodo.org/records/13962272)):

```bash
mkdir -p data/gaps && curl -L -o data/gaps/gaps_v1_no_audio.zip \
  "https://zenodo.org/records/13962272/files/gaps_v1_no_audio.zip?download=1"
(cd data/gaps && unzip -o gaps_v1_no_audio.zip "gaps_v1/*.csv")   # only the csv files: see the note below
.venv/bin/python scripts/gaps_extract.py          # MIDI / MusicXML with case-safe file names
.venv/bin/python scripts/gaps_download.py         # audio from the dataset's YouTube links; run twice to retry errors
.venv/bin/python scripts/gaps_labels.py           # string/fret labels from the score's tab staff, aligned to the MIDI
```

> GAPS ids differ only by letter case (`GM1wc` ≠ `gM1wc`), so a plain unzip on a case-insensitive file system overwrites files. Always extract with `scripts/gaps_extract.py`.

**SynthTab subset** (tabs rendered with sampled guitars, CC BY-NC 4.0, [SynthTab](https://github.com/AirLabUR/SynthTab)): 428 selected fingerstyle songs (about 21 h, 5.3 GB of audio and 1.1 GB of labels), fetched with HTTP range requests from the 2 TB dataset:

```bash
.venv/bin/python scripts/synthtab_fetch.py labels     # label archive
.venv/bin/python scripts/synthtab_fetch.py download   # the songs in data/synthtab/selection.csv (versioned)
```

**Custom dataset** (not distributed): real fingerstyle recordings with their tabs aligned to the audio. It is used by E17–E19, and through E19 by the distilled E24 and E25. The tools that built it are not published; `guitab/custom_data.py` documents the layout that training reads from `data/custom/`.

**Feature caches**:

```bash
.venv/bin/python scripts/build_cache.py --workers 4                    # CQT + frame labels (~10 min, 2.6 GB)
.venv/bin/python scripts/build_aug_cache.py --copies 2 --close --workers 4   # augmented training copies (phone/room, close mic)
.venv/bin/python scripts/build_aug_cache.py --degraded --workers 4     # held-out degraded copies for robustness tests
```

### 4.2 Train

The exact command lines of every training run are in `experiments/` (E8 to E25, described in EXPERIMENTS.md).

The public-data model E15 is trained in three stages, each after the previous one has finished (about 5 h, 3 h and 2 h on an Apple M3 Pro laptop):

```bash
experiments/train_mid_gaps.sh     > runs/tabmodel_mid_gaps.log 2>&1       # E10: GuitarSet + GAPS, from scratch
experiments/train_mid_gaps_aug.sh > runs/tabmodel_mid_gaps_aug.log 2>&1   # E12: + phone/room audio augmentation
experiments/train_mid_close.sh    > runs/tabmodel_mid_close.log 2>&1     # E15: + SynthTab, close-mic augmentation
grep EVAL runs/tabmodel_mid_close.log                                   # validation every 1000 steps
.venv/bin/python scripts/train_tablm.py corpus && .venv/bin/python scripts/train_tablm.py train   # tab LM for --music_lm
```

E25 needs the custom dataset. Train E18 (`experiments/train_large_all.sh`) and E19 (`experiments/train_mid_all.sh`), then distil them into the small model, which takes about 11 h because the teachers run on every batch:

```bash
SIZE=small experiments/train_distill.sh > runs/tabmodel_small_distill.log 2>&1   # E25 (SIZE=half: E24, 2.9M)
```

Distilled models are under-confident on note attacks. Pick their onset threshold on validation and store it in the checkpoint as `"decoding": {"onset": 0.45}`; every command and the web app apply it.

- `scripts/train.py` options: `--device`, `--max_memory`, `--size half|small|mid|large`, `--steps`, `--batch`, `--lr` and `--accum` (gradient accumulation); the data shares `--gaps_share`, `--synth_share` and `--ss_share` (custom dataset) with their string-label weights `--*_tab_weight`; audio augmentation with `--aug_copies` and `--aug_share`; `--init` (fine-tune from a checkpoint); `--eval_degraded`; `--teacher` (repeatable) and `--distill` (distillation weight).
- The launchers cap GPU memory (`--max_memory 0.35`, a fraction of the GPU memory) and run at low priority, so the machine stays usable. `DEVICE=cuda experiments/...` picks the device.
- Re-running a launcher resumes from `last.pt`, and `STOP_AT=18:30 experiments/...` stops at the first evaluation after 18:30.
- The outputs go to `runs/<name>/`: `best.pt` (best validation score), `last.pt` (for resuming) and `log.jsonl`.

## 5. Evaluation

```bash
.venv/bin/python scripts/evaluate.py --checkpoint runs/tabmodel_mid_close/best.pt --split test             # GuitarSet test
.venv/bin/python scripts/evaluate.py --checkpoint runs/tabmodel_mid_close/best.pt --split test --variant deg   # degraded copy
.venv/bin/python scripts/evaluate.py --checkpoint runs/tabmodel_mid_close/best.pt --split validation --tune   # threshold search
.venv/bin/python scripts/evaluate_gaps.py --checkpoints runs/tabmodel_mid_close/best.pt                    # GAPS test
.venv/bin/python scripts/sweep.py --checkpoint runs/tabmodel_mid_close/best.pt --out runs/sweep.json       # P/R sweep + frame metrics
```

- Choose checkpoints and settings on validation only (GuitarSet player 04, GAPS validation), and report the test sets once per model.
- GuitarSet is scored on the fixed 15 s chunks of `data/eval_chunks.json`, frozen so that results stay comparable.
- `scripts/analyze_errors.py` breaks the errors down by category, and `scripts/evaluate_song.py` is the private benchmark on the author's own recording.

## 6. Repository layout

```
guitab/            the library: audio, model, inference, tablm, tab, fingering, gp5, pipeline, cli, app (+ web/),
                   device, data, augment, guitarset, gaps, synthtab, custom_data, evaluation, paths
scripts/           command-line tools: data preparation, caches, training, evaluation
experiments/       exact launchers of the training runs (see EXPERIMENTS.md)
assets/            the logo
data/              datasets and caches (not versioned except eval_chunks.json and the subset selections)
runs/              trained checkpoints and logs (not versioned)
weights/           the published weights downloaded from Hugging Face (not versioned)
out/               transcriptions (not versioned)
```


## Licence and credits

The code is MIT licensed (see `LICENSE`). The datasets keep their own licences: GuitarSet CC BY 4.0, GAPS CC BY-NC-SA 4.0, SynthTab CC BY-NC 4.0. The models are trained on them and inherit their non-commercial terms, so the published weights are CC BY-NC-SA 4.0, for research and personal use.

Beat tracking is done by [Beat This!](https://github.com/CPJKU/beat_this). The research started from [Audio2Tab](https://github.com/JackJamesLoth/Audio2Tab), whose schedule the GuitarSet evaluation chunks follow.
