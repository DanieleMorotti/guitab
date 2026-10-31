# Changelog

What changed for people who use Guitab, one version at a time. The experiments behind each change, with all the numbers, are in [EXPERIMENTS.md](EXPERIMENTS.md). Versions follow [semantic versioning](https://semver.org): a new major version changes the results or the commands in a way you may notice, a minor version adds features or improves the results, a patch fixes bugs.

Published weights never change under the same file name: a new or retrained model gets a new file on [Hugging Face](https://huggingface.co/danmor/guitab), so results stay reproducible and the browser version never keeps a stale copy.

## 1.2.0 — 2026-10-31

- **Chord names above the tab.** A small chord model (135k parameters, 0.5 MB) reads the transcribed notes on the beat grid and names the chords: on real fingerstyle songs about 3 in 4 shown chords are right, on strummed accompaniment about 9 in 10. Where it is unsure, typically on a melody alone, it shows nothing. They are in the .gp5 as chord names (no diagrams), the same in the `guitab` command, the local app and the browser version (byte for byte).
- **Chords switch** in the local app and the browser version to show or hide them, with a note that they are approximate. `guitab --no_chords` leaves them out.
- **New files** on Hugging Face: `chord_model.safetensors` and `chord_model.onnx`, downloaded automatically on first use. The tab models are unchanged, and so are the notes of every tab.

## 1.1.1 — 2026-10-27

- **Download counts on Hugging Face.** The model page now shows how many times a model was downloaded: the first time a machine or a browser downloads a tab model, Guitab requests the small `config.json` file, which is what the Hub counts. Nothing else is sent, and later runs with the model already downloaded are not counted. `hf download danmor/guitab` counts too.
- **New file** on Hugging Face: `config.json` (the version, the models and their files).

## 1.1.0 — 2026-10-26

- **Better rhythm in the tab.** The beats now come from the beat tracker's activation together with the model's own note onsets, tracked by dynamic programming, so they stay regular but follow the player when the tempo changes. Notes get the right note value more often: +5 points on steady-tempo songs and +13 on free-tempo classical pieces (GuitarSet and GAPS test sets), and the bar lines fall in the right place more often.
- **Smaller and faster beat tracking.** The small Beat This! model (3.6 MB) replaces the large one (21 MB). A 2:23 song now takes about 16 s in the browser instead of 26 s, and 6 s instead of 11 s with the `guitab` command. The large model can still be selected with `BEAT_MODEL` in `guitab/paths.py`.
- **New file** on Hugging Face: `beat_this_small0_int8.onnx`, downloaded automatically on first use. The tab models are unchanged.

## 1.0.0 — 2026-10-23

First public release.

- **Models:** E25 (the default, 6.2M parameters), E24 (2.9M, the fastest) and E19 (24M), plus an optional tab language model for fuller chords (`--music_lm`). Weights on Hugging Face as `.safetensors`, downloaded automatically on first use.
- **`guitab` command:** any audio or video file in, a Guitar Pro `.gp5` tab out, with capo, fingering, tempo and grid options.
- **Local web app** (`guitab-app`): record or load a song, transcribe it, view and play the tab back.
- **Browser version** at [danielemorotti.github.io/guitab](https://danielemorotti.github.io/guitab/): the whole transcription runs in the browser, giving the same `.gp5` as the Python pipeline.
