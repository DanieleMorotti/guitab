# Guitar audio → tab: experiment log

> Entries before 2026-10-06 name scripts by their old `tools/` paths; see the table in the last entry for the
> current `guitab/` and `scripts/` names.

Each entry gives what was run, the exact command or script, and the result, so it can be repeated.
Add new entries at the bottom. All paths are relative to `guitab/`.

## Evaluation protocol (shared by all entries)

- **Data split:** GuitarSet split by player: 00–03 train, 04 validation, 05 test (60 tracks per player).
  Settings are chosen on validation only; test is reported once per finished model.
- **Main metric:** the Audio2Tab repo's `compute_note_detection_metrics` on the repo's fixed schedule
  of 15 s chunks (146 test chunks, 60 tracks; frozen in `data/eval_chunks.json`; the metric now lives in `tools/note_metrics.py`).
  It reports note F1 with onset matching:
  - `time_*`: onsets within ±50 ms.
  - `tick_*`: onsets within ±120 ticks (a 32nd note).
  - `*_pitch`: the pitch must match.
  - `*_string_fret`: the exact string and fret must match.
- **Non-token models:** their notes are written as tokens at the reference tempo with exact waits
  (`tools/eval_tabmodel.py`, `chunk_text`), so the same metric applies.
- **Benchmark-style metrics:** `tools/sweep_tabmodel.py` also gives the TabCNN definitions
  (frame-level tablature and multi-pitch P/R/F) and note-level F1 on full tracks.
  Caveat: published papers use 6-fold leave-one-player-out cross-validation. We use a single split,
  so our numbers are not directly comparable.
- **Environment:** macOS (M3 Pro, 36 GB), MPS.
  - Main venv: `.venv` (`requirements.txt`, torch 2.10). Until 2026-10-06 it was `Audio2Tab/.venv`, which also had transformers 5.3 for the Whisper runs.
  - Basic Pitch needs its own venv, `.venv_bp`.
- **Memory caps for training next to normal laptop use:**
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO` / `LOW` = 0.35 / 0.25 for the tab model, and 0.7 / 0.5 for Whisper.
  Launch long jobs with `nohup`, because background shells die after 2 h.

## Data

| Name | How it was made | Manifest / location |
|---|---|---|
| GuitarSet (mic audio, JAMS) | downloaded; `tools/guitarset_to_tokens.py` makes DadaGP tokens at 960 ticks per quarter | `data/guitarset.csv`, `data/guitarset/` |
| GuitarSet pitch-shifted +1/+2 semitones (train players only) | `tools/guitarset_shift.py` (librosa, 16 kHz), frets shifted to match | `data/guitarset_shift.csv` |
| Procedural tabs (deleted 2026-10-06) | `tools/gen_tabs.py` (6 textures + chord-melody), rendered with `tools/render_tabs.py` (tinysoundfont, 2 soundfonts × 6 programs, humanization, EQ, reverb, noise) | `data/procedural*.csv` |
| Chord-focused procedural tabs, 6000 (deleted) | `gen_tabs.py --chord_focus` | `data/procedural3.csv` |
| Classical gp5 renders (deleted) | `render_tabs.py` | `data/classical_synth*.csv` — **classical_synth2 test labels are time-misaligned; don't use for evaluation** |
| CQT cache for the tab model | `python tools/tabdata.py --workers 4` (about 5 min) | `data/tabmodel_cache/*.npz` |
| Mixes for Whisper (deleted) | combinations of the above | `data/mix_v2.csv`, `mix_v3.csv`, `mix_v4.csv` |

## Experiments

### E1 — Audio2Tab replication (Whisper-small → DadaGP tokens)
- **Model:** `classical_v1`, trained on classical renders only at learning rate 1e-5 (the repo default).
- **Result:** collapsed; all metrics about 0.
- **Lesson:** a learning rate of 1e-5 is too low for the new 653-token vocabulary; use 1e-4.

### E2 — Whisper `mix_v2` (procedural + classical + GuitarSet × 4)
- **Run:** `runs/mix_v2`, resumed via `runs/resume_mix_v2.sh`; batch 4 × gradient accumulation 2, gradient checkpointing, bf16.
- **Validation loss:** 2.27 → 1.44 by step 2500.
- **Test, checkpoint 3500, free decoding:** time pitch F1 0.130.

### E3 — Whisper `mix_v3` (fresh procedural data + GuitarSet + shifted copies)
- **Run:** `runs/mix_v3/checkpoint-1069`; validation loss 1.22. This was the best Whisper model.
- **Test, free decoding:** time pitch 0.177, time string/fret 0.132, tick pitch 0.269, tick string/fret 0.205; tempo accuracy 43%.
- **Test, true tempo forced** (`tools/final_eval.py`): 0.186 / 0.132 / 0.340 / 0.256.
- **Gotcha:** `generate()` strips the decoder prompt, so the forced tempo header must be added back.

### E4 — Tempo detection (`tools/tempo_eval.py`, test player)
- **Accuracy (exact / including ×2 errors):** Beat This! 70% / 80%, librosa 50% / 53%, the Whisper model's own guess 45% / 45%.
- **Adopted:** Beat This!, weights in `data/models/beat_this-final0.ckpt` (downloaded with curl because of SSL inspection).

### E5 — Post-processing for Whisper output (`tools/tune_post.py`, tuned on validation)
- **Basic Pitch chord fusion** (`tools/fuse.py`) plus the **Viterbi fingering** step (`tools/fingering.py`):
  test results 0.233 / 0.150 / 0.415 / 0.284.
- **Caveat:** Basic Pitch was partly trained on GuitarSet.

### E6 — Whisper `mix_v4` (+ 6000 chord-focused tabs)
- **Run:** `runs/mix_v4/checkpoint-944`; validation loss 1.294, worse than v3's 1.22.
- **Test, tempo forced:** 0.187 / 0.138 / 0.344 / 0.240.
- **Recall on chords of 3+ notes:** 0.442, against 0.434 for v3.
- **Conclusion:** more synthetic chords did not help, and string/fret got worse.

### E7 — Basic Pitch alone + fingering (`tools/transcribe.py --engine basic_pitch`)
- **Test chunks, amplitude ≥ 0.4:** time pitch 0.712, tick pitch 0.730, time string/fret 0.320, tick string/fret 0.331 (with fingering).
- **Diagnosis:** Whisper's spectrogram has 40 Hz frequency resolution (`n_fft = 400` at 16 kHz), so low notes can't be told apart.
- **Decision:** switch to a CQT frame/onset model.

### E8 — CQT frame/onset tab model, `mid` (`runs/tabmodel_mid`)
- **Input:** log-CQT, 36 bins per octave from E1, 288 bins, 22.05 kHz, hop 256 (11.6 ms).
  The harmonic stack (h = 0.5, 1, 2, 3, 4, 5) is built inside the model.
- **Model** (`tools/tabmodel.py`, 23.8M parameters):
  - 6 conv blocks (64 / 128 / 256 channels, frequency pooled 144 → 12);
  - transformer encoder (d = 512, 6 layers, 8 heads);
  - heads: tab (6 strings × 21 classes), onset (6), and auxiliary pitch (48).
- **Training** (`runs/train_tabmodel_mid.sh` → `tools/train_tabmodel.py`):
  - 12k steps, batch 16, 5 s crops; AdamW at 3e-4 with 1k warm-up steps and cosine decay.
  - Augmentation on the CQT: pitch roll 0–2 semitones, level, tilt, noise floor, time masks.
  - About 8 GB of GPU memory, 1.1 s per step, about 3.7 h in total.
  - Training data: players 00–03 plus the shifted copies, 6.1 h of audio (only about 2 h is real audio).
- **Validation, full-track string/fret F1 by step:**

  | Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | F1 | .539 | .667 | .714 | .737 | .723 | .749 | .734 | .755 | .753 | **.757** | .751 | .752 |

  Best is step 10000 → `best.pt`. Training tab loss reached 0.08 while validation flattened, so the model
  is limited by data, not by size.
- **Note extraction tuned on validation** (`eval_tabmodel.py --tune`): onset 0.6, frame 0.6, legato 8, orphan off.
- **Test chunks:**
  - time pitch 0.836, time string/fret 0.811, tick pitch 0.845, tick string/fret 0.818.
  - With the fingering step on top: 0.808 (`auto`) / 0.809 (`strong`), so it slightly hurts.
- **User audio:** `out/my_video/my_video_E_tabmodel_{model,low}.gp5`.

### E9 — Confidence / fingering sweep (`tools/sweep_tabmodel.py`)
- **Command:**
  `python tools/sweep_tabmodel.py --checkpoint runs/tabmodel_mid/best.pt --out runs/sweep_tabmodel_mid.json`
  (about 25 min). The full table, with notes per chunk, is in the JSON.
- **Results** for time ±50 ms, as precision / recall / F1, model strings, "orphan" off:

  | Onset threshold | Val pitch | Val string/fret | Test string/fret | Notes per val chunk |
  |---|---|---|---|---|
  | 0.2 | .642 / .775 / .702 | .625 / .756 / .685 | .700 / .829 / .759 | 93.6 |
  | 0.3 | .688 / .761 / .723 | .670 / .742 / .705 | .744 / .821 / .781 | 85.7 |
  | 0.4 | .723 / .750 / .736 | .705 / .731 / .718 | .777 / .814 / .795 | 80.3 |
  | 0.5 | .751 / .736 / .744 | .731 / .717 / .724 | .807 / .804 / .806 | 75.9 |
  | **0.6** (default) | .785 / .721 / .752 | .764 / .702 / .731 | .830 / .792 / .811 | 71.1 |

- **Findings:**
  - Lowering the threshold from 0.6 to 0.2 raises recall by only about 0.05 but costs about 0.14 precision.
    Most missed notes are not low-confidence notes; the model doesn't detect them at all.
  - Orphan notes (onset-less) lower F1 by 0.03–0.05 at every threshold.
  - Fingering:
    - `auto` costs about 0.004.
    - `low` (forced first position) halves string/fret F1 on GuitarSet (.81 → .36), because GuitarSet
      players use the whole neck. So `low` is a playing-style prior, and only a player's own data can say if it's right.
- **Benchmark-style metrics, test player 05, full tracks, best.pt:**
  - Frame-level tablature P/R/F: .840 / .872 / **.856**.
  - Frame-level multi-pitch P/R/F: .880 / .900 / **.890**.
  - Note-level, onset ±50 ms, mean F1 per track: pitch **.889**, string+fret **.860** (our greedy matcher, not mir_eval).
- **Published reference points** (different protocols, so only a rough comparison):
  - TabCNN, 2019, 6-fold player cross-validation, frame-level: tab .748, multi-pitch .826.
  - TabCNN as reported in SynthTab, random 8:1:1 track split: tab .801, multi-pitch .845.
  - FretNet and Noise2Fret (2026), 6-fold, note-level: tab .727 and .845; pitch .818 and .856.
  - GAPS paper, GuitarSet test, note onset F1 (pitch only): Riley et al. .897, GAPS model .912.

### E10 — Adding fingerstyle data: GuitarSet + GAPS (`runs/tabmodel_mid_gaps`) — DONE, new default model
**Choice of data (2026-10-05):** the user's focus is fingerstyle guitar. I reviewed the open datasets:
- **[GAPS](https://zenodo.org/records/13962272) — chosen.** Classical nylon-string guitar from 200+ players in varied rooms and
  on varied microphones, all fingerpicked. Note-level MIDI is aligned to YouTube audio, and the MusicXML scores usually carry
  a tab staff (`<string>`/`<fret>`). Licence CC BY-NC-SA, which is fine for this personal project.
- **Not chosen now:**
  - Guitar-TECHS and EGDB: electric guitar, mostly techniques, scales and rock playing.
  - IDMT-SMT-Guitar: mostly isolated notes and licks.
  - SynthTab: synthetic, about 2 TB. Kept as a later option for pretraining.
  - GOAT: restricted access.

**GAPS preparation (repeat in this order):**
1. `curl` the zip from Zenodo into `data/gaps/`, then run `python tools/gaps_common.py`.
   - GAPS ids are case-sensitive (`GM1wc` ≠ `gM1wc`). A plain unzip on macOS silently overwrote 278 files.
   - `gaps_common.extract()` writes every file under the case-safe name `safe(h) = h_md5(h)[:6]` in `data/gaps/safe/`.
2. Run `python tools/gaps_download.py --workers 3` (yt-dlp, 22.05 kHz mono), twice to retry 403 errors.
   - The corporate HTTPS proxy needs `SSL_CERT_FILE=data/certs/macos_trusted.pem` (exported from the macOS keychains)
     plus `--compat-options no-certifi`.
   - Result: 296 of 301 recordings downloaded; 5 are unavailable or blocked.
   - The MIDI timeline matches the full YouTube audio (offset 0.02 s, checked on `mvswc`).
3. Run `python tools/gaps_labels.py --workers 4`. String/fret per MIDI note comes from the score's tab staff:
   - Score and MIDI chord sequences are aligned by note order (an LCS-style dynamic programme on pitch overlap),
     then string/fret is copied by pitch.
   - Interpolating the syncpoints was unreliable. They count bars with repeats unfolded, and music21 cannot expand
     some scores' repeats.
   - Result: median match rate 0.57 (share of MIDI notes given a string).
   - 63 pieces use non-standard tuning (mostly drop-D) and are excluded for now. 3 pieces have MIDI that doesn't match the score.
4. Selection, in `tabdata.gaps_selection`: standard tuning, content similarity ≥ 0.9, match rate ≥ 0.5, audio present.
   - Result: 156 pieces. 123 train (6.53 h), 14 validation (0.72 h, every 10th train piece sorted by id),
     and 19 test (1.09 h, the official GAPS test split).
5. Run `python tools/tabdata.py --workers 4` to cache CQT features for the GAPS pieces.

**Label handling:**
- Notes without a string label (37% of active frames in GAPS train) are masked:
  - the tab head ignores strings not otherwise labelled active during such notes;
  - onsets are ignored around their attack;
  - the pitch head ignores those frames.
- GAPS string labels are score fingerings, not necessarily what was played, so they get tab-loss weight 0.5.
  Pitch and onset supervision from the aligned MIDI is used at full weight.

**Baseline (E8 model `tabmodel_mid/best.pt`, zero-shot), with `python tools/eval_gaps.py`:**
- GAPS validation: pitch F1 .629, string agreement .720.
- GAPS test: pitch F1 .594, string agreement .690.
- For comparison, GuitarSet test pitch is .836, so the model suffers a large domain gap on fingerstyle recordings.

**Training** (`runs/train_tabmodel_mid_gaps.sh`):
- Same mid architecture and augmentations as E8, trained from scratch.
- 16k steps, batch 16, learning rate 3e-4 with cosine decay; 50% of crops from GAPS (`--gaps_share 0.5`), `--gaps_tab_weight 0.5`.
- About 8 GB of GPU memory, 1.14 s per step, about 5 h in total.
- Validation every 1k steps on both GuitarSet player 04 (string/fret, pitch) and GAPS validation (pitch, string agreement).
  `best.pt` maximises the mean of GuitarSet string/fret F1 and GAPS pitch F1.
- The E8 weights in `runs/tabmodel_mid/` are kept unchanged.

**Validation by step** (GuitarSet string/fret F1, full tracks / GAPS pitch F1 / GAPS string agreement):

| Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k | 14k | 15k | 16k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GuitarSet string/fret | .473 | .699 | .739 | .753 | .776 | .772 | .760 | .777 | .792 | .794 | .804 | .808 | .810 | .807 | .809 | .807 |
| GAPS pitch | .576 | .740 | .755 | .790 | .779 | .821 | .822 | .833 | .832 | .840 | .840 | .840 | .843 | .845 | .845 | .846 |
| GAPS string agreement | .775 | .795 | .874 | .899 | .908 | .894 | .885 | .888 | .907 | .899 | .890 | .903 | .900 | .899 | .903 | .904 |

- **Best:** step 15000 → `best.pt` (selection score .827).
- **Comparison with E8:** from step 2 onward the GuitarSet scores beat the GuitarSet-only run, so the extra real fingerstyle audio
  helps GuitarSet as well. The run is converged by 13k–16k steps.
- **Note extraction:** re-tuned on GuitarSet validation; the optimum is unchanged (onset 0.6, frame 0.6, legato 8, orphan off).

**Test results** (models chosen on validation only):

| Test set | Metric | E8 `tabmodel_mid` | **E10 `tabmodel_mid_gaps`** |
|---|---|---|---|
| GuitarSet player 05, 146 chunks | time pitch F1 | .836 | **.851** |
| | time string/fret F1 | .811 | **.826** |
| | tick pitch F1 | .845 | **.861** |
| | tick string/fret F1 | .818 | **.836** |
| GAPS test, 19 pieces, 15.7k notes | pitch F1 (all MIDI notes) | .594 | **.787** |
| | string agreement with score fingering | .690 | **.864** |

**Conclusion:** adding GAPS:
- raises fingerstyle pitch F1 on unseen players, rooms and microphones by +.19;
- improves GuitarSet by +.015 as well.

E10 is the new default model.

**User audio:** `out/my_video/my_video_G_tabmodel_gaps_{model,low,onset04}.gp5`.
- Default settings: 494 notes. With `--onset 0.4`: 587 notes.
- The model now puts the opening C-major shape in first position by itself (B string fret 1, open G), where E8 needed `--fingering low`.

**Next options:**
- Drop-D and other non-standard tunings: 63 GAPS pieces, which needs a tuning-aware label scheme.
- SynthTab acoustic pretraining.
- The large model, now that there is more data.
- A labelled clip from the user, to tune the settings for their guitar.

### Clean-up — 2026-10-06
The Whisper / Audio2Tab approach (E1–E7) is outperformed by the CQT tab model, so all its artifacts were deleted, freeing about 84 GB.
The results above remain the record of it.

**Deleted:**
- **Runs:** `runs/mix_v2`, `mix_v3`, `mix_v4`, `classical_v1`; benchmark, overfit and smoke runs; their logs, predictions and evaluation JSONs.
- **Synthetic data:** `data/procedural*`, `classical*`, `soundfonts`, `test_render` and the `mix_v*.csv` manifests.
- **Whisper-only scripts:** `gen_tabs.py`, `render_tabs.py`, `render_gp5.py`, `final_eval.py`, `tempo_eval.py`, `tune_post.py`,
  `fuse.py`, `analyze_preds.py`, `diagnose.py`.
- **Whisper outputs on the user audio:** versions A and B, and the first `my_video.gp5`.
- **Tab-model snapshots:** the non-best `last.pt` and `step*.pt` checkpoints.

**Changed:**
- `tools/transcribe.py` no longer has the `whisper` engine or Basic Pitch chord fusion. The default is `--engine tabmodel` with
  `runs/tabmodel_mid_gaps/best.pt`.
- Evaluation reads the fixed GuitarSet chunks from `data/eval_chunks.json`, rebuilt from GuitarSet alone.
  Before deletion I verified them identical to the old reference files: 146 test and 147 validation chunks, same texts, paths and starts.
  The E10 test result reproduced exactly (.8505 / .8258), and so did the user tabs.

**Kept at the time:** the `Audio2Tab/` repo, for the venv, metrics, chunk schedule and gp5 writer. It was removed in the next entry.

### Audio2Tab repo removed — 2026-10-06
The pipeline used only three things from `Audio2Tab/`, so the repo was removed and they were taken into ours:

| From Audio2Tab | Now | Check |
|---|---|---|
| `src/metrics.py` + `src/tab.py`: `compute_note_detection_metrics` | `tools/note_metrics.py` (~140 lines) | Identical scores to the original on all 293 eval chunks, against randomly perturbed predictions |
| `src/dadagp_utils/` (DadaGP converter, gp5 writer) | `tools/dadagp_utils/`, copied unchanged | `.gp5` output byte-identical |
| `src/data.py` chunk schedule (only used to build the eval chunks) | not copied; `data/eval_chunks.json` is now a frozen artifact | — |

**Environment:**
- The old env was frozen to `env_freeze_audio2tab_old.txt` (114 packages).
- The new `.venv` was installed from `requirements.in` (direct dependencies), using the old freeze as constraints,
  then locked to `requirements.txt` (61 packages). Every package is a subset of the old env at the same version.

**Verified in the new env:**
- GuitarSet test .8505 / .8258.
- GAPS test .787 / .864.
- User tabs byte-identical for both engines.
- A short training run with GAPS works, and all scripts import.

Audio2Tab has no licence file. The copied code keeps attribution headers; check licences before publishing anything.

### Simplification — 2026-10-06
- Removed the Basic Pitch engine (E7 baseline, outperformed by the tab model everywhere): `tools/run_basic_pitch.py`,
  the `--engine` option of `transcribe.py`, the `.venv_bp` environment and the `data/basic_pitch` cache.
  E7 numbers remain above as the record.
- Requirements reduced to one file, `requirements.txt`: exact pins, with the direct dependencies listed in its header.
  `requirements.in` and the old Audio2Tab freeze were dropped.
- The project is now a git repository. Data, weights, outputs and environments are excluded (`.gitignore`);
  only code, docs and the frozen `data/eval_chunks.json` are tracked.

### Refactor into a package — 2026-10-06
The flat `tools/` folder was split into a library package, `guitab/`, and command-line entry points, `scripts/`.
Results are unchanged; verification is below.

| Old | New |
|---|---|
| `tools/tabmodel.py` | `guitab/model.py` |
| `tools/tabdata.py` | `guitab/data.py`, `guitab/audio.py` (CQT), `guitab/guitarset.py` (JAMS notes); `main()` → `scripts/build_cache.py` |
| `tools/tabmodel_infer.py` | `guitab/inference.py`; `match_f1` / `ref_notes` → `guitab/evaluation.py` |
| `tools/notes_to_tab.py` (+ `build_tokens` from `guitarset_to_tokens.py`, `decode_to_gp5` from `transcribe.py`) | `guitab/tab.py` |
| `tools/note_metrics.py`, `tools/fingering.py`, `tools/dadagp_utils/` | `guitab/metrics.py`, `guitab/fingering.py` (+ `FINGERING_PRESETS`), `guitab/dadagp_utils/` |
| `tools/gaps_common.py` + label logic of `tools/gaps_labels.py` | `guitab/gaps.py`; CLIs `scripts/gaps_extract.py`, `scripts/gaps_labels.py`, `scripts/gaps_download.py` |
| `tools/train_tabmodel.py`, `eval_tabmodel.py`, `eval_gaps.py`, `sweep_tabmodel.py`, `transcribe.py` | `scripts/train.py`, `evaluate.py`, `evaluate_gaps.py`, `sweep.py`, `transcribe.py` (evaluation helpers → `guitab/evaluation.py`) |
| `tools/guitarset_to_tokens.py`, `tools/guitarset_shift.py` | `scripts/prepare_guitarset.py` (writes only the track list; the reference token files were used only to build the frozen eval chunks), `scripts/guitarset_shift.py` |
| `runs/train_tabmodel_mid*.sh` | `scripts/train_mid.sh`, `scripts/train_mid_gaps.sh` |

**Other changes:**
- **Shared constants:** `OPEN_PITCH`, `STRINGS` and the fret classes now live in `guitab/guitar.py`, instead of being defined in four places.
  All paths are in `guitab/paths.py`.
- **Relative paths:** `data/guitarset.csv`, `data/guitarset_shift.csv` and `data/eval_chunks.json` now store repo-relative audio paths
  instead of absolute `/Users/...` ones. The track lists dropped their unused columns.
- **Bug fix:** `scripts/gaps_labels.py --only` used to overwrite `labels_summary.csv` with only the requested pieces,
  which would silently shrink the GAPS selection. It now updates just those rows.

**Verification** (old code snapshot vs new code, same data and weights):
- Every evaluation output is identical:
  - `evaluate.py` on validation and test for both models, all four metrics, with and without fingering;
  - `evaluate_gaps.py` on GAPS test and validation.
- Transcriptions of the demo recording with 5 option sets are byte-identical (`.gp5` and tokens):
  default; `--fingering low`; `--fingering auto --orphan`; `--onset 0.4 --tempo 90 --grid 120`; the E8 checkpoint.
- Function-level checks all pass:
  - `tracks()` (996 tracks) is identical, and so are the `load_split` arrays for 3 splits;
  - CQT features and frame labels recomputed by the new code equal the cached files written by the old code
    (4 GuitarSet/shifted tracks, 3 GAPS pieces);
  - 5 seeded training batches with augmentation, plus the model outputs and loss values, are identical;
  - GAPS labels for 6 pieces and their summary rows are identical;
  - case-safe extraction (1103 files) and the download list are identical.
- The regenerated track lists match the old ones (same tracks, order, audio files and splits).
- Training, evaluation, checkpoint saving and resume work through the new CLI.
- The full `sweep.py` output on E8 is identical to `runs/sweep_tabmodel_mid.json`.
