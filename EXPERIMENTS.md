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

### A1 — Error analysis of E10 (`scripts/analyze_errors.py`) — 2026-10-06
- **Command:** `python scripts/analyze_errors.py --checkpoint runs/tabmodel_mid_gaps/best.pt --out runs/errors_tabmodel_mid_gaps.json`
  (about 5 min). Validation and train subsets only; no test data.
- **Protocol:** full tracks, default note extraction, greedy matching (onset ±50 ms), notes pooled over tracks
  (so numbers differ slightly from the per-track means logged in training). Each miss / false note gets the first
  category that applies.

| | GuitarSet val | GuitarSet train (1/4) | GAPS val | GAPS train (1/9) |
|---|---|---|---|---|
| pitch P / R / F1 | .798 / .832 / .815 | .862 / .923 / .891 | .819 / .858 / .838 | .823 / .833 / .828 |
| string/fret F1 | .798 | .891 | – | – |
| string correct, given correct pitch | .960 | .998 | .859 (vs score fingering) | .916 |
| frame multi-pitch F1 | .878 | .937 | – | – |
| onset error median / share ≤ 25 ms | 4.8 ms / .97 | 4.9 ms / .98 | 6.0 ms / .77 | 4.3 ms / .97 |

**Missed notes** (share of misses): GuitarSet val — not heard at all .31, found 50–150 ms off .24, re-attack of a
ringing same pitch merged .13, semitone/tone confusion .10, weak frame evidence .08, chord collision .07,
octave/harmonic .05, heard but onset < 0.6 only .025. GAPS val — not heard at all .41, weak evidence .19,
heard but onset < 0.6 .09, chord collision .09, others ≤ .07.
Miss rate by context (GuitarSet val): notes < 120 ms .25–.36 vs .10–.18 for longer notes. GAPS val: .40 when 4+ notes sound.

**False notes** (share of false notes): a new attack of a pitch that is still ringing in the reference is the largest class,
.33 on GuitarSet and .58 on GAPS. Almost all of them are real onset peaks ≥ 0.6, a median 0.29 s (GAPS 0.44 s) after the
reference attack, not legato splits. Then timing 50–150 ms (.21 / .09), octave/harmonic (.15 / .11), semitone (.12 / .10).

**Worst GuitarSet val tracks:** `04_BN3-154-E_comp` (.22, fast bossa comping; matches spread over ±300 ms),
`04_Jazz1-200-B_comp` (.31); every other track is ≥ .67. Comping .79 vs solo .86 pitch F1.

**Findings:**
- **Strings are nearly solved on GuitarSet:** given the right pitch, the string is right 96% of the time, so a perfect fingering
  step could add at most about .017 string/fret F1. The errors are pitch and onset errors.
- **GuitarSet is data-limited:** train .891 vs validation .815 (an unseen player).
- **GAPS is underfitted:** train .828 vs validation .838, with no generalisation gap. A larger model or longer training
  should help here; label noise (alignment, unmatched notes) also caps it.
- **Thresholds won't recover the misses:** only about 2.5% (GuitarSet) / 9% (GAPS) of misses have frame evidence and a weak
  onset, consistent with E9.
- **Note decoding loses information:** frame multi-pitch F1 (.878) is well above note pitch F1 (.815). Re-attacks (merged or
  spurious) and short notes are the weak points of the onset head plus the hand-written extraction.
- **Even on training tracks, GuitarSet note F1 is only .89:** 30% of those misses are 50–150 ms timing differences, so part of
  the ceiling is annotation noise rather than the model.

### D1 — Note extraction: drop same-pitch duplicates on two strings — 2026-10-06, ADOPTED
Post-processing only, no retraining; tested on the E10 predictions, chosen on validation, then test once.
- **Problem (from A1):** the model sometimes starts the same pitch on two strings at once (unsure which string),
  which gives one true note plus one false note. This is frequent on GAPS: 924 known validation notes had such a pair.
- **Rule** (`dedup` in `DEFAULT_PARAMS`, `guitab/inference.py`): if the same pitch starts on two strings within 4 frames (46 ms),
  keep only the note whose fret posterior, averaged over the note, is higher. `dedup: 0` turns it off and reproduces the
  previous results exactly (checked: E10 validation chunks .7603 / .7439).
- **Choices on validation:**
  - window 2, 4, 6 or 8 frames: all within ±.0003 → 4;
  - which note to keep: fret posterior (GuitarSet string/fret .8127) beats attack confidence (.8101), max(onset, active) (.8105)
    and lowest fret (.8043); duration ties (.8129) → fret posterior, which also gave the best GAPS string agreement;
  - re-tuned thresholds: unchanged (onset 0.6, frame 0.6, legato 8, orphan off).
- **Rejected:** requiring an energy rise at the pitch for a re-attack of a still-ringing pitch. It gave +.004 GAPS pitch but cost
  .004–.015 on GuitarSet at every threshold. Real re-plucks and false ones have the same onset confidence (median .96 vs .95),
  so this needs the model (release head, E11), not a rule.
- **Validation:** GuitarSet chunks pitch .7603 → .7672, string/fret .7439 → .7465; GuitarSet full tracks (training eval)
  string/fret .8087 → .8127, pitch .8261 → .8346; GAPS validation pitch .8454 → .8824.
- **Test (once):**

  | Test set | Metric | E8 before → after | E10 before → after |
  |---|---|---|---|
  | GuitarSet player 05 chunks | time pitch F1 | .836 → .846 | .851 → **.859** |
  | | time string/fret F1 | .811 → .812 | .826 → **.828** |
  | | tick pitch / string-fret F1 | .845 / .818 → .854 / .820 | .861 / .836 → **.870 / .839** |
  | GAPS test | pitch F1 | .594 → .622 | .787 → **.811** |
  | | string agreement | .690 → .644 | .864 → .840 |

- **About the GAPS string agreement drop:** this is not a regression. Agreement is measured on pitch-correct notes, and a duplicate pair
  is two guesses of the string: in 898 of the 924 validation pairs one of the two was on the score's string. After dedup the kept
  note is on the score string in 62% of those cases, while pitch hits are unchanged (8696 → 8695). So .840 is the honest figure,
  and the earlier .864 was inflated by duplicates.
- **Demo recording:** one duplicate removed (472 → 471 notes), `out/my_video/my_video_H_dedup.gp5`.
- **Note:** training validation (`evaluate_guitarset` / `evaluate_gaps`) uses `DEFAULT_PARAMS`, so from E11 on the logged
  validation numbers include dedup; compare with the "after" figures above, not with E10's training log.

### E11 — Precise attack targets + release head (`runs/tabmodel_mid_gaps_reg`) — ABANDONED, code removed
**Goal:** attack the error classes from A1 that a rule could not fix (D1): spurious and merged re-attacks of a ringing pitch,
attacks 50–150 ms off, short notes. The backbone, data, sampling, augmentation, steps and seed are the same as E10.
The change is only in what the heads learn and how notes are built:
- **Attack head:** the target is a triangle, `max(0, 1 − |t − attack| / 4 frames)`, around the exact attack time
  (Kong et al. 2021, high-resolution piano transcription), learned with plain BCE. E10 used 1 on the attack frame,
  0.5 on its ±1 neighbours and positive weight 5.
- **New release head** (6 outputs, about 3k parameters): the same triangle around each note's end, per string.
  GAPS releases are ignored, because GAPS MIDI durations come from the score's note values, not from when the string stops.
  GuitarSet ends come from the hexaphonic pickup.
- **Unknown-string GAPS notes:** attack targets are ignored ±4 frames around their attack on strings not otherwise targeted.
- **Note building** (`guitab/inference.py`, used only when a checkpoint has a release head):
  - attack time refined within the frame by a parabola through the onset peak (`subframe`);
  - a note also ends at a release peak ≥ `offset`.
- **Old checkpoints are unaffected** (no release head → same path). Checked after the change:
  - 3 seeded training batches with the model outputs and losses are identical;
  - E10 validation chunks are identical (.7672 / .7465), and so is GAPS validation (.8824 / .8639);
  - the demo tab is byte-identical.
- **Command:** `nohup scripts/train_mid_gaps_reg.sh > runs/tabmodel_mid_gaps_reg.log 2>&1 &`
  - 16k steps, batch 16, `--gaps_share 0.5 --gaps_tab_weight 0.5 --onset_target regression`;
  - validation during training uses `{"onset": 0.4, "subframe": true}`, since triangle targets peak below 1;
  - thresholds are tuned on validation after training (`evaluate.py --tune`, which also searches `offset` and `subframe`).
- **Comparison point:** E10 + D1 on the same in-training protocol: GuitarSet validation string/fret .8127, pitch .8346;
  GAPS validation pitch .8824, string agreement .8639. The final comparison uses tuned settings for both models.
- **Decoding bug, fixed at step 4000:**
  - **Bug:** a note ended at a release peak while the string still looked active, and the legato rule then started a new note
    there (fake notes).
  - **Cost:** on the step-3000 checkpoint (GuitarSet validation, onset 0.4) it cost .04: string/fret .690 with releases
    vs .733 without them.
  - **Fix:** a legato continuation now happens only after a fret change, never after a release.
  - **Result:** releases are now neutral for attack F1 (.7322–.7337 for release thresholds 0.5–0.7 vs .7331 without them).
    That is expected: they change note lengths, not attacks.
  - **Old checkpoints:** unaffected (E10 validation chunks still .7672 / .7465).
  - **The run:** stopped right after the step-4000 checkpoint and resumed from it with the fixed code, so validation
    scores up to step 4000 in the log use the buggy decoding.
  - **Side effect of the resume:** data sampling is reseeded at the resume step (`rng.seed(step)`), so the batches after 4000
    differ from an uninterrupted run.

**Validation by step** (in-training protocol, onset 0.4 + subframe + dedup; steps 1k–4k used the buggy release decoding):

| Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GuitarSet string/fret | .390 | .634 | .690 | .710 | .785 | .797 | .792 | .804 | .811 | .810 | .816 | **.818** | .818 |
| GAPS pitch | .379 | .688 | .673 | .793 | .817 | .838 | .866 | .849 | .864 | .874 | .862 | **.871** | .869 |
| GAPS string agreement | .691 | .739 | .802 | .830 | .832 | .819 | .831 | .841 | .850 | .855 | .861 | .853 | .862 |

- **Stopped after the step-13000 eval** (laptop shutdown). By then the learning rate was 4e-5 on its cosine decay and the selection score
  had flattened (.8444 at 12k, .8436 at 13k). `best.pt` = step 12000. `last.pt` is kept, so the run can be resumed to 16k
  with `scripts/train_mid_gaps_reg.sh`.
- **Tuning on validation** (`evaluate.py --tune`, release grid): onset 0.4, frame 0.5, legato 8, orphan off, dedup 4, offset 0.7, subframe on.
  - The release threshold makes no difference to attack-based F1: 0.7 and "off" (1.01) give the same scores.
  - Sub-frame attack times were selected.

**E11 vs E10 + D1** (each with its own tuned settings):

| Set | Metric | E10 + D1 | E11 |
|---|---|---|---|
| GuitarSet validation chunks | time pitch / string-fret F1 | .7672 / .7465 | .7710 / .7484 |
| GAPS validation | pitch F1 / string agreement | .8824 / .8639 | .8698 / .8539 |
| **GuitarSet test chunks** | time pitch / string-fret F1 | .8593 / .8283 | **.8622 / .8306** |
| | tick pitch / string-fret F1 | .8696 / .8385 | **.8719 / .8400** |
| **GAPS test** | pitch F1 / string agreement | .8108 / .8403 | .8106 / **.8447** |

**Conclusion:**
- On attack-based note F1, E11 is roughly level with E10: +.002–.003 on GuitarSet test, equal on GAPS test pitch,
  −.013 on GAPS validation pitch. That is within run-to-run noise, so the new targets did not fix the missed and
  re-attacked notes found in A1.
- E11 also had 3k fewer training steps; E10's best was at step 15000.
- The release head's real effect is on note lengths (rests, let-ring), which none of our metrics measure. Next step:
  add an attack + release F1 (mir_eval style: release within max(50 ms, 20% of the note length)) and compare E10 and E11
  on it, together with a listening check on the demo, before choosing the default model.
- `DEFAULT_CHECKPOINT` stays E10 (`runs/tabmodel_mid_gaps/best.pt`). E11's tuned settings are the ones listed above; pass them
  with `--params` (or via `transcribe.py --onset 0.4`, plus the release settings, once that is wired).
- **Demo recording:** `out/my_video/my_video_I_e11.gp5` (E11, tuned settings) vs `my_video_H_dedup.gp5` (E10 + D1, the default):
  ```
  .venv/bin/python scripts/transcribe.py --audio out/my_video/my_video.wav --out_dir out/my_video --name my_video_I_e11 \
    --checkpoint runs/tabmodel_mid_gaps_reg/best.pt --params '{"onset": 0.4, "frame": 0.5, "offset": 0.7, "subframe": true}'
  ```
  `transcribe.py --params` was added for this; without it, the output is unchanged (E10 demo byte-identical).
- **Abandoned (2026-10-07):** the user compared the two demo tabs and preferred E10. The scores were level, and the other
  options for this approach are small: the last 3k steps, putting the positive weight back on the attack target, or tweaking
  the release decoding. Each would move results by about ±.005, within noise. The release head also learns only from
  GuitarSet's ~2 h of real audio (GAPS note lengths come from the score).
  - **Removed:** the E11 code (regression targets, release head, `--onset_target`, the release decoding and tuning grid,
    `scripts/train_mid_gaps_reg.sh`), `runs/tabmodel_mid_gaps_reg/` (546 MB) and the E11 demo tab.
  - **Kept:** D1 (dedup), `scripts/analyze_errors.py` and the generic `transcribe.py --params` option.
  - **Code at the time of the run:** commits `335a11f` and `37fe2b9`; it can be restored from git if needed.
  - **Checked after the removal:** the code under `guitab/` and `scripts/` (except `transcribe.py`) is byte-identical to the D1 commit;
    E10 validation chunks .7672 / .7465; GAPS validation .8824 / .8639; the demo tab `my_video_H_dedup.gp5` is byte-identical.

### E12 — Audio augmentation for phone / room recordings (`runs/tabmodel_mid_gaps_aug`) — DONE, new default model
**Goal:** make the model robust to recordings like the user's (phone / camera audio in a room). The test sets are cleaner recordings,
so robustness is measured on held-out degraded copies, built with different effects from the training ones.

**Effects** (`guitab/augment.py`). Every effect keeps timing (0 ms shift measured with onset-envelope cross-correlation)
and length, so the original labels are reused.
- **Training copies** (`train_chain`, pedalboard 0.9.25): 2 copies per training track (720 GuitarSet incl. shifted copies,
  123 GAPS), each with its own random chain, seeded by name + copy and listed in `data/tabmodel_cache_aug/chains.tsv`:
  - algorithmic reverb (p .5, room size .1–.7, wet .05–.35);
  - high-pass 60–300 Hz + low-pass 3.5–10.5 kHz (p .8);
  - 1–3 EQ peaks of ±8 dB between 150 Hz and 6 kHz (p .7);
  - compressor (p .6);
  - white / pink / brown noise at 15–40 dB SNR (p .5);
  - MP3 at VBR quality 2–9 (p .5).
  - Resulting mix over the 1,686 copies: band-limiting 79%, compression 62%, MP3 52%, reverb 49%, noise 48%; 7 copies untouched.
- **Held-out degradation** (`degrade`), used only for evaluation, on the validation and test splits (153 tracks):
  - convolution with a synthetic room impulse response (RT60 0.4–0.9 s);
  - zero-phase Butterworth band-pass 250 Hz–7 kHz;
  - pink noise at 20 dB SNR;
  - AAC at 64 kbit/s (ffmpeg).
  - The reverb smears attacks by about 6 ms, which is well inside the ±50 ms tolerance.
- **Storage:** features in `data/tabmodel_cache_aug/` (4.7 GB, float16 `.npy`, memory-mapped in training);
  built with `scripts/build_aug_cache.py --copies 2` and `--degraded` (about 15 min, 4 workers).

**E10 on the degraded validation copies** (baseline; clean in brackets):
- GuitarSet chunks: time pitch .5932 (.7672), time string/fret .4676 (.7465).
- GAPS validation: pitch .7173 (.8824), string agreement .7875 (.8639).
So phone / room conditions cost E10 .17–.28 F1.

**Training** (`scripts/train_mid_gaps_aug.sh`): fine-tune from E10 `best.pt` (`--init`, new optimizer).
- 6k steps, learning rate 1e-4 (E10 used 3e-4) with 300 warm-up steps and cosine decay, batch 16, `--gaps_share 0.5`.
- `--aug_copies 2 --aug_share 0.5`: half the crops use an augmented copy. The CQT-domain augmentations still apply on top.
- `--eval_degraded`: every 1k steps it also scores the degraded validation copies.
  `best.pt` maximises the mean of the clean and degraded selection scores.
- With augmentation off, training batches and losses are identical to before (seeded comparison with the previous `train.py`).

**Adoption rule** (fixed before the results):
- clean validation within about .005 of E10;
- a clear gain on degraded validation;
- then test once, clean and degraded, plus the demo tab for a listening check.

**Caveat:** there is no control run (E10 fine-tuned for 6k more steps on clean data only), so a small clean change
can't be pinned on augmentation versus the extra steps. Gains on the degraded sets are attributable to augmentation.

**Validation by step** (full tracks, in-training protocol; E10 in the first column):

| | E10 | 1k | 2k | 3k | 4k | 5k | 6k |
|---|---|---|---|---|---|---|---|
| clean GuitarSet string/fret | .813 | .807 | .809 | .806 | .811 | .813 | .812 |
| clean GAPS pitch | .882 | .883 | .881 | .879 | .881 | .884 | .883 |
| degraded GuitarSet string/fret | .563 | .607 | .597 | .604 | .629 | .620 | .625 |
| degraded GuitarSet pitch | .699 | .726 | .704 | .725 | .725 | .724 | .725 |
| degraded GAPS pitch | .717 | .741 | .714 | .723 | .725 | .732 | .732 |

- **Best checkpoint:** step 6000 → `best.pt` (selection score .7630).
- **Re-tuning:** thresholds re-tuned on clean validation; the optimum is unchanged (DEFAULT_PARAMS).
- **Validation chunks:**
  - clean: .7661 / .7449 (E10 .7672 / .7465);
  - degraded: .6210 / .5346 (E10 .5932 / .4676).
- **Note on the run:** an ablation launched during training crashed (a multiprocessing pool started from a stdin script) and slowed
  steps 3.3k–3.7k. Training itself was unaffected.

**Test (once):**

| Test set | Metric | E10 + D1 | **E12** |
|---|---|---|---|
| GuitarSet player 05 chunks, clean | time pitch / string-fret F1 | .8593 / .8283 | .8595 / .8265 |
| | tick pitch / string-fret F1 | .8696 / .8385 | .8690 / .8358 |
| GAPS test, clean (real YouTube recordings) | pitch F1 / string agreement | .8108 / .8403 | **.8147 / .8493** |
| GuitarSet test, degraded | time pitch / string-fret F1 | .7008 / .6092 | **.7295 / .6448** |
| GAPS test, degraded | pitch F1 / string agreement | .6462 / .7374 | **.6740 / .7573** |

**Which degradation hurts?** Pitch F1 with one held-out effect at a time (`scripts/build_aug_cache.py` recipe split up;
GAPS validation, and every 3rd GuitarSet validation track):

| Model | Set | clean | room (RT60 .4–.9 s) | band-pass 250 Hz–7 kHz | pink noise 20 dB | AAC 64k |
|---|---|---|---|---|---|---|
| E10 | GAPS val | .882 | .832 | .794 | .876 | .880 |
| E12 | GAPS val | .883 | .839 | .806 | .880 | .882 |
| E10 | GuitarSet val / 3 | .860 | .805 | .802 | .865 | .860 |
| E12 | GuitarSet val / 3 | .862 | .811 | .832 | .867 | .862 |

- **What hurts:** cutting the lows and room reverb. Noise at 20 dB and AAC at 64k barely matter.
- **Why the lows matter:** a 250 Hz high-pass removes the fundamentals of the four lowest open strings (82–196 Hz).
- **Training coverage:** the high-pass range was 60–300 Hz, median about 134 Hz, so cuts at 250 Hz and above were rare.
  The algorithmic reverb is shorter than RT60 0.4–0.9 s.

**Conclusion:**
- **Robustness:** E12 is clearly more robust to phone/room audio, +.03–.04 on degraded test.
- **Clean scores:** unchanged on GuitarSet (−.002 string/fret, +.000 pitch) and better on the real, varied GAPS recordings (+.004 pitch, +.009 agreement).
- **Adoption:** this meets the rule fixed beforehand, so E12 is the new default (`DEFAULT_CHECKPOINT`). E10's weights are kept.
- **No control run:** the GAPS clean gain could partly come from the 6k extra steps.

**Demo recording:** `out/my_video/my_video_J_aug.gp5` (E12, default settings) vs `my_video_H_dedup.gp5` (E10).
463 vs 471 notes, 6 vs 2 rests, 44 changed note lines.

**Next options:**
- Stronger low cut (high-pass up to about 400 Hz) and longer convolution reverbs in training. Real phone microphones do roll off
  the lows. To avoid tuning training to our own held-out recipe, that run would need a new held-out recipe, for example
  real recorded impulse responses.

### E13 — Adding a SynthTab fingerstyle subset (`runs/tabmodel_mid_gaps_synth`) — DONE, not adopted yet (mixed; awaiting the listening check)
**Goal:** improve dense fingerstyle chords. The user hears good melodies but weak multi-note passages on their recording,
and A1 agrees: GAPS validation misses 40% of notes when 4+ notes sound together, vs ~8% for single notes.
That needs audio with *correct* labels for dense polyphony. Self-training cannot provide it (the teacher is weakest exactly there),
so we use SynthTab: DadaGP tabs rendered with sampled guitars.

**Data** (`scripts/synthtab_fetch.py`, `guitab/synthtab.py`; budget about the size of the existing data):
- **Download:** SynthTab is 2 TB (acoustic 801 GB: 7 timbres × 4 zip parts of 23–34 GB). Its zips store FLAC uncompressed,
  so single songs are fetched with HTTP range requests; nothing else is downloaded.
- **Timbres:** the four fingerpicked acoustic ones, `martin_finger`, `taylor_finger`, `luthier_finger`, `gibson_thumb`
  (10,584 songs each, same songs).
- **Selection** (from the JAMS labels): 9,313 songs in standard tuning (399 h). 895 pass the fingerstyle filter:
  - 60–400 s long, < 15% silence;
  - mean polyphony 2.2–5;
  - 15–85% of attacks are chords;
  - 3–14 notes/s;
  - 20–65% of notes on strings 4–6;
  - frets ≤ 19.
- **Sampling:** random (seed 0), one version per song, ≤ 3 songs per artist. Result: 428 songs, 329 artists
  (classical, flamenco, acoustic pop / rock):
  - **synth_train:** 414 songs, 20.6 h; **synth_validation:** 14 songs, 0.7 h;
  - timbres round-robin (validation happens to use only martin / luthier);
  - `data/synthtab/selection.csv` is versioned.
- **Disk:** 5.3 GB audio + 1.1 GB label archive; features +0.7 GB (cache), 1 augmented copy +3.4 GB.
- **Labels:** per-string MIDI (the files the audio was rendered from): string, fret = pitch − open pitch, exact timing.
  Mean 2.74 strings sounding when any sounds (GAPS train 1.48).
- **Rendering latency (found and fixed):**
  - **Found:** E12 predictions on downloaded songs were systematically late vs the MIDI, by a different amount per timbre.
    Median over 12 training songs each: taylor 57.5 ms (beyond the ±50 ms tolerance; E12 pitch F1 ≈ .04–.26 on those songs),
    martin 24.1, gibson 21.4, luthier 16.7 ms. The spread within a timbre is about ±4 ms.
  - **Not a model bias:** E12's own signed bias on real audio is ~0 (GuitarSet validation −0.7 ms, GAPS validation +0.6 ms),
    so the delay is in the rendered samples.
  - **Fix:** labels are shifted by `LATENCY_MS` per timbre. After the shift, held-out validation songs line up at +0.4 ms.
  - **Other fixes:** songs with an unplayed string have no MIDI file for it (handled). The download retries after TLS drops
    from the office proxy (one drop stopped the first attempt at 214 songs).

**Training choice: mixed fine-tune from E12, not sequential pretraining.**
- **Too little data for a pretraining stage:** 20 h of synthetic audio is not a pretraining corpus; the SynthTab paper pretrains on thousands of hours.
- **A sequential run would forget the chords:** the real-only second stage would mostly overwrite the dense-chord knowledge, because
  real data has few such passages.
- **Mixing keeps both:** chord supervision throughout, while real audio keeps the sound realistic.
- **Starting point:** from E12, which keeps its phone/room robustness; about 3 h instead of 5+ h from scratch.

**Run** (`scripts/train_mid_gaps_synth.sh`):
- `--init` E12 `best.pt`, 8k steps, learning rate 1e-4, 300 warm-up steps, batch 16;
- crops: GuitarSet .35 / GAPS .35 / SynthTab .3 (`--synth_share 0.3`); SynthTab tab-loss weight 1 (exact labels);
- `--aug_copies 2 --aug_share 0.5` (SynthTab has 1 augmented copy);
- validation as in E12 (clean + degraded real audio decide `best.pt`), plus SynthTab validation, reported only.
- With `--synth_share 0`, training batches and losses are identical to before (seeded comparison, augmentation share 0 and 0.5).

**Baselines (E12):**
- training-protocol validation:
  - clean GuitarSet string/fret / pitch .8115 / .8342, GAPS pitch .8833;
  - degraded GuitarSet .6253 / .7249, degraded GAPS pitch .7320;
- SynthTab validation: string/fret .6998, pitch .7886.

**Adoption rule** (fixed before results):
- clean and degraded real validation not worse than E12 by more than about .005;
- a clear gain on dense polyphony: the A1 miss rate for 4+ simultaneous notes on GAPS validation (E10: .40);
- then test once, plus the demo tab.

**Validation by step** (full tracks; E12 in the first column):

| | E12 | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k |
|---|---|---|---|---|---|---|---|---|---|
| clean GuitarSet string/fret | .812 | .790 | .805 | .799 | .798 | .806 | .808 | .810 | .807 |
| clean GuitarSet pitch | .834 | .825 | .841 | .829 | .830 | .834 | .839 | .840 | .838 |
| clean GAPS pitch | .883 | .885 | .883 | .883 | .886 | .886 | .883 | .888 | .888 |
| degraded GuitarSet string/fret | .625 | .530 | .560 | .581 | .587 | .610 | .578 | .595 | .599 |
| degraded GAPS pitch | .732 | .728 | .721 | .715 | .737 | .738 | .726 | .740 | .740 |
| SynthTab string/fret | .700 | .890 | .896 | .881 | .906 | .911 | .907 | .911 | .909 |

- **Best checkpoint:** step 5000 → `best.pt` (selection .7600; 7k and 8k were close at .758).
- **Re-tuning on clean validation:** onset 0.6, frame 0.5 (default 0.6), legato 8, orphan off.
  - Tuned validation chunks: .7699 / .7440 (E12 .7661 / .7449). Degraded: .6163 / .5174 (E12 .6210 / .5346).
  - The test below used `DEFAULT_PARAMS` (frame 0.6), like E12's test; in earlier tunings, frame 0.5 vs 0.6 changed scores by < .001.

**Dense polyphony** (A1 miss rates; `poly>=4` = 4+ reference notes sounding at the attack):

| Set | Model | pitch P / R / F1 | miss, 1 note (long) | miss, 2–3 (long) | **miss, 4+ (long)** | **miss, 4+ (< 120 ms)** |
|---|---|---|---|---|---|---|
| GuitarSet validation | E12 | .815 / .828 / .822 | .104 | .183 | .154 | .335 |
| | E13 | .795 / .854 / .823 | .094 | .165 | **.122** | **.265** |
| GAPS validation | E12 | .898 / .859 / .878 | .078 | .114 | .401 | .533 |
| | E13 | .891 / .869 / .880 | .073 | .110 | **.345** | .533 |
| GAPS test | E12 | .874 / .761 / .813 | .137 | .177 | .394 | .460 |
| | E13 | .853 / .786 / .818 | .130 | .166 | **.319** | **.260** |

**Test (once):**

| Test set | Metric | E12 | E13 |
|---|---|---|---|
| GuitarSet player 05 chunks, clean | time pitch / string-fret F1 | .8595 / .8265 | .8566 / .8201 |
| GuitarSet test, degraded | time pitch / string-fret F1 | .7295 / .6448 | .7329 / .6296 |
| GAPS test, clean | pitch F1 / string agreement | .8147 / .8493 | **.8199 / .8786** |
| GAPS test, degraded | pitch F1 / string agreement | .6740 / .7573 | .6738 / .7483 |

**Reading:**
- **The goal is met: dense chords.** With 4+ notes sounding, misses drop by about a quarter to a half on every set
  (GAPS test, long notes .394 → .319; short notes .460 → .260). Recall rises everywhere: GuitarSet validation .828 → .854, GAPS test .761 → .786.
- **Cost: precision.** GuitarSet validation .815 → .795, GAPS test .874 → .853: more extra notes. Net F1 is about flat.
- **Strings:** GAPS string agreement jumps (+.029 on clean test), but GuitarSet string/fret is a little worse
  (clean −.006, degraded −.015).
  - **Likely cause:** SynthTab renders each string's MIDI through a sampled-guitar plugin (Ample Guitar via DawDreamer,
    `MIDI_to_Audio/render_acoustic.py`). Such plugins choose their own string position, so SynthTab string labels are tab-writer
    choices that are probably not audible. That suits GAPS (also score fingerings) and not GuitarSet (real players).
- **Phone/room robustness:** slightly worse on GuitarSet string/fret. SynthTab is clean studio audio with only 1 augmented copy.
- **Adoption rule:** dense-polyphony gain yes; "real validation within .005 of E12" no (degraded GuitarSet string/fret −.015 to −.017).
  So E13 is **not** made the default automatically. `DEFAULT_CHECKPOINT` stays E12.

**Demo recording:** `out/my_video/my_video_K_synth.gp5`: 544 notes vs 463 for E12 (`my_video_J_aug.gp5`), consistent with the higher recall.
The listening check decides whether the extra notes are the missing chord notes or false ones.

**From scratch with all data?** Not started. The two weaknesses come from properties of the SynthTab data (inaudible string labels,
clean audio), which a from-scratch run would inherit. Nothing here shows fine-tuning as the limit: E13 converged and its 5k–8k
validation is flat. A corrected recipe should be tested first (next options in the summary to the user).

### E14 — Time-sharp low band for bass attacks (`runs/tabmodel_mid_gaps_sharp`) — NO GAIN, code removed

**Motivation (user):** in fingerpicking, two notes struck together (thumb bass + finger) often lose the bass.
The user proposed detecting bass and melody separately. Measured on E12, validation:
- **Bass is the weak voice of a pinch:** lower note missed 29.6% (GuitarSet) / 25.4% (GAPS), upper note 20.7% / 13.6%.
- **Low E is the weak string:** string 6 missed 34% on GuitarSet (strings 2–4: 13–16%); fresh (not repeated) notes ≤ A2 missed 45%.
- **Not a decoding problem:** lowering the onset / frame thresholds of strings 4–6 recovers bass notes (pinch-bass miss
  .256 → .167) but adds more false notes; F1 drops at every setting.
- **Suspected cause, the input:** at 36 bins/oct a CQT bin's window is ~51 periods (0.62 s at E2). With a synthetic tone,
  an E2 attack starts rising 25 frames early. Fresh bass attacks gain only +2.5 dB in their own bin (45% under 2 dB); with
  4× shorter filters, +6.7 dB.

So separate output heads were not the target (the tab head is already per string). The idea became a separate input view.

**Setup:**
- The 5 octaves from E1 recomputed with `filter_scale` 0.25 (180 bins, `guitab/audio.py`, cache `data/tabmodel_cache_sharp`, 4.5 GB).
- Fed as a second 6-channel harmonic stack; the new first-conv weights start at zero (identical outputs to E12 at step 0).
- Fine-tuned from E12 with E12's recipe (`scripts/train_mid_gaps_sharp.sh`, 6k steps). Best checkpoint: step 6000.

**Validation, bass miss rates** (lower is better; "zeroed" = E14 with the sharp input set to 0):

| Set | Notes | E12 | E14 | E14 zeroed |
|---|---|---|---|---|
| GuitarSet val | lower note of a pinch | .296 | .289 | .314 |
| | fresh ≤ A2 | .454 | .451 | .464 |
| | string 6 | .343 | .341 | .359 |
| | above E3 (control) | .141 | .140 | .146 |
| GAPS val | lower note of a pinch | .254 | .250 | .264 |
| | fresh ≤ A2 | .144 | .142 | .162 |
| | string 6 | .152 | .149 | .173 |

**Test (once):**

| Test set | Metric | E12 | E14 |
|---|---|---|---|
| GuitarSet player 05 chunks, clean | time pitch / string-fret F1 | .8595 / .8265 | .8599 / .8253 |
| GuitarSet test, degraded | time pitch / string-fret F1 | .7295 / .6448 | .7279 / .6423 |
| GAPS test, clean | pitch F1 / string agreement | .8147 / .8493 | .8132 / .8529 |
| GAPS test, degraded | pitch F1 / string agreement | .6740 / .7573 | .6758 / .7590 |

**Reading:**
- **No gain.** Bass misses move by under a point, and test scores tie. Demo (`my_video_L_sharp.gp5`): 147 notes on strings 4–6 vs 144 for E12.
- **The model uses the band:** zeroing it costs 1.5–2.5 points of bass miss rate. But it used the band in place of what it
  already inferred (harmonics, context), not on top of it.
- **Why the missed bass notes stay missed:** in the sharp band itself, missed fresh bass notes have a median attack contrast
  of +3.6 dB on GuitarSet (40% under 2 dB) vs +10.7 dB for detected ones; on GAPS +5.7 dB (34% under 2 dB) vs +12.8 dB.
  They are also ~7 dB quieter. These are soft thumb strokes, attacks masked by ringing low strings, or (GuitarSet)
  notes the hexaphonic pickup labels but the microphone barely hears.
  The information is weak in the audio, not hidden by the representation, so a from-scratch run with the band is not expected to help.

Code removed (reverts commit c429a77; recoverable from it); run, feature cache and demo deleted.
