# Guitab: experiment log

> Entries before 2026-10-06 name scripts by their old `tools/` paths; see the table in the last entry for the current `guitab/` and `scripts/` names.

Each entry gives what was run, the exact command or script, and the result, so it can be repeated. Add new entries at the bottom. All paths are relative to the repository root.

## Evaluation protocol (shared by all entries)

- **Data split:** GuitarSet split by player: 00–03 train, 04 validation, 05 test (60 tracks per player). Settings are chosen on validation only; test is reported once per finished model.
- **Main metric:** the Audio2Tab repo's `compute_note_detection_metrics` on the repo's fixed schedule of 15 s chunks (146 test chunks, 60 tracks; frozen in `data/eval_chunks.json`; the metric now lives in `tools/note_metrics.py`). It reports note F1 with onset matching: `time_*`: onsets within ±50 ms; `tick_*`: onsets within ±120 ticks (a 32nd note); `*_pitch`: the pitch must match; `*_string_fret`: the exact string and fret must match.
- **Non-token models:** their notes are written as tokens at the reference tempo with exact waits (`tools/eval_tabmodel.py`, `chunk_text`), so the same metric applies.
- **Benchmark-style metrics:** `tools/sweep_tabmodel.py` also gives the TabCNN definitions (frame-level tablature and multi-pitch P/R/F) and note-level F1 on full tracks. Caveat: published papers use 6-fold leave-one-player-out cross-validation. We use a single split, so our numbers are not directly comparable.
- **Environment:** macOS (M3 Pro, 36 GB), MPS.
  - Main venv: `.venv` (`requirements.txt`, torch 2.10). Until 2026-10-06 it was `Audio2Tab/.venv`, which also had transformers 5.3 for the Whisper runs.
  - Basic Pitch needs its own venv, `.venv_bp`.
- **Memory caps for training:** `PYTORCH_MPS_HIGH_WATERMARK_RATIO` / `LOW` = 0.35 / 0.25 for the tab model, and 0.7 / 0.5 for Whisper. Launch long jobs with `nohup`.

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
- **Adopted:** Beat This!, weights in `data/models/beat_this-final0.ckpt`.

### E5 — Post-processing for Whisper output (`tools/tune_post.py`, tuned on validation)
- **Basic Pitch chord fusion** (`tools/fuse.py`) plus the **Viterbi fingering** step (`tools/fingering.py`): test results 0.233 / 0.150 / 0.415 / 0.284.
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
- **Input:** log-CQT, 36 bins per octave from E1, 288 bins, 22.05 kHz, hop 256 (11.6 ms). The harmonic stack (h = 0.5, 1, 2, 3, 4, 5) is built inside the model.
- **Model** (`tools/tabmodel.py`, 23.8M parameters): 6 conv blocks (64 / 128 / 256 channels, frequency pooled 144 → 12); transformer encoder (d = 512, 6 layers, 8 heads); heads: tab (6 strings × 21 classes), onset (6), and auxiliary pitch (48).
- **Training** (`runs/train_tabmodel_mid.sh` → `tools/train_tabmodel.py`):
  - 12k steps, batch 16, 5 s crops; AdamW at 3e-4 with 1k warm-up steps and cosine decay.
  - Augmentation on the CQT: pitch roll 0–2 semitones, level, tilt, noise floor, time masks.
  - About 8 GB of GPU memory, 1.1 s per step, about 3.7 h in total.
  - Training data: players 00–03 plus the shifted copies, 6.1 h of audio (only about 2 h is real audio).
- **Validation, full-track string/fret F1 by step:**

  | Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | F1 | .539 | .667 | .714 | .737 | .723 | .749 | .734 | .755 | .753 | **.757** | .751 | .752 |

  Best is step 10000 → `best.pt`. Training tab loss reached 0.08 while validation flattened, so the model is limited by data, not by size.
- **Note extraction tuned on validation** (`eval_tabmodel.py --tune`): onset 0.6, frame 0.6, legato 8, orphan off.
- **Test chunks:** time pitch 0.836, time string/fret 0.811, tick pitch 0.845, tick string/fret 0.818. With the fingering step on top: 0.808 (`auto`) / 0.809 (`strong`), so it slightly hurts.

### E9 — Confidence / fingering sweep (`tools/sweep_tabmodel.py`)
- **Command:** `python tools/sweep_tabmodel.py --checkpoint runs/tabmodel_mid/best.pt --out runs/sweep_tabmodel_mid.json` (about 25 min). The full table, with notes per chunk, is in the JSON.
- **Results** for time ±50 ms, as precision / recall / F1, model strings, "orphan" off:

  | Onset threshold | Val pitch | Val string/fret | Test string/fret | Notes per val chunk |
  |---|---|---|---|---|
  | 0.2 | .642 / .775 / .702 | .625 / .756 / .685 | .700 / .829 / .759 | 93.6 |
  | 0.3 | .688 / .761 / .723 | .670 / .742 / .705 | .744 / .821 / .781 | 85.7 |
  | 0.4 | .723 / .750 / .736 | .705 / .731 / .718 | .777 / .814 / .795 | 80.3 |
  | 0.5 | .751 / .736 / .744 | .731 / .717 / .724 | .807 / .804 / .806 | 75.9 |
  | **0.6** (default) | .785 / .721 / .752 | .764 / .702 / .731 | .830 / .792 / .811 | 71.1 |

- **Findings:**
  - Lowering the threshold from 0.6 to 0.2 raises recall by only about 0.05 but costs about 0.14 precision. Most missed notes are not low-confidence notes; the model doesn't detect them at all.
  - Orphan notes (onset-less) lower F1 by 0.03–0.05 at every threshold.
  - Fingering: `auto` costs about 0.004; `low` (forced first position) halves string/fret F1 on GuitarSet (.81 → .36), because GuitarSet players use the whole neck. So `low` is a playing-style prior, and only a player's own data can say if it's right.
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
**Choice of data (2026-10-05):** the project's focus is fingerstyle guitar. I reviewed the open datasets:
- **[GAPS](https://zenodo.org/records/13962272) — chosen.** Classical nylon-string guitar from 200+ players in varied rooms and on varied microphones, all fingerpicked. Note-level MIDI is aligned to YouTube audio, and the MusicXML scores usually carry a tab staff (`<string>`/`<fret>`). Licence CC BY-NC-SA, which is fine for this personal project.
- **Not chosen now:**
  - Guitar-TECHS and EGDB: electric guitar, mostly techniques, scales and rock playing.
  - IDMT-SMT-Guitar: mostly isolated notes and licks.
  - SynthTab: synthetic, about 2 TB. Kept as a later option for pretraining.
  - GOAT: restricted access.

**GAPS preparation (repeat in this order):**
1. Download the zip from Zenodo into `data/gaps/`, then run `python tools/gaps_common.py`.
   - GAPS ids are case-sensitive (`GM1wc` ≠ `gM1wc`). A plain unzip on macOS silently overwrote 278 files.
   - `gaps_common.extract()` writes every file under the case-safe name `safe(h) = h_md5(h)[:6]` in `data/gaps/safe/`.
2. Run `python tools/gaps_download.py --workers 3` (yt-dlp, 22.05 kHz mono), twice to retry 403 errors.
   - Result: 296 of 301 recordings downloaded; 5 are unavailable or blocked.
   - The MIDI timeline matches the full YouTube audio (offset 0.02 s, checked on `mvswc`).
3. Run `python tools/gaps_labels.py --workers 4`. String/fret per MIDI note comes from the score's tab staff:
   - Score and MIDI chord sequences are aligned by note order (an LCS-style dynamic programme on pitch overlap), then string/fret is copied by pitch.
   - Interpolating the syncpoints was unreliable. They count bars with repeats unfolded, and music21 cannot expand some scores' repeats.
   - Result: median match rate 0.57 (share of MIDI notes given a string).
   - 63 pieces use non-standard tuning (mostly drop-D) and are excluded for now. 3 pieces have MIDI that doesn't match the score.
4. Selection, in `tabdata.gaps_selection`: standard tuning, content similarity ≥ 0.9, match rate ≥ 0.5, audio present.
   - Result: 156 pieces. 123 train (6.53 h), 14 validation (0.72 h, every 10th train piece sorted by id), and 19 test (1.09 h, the official GAPS test split).
5. Run `python tools/tabdata.py --workers 4` to cache CQT features for the GAPS pieces.

**Label handling:**
- Notes without a string label (37% of active frames in GAPS train) are masked: the tab head ignores strings not otherwise labelled active during such notes; onsets are ignored around their attack; the pitch head ignores those frames.
- GAPS string labels are score fingerings, not necessarily what was played, so they get tab-loss weight 0.5. Pitch and onset supervision from the aligned MIDI is used at full weight.

**Baseline (E8 model `tabmodel_mid/best.pt`, zero-shot), with `python tools/eval_gaps.py`:**
- GAPS validation: pitch F1 .629, string agreement .720.
- GAPS test: pitch F1 .594, string agreement .690.
- For comparison, GuitarSet test pitch is .836, so the model suffers a large domain gap on fingerstyle recordings.

**Training** (`runs/train_tabmodel_mid_gaps.sh`):
- Same mid architecture and augmentations as E8, trained from scratch.
- 16k steps, batch 16, learning rate 3e-4 with cosine decay; 50% of crops from GAPS (`--gaps_share 0.5`), `--gaps_tab_weight 0.5`.
- About 8 GB of GPU memory, 1.14 s per step, about 5 h in total.
- Validation every 1k steps on both GuitarSet player 04 (string/fret, pitch) and GAPS validation (pitch, string agreement). `best.pt` maximises the mean of GuitarSet string/fret F1 and GAPS pitch F1.
- The E8 weights in `runs/tabmodel_mid/` are kept unchanged.

**Validation by step** (GuitarSet string/fret F1, full tracks / GAPS pitch F1 / GAPS string agreement):

| Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k | 14k | 15k | 16k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GuitarSet string/fret | .473 | .699 | .739 | .753 | .776 | .772 | .760 | .777 | .792 | .794 | .804 | .808 | .810 | .807 | .809 | .807 |
| GAPS pitch | .576 | .740 | .755 | .790 | .779 | .821 | .822 | .833 | .832 | .840 | .840 | .840 | .843 | .845 | .845 | .846 |
| GAPS string agreement | .775 | .795 | .874 | .899 | .908 | .894 | .885 | .888 | .907 | .899 | .890 | .903 | .900 | .899 | .903 | .904 |

- **Best:** step 15000 → `best.pt` (selection score .827).
- **Comparison with E8:** from step 2 onward the GuitarSet scores beat the GuitarSet-only run, so the extra real fingerstyle audio helps GuitarSet as well. The run is converged by 13k–16k steps.
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

**Conclusion:** adding GAPS: raises fingerstyle pitch F1 on unseen players, rooms and microphones by +.19; improves GuitarSet by +.015 as well.

E10 is the new default model.

**My recording:** 494 notes with the default settings, 587 with `--onset 0.4`. The model now puts the opening C-major shape in first position by itself (B string fret 1, open G), where E8 needed `--fingering low`.

**Next options:**
- Drop-D and other non-standard tunings: 63 GAPS pieces, which needs a tuning-aware label scheme.
- SynthTab acoustic pretraining.
- The large model, now that there is more data.
- A labelled clip of my own playing, to tune the settings for my guitar.

### Clean-up — 2026-10-06
The Whisper / Audio2Tab approach (E1–E7) is outperformed by the CQT tab model, so all its artifacts were deleted, freeing about 84 GB. The results above remain the record of it.

**Deleted:**
- **Runs:** `runs/mix_v2`, `mix_v3`, `mix_v4`, `classical_v1`; benchmark, overfit and smoke runs; their logs, predictions and evaluation JSONs.
- **Synthetic data:** `data/procedural*`, `classical*`, `soundfonts`, `test_render` and the `mix_v*.csv` manifests.
- **Whisper-only scripts:** `gen_tabs.py`, `render_tabs.py`, `render_gp5.py`, `final_eval.py`, `tempo_eval.py`, `tune_post.py`, `fuse.py`, `analyze_preds.py`, `diagnose.py`.
- **Whisper outputs on my recording.**
- **Tab-model snapshots:** the non-best `last.pt` and `step*.pt` checkpoints.

**Changed:** `tools/transcribe.py` no longer has the `whisper` engine or Basic Pitch chord fusion. The default is `--engine tabmodel` with `runs/tabmodel_mid_gaps/best.pt`. Evaluation reads the fixed GuitarSet chunks from `data/eval_chunks.json`, rebuilt from GuitarSet alone. Before deletion I verified them identical to the old reference files: 146 test and 147 validation chunks, same texts, paths and starts. The E10 test result reproduced exactly (.8505 / .8258), and so did the transcriptions of my recording.

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
- The new `.venv` was installed from `requirements.in` (direct dependencies), using the old freeze as constraints, then locked to `requirements.txt` (61 packages). Every package is a subset of the old env at the same version.

**Verified in the new env:**
- GuitarSet test .8505 / .8258.
- GAPS test .787 / .864.
- Transcriptions of my recording byte-identical for both engines.
- A short training run with GAPS works, and all scripts import.

Audio2Tab has no licence file, so the copied code kept its attribution headers; it was replaced by our own code before the public release ("Release clean-up" below).

### Simplification — 2026-10-06
- Removed the Basic Pitch engine (E7 baseline, outperformed by the tab model everywhere): `tools/run_basic_pitch.py`, the `--engine` option of `transcribe.py`, the `.venv_bp` environment and the `data/basic_pitch` cache. E7 numbers remain above as the record.
- Requirements reduced to one file, `requirements.txt`: exact pins, with the direct dependencies listed in its header. `requirements.in` and the old Audio2Tab freeze were dropped.
- The project is now a git repository. Data, weights, outputs and environments are excluded (`.gitignore`); only code, docs and the frozen `data/eval_chunks.json` are tracked.

### Refactor into a package — 2026-10-06
The flat `tools/` folder was split into a library package, `guitab/`, and command-line entry points, `scripts/`. Results are unchanged; verification is below.

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
| `runs/train_tabmodel_mid*.sh` | `experiments/train_mid.sh`, `experiments/train_mid_gaps.sh` |

**Other changes:**
- **Shared constants:** `OPEN_PITCH`, `STRINGS` and the fret classes now live in `guitab/guitar.py`, instead of being defined in four places. All paths are in `guitab/paths.py`.
- **Relative paths:** `data/guitarset.csv`, `data/guitarset_shift.csv` and `data/eval_chunks.json` now store repo-relative audio paths instead of absolute `/Users/...` ones. The track lists dropped their unused columns.
- **Bug fix:** `scripts/gaps_labels.py --only` used to overwrite `labels_summary.csv` with only the requested pieces, which would silently shrink the GAPS selection. It now updates just those rows.

**Verification** (old code snapshot vs new code, same data and weights):
- Every evaluation output is identical: `evaluate.py` on validation and test for both models, all four metrics, with and without fingering; `evaluate_gaps.py` on GAPS test and validation.
- Transcriptions of the demo recording with 5 option sets are byte-identical (`.gp5` and tokens): default; `--fingering low`; `--fingering auto --orphan`; `--onset 0.4 --tempo 90 --grid 120`; the E8 checkpoint.
- Function-level checks all pass: `tracks()` (996 tracks) is identical, and so are the `load_split` arrays for 3 splits; CQT features and frame labels recomputed by the new code equal the cached files written by the old code (4 GuitarSet/shifted tracks, 3 GAPS pieces); 5 seeded training batches with augmentation, plus the model outputs and loss values, are identical; GAPS labels for 6 pieces and their summary rows are identical; case-safe extraction (1103 files) and the download list are identical.
- The regenerated track lists match the old ones (same tracks, order, audio files and splits).
- Training, evaluation, checkpoint saving and resume work through the new CLI.
- The full `sweep.py` output on E8 is identical to `runs/sweep_tabmodel_mid.json`.

### A1 — Error analysis of E10 (`scripts/analyze_errors.py`) — 2026-10-06
- **Command:** `python scripts/analyze_errors.py --checkpoint runs/tabmodel_mid_gaps/best.pt --out runs/errors_tabmodel_mid_gaps.json` (about 5 min). Validation and train subsets only; no test data.
- **Protocol:** full tracks, default note extraction, greedy matching (onset ±50 ms), notes pooled over tracks (so numbers differ slightly from the per-track means logged in training). Each miss / false note gets the first category that applies.

| | GuitarSet val | GuitarSet train (1/4) | GAPS val | GAPS train (1/9) |
|---|---|---|---|---|
| pitch P / R / F1 | .798 / .832 / .815 | .862 / .923 / .891 | .819 / .858 / .838 | .823 / .833 / .828 |
| string/fret F1 | .798 | .891 | – | – |
| string correct, given correct pitch | .960 | .998 | .859 (vs score fingering) | .916 |
| frame multi-pitch F1 | .878 | .937 | – | – |
| onset error median / share ≤ 25 ms | 4.8 ms / .97 | 4.9 ms / .98 | 6.0 ms / .77 | 4.3 ms / .97 |

**Missed notes** (share of misses): GuitarSet val — not heard at all .31, found 50–150 ms off .24, re-attack of a ringing same pitch merged .13, semitone/tone confusion .10, weak frame evidence .08, chord collision .07, octave/harmonic .05, heard but onset < 0.6 only .025. GAPS val — not heard at all .41, weak evidence .19, heard but onset < 0.6 .09, chord collision .09, others ≤ .07. Miss rate by context (GuitarSet val): notes < 120 ms .25–.36 vs .10–.18 for longer notes. GAPS val: .40 when 4+ notes sound.

**False notes** (share of false notes): a new attack of a pitch that is still ringing in the reference is the largest class, .33 on GuitarSet and .58 on GAPS. Almost all of them are real onset peaks ≥ 0.6, a median 0.29 s (GAPS 0.44 s) after the reference attack, not legato splits. Then timing 50–150 ms (.21 / .09), octave/harmonic (.15 / .11), semitone (.12 / .10).

**Worst GuitarSet val tracks:** `04_BN3-154-E_comp` (.22, fast bossa comping; matches spread over ±300 ms), `04_Jazz1-200-B_comp` (.31); every other track is ≥ .67. Comping .79 vs solo .86 pitch F1.

**Findings:**
- **Strings are nearly solved on GuitarSet:** given the right pitch, the string is right 96% of the time, so a perfect fingering step could add at most about .017 string/fret F1. The errors are pitch and onset errors.
- **GuitarSet is data-limited:** train .891 vs validation .815 (an unseen player).
- **GAPS is underfitted:** train .828 vs validation .838, with no generalisation gap. A larger model or longer training should help here; label noise (alignment, unmatched notes) also caps it.
- **Thresholds won't recover the misses:** only about 2.5% (GuitarSet) / 9% (GAPS) of misses have frame evidence and a weak onset, consistent with E9.
- **Note decoding loses information:** frame multi-pitch F1 (.878) is well above note pitch F1 (.815). Re-attacks (merged or spurious) and short notes are the weak points of the onset head plus the hand-written extraction.
- **Even on training tracks, GuitarSet note F1 is only .89:** 30% of those misses are 50–150 ms timing differences, so part of the ceiling is annotation noise rather than the model.

### D1 — Note extraction: drop same-pitch duplicates on two strings — 2026-10-06, ADOPTED
Post-processing only, no retraining; tested on the E10 predictions, chosen on validation, then test once.
- **Problem (from A1):** the model sometimes starts the same pitch on two strings at once (unsure which string), which gives one true note plus one false note. This is frequent on GAPS: 924 known validation notes had such a pair.
- **Rule** (`dedup` in `DEFAULT_PARAMS`, `guitab/inference.py`): if the same pitch starts on two strings within 4 frames (46 ms), keep only the note whose fret posterior, averaged over the note, is higher. `dedup: 0` turns it off and reproduces the previous results exactly (checked: E10 validation chunks .7603 / .7439).
- **Choices on validation:** window 2, 4, 6 or 8 frames: all within ±.0003 → 4; which note to keep: fret posterior (GuitarSet string/fret .8127) beats attack confidence (.8101), max(onset, active) (.8105) and lowest fret (.8043); duration ties (.8129) → fret posterior, which also gave the best GAPS string agreement; re-tuned thresholds: unchanged (onset 0.6, frame 0.6, legato 8, orphan off).
- **Rejected:** requiring an energy rise at the pitch for a re-attack of a still-ringing pitch. It gave +.004 GAPS pitch but cost .004–.015 on GuitarSet at every threshold. Real re-plucks and false ones have the same onset confidence (median .96 vs .95), so this needs the model (release head, E11), not a rule.
- **Validation:** GuitarSet chunks pitch .7603 → .7672, string/fret .7439 → .7465; GuitarSet full tracks (training eval) string/fret .8087 → .8127, pitch .8261 → .8346; GAPS validation pitch .8454 → .8824.
- **Test (once):**

  | Test set | Metric | E8 before → after | E10 before → after |
  |---|---|---|---|
  | GuitarSet player 05 chunks | time pitch F1 | .836 → .846 | .851 → **.859** |
  | | time string/fret F1 | .811 → .812 | .826 → **.828** |
  | | tick pitch / string-fret F1 | .845 / .818 → .854 / .820 | .861 / .836 → **.870 / .839** |
  | GAPS test | pitch F1 | .594 → .622 | .787 → **.811** |
  | | string agreement | .690 → .644 | .864 → .840 |

- **About the GAPS string agreement drop:** this is not a regression. Agreement is measured on pitch-correct notes, and a duplicate pair is two guesses of the string: in 898 of the 924 validation pairs one of the two was on the score's string. After dedup the kept note is on the score string in 62% of those cases, while pitch hits are unchanged (8696 → 8695). So .840 is the honest figure, and the earlier .864 was inflated by duplicates.
- **My recording:** one duplicate removed (472 → 471 notes).
- **Note:** training validation (`evaluate_guitarset` / `evaluate_gaps`) uses `DEFAULT_PARAMS`, so from E11 on the logged validation numbers include dedup; compare with the "after" figures above, not with E10's training log.

### E11 — Precise attack targets + release head (`runs/tabmodel_mid_gaps_reg`) — ABANDONED, code removed
**Goal:** attack the error classes from A1 that a rule could not fix (D1): spurious and merged re-attacks of a ringing pitch, attacks 50–150 ms off, short notes. The backbone, data, sampling, augmentation, steps and seed are the same as E10. The change is only in what the heads learn and how notes are built:
- **Attack head:** the target is a triangle, `max(0, 1 − |t − attack| / 4 frames)`, around the exact attack time (Kong et al. 2021, high-resolution piano transcription), learned with plain BCE. E10 used 1 on the attack frame, 0.5 on its ±1 neighbours and positive weight 5.
- **New release head** (6 outputs, about 3k parameters): the same triangle around each note's end, per string. GAPS releases are ignored, because GAPS MIDI durations come from the score's note values, not from when the string stops. GuitarSet ends come from the hexaphonic pickup.
- **Unknown-string GAPS notes:** attack targets are ignored ±4 frames around their attack on strings not otherwise targeted.
- **Note building** (`guitab/inference.py`, used only when a checkpoint has a release head): attack time refined within the frame by a parabola through the onset peak (`subframe`); a note also ends at a release peak ≥ `offset`.
- **Old checkpoints are unaffected** (no release head → same path). Checked after the change: 3 seeded training batches with the model outputs and losses are identical; E10 validation chunks are identical (.7672 / .7465), and so is GAPS validation (.8824 / .8639); the transcription of my recording is byte-identical.
- **Command:** `nohup experiments/train_mid_gaps_reg.sh > runs/tabmodel_mid_gaps_reg.log 2>&1 &`, with 16k steps, batch 16 and `--gaps_share 0.5 --gaps_tab_weight 0.5 --onset_target regression`. Validation during training uses `{"onset": 0.4, "subframe": true}`, since triangle targets peak below 1; thresholds are tuned on validation after training (`evaluate.py --tune`, which also searches `offset` and `subframe`).
- **Comparison point:** E10 + D1 on the same in-training protocol: GuitarSet validation string/fret .8127, pitch .8346; GAPS validation pitch .8824, string agreement .8639. The final comparison uses tuned settings for both models.
- **Decoding bug, fixed at step 4000:**
  - **Bug:** a note ended at a release peak while the string still looked active, and the legato rule then started a new note there (fake notes).
  - **Cost:** on the step-3000 checkpoint (GuitarSet validation, onset 0.4) it cost .04: string/fret .690 with releases vs .733 without them.
  - **Fix:** a legato continuation now happens only after a fret change, never after a release.
  - **Result:** releases are now neutral for attack F1 (.7322–.7337 for release thresholds 0.5–0.7 vs .7331 without them). That is expected: they change note lengths, not attacks.
  - **Old checkpoints:** unaffected (E10 validation chunks still .7672 / .7465).
  - **The run:** stopped right after the step-4000 checkpoint and resumed from it with the fixed code, so validation scores up to step 4000 in the log use the buggy decoding.
  - **Side effect of the resume:** data sampling is reseeded at the resume step (`rng.seed(step)`), so the batches after 4000 differ from an uninterrupted run.

**Validation by step** (in-training protocol, onset 0.4 + subframe + dedup; steps 1k–4k used the buggy release decoding):

| Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GuitarSet string/fret | .390 | .634 | .690 | .710 | .785 | .797 | .792 | .804 | .811 | .810 | .816 | **.818** | .818 |
| GAPS pitch | .379 | .688 | .673 | .793 | .817 | .838 | .866 | .849 | .864 | .874 | .862 | **.871** | .869 |
| GAPS string agreement | .691 | .739 | .802 | .830 | .832 | .819 | .831 | .841 | .850 | .855 | .861 | .853 | .862 |

- **Stopped after the step-13000 eval**. By then the learning rate was 4e-5 on its cosine decay and the selection score had flattened (.8444 at 12k, .8436 at 13k). `best.pt` = step 12000. `last.pt` is kept, so the run can be resumed to 16k with `experiments/train_mid_gaps_reg.sh`.
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
- On attack-based note F1, E11 is roughly level with E10: +.002–.003 on GuitarSet test, equal on GAPS test pitch, −.013 on GAPS validation pitch. That is within run-to-run noise, so the new targets did not fix the missed and re-attacked notes found in A1.
- E11 also had 3k fewer training steps; E10's best was at step 15000.
- The release head's real effect is on note lengths (rests, let-ring), which none of our metrics measure. Next step: add an attack + release F1 (mir_eval style: release within max(50 ms, 20% of the note length)) and compare E10 and E11 on it, together with a listening check on the demo, before choosing the default model.
- `DEFAULT_CHECKPOINT` stays E10 (`runs/tabmodel_mid_gaps/best.pt`). E11's tuned settings are the ones listed above; pass them with `--params` (or via `transcribe.py --onset 0.4`, plus the release settings, once that is wired).
- **Listening check:** my recording transcribed by E11 with `--params '{"onset": 0.4, "frame": 0.5, "offset": 0.7, "subframe": true}'` vs E10 + D1. `transcribe.py --params` was added for this; without it, the output is unchanged.
- **Abandoned (2026-10-07):** in the listening comparison of the two tabs E10 was preferred. The scores were level, and the other options for this approach are small: the last 3k steps, putting the positive weight back on the attack target, or tweaking the release decoding. Each would move results by about ±.005, within noise. The release head also learns only from GuitarSet's ~2 h of real audio (GAPS note lengths come from the score).
  - **Removed:** the E11 code (regression targets, release head, `--onset_target`, the release decoding and tuning grid, `experiments/train_mid_gaps_reg.sh`), `runs/tabmodel_mid_gaps_reg/` (546 MB) and the E11 tab.
  - **Kept:** D1 (dedup), `scripts/analyze_errors.py` and the generic `transcribe.py --params` option.
  - **Code at the time of the run:** commits `335a11f` and `37fe2b9`; it can be restored from git if needed.
  - **Checked after the removal:** the code under `guitab/` and `scripts/` (except `transcribe.py`) is byte-identical to the D1 commit; E10 validation chunks .7672 / .7465; GAPS validation .8824 / .8639; the transcription of my recording is byte-identical.

### E12 — Audio augmentation for phone / room recordings (`runs/tabmodel_mid_gaps_aug`) — DONE, new default model
**Goal:** make the model robust to recordings like mine (phone / camera audio in a room). The test sets are cleaner recordings, so robustness is measured on held-out degraded copies, built with different effects from the training ones.

**Effects** (`guitab/augment.py`). Every effect keeps timing (0 ms shift measured with onset-envelope cross-correlation) and length, so the original labels are reused.
- **Training copies** (`train_chain`, pedalboard 0.9.25): 2 copies per training track (720 GuitarSet incl. shifted copies, 123 GAPS), each with its own random chain, seeded by name + copy and listed in `data/tabmodel_cache_aug/chains.tsv`: algorithmic reverb (p .5, room size .1–.7, wet .05–.35); high-pass 60–300 Hz + low-pass 3.5–10.5 kHz (p .8); 1–3 EQ peaks of ±8 dB between 150 Hz and 6 kHz (p .7); compressor (p .6); white / pink / brown noise at 15–40 dB SNR (p .5); MP3 at VBR quality 2–9 (p .5). Resulting mix over the 1,686 copies: band-limiting 79%, compression 62%, MP3 52%, reverb 49%, noise 48%; 7 copies untouched.
- **Held-out degradation** (`degrade`), used only for evaluation, on the validation and test splits (153 tracks): convolution with a synthetic room impulse response (RT60 0.4–0.9 s); zero-phase Butterworth band-pass 250 Hz–7 kHz; pink noise at 20 dB SNR; AAC at 64 kbit/s (ffmpeg). The reverb smears attacks by about 6 ms, which is well inside the ±50 ms tolerance.
- **Storage:** features in `data/tabmodel_cache_aug/` (4.7 GB, float16 `.npy`, memory-mapped in training); built with `scripts/build_aug_cache.py --copies 2` and `--degraded` (about 15 min, 4 workers).

**E10 on the degraded validation copies** (baseline; clean in brackets):
- GuitarSet chunks: time pitch .5932 (.7672), time string/fret .4676 (.7465).
- GAPS validation: pitch .7173 (.8824), string agreement .7875 (.8639). So phone / room conditions cost E10 .17–.28 F1.

**Training** (`experiments/train_mid_gaps_aug.sh`): fine-tune from E10 `best.pt` (`--init`, new optimizer).
- 6k steps, learning rate 1e-4 (E10 used 3e-4) with 300 warm-up steps and cosine decay, batch 16, `--gaps_share 0.5`.
- `--aug_copies 2 --aug_share 0.5`: half the crops use an augmented copy. The CQT-domain augmentations still apply on top.
- `--eval_degraded`: every 1k steps it also scores the degraded validation copies. `best.pt` maximises the mean of the clean and degraded selection scores.
- With augmentation off, training batches and losses are identical to before (seeded comparison with the previous `train.py`).

**Adoption rule** (fixed before the results): clean validation within about .005 of E10; a clear gain on degraded validation; then test once, clean and degraded, plus a listening check on my recording.

**Caveat:** there is no control run (E10 fine-tuned for 6k more steps on clean data only), so a small clean change can't be pinned on augmentation versus the extra steps. Gains on the degraded sets are attributable to augmentation.

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
- **Validation chunks:** clean: .7661 / .7449 (E10 .7672 / .7465); degraded: .6210 / .5346 (E10 .5932 / .4676).
- **Note on the run:** an ablation launched during training crashed (a multiprocessing pool started from a stdin script) and slowed steps 3.3k–3.7k. Training itself was unaffected.

**Test (once):**

| Test set | Metric | E10 + D1 | **E12** |
|---|---|---|---|
| GuitarSet player 05 chunks, clean | time pitch / string-fret F1 | .8593 / .8283 | .8595 / .8265 |
| | tick pitch / string-fret F1 | .8696 / .8385 | .8690 / .8358 |
| GAPS test, clean (real YouTube recordings) | pitch F1 / string agreement | .8108 / .8403 | **.8147 / .8493** |
| GuitarSet test, degraded | time pitch / string-fret F1 | .7008 / .6092 | **.7295 / .6448** |
| GAPS test, degraded | pitch F1 / string agreement | .6462 / .7374 | **.6740 / .7573** |

**Which degradation hurts?** Pitch F1 with one held-out effect at a time (`scripts/build_aug_cache.py` recipe split up; GAPS validation, and every 3rd GuitarSet validation track):

| Model | Set | clean | room (RT60 .4–.9 s) | band-pass 250 Hz–7 kHz | pink noise 20 dB | AAC 64k |
|---|---|---|---|---|---|---|
| E10 | GAPS val | .882 | .832 | .794 | .876 | .880 |
| E12 | GAPS val | .883 | .839 | .806 | .880 | .882 |
| E10 | GuitarSet val / 3 | .860 | .805 | .802 | .865 | .860 |
| E12 | GuitarSet val / 3 | .862 | .811 | .832 | .867 | .862 |

- **What hurts:** cutting the lows and room reverb. Noise at 20 dB and AAC at 64k barely matter.
- **Why the lows matter:** a 250 Hz high-pass removes the fundamentals of the four lowest open strings (82–196 Hz).
- **Training coverage:** the high-pass range was 60–300 Hz, median about 134 Hz, so cuts at 250 Hz and above were rare. The algorithmic reverb is shorter than RT60 0.4–0.9 s.

**Conclusion:**
- **Robustness:** E12 is clearly more robust to phone/room audio, +.03–.04 on degraded test.
- **Clean scores:** unchanged on GuitarSet (−.002 string/fret, +.000 pitch) and better on the real, varied GAPS recordings (+.004 pitch, +.009 agreement).
- **Adoption:** this meets the rule fixed beforehand, so E12 is the new default (`DEFAULT_CHECKPOINT`). E10's weights are kept.
- **No control run:** the GAPS clean gain could partly come from the 6k extra steps.

**My recording:** E12 writes 463 notes vs 471 for E10, with 6 vs 2 rests and 44 changed note lines.

**Next options:**
- Stronger low cut (high-pass up to about 400 Hz) and longer convolution reverbs in training. Real phone microphones do roll off the lows. To avoid tuning training to our own held-out recipe, that run would need a new held-out recipe, for example real recorded impulse responses.

### E13 — Adding a SynthTab fingerstyle subset (`runs/tabmodel_mid_gaps_synth`) — DONE, superseded by E15
**Goal:** improve dense fingerstyle chords. My recording comes out with good melodies but weak multi-note passages, and A1 agrees: GAPS validation misses 40% of notes when 4+ notes sound together, vs ~8% for single notes. That needs audio with *correct* labels for dense polyphony. Self-training cannot provide it (the teacher is weakest exactly there), so we use SynthTab: DadaGP tabs rendered with sampled guitars.

**Data** (`scripts/synthtab_fetch.py`, `guitab/synthtab.py`; budget about the size of the existing data):
- **Download:** SynthTab is 2 TB (acoustic 801 GB: 7 timbres × 4 zip parts of 23–34 GB). Its zips store FLAC uncompressed, so single songs are fetched with HTTP range requests; nothing else is downloaded.
- **Timbres:** the four fingerpicked acoustic ones, `martin_finger`, `taylor_finger`, `luthier_finger`, `gibson_thumb` (10,584 songs each, same songs).
- **Selection** (from the JAMS labels): 9,313 songs in standard tuning (399 h). 895 pass the fingerstyle filter: 60–400 s long, < 15% silence; mean polyphony 2.2–5; 15–85% of attacks are chords; 3–14 notes/s; 20–65% of notes on strings 4–6; frets ≤ 19.
- **Sampling:** random (seed 0), one version per song, ≤ 3 songs per artist. Result: 428 songs, 329 artists (classical, flamenco, acoustic pop / rock): **synth_train** 414 songs (20.6 h), **synth_validation** 14 songs (0.7 h); timbres round-robin (validation happens to use only martin / luthier); `data/synthtab/selection.csv` is versioned.
- **Disk:** 5.3 GB audio + 1.1 GB label archive; features +0.7 GB (cache), 1 augmented copy +3.4 GB.
- **Labels:** per-string MIDI (the files the audio was rendered from): string, fret = pitch − open pitch, exact timing. Mean 2.74 strings sounding when any sounds (GAPS train 1.48).
- **Rendering latency (found and fixed):**
  - **Found:** E12 predictions on downloaded songs were systematically late vs the MIDI, by a different amount per timbre. Median over 12 training songs each: taylor 57.5 ms (beyond the ±50 ms tolerance; E12 pitch F1 ≈ .04–.26 on those songs), martin 24.1, gibson 21.4, luthier 16.7 ms. The spread within a timbre is about ±4 ms.
  - **Not a model bias:** E12's own signed bias on real audio is ~0 (GuitarSet validation −0.7 ms, GAPS validation +0.6 ms), so the delay is in the rendered samples.
  - **Fix:** labels are shifted by `LATENCY_MS` per timbre. After the shift, held-out validation songs line up at +0.4 ms.
  - **Other fixes:** songs with an unplayed string have no MIDI file for it (handled). The download retries after dropped TLS connections (one stopped the first attempt at 214 songs).

**Training choice: mixed fine-tune from E12, not sequential pretraining.**
- **Too little data for a pretraining stage:** 20 h of synthetic audio is not a pretraining corpus; the SynthTab paper pretrains on thousands of hours.
- **A sequential run would forget the chords:** the real-only second stage would mostly overwrite the dense-chord knowledge, because real data has few such passages.
- **Mixing keeps both:** chord supervision throughout, while real audio keeps the sound realistic.
- **Starting point:** from E12, which keeps its phone/room robustness; about 3 h instead of 5+ h from scratch.

**Run** (`experiments/train_mid_gaps_synth.sh`): `--init` E12 `best.pt`, 8k steps, learning rate 1e-4, 300 warm-up steps, batch 16; crops: GuitarSet .35 / GAPS .35 / SynthTab .3 (`--synth_share 0.3`); SynthTab tab-loss weight 1 (exact labels); `--aug_copies 2 --aug_share 0.5` (SynthTab has 1 augmented copy); validation as in E12 (clean + degraded real audio decide `best.pt`), plus SynthTab validation, reported only. With `--synth_share 0`, training batches and losses are identical to before (seeded comparison, augmentation share 0 and 0.5).

**Baselines (E12):** training-protocol validation: clean GuitarSet string/fret / pitch .8115 / .8342, GAPS pitch .8833; degraded GuitarSet .6253 / .7249, degraded GAPS pitch .7320; SynthTab validation: string/fret .6998, pitch .7886.

**Adoption rule** (fixed before results): clean and degraded real validation not worse than E12 by more than about .005; a clear gain on dense polyphony: the A1 miss rate for 4+ simultaneous notes on GAPS validation (E10: .40); then test once, plus a listening check.

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
- **The goal is met: dense chords.** With 4+ notes sounding, misses drop by about a quarter to a half on every set (GAPS test, long notes .394 → .319; short notes .460 → .260). Recall rises everywhere: GuitarSet validation .828 → .854, GAPS test .761 → .786.
- **Cost: precision.** GuitarSet validation .815 → .795, GAPS test .874 → .853: more extra notes. Net F1 is about flat.
- **Strings:** GAPS string agreement jumps (+.029 on clean test), but GuitarSet string/fret is a little worse (clean −.006, degraded −.015).
  - **Likely cause:** SynthTab renders each string's MIDI through a sampled-guitar plugin (Ample Guitar via DawDreamer, `MIDI_to_Audio/render_acoustic.py`). Such plugins choose their own string position, so SynthTab string labels are tab-writer choices that are probably not audible. That suits GAPS (also score fingerings) and not GuitarSet (real players).
- **Phone/room robustness:** slightly worse on GuitarSet string/fret. SynthTab is clean studio audio with only 1 augmented copy.
- **Adoption rule:** dense-polyphony gain yes; "real validation within .005 of E12" no (degraded GuitarSet string/fret −.015 to −.017). So E13 is **not** made the default automatically. `DEFAULT_CHECKPOINT` stays E12.

**My recording:** 544 notes vs 463 for E12, consistent with the higher recall. A listening check decides whether the extra notes are the missing chord notes or false ones.

**From scratch with all data?** Not started. The two weaknesses come from properties of the SynthTab data (inaudible string labels, clean audio), which a from-scratch run would inherit. Nothing here shows fine-tuning as the limit: E13 converged and its 5k–8k validation is flat. A corrected recipe should be tested first.

### E14 — Time-sharp low band for bass attacks (`runs/tabmodel_mid_gaps_sharp`) — NO GAIN, code removed

**Motivation:** in fingerpicking, two notes struck together (thumb bass + finger) often lose the bass. One idea was to detect bass and melody separately. Measured on E12, validation:
- **Bass is the weak voice of a pinch:** lower note missed 29.6% (GuitarSet) / 25.4% (GAPS), upper note 20.7% / 13.6%.
- **Low E is the weak string:** string 6 missed 34% on GuitarSet (strings 2–4: 13–16%); fresh (not repeated) notes ≤ A2 missed 45%.
- **Not a decoding problem:** lowering the onset / frame thresholds of strings 4–6 recovers bass notes (pinch-bass miss .256 → .167) but adds more false notes; F1 drops at every setting.
- **Suspected cause, the input:** at 36 bins/oct a CQT bin's window is ~51 periods (0.62 s at E2). With a synthetic tone, an E2 attack starts rising 25 frames early. Fresh bass attacks gain only +2.5 dB in their own bin (45% under 2 dB); with 4× shorter filters, +6.7 dB.

So separate output heads were not the target (the tab head is already per string). The idea became a separate input view.

**Setup:**
- The 5 octaves from E1 recomputed with `filter_scale` 0.25 (180 bins, `guitab/audio.py`, cache `data/tabmodel_cache_sharp`, 4.5 GB).
- Fed as a second 6-channel harmonic stack; the new first-conv weights start at zero (identical outputs to E12 at step 0).
- Fine-tuned from E12 with E12's recipe (`experiments/train_mid_gaps_sharp.sh`, 6k steps). Best checkpoint: step 6000.

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
- **No gain.** Bass misses move by under a point, and test scores tie. My recording: 147 notes on strings 4–6 vs 144 for E12.
- **The model uses the band:** zeroing it costs 1.5–2.5 points of bass miss rate. But it used the band in place of what it already inferred (harmonics, context), not on top of it.
- **Why the missed bass notes stay missed:** in the sharp band itself, missed fresh bass notes have a median attack contrast of +3.6 dB on GuitarSet (40% under 2 dB) vs +10.7 dB for detected ones; on GAPS +5.7 dB (34% under 2 dB) vs +12.8 dB. They are also ~7 dB quieter. These are soft thumb strokes, attacks masked by ringing low strings, or (GuitarSet) notes the hexaphonic pickup labels but the microphone barely hears. The information is weak in the audio, not hidden by the representation, so a from-scratch run with the band is not expected to help.

Code removed (the revert of the E14 commit); run and feature cache deleted.

### D2 — Decoding for missed bass notes (beat prior, weak attacks) — 2026-10-08, not adopted (no code kept)

What E12 outputs at a missed bass note (string 4–6, validation):

| At the missed bass note | GuitarSet | GAPS |
|---|---|---|
| same pitch 50–200 ms off | .187 | .079 |
| a note with another pitch (other / semitone / fifth / octave) | .277 | .264 |
| same pitch already ringing (re-attack merged) | .082 | .059 |
| nothing; frame > .5 but onset .3–.6 | .063 | .172 |
| nothing; no frame or onset evidence | .197 | .147 |

- The timing misses are GuitarSet strumming: 203 of 229 are in comp tracks, median +70 ms. Not the fingerstyle case.
- **Beat prior** (Beat This! beats; add the best sub-threshold bass candidate on beats with no bass note): fails its premise. Missed bass notes are on a beat less often than hit ones (GuitarSet .39 vs .47, GAPS .49 vs .59). Best setting: pinch-bass miss −1 point, GAPS pitch +.001, GuitarSet −.001.
- **Weak-attack rule** (an onset peak .3–.6 starts a note if the string then sounds for 3 frames and it is a new note on it): GAPS pitch .8833 → .8868, pinch-bass miss .254 → .240; GuitarSet pitch −.001, string/fret −.0016. Small; to retest on the next model.
- **E13 (SynthTab) already cuts bass misses more than any of these:** GuitarSet pinch-bass .296 → .252, all bass .222 → .187; GAPS pinch-bass .254 → .238. Data with many pinches is the lever.

### E15 — SynthTab with down-weighted strings + close-mic phone augmentation (`runs/tabmodel_mid_close`) — DONE, new default model

**Why:** the tab of my recording misses many bass notes. Findings that set up this run:
- E13 (SynthTab) cut bass misses the most of anything tried (D2), but hurt GuitarSet strings and robustness.
- My recording is not like the training audio. Its long-term spectrum vs GuitarSet: +10 dB at 70–100 Hz, +4–5 dB at 100–200 Hz, −8 dB at 1–4 kHz (a phone close to the soundhole: boomy, dark). Validation reshaped to that spectrum (per-track linear-phase EQ) makes E12 miss more pinch basses: GuitarSet .296 → .352.
- Converting notes to the tab loses almost nothing (E15 on the video: 506 notes from the model, 502 in the tab).

**Setup** (`experiments/train_mid_close.sh`, fine-tune of E12, 8k steps):
- E13's recipe (GAPS .35, SynthTab .3), with SynthTab tab-loss weight 0.3 (`--synth_tab_weight`; pitches and attacks keep full weight).
- A third augmented copy `aug2` for every training track, including SynthTab (`augment.close_chain`, `build_aug_cache.py --close`): low shelf +3..+10 dB at 100–220 Hz, body peak +2..+8 dB at 80–150 Hz, high shelf −4..−12 dB from 1–3 kHz, optional low-pass, light room, phone compression, noise, MP3. It matches my recording within ~3 dB per band.
- `--aug_copies 3 --aug_share 0.6`. Best checkpoint: step 8000 (selection .7678 vs E12 .7630).

**Test (once):**

| Test set | Metric | E12 | E15 |
|---|---|---|---|
| GuitarSet player 05 chunks, clean | time pitch / string-fret F1 | .8595 / .8265 | **.8621 / .8319** |
| GuitarSet test, degraded | time pitch / string-fret F1 | .7295 / .6448 | **.7433 / .6596** |
| GAPS test, clean | pitch F1 / string agreement | .8147 / .8493 | **.8232 / .8805** |
| GAPS test, degraded | pitch F1 / string agreement | .6740 / .7573 | **.6810 / .7594** |

**Bass, validation, original and reshaped to sound like my recording** (miss rates; lower is better):

| Set | Audio | Model | pitch F1 | all bass | pinch bass | treble |
|---|---|---|---|---|---|---|
| GuitarSet | original | E12 / E13 / E15 | .834 / .834 / **.842** | .222 / **.187** / .195 | .296 / **.252** / .262 | .137 / .119 / .120 |
| | video-like | E12 / E13 / E15 | .833 / .832 / **.838** | .260 / **.206** / .225 | .352 / **.287** / .318 | .185 / .142 / .146 |
| GAPS | original | E12 / E13 / E15 | .883 / .886 / **.889** | .120 / .116 / **.115** | .254 / .238 / **.229** | .150 / .138 / .136 |
| | video-like | E12 / E13 / E15 | .883 / **.888** / .887 | .136 / **.109** / .114 | .266 / **.200** / .214 | .148 / .120 / .128 |

**Reading:**
- **Adopted:** better than E12 on every test number, clean and degraded. `DEFAULT_CHECKPOINT` is now E15.
- **Bass:** 3–5 points fewer pinch-bass misses than E12, on both sets and both tones. E13 is still slightly better on bass but worse on strings and robustness. The fixes for E13's problems worked: GuitarSet string/fret and GAPS string agreement both beat E12.
- **My recording:** 149 notes on strings 4–6 (E12: 144), not a large change. There is no ground truth for my playing yet, so how many bass notes are missing there is still unmeasured (see the next entry).

### Benchmark song: my recording vs its reference tab — 2026-10-08

I learnt "Remember Me (Coco)" (Sungha Jung's arrangement) from a published tab (43 measures, 2 voices), and the tab was aligned to my recording:
- **Score to seconds:** with its tempo map.
- **DTW:** between the score's pitch roll (activity + attacks, E2–D#6) and the mean pitch roll of E10/E12/E13/E15 on the recording. A chroma-only DTW was too loose (±300 ms).
- **Snap:** each note to the nearest model-free audio onset within 50 ms (461/522 snapped).
- **Ending dropped:** the last two measures (I end differently). This leaves 507 reference notes, 171 on strings 4–6.
- **Saved** as `data/my_video_reference.json` (not versioned).
- **Caveats:** my playing may deviate from the tab, and the alignment uses the models (averaged over four).

**Models on my song (±100 ms, default extraction before D3):**

| | E10 (H) | E12 (J) | E13 (K) | E15 (M) |
|---|---|---|---|---|
| pitch F1 | .710 | .711 | .704 | **.713** |
| tab (string + fret) F1 | .682 | .678 | .677 | **.691** |
| bass recall (strings 4–6) | .63 | .61 | **.67** | .64 |
| pinch-bass recall | .51 | .49 | **.61** | .56 |

**E15 errors on the song:**
- **False notes (124):** 84 are re-triggers of a ringing note, heard as "more notes than played", e.g. E4 E4 → E4 ×4 at the start.
- **Missed bass (61):** mostly F2 (15) and F3 (12).
  - 28% have the right pitch 100–300 ms off (rolled pinches or alignment).
  - 28% have no model evidence; 20% are heard sustaining with no attack.
  - Missed F2/F3 notes are as loud as hit ones but barely rise at the attack (+4.4 dB vs +15.9 dB; 26% under 1 dB). Many are probably let ring rather than re-struck.

### D3 — Merge re-attacks that split a ringing note (`reattack` 0.75) — 2026-10-08, ADOPTED

A note that starts exactly where the previous note on the same string and fret ends is kept only if its onset probability is ≥ `reattack`; otherwise the two are merged (`inference.merge_reattacks`).

**Choosing the threshold:** chosen on validation (sum of four pitch F1 changes): 0.7 +.0078, **0.75 +.0102**, 0.8 +.0083.

**E15, test once, without → with D3:**

| Test set | time pitch / string-fret F1 |
|---|---|
| GuitarSet clean | .8621 / .8319 → **.8662 / .8360** |
| GuitarSet degraded | **.7433 / .6596** → .7349 / .6503 |

| Test set | pitch F1 / string agreement |
|---|---|
| GAPS clean | .8232 / .8805 → **.8268** / .8804 |
| GAPS degraded | .6810 / .7594 → **.6841** / .7591 |

| My song (±100 ms) | before D3 | with D3 |
|---|---|---|
| pitch F1 | .713 | **.725** |
| tab F1 | .691 | **.703** |
| notes (fewer, almost all false) | 506 | 486 |
| pinch-bass recall | .56 | .54 |

- **Cost:** heavily reverberant audio (GuitarSet degraded, −.008), where real re-attacks have less confident onsets.
- **Adopted** because it fixes the most audible error: the opening of my song is now E4 E4 C4 | C3+G3 ..., as in the tab (E15 without D3 wrote E4 four times).

### E16 — Crop mixing: soft notes under loud ones (`runs/tabmodel_mid_mix`) — NO GAIN, code removed

**Why:** on my song (aligned reference tab), E15 + D3 completes only 6% of the 47 chords of 3–4 notes struck together (note recall .46; 2-note groups: 55% complete). Missed chord notes are present in the audio but ~16 dB softer than found ones (median −30 vs −14 dB re track max), mostly inner voices (G string 25 of 84). Equalising the recording to the training spectrum at inference only moved 3+ note recall .46 → .51.

**Setup:**
- `--mix_share 0.35`: a crop gets a second random crop added 6–25 dB quieter, summed as power from the dB features.
- Labels: tab ignored where both crops use the same string; onsets merged; pitch loss skipped on those frames.
- Fine-tune of E15 with E15's recipe, 6k steps (`experiments/train_mid_mix.sh`). Selected checkpoint: step 3000 (selection .7571).

**Results (E15 + D3 → E16 + D3):**

| | E15 | E16 (step 3000) | E16 (step 6000) |
|---|---|---|---|
| song: 3+ note chords complete / note recall | .06 / .46 | .11 / .54 | .06 / .52 |
| song: 2-note groups complete | .55 | .60 | .57 |
| song: bass recall | .64 | .69 | .67 |
| song: pitch F1 / tab F1 (±100 ms) | **.739 / .717** | .717 / .682 | .718 / .687 |
| song: notes output (reference 507) | 467 | 534 | 518 |

| Test (once, E16 step 3000) | Metric | E15 + D3 | E16 + D3 |
|---|---|---|---|
| GuitarSet clean | time pitch / string-fret F1 | **.8662 / .8360** | .8602 / .8270 |
| GuitarSet degraded | time pitch / string-fret F1 | **.7349 / .6503** | .7229 / .6348 |
| GAPS clean | pitch F1 | **.8268** | .8192 |
| GAPS degraded | pitch F1 | **.6841** | .6746 |

**Reading:**
- **On the song:** more chord notes are found (3+ note recall +.08, bass +.05), but more false notes too, so net F1 drops.
- **Validation:** clean scores were the best of any model (GuitarSet string/fret .8200 at 5k, GAPS pitch .8937 at 6k).
- **Robustness:** degraded validation stayed 2–5 points under E15 and swung between checkpoints. A likely cause is mixing a clean crop with a phone-processed one.
- **Test:** worse than E15 everywhere. Not adopted; code removed (the revert of the E16 commit), run deleted.

**Conclusion:** sensitivity to soft inner voices does not come from synthetic mixtures of unrelated crops. What is missing is real recordings of dense fingerstyle with exact labels.

### E17 — Real fingerstyle recordings labelled from reference tabs (`runs/tabmodel_mid_custom`) — DONE, not adopted (candidate)

**Data** (dataset tools not published; `guitab/custom_data.py`; audio and labels not distributed):
- **Selected:** 60 recordings of single acoustic guitar with their reference tabs (mostly by one fingerstyle arranger; standard tuning, ≥ 20% of attacks in chords). "Remember Me" (my benchmark song) excluded. 56 recordings available.
- **Labels:** pitches and strings from the tab, timed as follows.
  - Coarse: the tab's measure positions in the recording, linear inside a measure.
  - Refined: DTW against the mean pitch roll of E10/E12/E13/E15, keeping the best of banded ±0.5 s / global offset / free. Many syncs are 5–15 s off (video intros); free DTW recovers them.
  - Snapped: each onset moved to the nearest audio onset within 50 ms.
  - Label check on the arranger's own "Remember Me" recording (E15 pitch F1, ±100 ms): .45 coarse → .68 refined.
- **Kept:** songs whose labels agree with E15 at pitch F1 ≥ .5 (±100 ms): 47 of 56 (2.67 h; 42 ss_train, 5 ss_validation). Dropped: 2 recordings a semitone off the tab, 1 unalignable, 6 others.
- `scripts/evaluate_song.py`: the benchmark song as a script.

**Training:** fine-tune of E15, 6k steps (`experiments/train_mid_custom.sh`).
- Shares: custom-dataset .3 (string labels at weight .5), GAPS .3, SynthTab .2, GuitarSet .2.
- 3 augmented copies per track.
- Best checkpoint: step 6000 (selection .7683 vs E15 .7678).

**Validation, custom-dataset held-out songs (±50 ms):**

| | E12 | E15 | E17 |
|---|---|---|---|
| string/fret F1 | .527 | .528 | **.600** |
| pitch F1 | .657 | .650 | .653 |

**My song (±100 ms):**

| | E15 + D3 | E17 + D3 |
|---|---|---|
| pitch F1 / tab F1 | .739 / .717 | .739 / .717 |
| 2-note groups complete (notes) | .55 (.73) | **.64 (.78)** |
| 3+ note chords complete (notes) | .06 (.46) | .04 (.45) |
| bass recall | .64 | .64 |

**Test (once):**

| Test set | Metric | E15 + D3 | E17 + D3 |
|---|---|---|---|
| GuitarSet clean | time pitch / string-fret F1 | **.8662** / .8360 | .8647 / **.8365** |
| GuitarSet degraded | time pitch / string-fret F1 | .7349 / .6503 | **.7406 / .6632** |
| GAPS clean | pitch F1 / string agreement | **.8268** / .8804 | .8156 / .8804 |
| GAPS degraded | pitch F1 / string agreement | **.6841 / .7591** | .6813 / .7552 |

**Reading:**
- **Strings:** real fingerstyle data clearly improves strings on unseen fingerstyle recordings (+7 points).
- **2-note groups** on my song: +9 points complete.
- **Chords of 3+ notes:** not helped (still ~5% complete). 2.7 h is little next to GuitarSet / GAPS / SynthTab (~30 h).
- **Not adopted:** GAPS clean test pitch −.011 fails the adoption rule. E15 stays the default; E17 is kept as a candidate.

### D4 — Self-calibrated explaining-away for missed chord notes — 2026-10-12, promising, not integrated (no code kept)

**Idea:** learn the spectral fingerprint of each pitch on the player's own guitar / phone / room from the model's confident isolated notes in the same recording. Here 239 notes give 22 pitches directly; the others are shifted copies.
- At each attack group the model found, fit those notes' fingerprints (NNLS) to the energy the attack adds (linear CQT, frames +2..+8 minus −4..−1).
- Then add up to K missing pitches whose fingerprint explains ≥ θ of the residual energy, playable on a string not used by the group, where the model's fret posterior is ≥ ε.

**My song (E15 + D3, ±100 ms):**

| setting | pitch F1 / tab F1 | 2-note groups complete (notes) | 3+ chords complete (notes) | notes added |
|---|---|---|---|---|
| none | .739 / .717 | .55 (.73) | .06 (.46) | — |
| θ .1, ε .05, K 1 | .741 / .715 | .64 (.78) | .09 (.48) | 28 |
| θ .05, ε .01, K 1 | .739 / .712 | **.69 (.81)** | **.13 (.52)** | 57 |
| θ .05, ε 0, K 1 | .714 / .687 | .71 (.81) | .13 (.50) | 88 |

K 2 changes nothing.

**Chord completeness is mostly a property of my recording.** E15 on the custom-dataset held-out recordings (other players, aligned labels, ±100 ms): single notes .83, 2-note groups .55, **3+ chords .29 complete (notes .64)**. On my song: .89 / .55 / **.06 (.46)**.

The missed chord notes in my recording are ~16 dB softer than found ones and half have no visible attack (E16 analysis).
- **Likely causes:** inner voices plucked very softly or not re-struck, and the close boomy phone placement (+10 dB at 70–100 Hz).
- **Next:** a re-recording with the phone 30–50 cm away near the 12th fret would separate playing from recording.

### D5 — Benchmark of free adjustments (tone EQ, attack enhancement, explaining-away) — 2026-10-12, none adopted (no code kept)

**Benchmark** (E15 + D3, ±100 ms, aligned labels): the 5 custom-dataset held-out recordings; the same 5 through `augment.close_chain` (my phone-near-the-soundhole sound); my song.

**Adjustments:**
- Tone EQ: linear-phase EQ of each recording's long-term spectrum toward the median of 130 training tracks (strength .5 / 1, ±15 dB).
- Attack enhancement: + .5 × the HPSS percussive part.
- D4 explaining-away. Alone and combined.

| Set | Variant | pitch F1 | 2-note complete | 3+ complete (notes) | notes |
|---|---|---|---|---|---|
| custom-dataset | none | **.690** | .55 | .29 (.64) | 5170 |
| | D4 | .670 | .59 | .33 (.66) | 5698 |
| | EQ 1 | .684 | .55 | .36 (.67) | 5514 |
| | EQ 1 + D4 | .666 | .59 | .39 (.70) | 6064 |
| custom-dataset, close-mic | none | **.666** | .52 | .29 (.63) | 5272 |
| | D4 | .648 | .58 | .32 (.66) | 5912 |
| | EQ 1 | .629 | .57 | .38 (.68) | 6341 |
| My song | none | **.739** | .55 | .06 (.46) | 467 |
| | D4 | **.739** | .69 | .13 (.52) | 524 |
| | EQ .5 + D4 | .733 | .73 | .15 (.53) | 546 |
| | EQ 1 + D4 | .716 | .72 | .17 (.52) | 566 |

Attack enhancement: within ±.01 of none everywhere.

**Reading:**
- **A trade, not a gain.** Every adjustment raises chord recall and adds more false notes; overall pitch F1 is flat or worse. Nothing is adopted as default. D4 could be an opt-in "fuller chords" mode.
- **The phone sound is not the chord problem.** Simulating my phone placement on other recordings does not change their 3+ chord completeness (.29 → .29). My 3+ chord gap (.06) comes from the playing (soft or un-restruck inner notes) or from phone processing the chain does not model (automatic gain, noise suppression).

### A1 — Tab language model + joint decoding (`runs/tablm`, not versioned) — 2026-10-12, ADOPTED as opt-in `--music_lm`

**Idea:** speech-recognition-style fusion. The acoustic model (E15) scores candidate notes; a small tab language model, trained on symbolic tabs only, scores how plausible each chord is after the previous ones. Beam search picks the sequence.

**Language model:**
- Data: 12,737 acoustic / clean / jazz 6-string standard-tuning tracks from the DadaGP-derived SynthTab label archive (60,633 tracks in all; distortion, overdrive and bass excluded). My song and the 5 custom-dataset held-out titles are excluded.
- Tokens: notes of a chord (string, fret ≤ 19), low → high string, then end-of-chord. No rhythm. 16.9M training tokens.
- Model: causal transformer, 4 layers, d 256, 3.3M parameters. 8k steps × 64 × 256 tokens, ~40 min on an M3 Pro.
- Held-out perplexity: 1.8 per token.

**Decoder:**
- Attack groups: per-string onset peaks ≥ τ 0.2, merged across strings within 4 frames.
- Per string with a peak: no note, or one of its top-2 frets (fret posterior ≥ .1). At most 12 candidate chords per group, no duplicate pitches.
- Score: acoustic log-probabilities + λ · LM log P(chord | history, last 120 tokens) + β · notes. Beam 4.
- Notes without an onset peak (legato) pass through from the default extraction; then D3 re-attack merge.
- Cost: ~10–20 s extra per song.

**Tuning** (half of GAPS validation, a third of GuitarSet validation):
- λ 0.3 is the useful range; λ ≥ 0.6 lets the LM override the audio (GAPS pitch F1 down to .60–.86).
- The audio-only decoder (λ 0) is not better than the default; the gains come from the LM.
- Chosen: λ 0.3, β 1.0, 4-frame grouping window, D3 0.75.

**Evaluation** (default extraction → LM decoding):

| Set (tolerance) | pitch F1 | 3+ chords complete (notes) | 2-note groups complete |
|---|---|---|---|
| My song (±100 ms) | .739 → **.745** | .06 (.46) → **.11 (.48)** | .55 → **.59** |
| custom-dataset held-out (±100 ms) | **.690** → .678 | .29 (.64) → **.39 (.68)** | .55 → .54 |
| GAPS validation (±50 ms) | .892 → **.899** | .29 (.59) → **.38 (.64)** | .66 → **.70** |
| GuitarSet validation (±50 ms) | **.851** → .844 | .68 (.87) → **.72 (.88)** | .60 → **.65** |

β 1.25 gives slightly fuller chords (my song 2-note .64, 3+ .13) at ~−.005 F1.

**Ceiling:** share of notes in 3+ chords that are candidates at all (onset peak on the right string, fret in the top-k): my song: .48 (τ .2) to .58 (τ .05, top-3); the decoder finds .48; custom-dataset held-out: .70 to .76; the decoder finds .68.

**Reading:**
- The LM decoder recovers nearly every chord note the acoustic model gives any evidence for.
- On my recording ~42% of chord notes have no onset evidence on their string. That part needs better acoustic evidence (real labelled phone recordings, e.g. a hexaphonic pickup), not a better prior.
- Chords are more complete on every set; overall F1 moves ±.01.
- **Integrated as an option, not the default:** `transcribe.py --music_lm [--music_lm_weight .3 --music_lm_bonus 1.0]`, code in `guitab/tablm.py`; `scripts/train_tablm.py corpus|train` rebuilds the corpus (byte-identical) and the model; `evaluate_song.py --music_lm` reproduces the numbers above (.745 / .719); default output unchanged (my song's tokens identical).

### E18 — Large model from scratch on all data (`runs/tabmodel_large_all`) — 2026-10-12 → 10-14: kept as a backup, not the default

**Setup** (`experiments/train_large_all.sh`):
- **Model:** `large` preset, 66.1M parameters (mid: 23.8M); same CQT input and heads.
- **Data:** from scratch, all data mixed per crop: GuitarSet .25 (+ pitch-shifted copies), GAPS .3 (strings .5), SynthTab .2 (strings .3), custom-dataset .25 (strings .5).
- **Augmentation:** 3 augmented copies (phone/room ×2, close-mic) on 60% of crops.
- **Validation:** clean + degraded, selection as before.
- **Schedule:** 30k steps, lr 3e-4, warmup 1000, cosine.
- **Memory:** batch 16 does not fit in 9.8 GB, nor in 15 GB with fp32 or bf16. New `--accum 2` runs each batch as two chunks (same update, half the peak memory); memory cap .53 (15 GB). 2.4 s/step, ~20 h of training plus evaluations, over several sessions.
- **Why one mixed run rather than pretrain → fine-tune:** mixing sources per crop worked better than sequential stages (E13, E17). A single fixed schedule is also simpler to stop and resume.

**Run:** stopped after step 26000 (no improvement since step 22000). Best = step 22000 (selection .7816; E15 .768). Checkpoint kept weights-only (`best.pt`, optimizer state removed, predictions verified identical); `last.pt` deleted.

**Results (test sets evaluated once; E15 → E18):**

| Set | E15 | E18 |
|---|---|---|
| GuitarSet test, pitch / string+fret F1 | .8662 / .8360 | .8647 / .8372 |
| GuitarSet test degraded, pitch / string+fret | .7349 / .6503 | .7524 / .6750 |
| GAPS test, pitch F1 / string agreement | .8268 / .8804 | .8193 / .8791 |
| GAPS test degraded, pitch / agreement | .6841 / .7591 | .6913 / .7710 |
| My song: pitch / tab F1 | .739 / .717 | .745 / .729 |
| My song: bass recall, 2-note / 3+ complete | .64, .55 / .06 | .67, .67 / .09 |
| My song --music_lm: pitch / tab F1, 2-note / 3+ | .745 / .719, .59 / .11 | .742 / .726, .69 / .09 |
| custom-dataset held-out (5): pitch / tab F1 | .683 / .556 | .689 / .640 |
| custom-dataset held-out: 2-note / 3+ complete | .55 / .29 | .45 / .29 |
| custom-dataset held-out --music_lm: pitch / tab, 3+ | .671 / .554, .39 | .683 / .642, .38 |
| Time per test song / my song | 2.6 s / 6 s | 6.7 s / 15 s |

**Conclusion:** 2.8x the parameters and ~1 day of training buy ~+1 point on real recordings and +2-3 on degraded audio, nothing on the clean tests (GAPS pitch -.0075), at 2.5x the inference time. The model is not capacity-limited at this data size: the clean sets are near their label ceiling and the gains appear only where data variety matters (degraded audio, real custom-dataset recordings, string choice +8 on held-out custom-dataset). Next lever: more real labelled data. Not the default (E15 stays); usable as a backup with `--checkpoint runs/tabmodel_large_all/best.pt`.

### E19 — Enlarged custom dataset + mid model from scratch on all data (`runs/tabmodel_mid_all`) — 2026-10-14 → 10-15: NEW DEFAULT (after the paired comparison)

**Why:** E18 showed the model is not capacity-limited (2.8x parameters, ~+1 point). The remaining lever is more real labelled audio; the custom dataset is the only source we can scale.

**Data: custom dataset v2** (same selection and labelling rules as E17):
- **Select:** more fingerstyle arrangers and generic fingerstyle material (single acoustic guitar, standard tuning), append-only (the first 60 songs and their splits are unchanged); malformed tabs skipped. 360 songs selected (17.0 h; 40 ss_validation), 330 recordings available.
- **Label:** as in E17, ~30 s per song, in parallel shards. Songs whose audio is > 2x the tab duration + 60 s are skipped (a wrong recording; one 53-min compilation made the free DTW run out of memory).
- **Result:** 325 labelled, 165 pass the quality gate (E15 pitch F1 >= .5): **ss_train 146 songs (7.2 h of tab, 8.1 h of audio; 100 new), ss_validation 19 (0.95 h, 14 new)**. Real labelled fingerstyle audio 2.67 h → 8.2 h (3x, not the 6x hoped for).
- **Drop rate:** 58% of the new songs fail the gate vs 16% in the first batch. Scores are bimodal (114 songs < .3: wrong recording, other arrangement, vocals / band; 30 between .3 and .45; few borderline), so the threshold stays.
- **Caveat:** the gate keeps songs E15 already transcribes reasonably well; it biases the subset towards "easier" audio.
- The original 5 ss_validation songs stay in the held-out set; results will be reported on them (comparable with E15 / E18) and on all 19.

**Training setup** (`experiments/train_mid_all.sh`, run on 2026-10-15): mid (23.8M) **from scratch** (no `--init`), batch 16, no accumulation, usual memory cap .35 / .25; data shares: GuitarSet .25, GAPS .25 (strings .5), SynthTab .15 (strings .3), custom-dataset .35 (strings .5); 3 augmented copies (aug0/aug1 phone-room, aug2 close-mic) on 60% of crops; clean + degraded validation; 24k steps, lr 3e-4, warmup 1000. Feature caches built (CQT + aug0/aug1/aug2 for all 146 ss_train songs); `load_split` verified.

### D6 — Capo-aware decoding (`transcribe.py --capo N`) — 2026-10-14, ADOPTED (opt-in option)

**Problem:** with a capo at fret N, frets below N (and open strings) cannot be played, but the model does not know about the capo and often puts a pitch on a string where it would need such a fret. The tab also showed absolute frets, while guitarists read them relative to the capo.

**Method** (no retraining; any checkpoint): `inference.apply_capo`: a note below the capo keeps its pitch and moves to the string with the lowest playable fret (>= capo, <= 20) not sounding at its onset; it is dropped if no string can play it (a pitch below the capo'd low E); `tab.notes_to_tokens(..., capo)`: tokens hold frets relative to the capo (refingering then works in the capo'd tuning); `tab.decode_to_gp5(..., capo)`: the .gp5 track's capo is set (pyguitarpro `Track.offset`), so playback pitch is right.

**Results** (E15, the 12 ss_validation songs with a capo, labels with absolute frets, onset ±50 ms):

| | without | with --capo |
|---|---|---|
| string + fret F1 (mean) | .495 | **.596** |
| pitch F1 (mean) | .629 | .630 |

- 0–39% of the predicted notes were below the capo (more with capo 5). Capo 5 songs gain the most: "Time in a Bottle" .476 → .794, "Married Life" .366 → .600, "Kamado Tanjiro" .417 → .585.
- Few notes are dropped (at most 31 of 1929 on one song).
- On "Time in a Bottle" (capo 5) the first 8 notes of the `--capo` version match the reference tab exactly (string and relative fret).
- Without `--capo`, the tokens and .gp5 of my song are byte-identical to before.
- Not done: automatic capo detection (would need the capo guessed from the audio or the predictions).

**Run:** 1.17 s/step, 0.7 GB allocated between steps. Stopped at step 21100 (flat since 17k). Best = step 18000 (selection .7665; E18 .7816, E15 .768). Kept weights-only (`best.pt`, predictions verified identical); `last.pt` deleted. Validation vs E18 at the same step: equal or better on clean GuitarSet / GAPS / SynthTab / custom-dataset from ~12k on; consistently ~4 points behind on phone/room-degraded GuitarSet.

**Results (test sets evaluated once; E15 / E18 / E19):**

| Set | E15 | E18 (large) | E19 |
|---|---|---|---|
| GuitarSet test, pitch / string+fret F1 | .8662 / .8360 | .8647 / .8372 | .8656 / **.8433** |
| GuitarSet test degraded, pitch / string+fret | .7349 / .6503 | **.7524 / .6750** | .7210 / .6402 |
| GAPS test, pitch / agreement | **.8268 / .8804** | .8193 / .8791 | .8126 / .8713 |
| GAPS test degraded, pitch / agreement | .6841 / .7591 | **.6913 / .7710** | .6830 / .7461 |
| My song: pitch / tab F1 | .739 / .717 | **.745 / .729** | .725 / .703 |
| My song: bass recall, 2-note / 3+ complete | .64, .55 / .06 | .67, .67 / .09 | .63, **.73** / .09 |
| My song --music_lm: pitch / tab, 2-note / 3+ | .745 / .719, .59 / .11 | .742 / .726, .69 / .09 | .712 / .693, .72 / **.21** |
| custom-dataset orig 5 held-out (±50 ms): pitch / tab F1 | .650 / .528 | .656 / .609 | **.677 / .630** |
| custom-dataset all 19 held-out: pitch / tab F1 | .661 / .548 | .659 / .587 | **.673 / .617** |
| custom-dataset all 19: 2-note / 3+ complete | .57 / .29 | .48 / .26 | .57 / .26 |

(custom-dataset rows recomputed for all three models with the same script, onset ±50 ms; the E18 entry's custom-dataset numbers used a different tolerance.)

**Conclusion:** 3x the real custom data helps on custom-dataset-like recordings (tab F1 +7 over E15, +3 over E18 on the 19 held-out songs) and clean GuitarSet strings (.843, best so far), but not on anything out of that domain: GAPS pitch -1.4 and degraded GuitarSet -1 vs E15, and my song -1.4 tab F1. Against E18 it loses ~3.5 points on degraded audio. Not the default (GAPS pitch drop > .5). With the music LM it completes the most 3+ chords on my song (.21) but at lower F1. Neither scaling the model (E18) nor scaling the custom data (E19) moved the general benchmarks by more than ~1 point; the custom-dataset gains are partly in-domain (same arrangers, same alignment pipeline and label noise). Usable with `--checkpoint runs/tabmodel_mid_all/best.pt`.

**Paired per-recording comparison** (2026-10-15; full tracks, onset ±50 ms, Wilcoxon signed-rank):

| Set (recordings) | E15 | E18 | E19 | E19 vs E15 | E19 vs E18 |
|---|---|---|---|---|---|
| GuitarSet test, string+fret (60) | .887 | .887 | **.899** | wins 37/60, +.011, p=.004 | wins 40/60, +.012, p=.002 |
| GuitarSet test phone/room, string+fret (60) | .724 | **.751** | .736 | wins 32/60, +.012, p=.20 | wins 33/60, -.015, p=.78 |
| GAPS test (classical), pitch (19) | **.827** | .819 | .813 | wins 4/19, -.014, p=.016 | wins 7/19, -.007, p=.29 |
| custom-dataset held-out, string+fret (19) | .548 | .587 | **.617** | wins 16/19, +.068, p=.001 | wins 12/19, +.030, p=.08 |

E19 is significantly better than E15 on steel-string material (GuitarSet, custom-dataset) and significantly worse only on GAPS (classical / nylon repertoire); against E18 it is better on clean steel-string audio and not significantly different elsewhere, at a third of the size. For the project's target (steel-string fingerstyle) it is the best-supported checkpoint; single songs (my recording, a random custom-dataset song) can go either way.

**Decision (2026-10-15): E19 is the new default** (`DEFAULT_CHECKPOINT`), overriding the GAPS rule: the target is steel-string fingerstyle, where the paired comparison favours E19; GAPS is classical nylon. Re-tuning the note thresholds on GuitarSet validation for E19 (`evaluate.py --tune`, 96 settings) keeps the current `DEFAULT_PARAMS` (onset .6, frame .6, legato 8, no orphans) as the best. The checkpoint-independent decoding steps all apply unchanged: duplicate removal (D1), re-attack merge (D3), `--capo` (D6), `--music_lm` (A1, tuned with E15 probabilities; not re-tuned). Default transcription verified identical to `--checkpoint runs/tabmodel_mid_all/best.pt`.

### Release clean-up — 2026-10-15

Preparing a public GitHub release (code only, MIT; no weights; no custom-dataset code or data). Every step verified identical: `guitab/metrics.py` (copied from Audio2Tab, no licence) → own `evaluation.token_metrics`: precision / recall / F1 bit-identical on test and validation for E15 and E19 (time-based onsets are summed wait by wait, as before); `guitab/dadagp_utils/` (~2,500 copied lines) → own `guitab/gp5.py` (~90 lines): 22 earlier transcriptions plus fresh ones identical in tempo, capo, tuning, measures, time signatures and every beat and note; no more debug printing; `match()` moved into `guitab.evaluation`; experiment launchers moved to `experiments/`; `pyproject.toml` with a `guitab` command (core vs `[train]` dependencies); ruff F/E/W/B clean; MIT `LICENSE`; README rewritten for outsiders; A public branch was prepared here; the final approach is described in "Public release" below.

### Device-neutral code — 2026-10-16

- Every script and the `guitab` command take `--device auto|cuda|mps|cpu` (auto = CUDA, else Apple GPU, else CPU; `guitab/device.py`). Training memory: `--max_memory <fraction>` replaces the `PYTORCH_MPS_*` variables in the launchers (on CUDA `set_per_process_memory_fraction`; on Apple GPUs the high watermark plus a low watermark at 70% of it - the cap alone ran out of memory at batch 16). Launchers take `DEVICE=...`.
- Verified on this Mac: my song's tokens identical (auto and mps), GuitarSet test and GAPS test identical, CPU transcription runs, training at 1.12 s/step under `--max_memory 0.35`.

### Public release — 2026-10-16

- Published history: every commit kept, with the custom dataset's acquisition tools and data files and the copied third-party code (Audio2Tab metrics, DadaGP converter) removed from all of them; older commits therefore do not run standalone (they imported the copied code).
- Default checkpoint: E15 (`experiments/train_mid_close.sh`), reproducible from public data; E19 needs the custom dataset.
- `guitab/custom_data.py` keeps only the layout that training reads; `train_tablm.py` works without the custom dataset.

### Light models — 2026-10-17

Question: how much of the 24M mid model is needed? The transformer is 18.9M of it (plus a 2.1M position table); the CNN front end 1.15M. Each variant uses E19's recipe unchanged (from scratch, all data, 24k steps, `experiments/train_light_all.sh SIZE=...`) and is compared with E19 by the paired per-recording test. Run one at a time. `build_model(size)` in `guitab/model.py`; E15 and E19 transcriptions verified byte-identical before/after the change.

| Size | Time model | Params | Train step (MPS, batch 16) | Inference per 5 s window |
|---|---|---|---|---|
| mid (E19) | 6-layer transformer, 512 wide | 23.8M | 1.09 s | 39 ms |
| small (E20) | 4-layer transformer, 256 wide (same CNN) | 6.2M | 0.69 s | 16 ms |
| gru (E21) | 2-layer bidirectional GRU, 256 units (same CNN) | 4.0M | 0.88 s | 139 ms |
| tcn (E22) | 8 residual dilated convolutions (dilation 1..128, +-3 s; same CNN) | 3.6M | 0.62 s | 12 ms |
| tiny (E23) | TinyTabModel: pitch-axis CNN shared across pitches, per-string 20-semitone fret windows | 0.105M | 0.61 s | 7 ms |

Expectations: small and gru within ~1 point of E19; tcn similar or slightly lower; tiny several points lower on string/fret, closer on pitch.

#### E20 — small transformer (`runs/tabmodel_small_all`) — 2026-10-17: matches E19 except clean GuitarSet (-1 point)

- 24k steps at 0.72 s/step (~5 h); selected checkpoint step 18000 (selection .7665). Validation curves tracked E19 within ~1-2 points at every eval from step 3000; degraded GAPS pitch stayed ~2-3 points lower from step 16000.
- Paired per-recording comparison (best.pt, onset +-50 ms, Wilcoxon signed-rank):

| Set (recordings) | E15 | E19 | E20 | E20 vs E19 | E20 vs E15 |
|---|---|---|---|---|---|
| GuitarSet test, string+fret (60) | .887 | **.899** | .888 | wins 21/60, -.010, p=.029 | wins 27/60, +.001, p=.95 |
| GuitarSet test phone/room, string+fret (60) | .724 | .736 | **.737** | wins 24/60, +.000, p=.42 | wins 33/60, +.013, p=.31 |
| GAPS test (classical), pitch (19) | **.827** | .813 | .808 | wins 8/19, -.004, p=.28 | wins 2/19, -.018, p=.002 |
| custom-dataset held-out, string+fret (19) | .548 | **.617** | .611 | wins 10/19, -.005, p=.62 | wins 15/19, +.063, p=.001 |

- Model time over all four sets (predict only, MPS): E19 138 s, E20 57 s (2.4x faster); 6.2M vs 23.8M parameters.
- Note-extraction thresholds re-tuned on GuitarSet validation (`evaluate.py --tune`, grid widened to onset / frame .3–.7, 150 settings), for E20 and E19: the defaults stay. E20's best (frame .7) beats the defaults by +.0003 string/fret F1 (noise), and E19's best is the defaults. On the validation chunks E20 and E19 are level (string/fret .739 / .739, pitch .767 / .769).

**Conclusion:** at a quarter of the parameters and 2.4x the speed, E20 is not significantly different from E19 on degraded GuitarSet, GAPS and the custom held-out set; it loses ~1 point on clean GuitarSet (significant, p=.03). The 19M transformer buys about one point on clean audio. Not adopted (yet): kept as the light candidate until the GRU / TCN / tiny runs.

#### E21 — bidirectional GRU (`--size gru`) — 2026-10-19: stopped at step 18000, level on public data, behind on steel-string, slower

- Same recipe as E19/E20 (`SIZE=gru experiments/train_light_all.sh`). 1.2 s/step excluding evals (another application shared the GPU), vs 0.72 s for E20.
- Evals were taking ~11 min each: `predict` ran one 432-frame window at a time, and the GRU has to step through the frames in order. `predict` now runs 16 windows per forward pass (identical probabilities and metrics for every size; GRU eval 467 s -> 73 s, E19 100 s -> 45 s, E20 55 s -> 32 s). Training resumed from step 4000 with the fix.
- Validation at the same steps (string/fret F1 unless noted):

| Step | clean val | clean val pitch | degraded val | GAPS pitch | GAPS string agree | custom val | SynthTab val |
|---|---|---|---|---|---|---|---|
| 3000 E21 / E19 / E20 | .707 / .628 / .625 | .801 / .773 / .767 | .539 / .521 / .512 | .811 / .785 / .784 | .793 / .777 / .780 | .532 / .519 / .491 | .855 / .825 / .828 |
| 8000 | .790 / .783 / .785 | .830 / .826 / .826 | .608 / .588 / .590 | .864 / .855 / .844 | .850 / .829 / .825 | .596 / .597 / .584 | .885 / .856 / .857 |
| 13000 | .804 / .811 / .817 | .844 / .842 / .838 | .611 / .626 / .632 | .885 / .870 / .867 | .849 / .865 / .854 | .589 / .597 / .604 | .906 / .892 / .883 |
| 18000 | .815 / .818 / .817 | .844 / .847 / .844 | .657 / .627 / .646 | .881 / .885 / .879 | .862 / .864 / .865 | .601 / .617 / .611 | .906 / .915 / .907 |

- The GRU learns faster early (ahead by 4-8 points up to step 3000), is level with the transformers from step ~5000 on the public sets, and stays flat on the custom validation set (.586-.604 from step 11000) while E19 reaches .617.

**Conclusion:** no better than the transformers (level on GuitarSet/GAPS/SynthTab, ~1.5-3 points lower on steel-string) and not lighter where it matters: fewer parameters (4.0M) but slower to train and ~9x slower per window at inference (sequential recurrence; also poorly supported by the browser backends). Stopped at step 18000 to save time; run deleted, no test-set evaluation.

#### E22 — dilated TCN (`--size tcn`) — 2026-10-19: stopped at step 10000, 3-14 points below the transformers

- Same recipe (`SIZE=tcn experiments/train_light_all.sh`). 0.63 s/step alone; from step 1000 it shared the GPU with E23 (tiny): 0.88 s/step for the TCN, 1.38 s/step for tiny (~14% more total throughput than running them in sequence).
- Validation at the same steps (string/fret F1 unless noted), E22 / E19 / E20:

| Step | clean val | clean val pitch | degraded val | GAPS pitch | GAPS string agree | custom val | SynthTab val |
|---|---|---|---|---|---|---|---|
| 3000 | .680 / .628 / .625 | .782 / .773 / .767 | .532 / .521 / .512 | .761 / .785 / .784 | .772 / .777 / .780 | .492 / .519 / .491 | .808 / .825 / .828 |
| 6000 | .712 / .755 / .742 | .796 / .822 / .814 | .524 / .526 / .551 | .794 / .840 / .841 | .816 / .827 / .833 | .529 / .579 / .572 | .839 / .856 / .856 |
| 8000 | .754 / .783 / .785 | .800 / .826 / .826 | .523 / .588 / .590 | .794 / .855 / .844 | .812 / .829 / .825 | .538 / .597 / .584 | .852 / .856 / .857 |
| 10000 | .761 / .793 / .795 | .809 / .837 / .831 | .482 / .621 / .606 | .774 / .874 / .853 | .844 / .849 / .847 | .527 / .595 / .581 | .863 / .879 / .887 |

- Ahead only at step 3000; from step 4000 behind everywhere except GAPS string agreement, and flat from step 8000 (clean .754-.763, custom .525-.541) while the transformers kept improving. Weakest on degraded audio and GAPS pitch.

**Conclusion:** the fixed +-3 s dilated-convolution context is clearly worse than attention (or the GRU) here: ~3 points lower on clean GuitarSet, ~5-7 on steel-string, ~8-14 on degraded audio and GAPS pitch, for a small inference gain (12 vs 16 ms per window for E20). Stopped at step 10000; no test-set evaluation.

#### E23 — tiny pitch-axis model (`--size tiny`, 0.105M params) — 2026-10-19: stopped at step 12000, pitch level, string/fret ~11-13 points lower

- Same recipe (`SIZE=tiny experiments/train_light_all.sh`). Shared the GPU with E22 up to step ~5500 (1.38 s/step), alone afterwards.
- Validation at the same steps (string/fret F1 unless noted), E23 / E19 / E20:

| Step | clean val | clean val pitch | degraded val | GAPS pitch | GAPS string agree | custom val | SynthTab val |
|---|---|---|---|---|---|---|---|
| 3000 | .601 / .628 / .625 | .797 / .773 / .767 | .356 / .521 / .512 | .743 / .785 / .784 | .763 / .777 / .780 | .460 / .519 / .491 | .778 / .825 / .828 |
| 6000 | .630 / .755 / .742 | .799 / .822 / .814 | .336 / .526 / .551 | .758 / .840 / .841 | .779 / .827 / .833 | .501 / .579 / .572 | .807 / .856 / .856 |
| 9000 | .689 / .783 / .781 | .822 / .828 / .822 | .396 / .552 / .507 | .790 / .869 / .848 | .786 / .843 / .854 | .521 / .581 / .570 | .828 / .885 / .884 |
| 12000 | .682 / .807 / .808 | .819 / .837 / .834 | .428 / .616 / .602 | .766 / .879 / .873 | .794 / .848 / .846 | .515 / .605 / .601 | .836 / .906 / .901 |

- Clean pitch stays within ~1-2 points of the transformers throughout (best at step 3000, ahead by ~2.5); string/fret stalls around .65-.69 on clean GuitarSet and .50-.53 on the custom set from step 8000, degraded audio around .40-.43.

**Conclusion:** the pitch-equivariant front end with 0.1M parameters detects notes nearly as well as the 6-24M models, but choosing the string needs more capacity/context than the per-string 20-semitone fret windows give (~12 points lower string/fret on clean GuitarSet, ~9 on steel-string, ~18 on degraded audio). A possible follow-up: this tiny front end with a larger string/fret head. Stopped at step 12000; run deleted, no test-set evaluation.

**Light-model summary:** E20 (small transformer, 6.2M) remains the light candidate; GRU (level but slower), TCN (3-14 points lower) and tiny (string/fret ~12 points lower) are not worth pursuing as tab models.

### E24 — Distillation: half-size student (2.9M) taught by the E18 + E19 average (`runs/tabmodel_half_distill`) — 2026-10-20: E20 quality at half the size

**Teacher check (validation, no training):** averaging the tab/onset probabilities of E18 and E19 beats either alone where it matters: clean string/fret .823 (E19 .818, E18 .815), degraded .662 (E19 .627, E18 .672), GAPS pitch / agreement .888 / .874 (E19 .885 / .864), custom-dataset .612 (E19 .617, E18 .587).

**Setup:** the student is `--size half` (`guitab/model.py`): CNN 48/96/192, transformer d_model 160, 4 layers, 2.94M parameters, 10.4 ms per 5 s window on MPS (E20 6.19M / 15.8 ms, E19 23.8M / 39 ms). With `scripts/train.py --teacher A --teacher B --distill 0.5` the frozen teachers run on every (augmented) batch, and the loss is 0.5 × the label loss + 0.5 × a soft loss (soft cross-entropy on the tab, BCE on onsets and pitches against the averaged teacher probabilities, on every frame). Otherwise it is E19's recipe (`experiments/train_distill.sh`), 24k steps at 1.5 s/step (the E18 teacher dominates), about 10.5 h. Best checkpoint (selection on the default thresholds): step 23000.

**Calibration:** the student's onset probabilities are lower than its teachers' (soft onset targets spread over neighbouring frames, without the label loss's positive weight), so the default onset threshold .6 under-counts notes. Its best threshold on validation is ~.45 early and .5 at the end (same on every set); from .6 to .5 at the end: clean .815 → .811, degraded .633 → .654, GAPS pitch .871 → .887, custom .597 → .601. E24 is scored with onset .5.

**Paired per-recording comparison** (test sets, onset ±50 ms, Wilcoxon signed-rank; E24 at onset .5):

| Set (recordings) | E19 | E20 | E24 | E24 vs E19 | E24 vs E20 |
|---|---|---|---|---|---|
| GuitarSet test, string+fret (60) | **.899** | .888 | .890 | wins 22/60, -.008, p=.028 | wins 26/60, +.002, p=.82 |
| GuitarSet test phone/room, string+fret (60) | .736 | .737 | **.740** | wins 26/60, +.004, p=.52 | wins 32/60, +.003, p=.17 |
| GAPS test (classical), pitch (19) | .813 | .808 | **.825** | wins 16/19, +.012, p<.001 | wins 17/19, +.016, p<.001 |
| custom-dataset held-out, string+fret (19) | **.617** | .611 | .601 | wins 5/19, -.016, p=.09 | wins 7/19, -.010, p=.18 |

Model time over all four sets (predict only, MPS): E19 66 s, E20 41 s, E24 28 s.

**Conclusion:** distillation is worth about a doubling of the model: the 2.9M student matches E20 (6.2M) on GuitarSet clean and phone/room and the custom dataset (differences not significant), and beats both E20 and E19 on GAPS (+1.2-1.6 points, significant, the teachers' average carrying E18's broader public-data skill). Against E19 it keeps the ~1 point loss on clean GuitarSet (p=.03) and is 1.6 points lower on the custom dataset (p=.09). It needs its own onset threshold (.5). Candidate for the browser version (smallest download, fastest), not yet in the app. Next: the same distillation into an E20-sized student, expected to reach E19 on GuitarSet / custom dataset at a quarter of its size.

### EGDB electric guitar check (no training) — 2026-10-20

Do the acoustic models transcribe electric guitar? EGDB (Chen et al., "Towards automatic transcription of polyphonic electric guitar music: a new dataset and a multi-loss transformer model", ICASSP 2022; https://ss12f32v.github.io/Guitar-Transcription/): 240 clips recorded with a hexaphonic pickup, the clean DI signal re-rendered through 5 amplifiers, labels as one MIDI track per string. Evaluated: the official test clips 216-240 (25 clips, ~12 min) as clean DI and through the two test amps (Fender JC Jazz clean, Plexi crunch), plus the 5 real recordings (`RealData`, pitch-only labels named `*_pred`, 190 notes). Data in `data/egdb` (not versioned, 378 MB); script kept out of the repo (scratch). Onset ±50 ms; string+fret assumes standard tuning (one label note below the open string dropped).

| Set | E15 string+fret / pitch | E19 | E20 | E24 (onset .5) |
|---|---|---|---|---|
| EGDB test, clean DI (25) | .636 / .833 | **.712 / .850** | .712 / .839 | .691 / .830 |
| EGDB test, JC Jazz amp (25) | .618 / .789 | .645 / **.800** | **.656** / .787 | .629 / .783 |
| EGDB test, Plexi amp (25) | .608 / **.779** | **.616** / .774 | .615 / .764 | .615 / .778 |
| EGDB real recordings (5), pitch | **.891** | .852 | .857 | .871 |

**Reading:** none of the models has seen an electric guitar, yet pitch F1 is .83-.85 on clean DI (GuitarSet test: .87) and ~.78 through amps, and .85-.89 on the 5 real recordings. String choice transfers less well: .64-.71 on clean DI (GuitarSet .84-.90), ~.62 through amps: electric necks, lighter strings and amp tone change the timbre cues the string decision relies on, and EGDB's rock/pop vocabulary (power chords, bends) differs from fingerstyle. Usable as a first draft for electric guitar (clean or crunch), with more string corrections to expect; no high-gain distortion was tested.

### E25 — Distillation into an E20-sized student (6.2M, `runs/tabmodel_small_distill`) — 2026-10-20 → 10-22: the new default, best on phone/room audio and GAPS

**Setup:** E24's recipe with `--size small` (`SIZE=small experiments/train_distill.sh`), teachers E18 + E19 averaged, 24k steps, 1.7 s/step. The first attempt ran out of GPU memory at step 2100 (the E18 teacher's CNN ran while the student's activations were held for backward); `train.py` now computes the teacher targets before the student's forward pass (same targets, lower peak) and the run resumed from step 2000. Paused and resumed twice (steps 5000 and 22000).

**Checkpoint and threshold (validation only):** `best.pt` (default-threshold selection) is step 16000, picked by a spike on degraded audio; the final step 24000 is better on clean GuitarSet and the custom set. Onset-threshold sweep, mean of clean string/fret, degraded string/fret, GAPS pitch and custom string/fret: step 16000 .7445, **step 24000 .7494** (both at onset .5, the same calibration offset as E24). Selected: `last.pt` (step 24000) at onset .5; validation clean .820, degraded .669, GAPS pitch .896, custom .614 (E19 at its best step: .818 / .627 / .885 / .617; teacher average .823 / .662 / .888 / .612). Caveat (noted 2026-10-21): the custom-dataset validation set is the same 19 songs as the custom "held-out" test row below, and it was part of this selection mean (and of E24's threshold choice), so E24 / E25's custom-dataset scores are slightly optimistic; GuitarSet test, its degraded copy and GAPS test were never used for any choice. Redone without it on 2026-10-22 (see "E24 / E25 re-selection" below): E25's threshold becomes .45 and its final numbers are those.

**Paired per-recording comparison** (test sets, onset ±50 ms, Wilcoxon signed-rank; E24 and E25 at onset .5):

| Set (recordings) | E19 (24M) | E20 (6.2M) | E24 (2.9M) | E25 (6.2M) | E25 vs E19 | E25 vs E20 | E25 vs E24 |
|---|---|---|---|---|---|---|---|
| GuitarSet test, string+fret (60) | **.899** | .888 | .890 | .895 | wins 25/60, -.003, p=.30 | wins 35/60, +.007, p=.16 | wins 32/60, +.005, p=.39 |
| GuitarSet test phone/room, string+fret (60) | .736 | .737 | .740 | **.761** | wins 35/60, +.024, p=.03 | wins 45/60, +.024, p<.001 | wins 37/60, +.021, p=.02 |
| GAPS test (classical), pitch (19) | .813 | .808 | .825 | **.826** | wins 17/19, +.013, p=.001 | wins 17/19, +.018, p=.001 | wins 15/19, +.001, p=.10 |
| custom-dataset held-out, string+fret (19) | **.617** | .611 | .601 | .614 | wins 12/19, -.003, p=.92 | wins 12/19, +.002, p=.13 | wins 13/19, +.013, p=.04 |

Model time over all four sets (predict only, MPS): E19 65 s, E20 40 s, E24 28 s, E25 40 s. (`best.pt`, step 16000, for reference: .886 / .763 / .823 / .604; selected against on validation.)

**EGDB electric guitar** (string+fret / pitch; see the EGDB entry): clean DI .721 / .845, JC Jazz .666 / .800, Plexi .646 / .782, real recordings pitch .880: the best string+fret of all models on every rendering.

**Conclusion:** the best model of the project. At a quarter of E19's parameters and 1.6x its speed, E25 is not significantly different from E19 on clean GuitarSet and the custom dataset (the two sets E19 won) and significantly better on phone/room audio (+2.4 points) and GAPS (+1.3); it matches the E18 + E19 teacher average it learned from. Over E24 (half its size) it gains on phone/room audio and the custom set (+2.1 and +1.3, significant) and ties elsewhere. Needs onset threshold .5. (Superseded by the re-selection below: onset .45 and the final numbers.)

**Adopted (2026-10-21): E25 is the default model.** `guitab/paths.py`: `DEFAULT_CHECKPOINT` = E25 when trained, else E15 (public users can only train E15: E25 needs the custom dataset and the E18 / E19 teachers); E25 and E24 added to the web app's models. Per-model decoding: a checkpoint may carry `"decoding"` (note-extraction settings over `DEFAULT_PARAMS`), which `load_model` exposes as `model.decoding` and the CLI, pipeline, tab-LM decoding, web app (note-sensitivity starts at the model's onset) and evaluation scripts apply under any explicit setting; E24 and E25 store `{"onset": 0.5}` (E25 later .45, see the re-selection). README protocol (scripts/evaluate.py, evaluate_gaps.py): GuitarSet test .843 / .865 (E19 .843 / .866, reproduced), degraded .680 / .750 (E19 .640 / .721), GAPS pitch .826 (E19 .813). Clean-up: the kept checkpoints (E15, E18, E19, E20, E24, E25) are weights-only `best.pt` files (identical predictions; E25's is the step-24000 checkpoint); the `last.pt` files and the intermediate runs (E8, E10, E12, E13 and E17) are deleted (runs/ 2.3 GB -> 0.5 GB).

### Weights release — 2026-10-21

Published on Hugging Face (one flat repo, uploaded by hand): `guitab_e25.pt` (E25, the default), `guitab_e24.pt` (E24, the tiny model: 2.9M parameters, 11 MB, 2.3x faster than E19, about one point behind on GuitarSet), `guitab_e19.pt` (E19, the strongest model trained on labels alone and E25's teacher) and `tablm.pt` (tab LM), licensed CC BY-NC-SA 4.0 like the non-commercial training data (GAPS, SynthTab); the code stays MIT. The other checkpoints stay local: E18 (2.5x slower, worse on clean audio), E20 (superseded by E25) and E15 (reproducible from the public data with the README's training commands). README-protocol numbers of E24: GuitarSet test .833 / .856, degraded .655 / .737, GAPS pitch .825. `guitab/paths.py`: `checkpoint(run, published)` returns `runs/<run>/best.pt` when trained locally, else `weights/<published file>`; `weights/` (downloads) and `huggingface/` (upload staging) are git-ignored. The custom dataset itself is still not distributed.

### E24 / E25 re-selection without the custom held-out songs — 2026-10-22

The custom-dataset validation songs are also the "held-out" test row, so the checkpoint and onset threshold of the distilled models are now chosen on GuitarSet validation (clean and degraded string+fret) and GAPS validation (pitch) only, the sets that are never used as tests (mean of the three; a scratch script, as `scripts/evaluate.py --tune` only covers clean GuitarSet):

| Onset threshold | .6 | .55 | .5 | .45 | .4 |
|---|---|---|---|---|---|
| E24 (best.pt, step 23000) | .77284 | .77984 | **.78391** | .78374 | .78255 |
| E25 (step 24000) | .78576 | .79184 | .79462 | **.79479** | .79359 |

- E24: unchanged (training selection step 23000, onset .5), so its results stand and are clean.
- E25: step 24000 still beats step 16000 (.7946 vs .7913 at .5); the threshold moves to **.45** (by .0002, a tie in practice, but the rule decides). Stored in the checkpoint (`"decoding": {"onset": 0.45}`).

E25 at onset .45, test sets (paired, full tracks, onset ±50 ms):

| Set (recordings) | E19 | E20 | E24 | E25 | E25 vs E19 | E25 vs E24 |
|---|---|---|---|---|---|---|
| GuitarSet test, string+fret (60) | **.899** | .888 | .890 | .891 | wins 22/60, -.008, p=.046 | wins 31/60, +.001, p=.92 |
| GuitarSet test phone/room (60) | .736 | .737 | .740 | **.764** | wins 35/60, +.028, p=.016 | wins 38/60, +.025, p=.004 |
| GAPS test, pitch (19) | .813 | .808 | .825 | **.830** | wins 17/19, +.017, p=.001 | wins 14/19, +.005, p=.04 |
| custom-dataset held-out (19) | **.617** | .611 | .601 | .612 | wins 9/19, -.004, p=.80 | wins 13/19, +.011, p=.09 |

Model time over the four sets: E25 43 s, E24 29 s, E19 66 s, E20 41 s, E18 151 s, E15 66 s. README protocol (chunks): GuitarSet test .838 / .861, degraded .685 / .757, GAPS pitch .830; EGDB clean DI .721 / .849, JC Jazz .664 / .799, Plexi .639 / .777, real recordings pitch .877.

**Conclusion (final):** with a selection that never touches a test set, E25 is 0.8 points below E19 on clean GuitarSet (just significant) and level on the custom songs, and clearly better on phone/room audio (+2.8) and GAPS (+1.7), at a quarter of the size and 1.5x the speed. It stays the default: the project's target is real recordings, mostly phone ones. E24 stays the tiny alternative.

### Browser version — 2026-10-17 → 10-22

`docs/` runs the whole pipeline in the browser: ONNX Runtime Web 1.22 (WebGPU, else WebAssembly) for the tab model and the Beat This! beat tracker (WebAssembly only: its WebGPU output differed, 226 vs 200 beats on a test song), and JavaScript ports of the CQT (librosa's sparse filter basis plus an FIR filter matching the soxr_hq halving between octaves), the note extraction, the beat grid, the fingering and the .gp5 writer (`docs/js/`). `scripts/export_web.py` exports the models with fixed input shapes (432 frames for the tab model, 1500 for the beats), checks them against PyTorch and writes `docs/models.json` and `docs/assets/dsp.json`.

The page offers E25, E19 and E24 (20, 84 and 9 MB). Each model is downloaded on its first use and kept in the browser cache; `models.json` carries each model's own decoding settings (E25 onset .45, E24 .5), which set the note-sensitivity slider. For GitHub Pages the .onnx files are fetched from the Hugging Face model repository (`export_web.py --base ...`), next to the .pt weights.

Checks: the JavaScript parts are identical to Python on unit cases (mel spectrogram, beat post-processing, note extraction, 24 / 24 token cases, 24 / 24 .gp5 files; CQT mean difference 9e-6). End to end, the browser's .gp5 is byte-identical to the Python pipeline's (see the beat tracker below). Audio at other sample rates is resampled by the browser rather than by librosa, so its results are close but not byte-identical.

**One beat tracker for both versions (2026-10-22): Beat This! final0 with int8 weights instead of small0, in Python too.** Beat F-measure (±70 ms) on the 60 GuitarSet test tracks against the JAMS beat annotations, each clip padded to the model's fixed 30 s input: small0 .683, final0 float32 .743, final0 with int8 weights .743 (mean difference to float32 .0005, at most .029 on a track). The weights are stored as int8 with a per-output-channel scale and a `DequantizeLinear` in front of them (`export_web.int8_weights`), so the arithmetic stays float32: 21 MB instead of 79 MB.

- Dynamic quantisation (`onnxruntime.quantization.quantize_dynamic`) was tried first: ONNX Runtime Web has no `ConvInteger`, and with `MatMul` only it rounds the activations to 8 bits, so the ~1e-6 float differences between the browser and Python moved some beats by a frame (194 vs 193 beats on the 2:23 song) and an extra weak beat shifted the bar grid from bar 27 on.
- Native onnxruntime's extended graph optimisations fuse the int8 weights into integer matrix products (max logit difference to the browser .13); with basic optimisations only, the two agree to 6e-6. `guitab.tab.detect_beats` now runs the same ONNX file (`weights/` or `docs/models/`) with basic optimisations, padding every chunk to 30 s, and both versions round the beat logits to 0.01 (float64, half up) before picking peaks. Without the ONNX file, Python falls back to the PyTorch final0, padded the same way.
- Result: the browser's .gp5 is byte-identical to the Python pipeline's for E25, E24 and E19 on the 2:23 song, and for E25 and E24 on a GAPS piece (110 s), a custom-dataset song (180 s) and an EGDB clip through the Plexi amp (20 s). A 2:23 song takes 22 s in the browser instead of 13 s with small0 (the larger beat tracker runs on WebAssembly). In PyTorch without padding, final0 scored .693: the GuitarSet clips are shorter than 30 s, and the padding helps.

**E25's re-attack merge (D3):** a false re-attack of a ringing open E4 at the start of my song survives at every onset threshold (onset probability .82 ≥ the D3 threshold .75). A higher D3 threshold costs E25 on validation (mean of GuitarSet clean / degraded string+fret and GAPS pitch: .7948 at .75, .7934 at .8, .7894 at .85, .7716 at .95), so it stays .75.

### Published weights as safetensors — 2026-10-22

The Hugging Face files are `.safetensors` instead of `.pt`: a `.pt` file is a pickle, which `torch.load` may execute, while safetensors holds only tensors. `scripts/export_weights.py` writes `guitab_e25`, `guitab_e24`, `guitab_e19` and `tablm.safetensors`, with the size, training step and decoding settings (E25 onset .45, E24 .5) as JSON metadata. `guitab.inference.read_checkpoint` reads both formats (`.pt` with `weights_only=True`, so locally trained checkpoints load without running code), and `load_model`, `checkpoint_decoding`, `load_tablm` and `train.py --init` use it. Checked: each file gives the same outputs as its `.pt` checkpoint, and a fresh clone with only the published files writes byte-identical .gp5 files for E25, E24, E19 and E25 with `--music_lm`.
