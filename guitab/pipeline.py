"""Audio file -> Guitar Pro tab: the steps shared by the `guitab` command and the web app (guitab/app.py)."""

from pathlib import Path

from .chords import suggest_chords
from .device import pick_device
from .fingering import FINGERING_PRESETS
from .gp5 import write_gp5
from .inference import apply_capo, transcribe_file
from .paths import DEFAULT_CHECKPOINT
from .tab import BeatGrid, detect_beats, notes_to_tokens


def transcribe_to_tab(audio_path, out_dir, stem=None, checkpoint=DEFAULT_CHECKPOINT, device="auto", params=None,
                      music_lm=None, capo=0, tempo="auto", fingering="model", grid=240, write_tokens=True, log=print,
                      chords=False):
    """Write <out_dir>/<stem>.gp5 (and the tab as text tokens, <stem>.tokens.txt, if write_tokens). params: note-extraction settings (inference.DEFAULT_PARAMS keys);
    music_lm: None or tab-LM decoding settings (lam, beta); tempo: "auto" (beat tracking) or BPM; fingering: model / auto /
    low; chords: chord names above the parts where the chord model is confident (guitab.chords; none without its
    weights or a connection for their first download). Returns {"gp5", "tokens", "notes", "tempo", "beats_per_measure"}; raises ValueError if no note was found."""
    audio_path = Path(audio_path).resolve()
    device = str(pick_device(device))
    notes, onset = transcribe_file(str(checkpoint), audio_path, device, params or {}, music_lm, with_onset=True)
    notes = apply_capo(notes, capo)
    log(f"{len(notes)} notes")
    if tempo != "auto":
        beat_grid = BeatGrid([], tempo=int(tempo))
        downbeats = []
    else:
        beats, downbeats = detect_beats(audio_path, device, onset)
        beat_grid = BeatGrid(beats, downbeats)
    log(f"tempo {beat_grid.tempo} bpm, {beat_grid.num_beats}/4")
    weights = None if fingering == "model" else FINGERING_PRESETS[fingering]
    stem = stem or audio_path.stem
    chord_list = suggest_chords(notes, beat_grid.beats, downbeats, capo) if chords else None
    if chord_list:
        log(f"{len(chord_list)} chord names")
    tokens = notes_to_tokens(notes, beat_grid, stem, grid, weights, capo=capo, chords=chord_list)
    if tokens is None:
        raise ValueError("No notes were transcribed.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = out_dir / f"{stem}.tokens.txt" if write_tokens else None
    if write_tokens:
        tokens_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    gp5_path = out_dir / f"{stem}.gp5"
    write_gp5(tokens, gp5_path, capo)
    log(f"-> {gp5_path}")
    return {"gp5": gp5_path, "tokens": tokens_path, "notes": len(notes), "tempo": beat_grid.tempo,
            "beats_per_measure": beat_grid.num_beats}
