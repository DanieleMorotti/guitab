"""Write tab tokens (tab.build_tokens) as a Guitar Pro 5 file with pyguitarpro.

Tokens: the song name, header tokens (downtune:, tempo:, start), then per beat its content - "note:s<string>:f<fret>"
(optionally followed by "nfx:tie") or "rest" - and one "wait:<ticks>" giving the beat's duration (960 ticks per
quarter); "new_measure" closes a measure and "end" the song. A measure's time signature is its length in quarters
(a shorter last measure in eighths, sixteenths or 32nds). One standard-tuned 6-string track, capo optional.
"""

import guitarpro as gp

from .guitar import OPEN_PITCH

QUARTER = 960
# beat length in ticks -> (Guitar Pro duration value, dotted)
DURATIONS = {3840: (1, False), 2880: (2, True), 1920: (2, False), 1440: (4, True), 960: (4, False), 720: (8, True),
             480: (8, False), 360: (16, True), 240: (16, False), 120: (32, False)}


def parse_measures(tokens):
    """[[(ticks, [(string, fret, tied)] or None for a rest)]] per measure, and the tempo."""
    tempo = next((int(t.split(":")[1]) for t in tokens if t.startswith("tempo:")), 120)
    measures, beats, content = [], [], []
    for t in tokens[tokens.index("start") + 1:]:
        if t.startswith("note:"):
            _, s, f = t.split(":")
            content.append([int(s[1:]), int(f[1:]), False])
        elif t == "nfx:tie":
            content[-1][2] = True
        elif t == "rest":
            content = None
        elif t.startswith("wait:"):
            beats.append((int(t.split(":")[1]), [tuple(n) for n in content] if content is not None else None))
            content = []
        elif t in ("new_measure", "end") and beats:
            measures.append(beats)
            beats = []
    return measures, tempo


def time_signature(ticks):
    for denominator in (4, 8, 16, 32):
        unit = QUARTER * 4 // denominator
        if ticks % unit == 0:
            return ticks // unit, denominator
    raise ValueError(f"measure of {ticks} ticks is not on the 32nd grid")


def write_gp5(tokens, path, capo=0):
    measures, tempo = parse_measures(tokens)
    song = gp.Song(title=tokens[0], tempo=tempo)
    track = song.tracks[0]
    track.name = "Guitar"
    track.offset = capo  # the Guitar Pro capo
    track.channel.instrument = 25  # acoustic guitar (steel)
    track.strings = [gp.GuitarString(s, OPEN_PITCH[s]) for s in sorted(OPEN_PITCH)]
    song.measureHeaders, track.measures = [], []
    start = QUARTER
    for number, beats in enumerate(measures, 1):
        numerator, denominator = time_signature(sum(ticks for ticks, _ in beats))
        header = gp.MeasureHeader(number=number, start=start,
                                  timeSignature=gp.TimeSignature(numerator, gp.Duration(value=denominator)))
        song.addMeasureHeader(header)
        measure = gp.Measure(track, header)
        voice = measure.voices[0]
        for ticks, notes in beats:
            value, dotted = DURATIONS[ticks]
            beat = gp.Beat(voice, duration=gp.Duration(value=value, isDotted=dotted),
                           status=gp.BeatStatus.rest if notes is None else gp.BeatStatus.normal)
            for string, fret, tied in notes or []:
                beat.notes.append(gp.Note(beat, value=fret, string=string,
                                          type=gp.NoteType.tie if tied else gp.NoteType.normal))
            voice.beats.append(beat)
        track.measures.append(measure)
        start += header.length
    gp.write(song, str(path), version=(5, 1, 0))
