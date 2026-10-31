// Tab tokens -> Guitar Pro 5 file bytes: port of guitab/gp5.py. The constant parts of the file (pyguitarpro's defaults:
// page setup, MIDI channels, track settings) come from assets/dsp.json ("gp5", written by scripts/export_web.py); this
// writes the title, tempo, capo, measure headers and beats in between, byte for byte as pyguitarpro does.

const QUARTER = 960;
// beat length in ticks -> [Guitar Pro duration value, dotted]
const DURATIONS = { 3840: [1, false], 2880: [2, true], 1920: [2, false], 1440: [4, true], 960: [4, false],
  720: [8, true], 480: [8, false], 360: [16, true], 240: [16, false], 120: [32, false] };

// [[[ticks, [[string, fret, tied], ...] or null for a rest, chord name or null], ...] per measure], tempo
export function parseMeasures(tokens) {
  const tempoToken = tokens.find((t) => t.startsWith("tempo:"));
  const tempo = tempoToken ? parseInt(tempoToken.split(":")[1], 10) : 120;
  const measures = [];
  let beats = [], content = [], chord = null;
  for (const t of tokens.slice(tokens.indexOf("start") + 1)) {
    if (t.startsWith("chord:")) chord = t.slice(6);
    else if (t.startsWith("note:")) {
      const [, s, f] = t.split(":");
      content.push([parseInt(s.slice(1), 10), parseInt(f.slice(1), 10), false]);
    } else if (t === "nfx:tie") content[content.length - 1][2] = true;
    else if (t === "rest") content = null;
    else if (t.startsWith("wait:")) {
      beats.push([parseInt(t.split(":")[1], 10), content !== null ? content.map((n) => n.slice()) : null, chord]);
      content = [];
      chord = null;
    } else if ((t === "new_measure" || t === "end") && beats.length) {
      measures.push(beats);
      beats = [];
    }
  }
  return [measures, tempo];
}

function timeSignature(ticks) {
  for (const denominator of [4, 8, 16, 32]) {
    const unit = (QUARTER * 4) / denominator;
    if (ticks % unit === 0) return [ticks / unit, denominator];
  }
  throw new Error(`measure of ${ticks} ticks is not on the 32nd grid`);
}

class Writer {
  constructor() { this.bytes = []; }
  byte(v) { this.bytes.push(v & 0xff); }
  short(v) { this.byte(v); this.byte(v >> 8); }
  int(v) { for (let i = 0; i < 4; i++) this.byte(v >> (8 * i)); }
  raw(hex) { for (let i = 0; i < hex.length; i += 2) this.bytes.push(parseInt(hex.slice(i, i + 2), 16)); }
  intByteSizeString(s) {
    const chars = [...s].map((c) => (c.charCodeAt(0) < 256 ? c.charCodeAt(0) : 63)); // cp1252-ish, "?" otherwise
    this.int(chars.length + 1);
    this.byte(chars.length);
    chars.forEach((c) => this.byte(c));
  }
  patchInt(offset, v) { for (let i = 0; i < 4; i++) this.bytes[offset + i] = (v >> (8 * i)) & 0xff; }
  // a chord name without diagram, as pyguitarpro writes gp.Chord(length=6, name=..., sharp=False, add=False,
  // firstFret=0, show=False, strings=[-1] * 6) (GP4 chord format)
  chord(name) {
    const chars = [...name].map((c) => (c.charCodeAt(0) < 256 ? c.charCodeAt(0) : 63));
    this.byte(1); // GP4 chord format
    this.byte(0); // sharp: no
    for (let i = 0; i < 6; i++) this.byte(0); // placeholder (3), root, type, extension
    this.int(0); this.int(0); // bass, tonality
    this.byte(0); // add: no
    this.byte(chars.length);
    for (let i = 0; i < 22; i++) this.byte(i < chars.length ? chars[i] : 0); // name, 22 bytes
    for (let i = 0; i < 3; i++) this.byte(0); // fifth, ninth, eleventh
    this.int(0); // first fret
    for (let i = 0; i < 7; i++) this.int(-1); // strings: not played
    this.byte(0); // barres: none
    for (let i = 0; i < 15; i++) this.byte(0); // barre frets, starts, ends
    for (let i = 0; i < 7; i++) this.byte(1); // omissions
    this.byte(0); // placeholder
    for (let i = 0; i < 7; i++) this.byte(0xfe); // fingerings: unknown (-2)
    this.byte(0); // show diagram: no
  }
}

export function writeGp5(tokens, template, capo = 0) {
  const [measures, tempo] = parseMeasures(tokens);
  const w = new Writer();
  w.raw(template.head1);
  w.intByteSizeString(tokens[0]); // title, then subtitle, artist, album, words, music, copyright, tab, instructions
  for (let i = 0; i < 8; i++) w.intByteSizeString("");
  w.int(0); // notice lines
  const head2 = w.bytes.length;
  w.raw(template.head2);
  w.patchInt(head2 + template.tempo_at, tempo);
  w.int(measures.length);
  w.int(1); // tracks

  let previous = null;
  for (const beats of measures) {
    const [num, den] = timeSignature(beats.reduce((a, [ticks]) => a + ticks, 0));
    let flags = 0;
    if (previous === null) flags = 0x01 | 0x02 | 0x40; // first header: time and key signature
    else {
      w.byte(0); // placeholder before every header but the first
      if (num !== previous[0]) flags |= 0x01;
      if (den !== previous[1]) flags |= 0x02;
    }
    w.byte(flags);
    if (flags & 0x01) w.byte(num);
    if (flags & 0x02) w.byte(den);
    if (flags & 0x40) { w.byte(0); w.byte(0); } // C major
    if (flags & 0x03) [2, 2, 2, 2].forEach((b) => w.byte(b)); // beams
    w.byte(0); // no repeat alternative
    w.byte(0); // triplet feel: none
    previous = [num, den];
  }

  const track = w.bytes.length;
  w.raw(template.track);
  w.patchInt(track + template.capo_at, capo);

  for (const beats of measures) {
    w.int(beats.length);
    for (const [ticks, notes, chord] of beats) {
      const [value, dotted] = DURATIONS[ticks];
      w.byte((dotted ? 0x01 : 0) | (chord ? 0x02 : 0) | (notes === null ? 0x40 : 0));
      if (notes === null) w.byte(2); // rest
      w.byte(Math.log2(value) - 2); // duration: whole -2 ... 32nd 3
      if (chord) w.chord(chord);
      const sorted = (notes || []).slice().sort((a, b) => a[0] - b[0]);
      w.byte(sorted.reduce((f, [s]) => f | (1 << (7 - s)), 0));
      for (const [, fret, tied] of sorted) {
        w.byte(0x20);
        w.byte(tied ? 2 : 1); // note type: normal / tie
        w.byte(tied ? 0 : fret);
        w.byte(0);
      }
      w.short(0); // beat display flags
    }
    w.int(0); // second voice: empty
    w.byte(0); // line break: none
  }
  return new Uint8Array(w.bytes);
}
