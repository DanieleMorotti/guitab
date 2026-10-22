// Notes in seconds -> quantized tab tokens following the beats: port of guitab/tab.py (BeatGrid, notes_to_tokens,
// build_tokens) and guitab/fingering.py (refinger_tokens).

import { OPEN_PITCH, roundHalfEven } from "./guitar.js";

const TICKS_PER_BEAT = 960;
const GRID = 120;
const MIN_REST_TICKS = 960;
const SUPPORTED_WAITS = [3840, 2880, 1920, 1440, 960, 720, 480, 360, 240, 120];

function median(values) {
  const v = values.slice().sort((a, b) => a - b), m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// numpy.interp(t, xs, arange(len(xs))) for increasing xs, t inside [xs[0], xs[-1]].
function interpIndex(t, xs) {
  let lo = 0, hi = xs.length - 1;
  if (t >= xs[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= t) lo = mid; else hi = mid;
  }
  return lo + (t - xs[lo]) / (xs[lo + 1] - xs[lo]);
}

export function beatsPerMeasure(beats, downbeats) {
  if (downbeats.length < 2) return 4;
  const idx = downbeats.map((d) => {
    let i = 0;
    while (i < beats.length && beats[i] < d - 1e-3) i++;
    return i;
  });
  const counts = new Array(8).fill(0); // np.bincount(..., minlength=8)[:8]
  for (let i = 1; i < idx.length; i++) {
    const d = idx[i] - idx[i - 1];
    if (d < 8) counts[d] += 1;
  }
  const total = counts.reduce((a, b) => a + b, 0);
  let mode = 0;
  for (let i = 1; i < 8; i++) if (counts[i] > counts[mode]) mode = i;
  // trust the detected meter only when one bar length clearly dominates
  return mode >= 2 && mode <= 7 && counts[mode] >= 0.6 * total ? mode : 4;
}

// Maps seconds to ticks through the beat sequence (linear extrapolation outside it).
export class BeatGrid {
  constructor(beats, downbeats = null, tempo = null) {
    if (beats.length < 2) {
      const period = 60 / (tempo || 120);
      beats = Array.from({ length: Math.ceil(3600 / period) }, (_, i) => i * period); // np.arange(0, 3600, period)
    }
    this.beats = beats;
    this.period = median(beats.slice(1).map((b, i) => b - beats[i]));
    this.tempo = roundHalfEven(60 / this.period);
    this.numBeats = downbeats !== null ? beatsPerMeasure(beats, downbeats) : 4;
    let firstDown = 0;
    if (downbeats !== null && downbeats.length) {
      let best = Infinity;
      beats.forEach((b, i) => {
        const d = Math.abs(b - downbeats[0]);
        if (d < best) { best = d; firstDown = i; }
      });
    }
    this.origin = firstDown % this.numBeats;
  }

  beatIndex(t) {
    const b = this.beats;
    let idx;
    if (t < b[0]) idx = (t - b[0]) / this.period;
    else if (t > b[b.length - 1]) idx = b.length - 1 + (t - b[b.length - 1]) / this.period;
    else idx = interpIndex(t, b);
    return idx - this.origin;
  }

  ticks(t, grid) {
    return roundHalfEven((this.beatIndex(t) * TICKS_PER_BEAT) / grid) * grid;
  }
}

function assignStrings(group, capo) {
  const used = new Set(group.filter((n) => n.string).map((n) => n.string));
  for (const n of group.slice().sort((a, b) => a.pitch - b.pitch)) {
    if (n.string) continue;
    const free = [6, 5, 4, 3, 2, 1].filter((s) => !used.has(s) && capo <= n.pitch - OPEN_PITCH[s] && n.pitch - OPEN_PITCH[s] <= 20);
    if (!free.length) { n.string = null; continue; }
    let s = free[0];
    for (const c of free) if (n.pitch - OPEN_PITCH[c] < n.pitch - OPEN_PITCH[s]) s = c;
    n.string = s;
    used.add(s);
  }
  return group.filter((n) => n.string);
}

export function notesToTokens(notes, grid, name = "song", gridTicks = 240, fingering = null, maxNotes = 6, capo = 0) {
  const groups = new Map();
  for (const n of notes) {
    const on = grid.ticks(n.onset, gridTicks);
    const off = Math.max(on + gridTicks, grid.ticks(n.offset, gridTicks));
    if (!groups.has(on)) groups.set(on, []);
    groups.get(on).push({ ...n, on, off });
  }
  const events = [];
  for (const on of [...groups.keys()].sort((a, b) => a - b)) {
    const best = new Map(); // duplicate pitches at one onset: keep the stronger
    for (const n of groups.get(on)) {
      if (!best.has(n.pitch) || (n.score ?? 0) > (best.get(n.pitch).score ?? 0)) best.set(n.pitch, n);
    }
    const group = [...best.values()].sort((a, b) => -(a.score ?? 0) - -(b.score ?? 0)).slice(0, maxNotes);
    const seen = new Set(), kept = []; // a string can sound once per onset
    for (let n of group) {
      if (n.string && seen.has(n.string)) n = { ...n, string: null };
      if (n.string) seen.add(n.string);
      kept.push(n);
    }
    events.push(...assignStrings(kept, capo));
  }
  if (!events.length) return null;
  const measure = grid.numBeats * TICKS_PER_BEAT;
  let shift = Math.min(...events.map((n) => n.on));
  shift = Math.floor(shift / measure) * measure; // keep barlines, drop leading empty measures
  const tabNotes = events.map((n) => [n.on - shift, n.off - shift, n.string, n.pitch - OPEN_PITCH[n.string] - capo]);
  let tokens = buildTokens(name, tabNotes, grid.tempo, grid.numBeats);
  if (fingering !== null) tokens = refingerTokens(tokens, fingering);
  return tokens;
}

function buildTokens(name, notes, tempo, numBeats) {
  const measureTicks = numBeats * TICKS_PER_BEAT;
  const byOnset = new Map();
  for (const [onset, offset, string, fret] of notes) {
    if (!byOnset.has(onset)) byOnset.set(onset, new Map());
    const group = byOnset.get(onset);
    if (!group.has(string) || offset > group.get(string)[1]) group.set(string, [fret, offset]);
  }
  const onsets = [...byOnset.keys()].sort((a, b) => a - b);
  const tokens = [name, "downtune:0", `tempo:${tempo}`, "start", "new_measure"];
  const sounding = new Map(); // string -> [fret, offset]
  let cursor = 0;
  const sortedKeys = (m) => [...m.keys()].sort((a, b) => a - b);

  // Waits from cursor to target as encodable note values, split at barlines; notes still sounding across a split are
  // re-emitted with nfx:tie.
  const advance = (target) => {
    while (cursor < target) {
      const nextBar = (Math.floor(cursor / measureTicks) + 1) * measureTicks;
      const remaining = Math.min(target, nextBar) - cursor;
      const step = SUPPORTED_WAITS.find((w) => w <= remaining);
      tokens.push(`wait:${step}`);
      cursor += step;
      for (const s of sortedKeys(sounding)) if (sounding.get(s)[1] <= cursor) sounding.delete(s);
      if (cursor === nextBar) tokens.push("new_measure");
      if (cursor < target) {
        if (sounding.size) {
          for (const s of sortedKeys(sounding)) tokens.push(`note:s${s}:f${sounding.get(s)[0]}`, "nfx:tie");
        } else tokens.push("rest");
      }
    }
  };
  const lastOff = () => Math.max(...[...sounding.values()].map(([, off]) => off));

  if (onsets.length && onsets[0] > 0) { tokens.push("rest"); advance(onsets[0]); }
  for (let i = 0; i < onsets.length; i++) {
    const onset = onsets[i];
    if (onset > cursor) advance(onset);
    const group = byOnset.get(onset);
    for (const s of sortedKeys(group)) {
      const [fret, offset] = group.get(s);
      tokens.push(`note:s${s}:f${fret}`);
      sounding.set(s, [fret, offset]);
    }
    const next = i + 1 < onsets.length ? onsets[i + 1] : null;
    if (next === null) {
      const last = sounding.size ? lastOff() : onset + GRID;
      advance(Math.max(last, onset + GRID));
      break;
    }
    const last = lastOff();
    if (next - last >= MIN_REST_TICKS) { advance(Math.max(last, onset + GRID)); tokens.push("rest"); }
  }
  if (tokens[tokens.length - 1] === "new_measure") tokens.pop();
  tokens.push("end");
  return tokens;
}

// ---- fingering (guitab/fingering.py) ----

const DEFAULT_WEIGHTS = { position: 0.25, high: 1.0, span: 0.6, move: 0.35, model: 0.8, max_fret: 20, max_span: 4 };
export const FINGERING_PRESETS = {
  auto: { position: 0.05, move: 0.35, model: 2.0 },
  low: { position: 0.6, move: 0.35, model: 0.3, high: 2.0 },
};
const NOTE = /^note:s([1-6]):f(\d+)$/;

function parseGroups(tokens) {
  const groups = [];
  let current = null;
  tokens.forEach((t, i) => {
    const m = NOTE.exec(t);
    if (m) {
      if (current === null) { current = { notes: [] }; groups.push(current); }
      const s = +m[1], f = +m[2];
      current.notes.push({ idx: i, string: s, fret: f, pitch: OPEN_PITCH[s] + f, tie: tokens[i + 1] === "nfx:tie" });
    } else if (t.startsWith("wait:") || t.startsWith("rest") || t.startsWith("new_measure")) current = null;
  });
  return groups;
}

const optionsForPitch = (pitch, maxFret) =>
  [6, 5, 4, 3, 2, 1].filter((s) => pitch - OPEN_PITCH[s] >= 0 && pitch - OPEN_PITCH[s] <= maxFret).map((s) => [s, pitch - OPEN_PITCH[s]]);

function handPosition(fingering) {
  const fretted = fingering.map(([, f]) => f).filter((f) => f > 0);
  return fretted.length ? Math.min(...fretted) : null;
}

function staticCost(combo, notes, w) {
  const fretted = combo.map(([, f]) => f).filter((f) => f > 0);
  const pos = fretted.length ? Math.min(...fretted) : 0;
  const span = fretted.length ? Math.max(...fretted) - Math.min(...fretted) : 0;
  const high = fretted.filter((f) => f > 12).length;
  const match = combo.filter(([s], i) => s === notes[i].string).length;
  return w.position * pos + w.span * span + w.high * high - w.model * match;
}

function* product(lists, prefix = []) {
  if (prefix.length === lists.length) { yield prefix; return; }
  for (const item of lists[prefix.length]) yield* product(lists, [...prefix, item]);
}

function candidates(notes, w, limit = 40) {
  const perNote = notes.map((n) => optionsForPitch(n.pitch, w.max_fret));
  const own = [notes.map((n) => [n.string, n.fret])];
  if (perNote.some((o) => !o.length)) return own;
  const combos = [];
  for (const combo of product(perNote)) {
    const strings = combo.map(([s]) => s);
    if (new Set(strings).size !== strings.length) continue;
    const fretted = combo.map(([, f]) => f).filter((f) => f > 0);
    const span = fretted.length ? Math.max(...fretted) - Math.min(...fretted) : 0;
    if (span > w.max_span) continue;
    combos.push(combo);
  }
  if (!combos.length) return own;
  const costs = new Map(combos.map((c) => [c, staticCost(c, notes, w)]));
  combos.sort((a, b) => costs.get(a) - costs.get(b));
  return combos.slice(0, limit);
}

export function refingerTokens(tokensIn, weights = null) {
  const w = { ...DEFAULT_WEIGHTS, ...(weights || {}) };
  const tokens = tokensIn.slice();
  const groups = parseGroups(tokens);
  const steps = groups
    .map((g) => [g, g.notes.filter((n) => !n.tie)])
    .filter(([, nf]) => nf.length)
    .map(([g, nf]) => [g, nf, candidates(nf, w)]);
  if (!steps.length) return tokens;

  let prevCost = steps[0][2].map((c) => staticCost(c, steps[0][1], w));
  let lastPos = steps[0][2].map(handPosition);
  const back = [];
  for (const [, nf, cands] of steps.slice(1)) {
    const costs = [], ptrs = [], poss = [];
    for (const c of cands) {
      const pos = handPosition(c);
      let best = null, arg = 0;
      prevCost.forEach((pc, j) => {
        const move = pos === null || lastPos[j] === null ? 0.0 : w.move * Math.abs(pos - lastPos[j]);
        const total = pc + move;
        if (best === null || total < best) { best = total; arg = j; }
      });
      costs.push(best + staticCost(c, nf, w));
      ptrs.push(arg);
      poss.push(pos !== null ? pos : lastPos[arg]);
    }
    back.push(ptrs);
    prevCost = costs;
    lastPos = poss;
  }
  let k = 0;
  prevCost.forEach((c, i) => { if (c < prevCost[k]) k = i; });
  const choice = [k];
  for (const ptrs of back.slice().reverse()) choice.push(ptrs[choice[choice.length - 1]]);
  choice.reverse();

  const chosen = new Map();
  steps.forEach(([, nf, cands], i) => nf.forEach((n, j) => chosen.set(n.idx, cands[choice[i]][j])));
  const lastStringForPitch = new Map();
  for (const g of groups) {
    const used = new Set();
    for (const n of g.notes) {
      let s, f;
      if (chosen.has(n.idx)) [s, f] = chosen.get(n.idx);
      else { // tied note: continue on the string this pitch was last assigned to
        s = lastStringForPitch.has(n.pitch) ? lastStringForPitch.get(n.pitch) : n.string;
        f = n.pitch - OPEN_PITCH[s];
        if (used.has(s) || !(f >= 0 && f <= 24)) { s = n.string; f = n.fret; }
      }
      used.add(s);
      lastStringForPitch.set(n.pitch, s);
      tokens[n.idx] = `note:s${s}:f${f}`;
    }
  }
  return tokens;
}
