// Model windows -> probabilities -> notes: port of guitab/inference.py (predict, extract_notes, drop_duplicates,
// merge_reattacks, apply_capo). Probabilities are frame-major: tab[t][col][class] in a Float32Array of T * 6 * 21.

import { FPS, OPEN_PITCH, STRINGS, roundHalfEven } from "./guitar.js";

export const DEFAULT_PARAMS = { onset: 0.6, frame: 0.6, min_frames: 3, legato: 8, orphan: false, fret_frames: 3,
  dedup: 4, reattack: 0.75 };
const N_CLASSES = 21;

// Overlapping 432-frame windows (hop 288), each weighted by a Hann taper; runWindow(Float32Array(win * 288)) resolves to
// {tab: Float32Array(win * 6 * 21) of probabilities, onset: Float32Array(win * 6)}. Inputs shorter than a window are
// padded with silence (the exported model has a fixed window).
export async function predict(feat, frames, nBins, runWindow, progress = () => {}, win = 432, hop = 288) {
  const tab = new Float32Array(frames * 6 * N_CLASSES), onset = new Float32Array(frames * 6);
  const weight = new Float32Array(frames);
  const starts = [];
  for (let s = 0; s <= Math.max(frames - win, 0); s += hop) starts.push(s);
  if (starts[starts.length - 1] + win < frames) starts.push(frames - win);
  const taper = new Float32Array(win);
  for (let i = 0; i < win; i++) taper[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 1)) / (win + 1)) + 1e-3;
  for (const [i, s] of starts.entries()) {
    const n = Math.min(win, frames - s);
    const x = new Float32Array(win * nBins);
    x.set(feat.subarray(s * nBins, (s + n) * nBins));
    const out = await runWindow(x);
    for (let t = 0; t < n; t++) {
      const w = taper[t];
      for (let j = 0; j < 6 * N_CLASSES; j++) tab[(s + t) * 6 * N_CLASSES + j] += out.tab[t * 6 * N_CLASSES + j] * w;
      for (let j = 0; j < 6; j++) onset[(s + t) * 6 + j] += out.onset[t * 6 + j] * w;
      weight[s + t] += w;
    }
    progress((i + 1) / starts.length);
  }
  for (let t = 0; t < frames; t++) {
    for (let j = 0; j < 6 * N_CLASSES; j++) tab[t * 6 * N_CLASSES + j] /= weight[t];
    for (let j = 0; j < 6; j++) onset[t * 6 + j] /= weight[t];
  }
  return { tab, onset, frames };
}

export function extractNotes({ tab, onset, frames: T }, params = {}) {
  const p = { ...DEFAULT_PARAMS, ...params };
  const at = (t, col, cls) => tab[(t * 6 + col) * N_CLASSES + cls];
  let notes = [];
  for (let col = 0; col < 6; col++) {
    const string = STRINGS[col];
    const active = new Float64Array(T), frets = new Int32Array(T), o = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      active[t] = 1 - at(t, col, 0);
      let best = 1;
      for (let c = 2; c < N_CLASSES; c++) if (at(t, col, c) > at(t, col, best)) best = c;
      frets[t] = best - 1;
      o[t] = onset[t * 6 + col];
    }
    const peakSet = new Set();
    for (let t = 0; t < T; t++) {
      if (o[t] >= p.onset && o[t] >= o[Math.max(t - 1, 0)] && o[t] > o[Math.min(t + 1, T - 1)]) peakSet.add(t);
    }
    let t = 0;
    while (t < T) {
      let start = null;
      if (peakSet.has(t)) start = t;
      else if (p.orphan && active[t] > p.frame && (t === 0 || active[t - 1] <= p.frame)) {
        if (t + p.min_frames <= T) {
          let ok = true;
          for (let k = t; k < t + p.min_frames; k++) if (!(active[k] > p.frame)) ok = false;
          if (ok) start = t;
        }
      }
      if (start === null) { t += 1; continue; }
      const headEnd = Math.min(start + p.fret_frames, T);
      let fret = 0, bestMean = -Infinity;
      for (let f = 0; f < N_CLASSES - 1; f++) {
        let s = 0;
        for (let k = start; k < headEnd; k++) s += at(k, col, f + 1);
        const mean = s / (headEnd - start);
        if (mean > bestMean) { bestMean = mean; fret = f; }
      }
      let end = start + 1, changed = 0;
      while (end < T) {
        if (peakSet.has(end) || active[end] <= p.frame) break;
        changed = frets[end] !== fret ? changed + 1 : 0;
        if (changed >= p.legato) { end -= changed - 1; break; }
        end += 1;
      }
      if (end - start >= 2 || peakSet.has(start)) {
        const confEnd = Math.max(end, start + 1);
        let s = 0;
        for (let k = start; k < confEnd; k++) s += at(k, col, fret + 1);
        notes.push({ onset: start / FPS, offset: end / FPS, string, fret, pitch: OPEN_PITCH[string] + fret,
          score: Math.max(o[start], active[start]), fret_conf: s / (confEnd - start) });
      }
      // legato change: the next note starts where the fret changed (no onset needed)
      if (end < T && !peakSet.has(end) && active[end] > p.frame) peakSet.add(end);
      t = Math.max(end, start + 1);
    }
  }
  notes = sortNotes(notes);
  if (p.dedup) notes = dropDuplicates(notes, p.dedup / FPS);
  for (const n of notes) delete n.fret_conf;
  if (p.reattack) notes = mergeReattacks(notes, onset, p.reattack);
  return notes;
}

function sortNotes(notes) {
  return notes.slice().sort((a, b) => a.onset - b.onset || a.string - b.string);
}

// Same pitch starting on two strings within `window` s: keep the note with the higher fret posterior.
function dropDuplicates(notes, window) {
  const keep = notes.map(() => true);
  for (let i = 0; i < notes.length; i++) {
    const a = notes[i];
    for (let j = i + 1; j < notes.length; j++) {
      const b = notes[j];
      if (b.onset - a.onset > window) break;
      if (keep[i] && keep[j] && a.pitch === b.pitch && a.string !== b.string) {
        if (a.fret_conf >= b.fret_conf) keep[j] = false;
        else keep[i] = false;
      }
    }
  }
  return notes.filter((_, i) => keep[i]);
}

// A note that starts where the previous note on the same string and fret ends is merged into it unless its onset
// probability reaches thr (the onset head split one ringing note in two).
function mergeReattacks(notes, onset, thr) {
  const out = [], last = {};
  for (const n of notes.slice().sort((a, b) => a.onset - b.onset)) {
    const prev = last[n.string];
    const t = roundHalfEven(n.onset * FPS), col = STRINGS.indexOf(n.string);
    if (prev && prev.fret === n.fret && Math.abs(prev.offset - n.onset) <= 1.5 / FPS && onset[t * 6 + col] < thr) {
      prev.offset = n.offset;
      continue;
    }
    out.push(n);
    last[n.string] = n;
  }
  return sortNotes(out);
}

// Notes below the capo move to the string with the lowest playable fret (>= capo) that is free at their onset; dropped
// if there is none. Frets stay absolute.
export function applyCapo(notes, capo, maxFret = 20) {
  if (!capo) return notes;
  const out = notes.filter((n) => n.fret >= capo).map((n) => ({ ...n })).sort((a, b) => a.onset - b.onset);
  for (const n of notes.filter((n) => n.fret < capo).sort((a, b) => a.onset - b.onset)) {
    const busy = new Set(out.filter((m) => m.onset - 0.03 <= n.onset && n.onset < m.offset).map((m) => m.string));
    const options = STRINGS.filter((s) => !busy.has(s) && capo <= n.pitch - OPEN_PITCH[s] && n.pitch - OPEN_PITCH[s] <= maxFret);
    if (options.length) {
      let s = options[0];
      for (const c of options) if (n.pitch - OPEN_PITCH[c] < n.pitch - OPEN_PITCH[s]) s = c;
      out.push({ ...n, string: s, fret: n.pitch - OPEN_PITCH[s] });
    }
  }
  return sortNotes(out);
}
