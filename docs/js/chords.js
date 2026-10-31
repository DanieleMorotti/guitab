// Chord names from the transcribed notes, as guitab.chords.suggest_chords finds them: segments of an eighth note on the
// beat grid, pitch-class features of the notes, the chord model (ONNX) and a Viterbi decoding with Chordonomicon's
// chord-to-chord statistics; consecutive chords of one family (root and major / minor) are merged and shown when the
// model is confident enough. Sums run in the same order as in Python and values are rounded where Python rounds them,
// so both versions name the same chords.

export const QUALITIES = ["maj", "min", "7", "maj7", "min7", "hdim7", "dim", "sus2", "sus4", "5"];
const SUFFIX = { maj: "", min: "m", "7": "7", maj7: "maj7", min7: "m7", hdim7: "m7b5", dim: "dim", sus2: "sus2",
  sus4: "sus4", "5": "5" };
const ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const NQ = QUALITIES.length, N_CHORDS = 12 * NQ, NO_CHORD = N_CHORDS;
const N_PC = 6, N_GLOBAL = 6, LOW_PITCH = 55;
const CHANGE = 2.0, PROGRESSION = 0.25, CONFIDENCE = 0.5;

const round = (x, step) => Math.floor(x / step + 0.5) * step;

// numpy.searchsorted: "right" -> first index with a[i] > v, "left" -> first index with a[i] >= v
function searchsorted(a, v, side) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (side === "right" ? a[mid] <= v : a[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function arange(start, stop, step) { // numpy.arange: start + i * step
  const n = Math.max(Math.ceil((stop - start) / step), 0);
  return Array.from({ length: n }, (_, i) => start + i * step);
}

function median(values) {
  const s = [...values].sort((a, b) => a - b), m = s.length;
  return m % 2 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
}

// guitab.chords.segment_bounds (two segments per beat)
export function segmentBounds(beatsIn, end) {
  let beats = beatsIn.filter((b) => b < end);
  if (beats.length < 2) beats = arange(0, end + 0.5, 0.5);
  const period = median(beats.slice(1).map((b, i) => b - beats[i]));
  const before = arange(beats[0] - period, -period, -period).reverse();
  const after = arange(beats[beats.length - 1] + period, end + period, period);
  const grid = [...before, ...beats, ...after];
  const edges = [];
  for (let i = 0; i < grid.length - 1; i++) {
    const d = grid[i + 1] - grid[i];
    edges.push(grid[i] + d * 0, grid[i] + d * 0.5);
  }
  edges.push(grid[grid.length - 1]);
  return [0, ...edges.filter((e) => e > 0 && e < end), end];
}

// guitab.chords.features: pc [S * 6 * 12] and global [S * 6] (float32, computed in float64 in the notes' order)
export function chordFeatures(notes, bounds, beats, downbeats, capo = 0) {
  const n = bounds.length - 1;
  const pcf = new Float64Array(n * N_PC * 12), glob = new Float64Array(n * N_GLOBAL);
  const low = new Int32Array(n).fill(999), high = new Int32Array(n).fill(-1);
  const P = (i, c, pc) => (i * N_PC + c) * 12 + pc;
  for (const note of notes) {
    const on = note.onset, off = note.offset, pitch = note.pitch, pc = ((pitch % 12) + 12) % 12;
    const first = Math.max(searchsorted(bounds, on, "right") - 1, 0);
    const last = Math.min(searchsorted(bounds, off, "left"), n);
    if (first < n && bounds[first] <= on && on < bounds[first + 1]) {
      pcf[P(first, 1, pc)] += 1;
      glob[first * N_GLOBAL + 1] += 1;
    }
    const isOpen = note.fret === capo;
    for (let i = first; i < Math.min(Math.max(last, first + 1), n); i++) {
      const length = bounds[i + 1] - bounds[i];
      const share = (Math.min(off, bounds[i + 1]) - Math.max(on, bounds[i])) / length;
      if (share <= 0) continue;
      pcf[P(i, 0, pc)] += share;
      glob[i * N_GLOBAL] += share;
      if (isOpen) pcf[P(i, 3, pc)] += share;
      if (pitch < LOW_PITCH) pcf[P(i, 4, pc)] += share;
      if (share >= 0.3) { low[i] = Math.min(low[i], pitch); high[i] = Math.max(high[i], pitch); }
    }
  }
  for (let i = 0; i < n; i++) {
    const has = low[i] < 999;
    if (has) { pcf[P(i, 2, low[i] % 12)] = 1; pcf[P(i, 5, high[i] % 12)] = 1; }
    for (const c of [0, 1, 3, 4]) for (let pc = 0; pc < 12; pc++) pcf[P(i, c, pc)] = Math.min(pcf[P(i, c, pc)], 2) / 2;
    const start = bounds[i];
    [[2, beats], [3, downbeats]].forEach(([col, times]) => {
      if (times.length > 1) {
        const j = Math.min(Math.max(searchsorted(times, start, "left"), 1), times.length - 1);
        glob[i * N_GLOBAL + col] = Math.min(Math.abs(times[j] - start), Math.abs(times[j - 1] - start)) < 0.03 ? 1 : 0;
      }
    });
    glob[i * N_GLOBAL] = Math.min(glob[i * N_GLOBAL], 6) / 6;
    glob[i * N_GLOBAL + 1] = Math.min(glob[i * N_GLOBAL + 1], 6) / 6;
    glob[i * N_GLOBAL + 4] = has ? 1 : 0;
    glob[i * N_GLOBAL + 5] = Math.min((bounds[i + 1] - bounds[i]) / 0.5, 2);
  }
  return { pcf: Float32Array.from(pcf), glob: Float32Array.from(glob), n };
}

// guitab.chords.probabilities: logits [S * (C+1)] -> rounded softmax rows (float64)
function probabilities(logits, n) {
  const c = N_CHORDS + 1, out = new Float64Array(n * c);
  for (let i = 0; i < n; i++) {
    const x = Array.from(logits.subarray(i * c, (i + 1) * c), (v) => round(v, 1e-3));
    let mx = -Infinity;
    for (const v of x) if (v > mx) mx = v;
    const e = x.map((v) => Math.exp(v - mx));
    let s = 0;
    for (const v of e) s += v;
    for (let k = 0; k < c; k++) out[i * c + k] = round(e[k] / s, 1e-4);
  }
  return out;
}

// guitab.chords._progression_scores from the Chordonomicon counts (dsp.json "chord_bigram")
function progressionScores(bigram) {
  const used = QUALITIES.map((q) => bigram.qualities.indexOf(q));
  const c = N_CHORDS + 1, out = new Float64Array(c * c);
  const count = (a, d, b) => bigram.count[used[a]][d][used[b]] + 1;
  const total = QUALITIES.map((_, a) => {
    let s = 0;
    for (let d = 0; d < 12; d++) for (let b = 0; b < NQ; b++) s += count(a, d, b);
    return s;
  });
  for (let i = 0; i < N_CHORDS; i++) {
    for (let j = 0; j < N_CHORDS; j++) {
      const qi = i % NQ, qj = j % NQ, d = (((Math.floor(j / NQ) - Math.floor(i / NQ)) % 12) + 12) % 12;
      out[i * c + j] = round(Math.log((count(qi, d, qj) / total[qi]) * N_CHORDS), 1e-9);
    }
  }
  return out;
}

// guitab.chords.decode
function decode(logProbs, n, scores) {
  const c = N_CHORDS + 1, move = new Float64Array(c * c);
  for (let i = 0; i < c * c; i++) move[i] = PROGRESSION * scores[i] - CHANGE;
  for (let i = 0; i < c; i++) move[i * c + i] = -Infinity;
  let total = Float64Array.from(logProbs.subarray(0, c));
  const back = new Int32Array(n * c);
  for (let t = 1; t < n; t++) {
    const next = new Float64Array(c);
    for (let j = 0; j < c; j++) {
      let best = -Infinity, from = 0;
      for (let i = 0; i < c; i++) {
        const v = total[i] + move[i * c + j];
        if (v > best) { best = v; from = i; }
      }
      const stay = total[j] >= best;
      back[t * c + j] = stay ? j : from;
      next[j] = (stay ? total[j] : best) + logProbs[t * c + j];
    }
    total = next;
  }
  const path = new Int32Array(n);
  let best = -Infinity;
  for (let j = 0; j < c; j++) if (total[j] > best) { best = total[j]; path[n - 1] = j; }
  for (let t = n - 1; t > 0; t--) path[t - 1] = back[t * c + path[t]];
  return path;
}

const FAMILY = { maj: 0, "7": 0, maj7: 0, min: 1, min7: 1 };
const family = (k) => (k === NO_CHORD ? 36 : Math.floor(k / NQ) * 3 + (FAMILY[QUALITIES[k % NQ]] ?? 2));

// guitab.chords.chord_spans (merge on): [[start s, end s, class]]
function chordSpans(bounds, probs, n, scores) {
  const c = N_CHORDS + 1, logProbs = probs.map((p) => round(Math.log(Math.max(p, 1e-9)), 1e-9));
  const path = decode(logProbs, n, scores);
  const out = [];
  let start = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || family(path[i]) !== family(path[start])) {
      const k = path[start];
      if (k !== NO_CHORD) {
        const same = [];
        for (let m = 0; m < N_CHORDS; m++) if (family(m) === family(k)) same.push(m);
        let total = 0;
        const per = new Float64Array(same.length);
        for (let r = start; r < i; r++) {
          same.forEach((m, j) => { total += probs[r * c + m]; per[j] += probs[r * c + m]; });
        }
        if (total / (i - start) >= CONFIDENCE) {
          let arg = 0;
          for (let j = 1; j < per.length; j++) if (per[j] > per[arg]) arg = j;
          out.push([bounds[start], bounds[i], same[arg]]);
        }
      }
      start = i;
    }
  }
  return out;
}

export function chordName(k) { return ROOT_NAMES[Math.floor(k / NQ)] + SUFFIX[QUALITIES[k % NQ]]; }

// guitab.chords.suggest_chords: [[start s, end s, name]]; run({pcf, glob, n}) resolves to the model's logits [S * (C+1)]
export async function suggestChords(notes, beats, downbeats, capo, run, bigram) {
  if (!notes.length || beats.length < 2) return [];
  const end = Math.max(...notes.map((x) => x.offset));
  const bounds = segmentBounds(beats, end);
  const f = chordFeatures(notes, bounds, beats, downbeats, capo);
  const probs = probabilities(await run(f), f.n);
  return chordSpans(bounds, probs, f.n, progressionScores(bigram)).map(([s, e, k]) => [s, e, chordName(k)]);
}
