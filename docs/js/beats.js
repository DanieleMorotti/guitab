// Beats and downbeats (B1), as guitab.tab.detect_beats finds them: the Beat This! logits (port of beat_this.inference:
// split_piece, aggregate_prediction with overlap_mode "keep_first"), then guitab.beats: the beat activation plus the
// tab model's onset probabilities tracked by dynamic programming, downbeats as one metre and phase per song. Sums run
// in the same order as in Python and the values are rounded where Python rounds them, so both find the same beats.

import { FPS, SR, roundHalfEven } from "./guitar.js";

const BEAT_FPS = 50, HOP = 256;
const TEMPO_WIN = 689, TEMPO_HOP = 16;
const TINY = 2.2250738585072014e-308; // numpy.finfo(float64).tiny

// spect: Float32Array(T * 128); runChunk(Float32Array(chunk * 128)) resolves to {beat, downbeat} logits of the chunk.
// Returns the logits per 50 fps frame, rounded to 0.01 (float64, round half up) as in Python: the browser's
// WebAssembly arithmetic differs from Python's native one by ~1e-6.
export async function beatLogits(spect, frames, runChunk, chunk = 1500, border = 6, nMels = 128) {
  const starts = [];
  for (let s = -border; s < frames - border; s += chunk - 2 * border) starts.push(s);
  if (frames > chunk - 2 * border) starts[starts.length - 1] = frames - (chunk - border);
  const beat = new Float32Array(frames).fill(-1000), down = new Float32Array(frames).fill(-1000);
  const preds = [];
  for (const start of starts) {
    // the chunk as split_piece builds it (zero padding before 0 and up to `border` frames after the end), then padded
    // with silence to the model's fixed length
    const left = Math.max(0, -start);
    const from = Math.max(start, 0), to = Math.min(start + chunk, frames);
    const x = new Float32Array(chunk * nMels);
    x.set(spect.subarray(from * nMels, to * nMels), left * nMels);
    const used = left + (to - from) + Math.max(0, Math.min(border, start + chunk - frames));
    const out = await runChunk(x);
    preds.push({ start, beat: out.beat.subarray(0, used), downbeat: out.downbeat.subarray(0, used) });
  }
  // keep_first: later chunks are written first, earlier ones overwrite them
  for (const { start, beat: b, downbeat: d } of preds.reverse()) {
    for (let i = border; i < b.length - border; i++) {
      const t = start + i;
      if (t >= 0 && t < frames) { beat[t] = b[i]; down[t] = d[i]; }
    }
  }
  const round = (a) => Float64Array.from(a, (v) => Math.floor(v * 100 + 0.5) / 100);
  return { beat: round(beat), downbeat: round(down) };
}

const roundTo = (x, step) => Math.floor(x / step + 0.5) * step;

function seqSum(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

function maxOf(a) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m;
}

// guitab.beats.activation: logits (50 fps) -> probabilities at the tab model's n frames.
export function activation(logits, n) {
  const p = Float64Array.from(logits, (x) => roundTo(1 / (1 + Math.exp(-x)), 1e-6));
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const pos = i * HOP / SR * BEAT_FPS;
    const j = Math.min(Math.floor(pos), p.length - 1);
    const frac = pos - j;
    const nxt = p[Math.min(j + 1, p.length - 1)];
    out[i] = roundTo(p[j] * (1 - frac) + nxt * frac, 1e-6);
  }
  return out;
}

// guitab.beats.envelope: beat activation + summed onset probabilities (onset: Float32Array(n * 6)), each scaled to a
// maximum of 1, rounded to 0.01.
export function envelope(act, onset = null) {
  const n = act.length, env = new Float64Array(n);
  const am = maxOf(act) + 1e-9;
  for (let i = 0; i < n; i++) env[i] = act[i] / am;
  if (onset) {
    const total = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = onset[i * 6];
      for (let c = 1; c < 6; c++) s = s + onset[i * 6 + c];
      total[i] = s;
    }
    const om = maxOf(total) + 1e-9;
    for (let i = 0; i < n; i++) env[i] = env[i] + total[i] / om;
  }
  for (let i = 0; i < n; i++) env[i] = roundTo(env[i], 0.01);
  return env;
}

// guitab.beats.beat_period: beat period in frames (mean normalised autocorrelation of Hann-windowed 8 s windows every
// 16 frames, log-normal prior around 120 BPM, tempi below 320 BPM).
export function beatPeriod(env) {
  const n = env.length, w = TEMPO_WIN, pad = Math.floor(w / 2);
  const padded = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) padded[i] = env[0] * i / pad;
  padded.set(env, pad);
  for (let i = 0; i < pad; i++) padded[pad + n + i] = env[n - 1] * (pad - i) / pad;
  const win = new Float64Array(w);
  for (let j = 0; j < w; j++) win[j] = roundTo(0.5 - 0.5 * Math.cos(2 * Math.PI * j / w), 1e-12);
  const tg = new Float64Array(w), frame = new Float64Array(w);
  let count = 0;
  for (let t = 0; t < n; t += TEMPO_HOP) {
    for (let j = 0; j < w; j++) frame[j] = padded[t + j] * win[j];
    let a0 = 0;
    for (let j = 0; j < w; j++) a0 += frame[j] * frame[j];
    if (a0 > TINY) {
      for (let k = 0; k < w; k++) {
        let a = 0;
        for (let j = 0; j + k < w; j++) a += frame[j] * frame[j + k];
        tg[k] += a / a0;
      }
    }
    count++;
  }
  let best = -Infinity, period = 1;
  const p120 = Math.log2(120);
  for (let lag = 1; lag < w; lag++) {
    const bpm = 60 * FPS / lag;
    if (bpm >= 320) continue;
    const d = Math.log2(bpm) - p120;
    const prior = roundTo(-0.5 * (d * d), 1e-9);
    const score = roundTo(Math.log1p(1e6 * roundTo(tg[lag] / count, 1e-6)) + prior, 1e-9);
    if (score > best) { best = score; period = lag; }
  }
  return period;
}

// guitab.beats.dp_beats: beat frames by dynamic programming (Ellis 2007, as librosa.beat.beat_track, trim=False).
export function dpBeats(env, k, tightness = 100) {
  const n = env.length;
  const mean = seqSum(env) / n;
  let ss = 0;
  for (let i = 0; i < n; i++) { const d = env[i] - mean; ss += d * d; }
  const std = Math.sqrt(ss / (n - 1));
  const onsets = Float64Array.from(env, (v) => v / (std + TINY));
  const K = 2 * k + 1, half = k;
  const window = new Float64Array(K);
  for (let j = 0; j < K; j++) { const x = (j - k) * 32.0 / k; window[j] = roundTo(Math.exp(-0.5 * (x * x)), 1e-12); }
  const local = new Float64Array(n);
  for (let j = 0; j < K; j++) {
    const lo = Math.max(0, j - half), hi = Math.min(n, n + j - half);
    for (let i = lo; i < hi; i++) local[i] += window[j] * onsets[i + half - j];
  }
  const first = Math.floor((k + 1) / 2), lastD = 2 * k;
  const penalty = new Float64Array(lastD);
  const lk = Math.log(k);
  for (let d = 1; d <= lastD; d++) { const x = Math.log(d) - lk; penalty[d - 1] = roundTo(tightness * (x * x), 1e-9); }
  const thresh = 0.01 * maxOf(local);
  const cum = new Float64Array(n), back = new Int32Array(n).fill(-1);
  let firstBeat = true;
  for (let i = 0; i < n; i++) {
    const lo = Math.max(i - lastD, 0), hi = i - first;
    let bestLoc = -1, best = -Infinity;
    for (let loc = hi; loc >= lo; loc--) {
      const s = cum[loc] - penalty[i - loc - 1];
      if (s > best) { best = s; bestLoc = loc; }
    }
    cum[i] = bestLoc >= 0 ? local[i] + best : local[i];
    if (firstBeat && local[i] < thresh) back[i] = -1;
    else { back[i] = bestLoc; firstBeat = false; }
  }
  const peak = new Uint8Array(n), vals = [];
  for (let i = 0; i < n; i++) {
    const prev = i > 0 ? cum[i - 1] : cum[0], next = i < n - 1 ? cum[i + 1] : cum[n - 1];
    if (cum[i] > prev && cum[i] >= next) { peak[i] = 1; vals.push(cum[i]); }
  }
  vals.sort((a, b) => a - b);
  const m = vals.length;
  const median = m ? (vals[(m - 1) >> 1] + vals[m >> 1]) / 2 : 0;
  let tail = n - 1;
  for (let t = n - 1; t >= 0; t--) if (peak[t] && cum[t] >= 0.5 * median) { tail = t; break; }
  const beats = new Uint8Array(n);
  for (let t = tail; t >= 0; t = back[t]) beats[t] = 1;
  for (let t = 0; t < n && local[t] <= 0; t++) beats[t] = 0;
  for (let t = n - 1; t >= 0 && local[t] <= 0; t--) beats[t] = 0;
  const out = [];
  for (let t = 0; t < n; t++) if (beats[t]) out.push(t);
  return out;
}

// guitab.beats.track_beats: beat times (s).
export function trackBeats(act, onset = null, tightness = 100) {
  const env = envelope(act, onset);
  if (!env.some((v) => v !== 0)) return [];
  return dpBeats(env, beatPeriod(env), tightness).map((f) => f * HOP / SR);
}

// guitab.beats.pick_downbeats: the bar length (2, 3 or 4 beats) and phase with the highest mean downbeat activation.
export function pickDownbeats(beats, downAct, lengths = [2, 3, 4]) {
  if (beats.length < 2) return beats.slice(0, 1);
  const act = beats.map((b) => downAct[Math.min(Math.max(roundHalfEven(b * FPS), 0), downAct.length - 1)]);
  let best = -Infinity, choice = [lengths[0], 0];
  for (const k of lengths) {
    for (let p = 0; p < Math.min(k, act.length); p++) {
      let s = 0, c = 0;
      for (let i = p; i < act.length; i += k) { s += act[i]; c++; }
      if (s / c > best) { best = s / c; choice = [k, p]; }
    }
  }
  return beats.filter((_, i) => i >= choice[1] && (i - choice[1]) % choice[0] === 0);
}

// Beats and downbeats (s) from the Beat This! logits and the tab model's onset probabilities (n frames).
export function detectBeats(logits, onset, n) {
  const beats = trackBeats(activation(logits.beat, n), onset);
  return { beats, downbeats: pickDownbeats(beats, activation(logits.downbeat, n)) };
}
