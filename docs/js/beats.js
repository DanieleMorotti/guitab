// Beat and downbeat times from the Beat This! model: port of beat_this.inference (split_piece, aggregate_prediction with
// overlap_mode "keep_first") and its "minimal" postprocessor (no DBN), as guitab.tab.detect_beats uses it.

const FPS = 50;

// spect: Float32Array(T * 128); runChunk(Float32Array(chunk * 128)) resolves to {beat, downbeat} logits of the chunk.
export async function detectBeats(spect, frames, runChunk, chunk = 1500, border = 6, nMels = 128) {
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
  // logits rounded to 0.01 (float64, round half up) before picking peaks, as guitab.tab.detect_beats does: the
  // browser's WebAssembly arithmetic differs from Python's native one by ~1e-6, which could otherwise move a near-tied
  // peak by a frame and shift the bar grid
  const round = (a) => Float64Array.from(a, (v) => Math.floor(v * 100 + 0.5) / 100);
  return postprocess(round(beat), round(down));
}

// Beat and downbeat times (s) from frame logits; downbeats moved onto the nearest beat.
export function postprocess(beat, down) {
  const beatTimes = peaks(beat).map((f) => f / FPS);
  let downTimes = peaks(down).map((f) => f / FPS);
  if (beatTimes.length) {
    downTimes = downTimes.map((d) => {
      let best = 0;
      beatTimes.forEach((b, i) => { if (Math.abs(b - d) < Math.abs(beatTimes[best] - d)) best = i; });
      return beatTimes[best];
    });
  }
  downTimes = [...new Set(downTimes)].sort((a, b) => a - b);
  return { beats: beatTimes, downbeats: downTimes };
}

// Frames that are the maximum within +-3 frames and above probability 0.5, adjacent peaks merged into their mean.
function peaks(logits) {
  const n = logits.length, frames = [];
  for (let t = 0; t < n; t++) {
    let m = -Infinity;
    for (let k = Math.max(0, t - 3); k <= Math.min(n - 1, t + 3); k++) m = Math.max(m, logits[k]);
    if (logits[t] === m && logits[t] > 0) frames.push(t);
  }
  const result = [];
  if (!frames.length) return result;
  let p = frames[0], c = 1;
  for (const p2 of frames.slice(1)) {
    if (p2 - p <= 1) { c += 1; p += (p2 - p) / c; }
    else { result.push(p); p = p2; c = 1; }
  }
  result.push(p);
  return result;
}
