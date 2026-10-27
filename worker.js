// Web Worker running the whole transcription in the browser: audio (22.05 kHz mono) -> CQT -> tab model (ONNX Runtime
// Web, WebGPU when available, else WebAssembly) -> notes -> beats (Beat This! + the model's onsets) -> tab tokens -> .gp5.
// Same steps as guitab/pipeline.py; models are downloaded on the first transcription that needs them and kept in the
// browser cache (Cache Storage) for the next visits.

import { cqtFeature, melSpectrogram } from "./dsp.js";
import { applyCapo, extractNotes, predict } from "./notes.js";
import { BeatGrid, FINGERING_PRESETS, notesToTokens } from "./tab.js";
import { beatLogits, detectBeats } from "./beats.js";
import { writeGp5 } from "./gp5.js";

const ORT_VERSION = "1.22.0";
const ORT_DIR = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const CACHE = "guitab-models-v1";

let ort = null, dsp = null, backend = null; // backend of the tab model: "webgpu" | "wasm"
const sessions = new Map();

const post = (msg) => self.postMessage(msg);
const progress = (stage, frac, text) => post({ type: "progress", stage, frac, text });

async function loadOrt() {
  if (ort) return;
  ort = await import(`${ORT_DIR}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_DIR;
  ort.env.logLevel = "error";
}

async function loadDsp(base) {
  if (!dsp) dsp = await (await fetch(new URL("assets/dsp.json", base))).json();
}

// Model bytes from the browser cache, else downloaded with progress and cached.
// the model's download URL: models.json points at the Hugging Face repository; a page served from this computer
// (localhost, 127.0.0.1) reads the same files from docs/models/ instead, so it works before they are uploaded
function modelUrl(url, base) {
  const local = ["localhost", "127.0.0.1"].includes(new URL(base).hostname);
  return new URL(local ? "models/" + url.split("/").pop() : url, base).href;
}

async function modelBytes(url, label) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache Storage (e.g. private mode) */ }
  const hit = cache && (await cache.match(url));
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not download ${label} (${res.status} ${res.statusText}) from ${url}`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  const reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    progress("download", total ? got / total : 0, `Downloading ${label}: ${(got / 2 ** 20).toFixed(0)}${total ? ` / ${(total / 2 ** 20).toFixed(0)}` : ""} MB`);
  }
  // a tab model downloaded from Hugging Face counts as one download on the model page: the Hub counts requests for
  // config.json (guitab.paths.count_download); best effort, nothing but the request itself is sent
  if (url.startsWith("https://huggingface.co/") && url.split("/").pop().startsWith("guitab_")) {
    fetch(url.replace(/[^/]+$/, "config.json"), { method: "HEAD" }).catch(() => {});
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const p of parts) { bytes.set(p, off); off += p.length; }
  if (cache) {
    try { await cache.put(url, new Response(bytes)); } catch { /* quota: just not cached */ }
  }
  return bytes;
}

// gpu: try WebGPU first (the tab model); the beat tracker always runs on WebAssembly, since its WebGPU output differs
// from PyTorch's (226 vs 200 beats on a test song; the WebAssembly output matches).
async function getSession(url, label, gpu) {
  if (sessions.has(url)) return sessions.get(url);
  await loadOrt();
  const bytes = await modelBytes(url, label);
  progress("load", 0, `Loading ${label}…`);
  let session = null;
  if (gpu && "gpu" in navigator && backend !== "wasm") {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
      backend = "webgpu";
    } catch (e) {
      console.warn("WebGPU unavailable, using WebAssembly:", e);
    }
  }
  if (!session) {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    if (gpu) backend = "wasm";
  }
  sessions.set(url, session);
  return session;
}

async function transcribe({ audio, base, opts }) {
  const t0 = performance.now();
  if (opts.backend && !backend) backend = opts.backend; // page URL ?backend=wasm forces the CPU
  await loadDsp(base);
  const tabUrl = modelUrl(opts.model.url, base);

  progress("features", 0, "Computing the spectrogram…");
  const feat = cqtFeature(audio, dsp, (f) => progress("features", f, "Computing the spectrogram…"));

  const tabSession = await getSession(tabUrl, `model ${opts.model.id} (${opts.model.mb} MB)`, true);
  const win = opts.model.window || 432, nBins = dsp.n_bins;
  const probs = await predict(feat.data, feat.frames, nBins, async (x) => {
    const out = await tabSession.run({ x: new ort.Tensor("float32", x, [1, win, nBins]) });
    return { tab: await out.tab.getData(), onset: await out.onset.getData() };
  }, (f) => progress("model", f, `Transcribing (${backend === "webgpu" ? "WebGPU" : "WebAssembly"})…`), win);

  // the model's own note-extraction settings (e.g. onset .45 for E25), under the page's note-sensitivity value
  const params = { ...(opts.model.decoding || {}), ...(opts.onset ? { onset: opts.onset } : {}) };
  const notes = applyCapo(extractNotes(probs, params), opts.capo);

  let grid;
  if (opts.tempo === "auto") {
    progress("beats", 0, "Finding the beats…");
    const beatModel = opts.beatModel;
    const beatSession = await getSession(modelUrl(beatModel.url, base), `beat tracker (${beatModel.mb} MB)`, false);
    const mel = melSpectrogram(audio, dsp);
    const logits = await beatLogits(mel.data, mel.frames, async (x) => {
      const out = await beatSession.run({ x: new ort.Tensor("float32", x, [1, beatModel.chunk, 128]) });
      return { beat: await out.beat.getData(), downbeat: await out.downbeat.getData() };
    }, beatModel.chunk, beatModel.border);
    const { beats, downbeats } = detectBeats(logits, probs.onset, probs.frames);
    grid = new BeatGrid(beats, downbeats);
    console.debug("beats", beats.length, "downbeats", downbeats.length, "tempo", grid.tempo);
  } else grid = new BeatGrid([], null, parseInt(opts.tempo, 10));

  const weights = opts.fingering === "model" ? null : FINGERING_PRESETS[opts.fingering];
  const tokens = notesToTokens(notes, grid, opts.name, opts.grid, weights, 6, opts.capo);
  if (tokens === null) throw new Error("No notes were transcribed.");
  const gp5 = writeGp5(tokens, dsp.gp5, opts.capo);
  return { gp5, notes: notes.length, tempo: grid.tempo, beats_per_measure: grid.numBeats, backend,
    seconds: Math.round((performance.now() - t0) / 100) / 10 };
}

self.onmessage = async ({ data }) => {
  if (data.type !== "transcribe") return;
  try {
    const result = await transcribe(data);
    self.postMessage({ type: "done", result }, [result.gp5.buffer]);
  } catch (e) {
    post({ type: "error", message: e && e.message ? e.message : String(e) });
  }
};
