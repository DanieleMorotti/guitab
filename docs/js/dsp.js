// Signal processing for the model inputs, ported from the Python pipeline (guitab/audio.py and Beat This!).
// The constants (CQT filters, halving filter, mel filterbank) come from assets/dsp.json (scripts/export_web.py).

class FFT {
  // In-place radix-2 complex FFT of a fixed power-of-two size.
  constructor(n) {
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = -Math.sin((2 * Math.PI * i) / n);
    }
  }

  forward(re, im) {
    const n = this.n, rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step], wi = this.sin[k * step];
          const a = start + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  inverse(re, im) {
    for (let i = 0; i < this.n; i++) im[i] = -im[i];
    this.forward(re, im);
    for (let i = 0; i < this.n; i++) { re[i] /= this.n; im[i] = -im[i] / this.n; }
  }
}

// Zero-phase FIR low-pass (matching librosa's soxr_hq 2:1 resampler) then every other sample, times sqrt(2)
// (librosa.resample(..., scale=True)). Overlap-add FFT convolution in blocks.
function halve(y, taps) {
  const L = taps.length, delay = (L - 1) / 2;
  const nfft = 1 << 15, block = nfft - L + 1;
  const fft = new FFT(nfft);
  const hr = new Float64Array(nfft), hi = new Float64Array(nfft);
  hr.set(taps);
  fft.forward(hr, hi);
  const full = new Float64Array(y.length + L - 1);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let start = 0; start < y.length; start += block) {
    re.fill(0); im.fill(0);
    const len = Math.min(block, y.length - start);
    for (let i = 0; i < len; i++) re[i] = y[start + i];
    fft.forward(re, im);
    for (let k = 0; k < nfft; k++) {
      const r = re[k] * hr[k] - im[k] * hi[k];
      im[k] = re[k] * hi[k] + im[k] * hr[k];
      re[k] = r;
    }
    fft.inverse(re, im);
    const end = Math.min(len + L - 1, full.length - start);
    for (let i = 0; i < end; i++) full[start + i] += re[i];
  }
  const out = new Float64Array(Math.ceil(y.length / 2));
  for (let i = 0; i < out.length; i++) out[i] = full[2 * i + delay] * Math.SQRT2;
  return out;
}

// Round to the nearest float16 value (the Python feature is stored as float16).
function toHalf(x) {
  if (x === 0) return 0;
  const a = Math.abs(x);
  const e = Math.max(Math.floor(Math.log2(a)), -14);
  const step = Math.pow(2, e - 10);
  let q = a / step;
  const f = Math.floor(q), d = q - f;
  q = d > 0.5 || (d === 0.5 && f % 2 === 1) ? f + 1 : f;
  return Math.sign(x) * q * step;
}

// Log-magnitude CQT, 36 bins/octave from E1, 288 bins, hop 256 at 22.05 kHz, scaled to [0, 1] (0 = 80 dB below the
// track maximum): guitab.audio.compute_cqt. Returns {data: Float32Array(T * 288), frames: T}.
export function cqtFeature(y, dsp, progress = () => {}) {
  const { n_fft: nfft, rows, cols, re: bre, im: bim, lengths } = dsp.cqt;
  const bpo = dsp.bins_per_octave, nBins = dsp.n_bins, octaves = nBins / bpo;
  const fft = new FFT(nfft);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  const accR = new Float64Array(bpo), accI = new Float64Array(bpo);
  let sig = Float64Array.from(y), hop = dsp.hop;
  const perOctave = [];
  for (let k = 0; k < octaves; k++) {
    const frames = 1 + Math.floor(sig.length / hop);
    const out = new Float64Array(frames * bpo);
    const scale = Math.sqrt(2 ** k), half = nfft / 2;
    for (let t = 0; t < frames; t++) {
      const s0 = t * hop - half;
      for (let i = 0; i < nfft; i++) {
        const j = s0 + i;
        re[i] = j >= 0 && j < sig.length ? sig[j] : 0;
        im[i] = 0;
      }
      fft.forward(re, im);
      accR.fill(0); accI.fill(0);
      for (let z = 0; z < rows.length; z++) {
        const r = rows[z], c = cols[z];
        accR[r] += bre[z] * re[c] - bim[z] * im[c];
        accI[r] += bre[z] * im[c] + bim[z] * re[c];
      }
      for (let r = 0; r < bpo; r++) out[t * bpo + r] = Math.hypot(accR[r], accI[r]) * scale;
    }
    perOctave.push({ out, frames });
    progress((k + 1) / octaves);
    if (k < octaves - 1) { sig = halve(sig, dsp.halving.taps); hop /= 2; }
  }
  const T = Math.min(...perOctave.map((o) => o.frames));
  const c = new Float64Array(T * nBins);
  let max = 0;
  perOctave.forEach(({ out }, k) => {
    const base = (octaves - 1 - k) * bpo;
    for (let t = 0; t < T; t++) {
      for (let r = 0; r < bpo; r++) {
        const v = out[t * bpo + r] / Math.sqrt(lengths[base + r]);
        c[t * nBins + base + r] = v;
        if (v > max) max = v;
      }
    }
  });
  // librosa.amplitude_to_db(c, ref=max + 1e-9, top_db=80), then (db + 80) / 80
  const amin = 1e-5, ref = 20 * Math.log10(Math.max(amin, max + 1e-9));
  const db = new Float64Array(c.length);
  let top = -Infinity;
  for (let i = 0; i < c.length; i++) {
    db[i] = 20 * Math.log10(Math.max(amin, c[i])) - ref;
    if (db[i] > top) top = db[i];
  }
  const floor = top - 80, data = new Float32Array(c.length);
  for (let i = 0; i < c.length; i++) data[i] = toHalf((Math.max(db[i], floor) + 80) / 80);
  return { data, frames: T };
}

// Beat This! input: log(1 + 1000 * mel), torchaudio MelSpectrogram(n_fft 1024, hop 441, 128 slaney mels 30 Hz - 11 kHz,
// magnitude, centred with reflect padding, normalized="frame_length"). Returns {data: Float32Array(T * 128), frames}.
export function melSpectrogram(y, dsp) {
  const { n_fft: nfft, hop, n_mels: nMels, log_multiplier: mult, fb } = dsp.mel;
  const fft = new FFT(nfft), half = nfft / 2;
  const win = new Float64Array(nfft);
  for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nfft);
  const n = y.length, frames = 1 + Math.floor(n / hop);
  const reflect = (j) => {
    while (j < 0 || j >= n) j = j < 0 ? -j : 2 * (n - 1) - j;
    return j;
  };
  const re = new Float64Array(nfft), im = new Float64Array(nfft), mag = new Float64Array(half + 1);
  const data = new Float32Array(frames * nMels);
  const norm = 1 / Math.sqrt(nfft);
  for (let t = 0; t < frames; t++) {
    const s0 = t * hop - half;
    for (let i = 0; i < nfft; i++) { re[i] = y[reflect(s0 + i)] * win[i]; im[i] = 0; }
    fft.forward(re, im);
    for (let k = 0; k <= half; k++) mag[k] = Math.hypot(re[k], im[k]) * norm;
    for (let m = 0; m < nMels; m++) {
      let s = 0;
      for (const [k, w] of fb[m]) s += mag[k] * w;
      data[t * nMels + m] = Math.log1p(mult * s);
    }
  }
  return { data, frames };
}
