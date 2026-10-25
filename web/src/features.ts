// Log-mel features, identical to the Python training code:
//   librosa.feature.melspectrogram(y, sr, n_fft, hop_length, n_mels, fmin, fmax)
//   (center=True, zero padding), then log(mel + 1e-6), then per-band (x - mean) / std
// The mel filterbank, mean and std come from the model's config json
// (train.py web / train_lung.py web).

/** What LogMel needs; the cough and lung configs both extend it. */
export interface MelConfig {
  n_fft: number
  hop: number
  n_mels: number
  frames: number
  log_eps: number
  mean: number[]
  std: number[]
  mel_fb: number[][] // [n_mels][n_fft / 2 + 1]
}

export interface CoughConfig extends MelConfig {
  sr: number
  threshold: number
}

export class LogMel {
  readonly cfg: MelConfig
  private nBins: number
  private win: Float64Array
  private fb: Float64Array // [n_mels][nBins]
  private frame: Float64Array
  private power: Float64Array
  private spectrum: (frame: Float64Array, power: Float64Array) => void

  constructor(cfg: MelConfig) {
    this.cfg = cfg
    const n = cfg.n_fft
    this.nBins = n / 2 + 1
    // periodic Hann (scipy get_window('hann', n, fftbins=True))
    this.win = new Float64Array(n)
    for (let i = 0; i < n; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n)
    this.fb = new Float64Array(cfg.n_mels * this.nBins)
    cfg.mel_fb.forEach((row, m) => row.forEach((v, k) => (this.fb[m * this.nBins + k] = v)))
    this.frame = new Float64Array(n)
    this.power = new Float64Array(this.nBins)
    this.spectrum = (n & (n - 1)) === 0 ? fftPower(n) : dftPower(n)
  }

  /** y: exactly the window length (frames - 1) * hop. Returns raw log-mel, [n_mels][frames]. */
  logmel(y: Float32Array): Float32Array {
    const { n_fft: n, hop, n_mels, frames, log_eps } = this.cfg
    const pad = n / 2
    const out = new Float32Array(n_mels * frames)
    for (let t = 0; t < frames; t++) {
      const start = t * hop - pad // position in y of frame sample 0 (zero padded outside)
      for (let i = 0; i < n; i++) {
        const j = start + i
        this.frame[i] = j >= 0 && j < y.length ? y[j] * this.win[i] : 0
      }
      this.spectrum(this.frame, this.power)
      for (let m = 0; m < n_mels; m++) {
        let s = 0
        const off = m * this.nBins
        for (let k = 0; k < this.nBins; k++) s += this.fb[off + k] * this.power[k]
        out[m * frames + t] = Math.log(s + log_eps)
      }
    }
    return out
  }

  /** Per-band normalization with the training-set mean/std, in place. */
  normalize(x: Float32Array): Float32Array {
    const { n_mels, frames, mean, std } = this.cfg
    for (let m = 0; m < n_mels; m++)
      for (let t = 0; t < frames; t++) x[m * frames + t] = (x[m * frames + t] - mean[m]) / std[m]
    return x
  }
}

/** Table-driven real DFT, for window sizes that aren't a power of two (cough: 400). */
function dftPower(n: number) {
  const nBins = n / 2 + 1
  const cos = new Float64Array(nBins * n)
  const sin = new Float64Array(nBins * n)
  for (let k = 0; k < nBins; k++)
    for (let i = 0; i < n; i++) {
      const a = (2 * Math.PI * k * i) / n
      cos[k * n + i] = Math.cos(a)
      sin[k * n + i] = Math.sin(a)
    }
  return (frame: Float64Array, power: Float64Array) => {
    for (let k = 0; k < nBins; k++) {
      let re = 0
      let im = 0
      const off = k * n
      for (let i = 0; i < n; i++) {
        re += frame[i] * cos[off + i]
        im -= frame[i] * sin[off + i]
      }
      power[k] = re * re + im * im
    }
  }
}

/** Iterative radix-2 FFT, for power-of-two window sizes (lung: 1024). */
function fftPower(n: number) {
  const bits = Math.log2(n)
  const rev = new Uint32Array(n)
  for (let i = 0; i < n; i++) {
    let r = 0
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b)
    rev[i] = r
  }
  const cos = new Float64Array(n / 2)
  const sin = new Float64Array(n / 2)
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n)
    sin[i] = -Math.sin((2 * Math.PI * i) / n)
  }
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  return (frame: Float64Array, power: Float64Array) => {
    for (let i = 0; i < n; i++) {
      re[i] = frame[rev[i]]
      im[i] = 0
    }
    for (let size = 2; size <= n; size *= 2) {
      const half = size / 2
      const step = n / size
      for (let s = 0; s < n; s += size)
        for (let j = 0; j < half; j++) {
          const wr = cos[j * step]
          const wi = sin[j * step]
          const a = s + j
          const b = a + half
          const tr = re[b] * wr - im[b] * wi
          const ti = re[b] * wi + im[b] * wr
          re[b] = re[a] - tr
          im[b] = im[a] - ti
          re[a] += tr
          im[a] += ti
        }
    }
    for (let k = 0; k <= n / 2; k++) power[k] = re[k] * re[k] + im[k] * im[k]
  }
}
