// Log-mel features, identical to train.py's logmel():
//   librosa.feature.melspectrogram(y, sr=16000, n_fft=400, hop_length=160, n_mels=64,
//                                  fmin=50, fmax=8000)   (center=True, zero padding)
//   log(mel + 1e-6), then per-band (x - mean) / std
// The mel filterbank, mean and std come from cough_config.json (train.py web).

export interface CoughConfig {
  sr: number
  n_fft: number
  hop: number
  n_mels: number
  frames: number
  log_eps: number
  threshold: number
  mean: number[]
  std: number[]
  mel_fb: number[][] // [n_mels][n_fft / 2 + 1]
}

export class LogMel {
  readonly cfg: CoughConfig
  private nBins: number
  private win: Float64Array
  private cos: Float64Array // [nBins][n_fft]
  private sin: Float64Array
  private fb: Float64Array // [n_mels][nBins]
  private frame: Float64Array
  private power: Float64Array

  constructor(cfg: CoughConfig) {
    this.cfg = cfg
    const n = cfg.n_fft
    this.nBins = n / 2 + 1
    // periodic Hann (scipy get_window('hann', n, fftbins=True))
    this.win = new Float64Array(n)
    for (let i = 0; i < n; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n)
    // n_fft = 400 is not a power of two, so use a table-driven real DFT
    this.cos = new Float64Array(this.nBins * n)
    this.sin = new Float64Array(this.nBins * n)
    for (let k = 0; k < this.nBins; k++)
      for (let i = 0; i < n; i++) {
        const a = (2 * Math.PI * k * i) / n
        this.cos[k * n + i] = Math.cos(a)
        this.sin[k * n + i] = Math.sin(a)
      }
    this.fb = new Float64Array(cfg.n_mels * this.nBins)
    cfg.mel_fb.forEach((row, m) => row.forEach((v, k) => (this.fb[m * this.nBins + k] = v)))
    this.frame = new Float64Array(n)
    this.power = new Float64Array(this.nBins)
  }

  /** y: exactly sr samples (1 s). Returns raw log-mel, row-major [n_mels][frames]. */
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
      for (let k = 0; k < this.nBins; k++) {
        let re = 0
        let im = 0
        const off = k * n
        for (let i = 0; i < n; i++) {
          re += this.frame[i] * this.cos[off + i]
          im -= this.frame[i] * this.sin[off + i]
        }
        this.power[k] = re * re + im * im
      }
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
