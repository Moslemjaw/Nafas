// Lung sound check: classifies breaths as normal / crackle / wheeze / both with the
// ICBHI-trained model (train_lung.py), the same way it was trained:
//   - the whole recording is peak-normalized to 0.9 (lung_prep.py)
//   - each breath is a 4 s window (the pacer makes one breath last 4 s)
//   - log-mel per window, then each band's mean over the whole recording is subtracted
//     (per-recording normalization, which removes the mic's "coloring")
//   - then the training mean/std, and the model
import { LogMel, type MelConfig } from './features'
import { ort } from './ort'

export interface LungConfig extends MelConfig {
  sr: number
  seconds: number
  labels: string[]
  recording_norm: boolean
}

export interface Breath {
  t: number // start, seconds into the recording
  label: string
  probs: number[] // per label, same order as cfg.labels
}

export interface LungResult {
  breaths: Breath[]
  counts: Record<string, number>
  tooQuiet: boolean
}

const QUIET_RMS = 0.0015 // below this the mic probably isn't touching the chest

export class LungModel {
  readonly cfg: LungConfig
  private lm: LogMel
  private session: ort.InferenceSession

  private constructor(cfg: LungConfig, session: ort.InferenceSession) {
    this.cfg = cfg
    this.lm = new LogMel(cfg)
    this.session = session
  }

  /** Loads the ~45 MB model, reporting download progress from 0 to 1. */
  static async load(base: string, onProgress: (f: number) => void = () => {}): Promise<LungModel> {
    const cfg: LungConfig = await (await fetch(`${base}model/lung_config.json`)).json()
    const res = await fetch(`${base}model/lung.onnx`)
    if (!res.ok || !res.body) throw new Error(`lung.onnx: HTTP ${res.status}`)
    const total = Number(res.headers.get('content-length')) || 0
    const reader = res.body.getReader()
    const parts: Uint8Array[] = []
    let got = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
      got += value.length
      if (total) onProgress(got / total)
    }
    const bytes = new Uint8Array(got)
    let off = 0
    for (const p of parts) {
      bytes.set(p, off)
      off += p.length
    }
    onProgress(1)
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
    return new LungModel(cfg, session)
  }

  /** y: 16 kHz mono. starts: where each breath begins, in samples (default: every 4 s). */
  async analyze(y: Float32Array, starts?: number[]): Promise<LungResult> {
    const { sr, seconds, n_mels, frames, labels } = this.cfg
    const n = seconds * sr
    if (!starts) {
      starts = []
      for (let s = 0; s + n <= y.length; s += n) starts.push(s)
      if (!starts.length && y.length > sr) starts.push(0) // under 4 s: repeat-pad below
    }

    let peak = 0
    let ss = 0
    for (const v of y) {
      peak = Math.max(peak, Math.abs(v))
      ss += v * v
    }
    const tooQuiet = Math.sqrt(ss / Math.max(y.length, 1)) < QUIET_RMS
    const gain = peak > 0 ? 0.9 / peak : 0

    const feats = starts.map((s) => {
      const w = new Float32Array(n)
      const len = Math.min(n, y.length - s)
      for (let i = 0; i < n; i++) w[i] = y[s + (i % len)] * gain // repeat-pad, like fit_cycle
      return this.lm.logmel(w)
    })

    if (this.cfg.recording_norm && feats.length) {
      const bandMean = new Float64Array(n_mels)
      for (const f of feats)
        for (let m = 0; m < n_mels; m++)
          for (let t = 0; t < frames; t++) bandMean[m] += f[m * frames + t]
      for (let m = 0; m < n_mels; m++) bandMean[m] /= feats.length * frames
      for (const f of feats)
        for (let m = 0; m < n_mels; m++)
          for (let t = 0; t < frames; t++) f[m * frames + t] -= bandMean[m]
    }

    const breaths: Breath[] = []
    for (let i = 0; i < feats.length; i++) {
      const x = this.lm.normalize(feats[i])
      const out = await this.session.run({
        logmel: new ort.Tensor('float32', x, [1, 1, n_mels, frames]),
      })
      const z = Array.from(out.logits.data as Float32Array)
      const zmax = Math.max(...z)
      const e = z.map((v) => Math.exp(v - zmax))
      const sum = e.reduce((a, b) => a + b, 0)
      const probs = e.map((v) => v / sum)
      breaths.push({ t: starts[i] / sr, label: labels[probs.indexOf(Math.max(...probs))], probs })
    }

    const counts = Object.fromEntries(labels.map((l) => [l, 0]))
    for (const b of breaths) counts[b.label]++
    return { breaths, counts, tooQuiet }
  }
}
