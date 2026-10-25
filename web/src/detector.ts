// Streaming cough detector: 16 kHz audio in, cough events out.
//
// Counting works the way cough_prep.py made the training clips: find loud events
// (librosa.effects.split: frames within 30 dB of the peak, at least 100 ms long),
// score the 1 s window centered on each one, and count it as a cough if it passes
// the threshold. Separately, a sliding 1 s window every HOP drives the live meter.
//
// cough_prep.py peak-normalized each *recording* to 0.9, so a live window is scaled
// by the loudest sample of the last PEAK_SECONDS instead (capped at MAX_GAIN, so
// silence is not blown up into noise).
import { LogMel, type CoughConfig } from './features'
import { ort } from './ort'

export const HOP = 4000 // live meter: one score every 0.25 s
const PEAK_SECONDS = 10
const TARGET_PEAK = 0.9
const MAX_GAIN = 50
const SILENCE_RMS = 0.005 // same gate as cough_prep.py, after normalization
// event finding, as librosa.effects.split(top_db=30) with its default frames
const FRAME = 2048
const FRAME_HOP = 512
const TOP_DB = 30
const MIN_EVENT = 1600 // 100 ms
const MIN_EVENT_RMS = 0.002 // about -54 dBFS: ignore "events" in a silent room
export const MIC_WARMUP = 8000 // live: ignore events in the first 0.5 s (mic start click)
// librosa compares each event with the whole recording's peak. Live, we wait this
// long after an event so its loudness is judged against what came next too.
const CONFIRM = 24000 // 1.5 s

export interface Score {
  t: number // seconds since start, end of the 1 s window
  p: number // cough probability, or 0 for skipped (silent) windows
  level: number // RMS of the raw window, for the level meter
}

export interface CoughEvent {
  t: number // seconds since start, center of the cough
  p: number
}

export class Model {
  readonly cfg: CoughConfig
  private lm: LogMel
  private session: ort.InferenceSession
  private queue: Promise<unknown> = Promise.resolve()
  pending = 0

  private constructor(cfg: CoughConfig, session: ort.InferenceSession) {
    this.cfg = cfg
    this.lm = new LogMel(cfg)
    this.session = session
  }

  static async load(base: string): Promise<Model> {
    const cfg: CoughConfig = await (await fetch(`${base}model/cough_config.json`)).json()
    const session = await ort.InferenceSession.create(`${base}model/cough.onnx`, {
      executionProviders: ['wasm'],
    })
    return new Model(cfg, session)
  }

  /** y: 1 s of audio, already normalized. Calls run one at a time. */
  prob(y: Float32Array): Promise<number> {
    this.pending++
    const run = this.queue.then(async () => {
      const x = this.lm.normalize(this.lm.logmel(y))
      const out = await this.session.run({
        logmel: new ort.Tensor('float32', x, [1, 1, this.cfg.n_mels, this.cfg.frames]),
      })
      return 1 / (1 + Math.exp(-(out.logit.data as Float32Array)[0]))
    })
    this.queue = run.catch(() => {}).finally(() => this.pending--)
    return run
  }
}

export class Detector {
  private model: Model
  private sr: number
  private warmup: number
  private buf: Float32Array // the last PEAK_SECONDS of audio, oldest first
  private total = 0 // samples received so far; buf ends at this index
  private nextHop: number
  private nextFrame = FRAME
  private frameRms: number[] = [] // recent envelope, for the "peak" in top_db
  private eventStart = -1 // first sample of the current loud run, or -1
  private eventEnd = 0
  private eventPeak = 0
  // events waiting for CONFIRM samples of audio after their center before scoring
  private waiting: { center: number; peak: number }[] = []
  private inflight = new Set<Promise<void>>()
  onScore: (s: Score) => void = () => {}
  onCough: (e: CoughEvent) => void = () => {}

  constructor(model: Model, warmup = 0) {
    this.model = model
    this.sr = model.cfg.sr
    this.warmup = warmup
    this.buf = new Float32Array(this.sr * PEAK_SECONDS)
    this.nextHop = this.sr
  }

  /** Feed 16 kHz mono samples. Results arrive through onScore / onCough. */
  push(chunk: Float32Array) {
    for (let off = 0; off < chunk.length; ) {
      // advance to the next frame or meter boundary, whichever comes first
      const n = Math.min(chunk.length - off, this.nextFrame - this.total, this.nextHop - this.total)
      this.append(chunk.subarray(off, off + n))
      off += n
      if (this.total === this.nextFrame) {
        this.envelope()
        this.nextFrame += FRAME_HOP
      }
      if (this.total === this.nextHop) {
        this.nextHop += HOP
        // the meter is best-effort: skip a window rather than fall behind
        if (this.model.pending < 2) this.track(this.meter())
      }
      while (this.waiting.length && this.waiting[0].center + CONFIRM <= this.total)
        this.track(this.scoreEvent(this.waiting.shift()!))
    }
  }

  /** Score a whole recording (file test). Returns the meter scores; coughs go to onCough. */
  async run(y: Float32Array): Promise<Score[]> {
    const scores: Score[] = []
    const prev = this.onScore
    this.onScore = (s) => scores.push(s)
    for (let s = 0; s < y.length; s += HOP) {
      this.push(y.subarray(s, Math.min(s + HOP, y.length)))
      await Promise.all(this.inflight) // offline: never skip meter windows
    }
    this.closeEvent() // a cough right at the end of the file
    for (const e of this.waiting.splice(0)) this.track(this.scoreEvent(e))
    await Promise.all(this.inflight)
    this.onScore = prev
    return scores
  }

  private track(p: Promise<void>) {
    this.inflight.add(p)
    void p.finally(() => this.inflight.delete(p))
  }

  private append(x: Float32Array) {
    const L = this.buf.length
    this.buf.copyWithin(0, x.length)
    this.buf.set(x, L - x.length)
    this.total += x.length
  }

  /** Samples [start, start + n) by absolute index, zero outside what we have. */
  private slice(start: number, n: number): Float32Array {
    const out = new Float32Array(n)
    const bufStart = this.total - this.buf.length
    for (let i = 0; i < n; i++) {
      const j = start + i
      if (j >= Math.max(0, bufStart) && j < this.total) out[i] = this.buf[j - bufStart]
    }
    return out
  }

  private gain(): number {
    let peak = 0
    const from = Math.max(0, this.buf.length - this.total)
    for (let i = from; i < this.buf.length; i++) peak = Math.max(peak, Math.abs(this.buf[i]))
    return peak > 0 ? Math.min(TARGET_PEAK / peak, MAX_GAIN) : 0
  }

  private envelope() {
    const f = this.slice(this.total - FRAME, FRAME)
    let ss = 0
    for (const v of f) ss += v * v
    const rms = Math.sqrt(ss / FRAME)
    this.frameRms.push(rms)
    if (this.frameRms.length > (PEAK_SECONDS * this.sr) / FRAME_HOP) this.frameRms.shift()
    const ref = Math.max(...this.frameRms)
    const loud = rms > MIN_EVENT_RMS && 20 * Math.log10(rms / ref) > -TOP_DB
    if (loud) {
      if (this.eventStart < 0) {
        this.eventStart = this.total - FRAME_HOP
        this.eventPeak = 0
      }
      this.eventEnd = this.total
      this.eventPeak = Math.max(this.eventPeak, rms)
    } else {
      this.closeEvent()
    }
  }

  private closeEvent() {
    if (this.eventStart >= this.warmup && this.eventEnd - this.eventStart >= MIN_EVENT)
      this.waiting.push({ center: Math.round((this.eventStart + this.eventEnd) / 2), peak: this.eventPeak })
    this.eventStart = -1
  }

  private async scoreEvent({ center, peak }: { center: number; peak: number }) {
    if (20 * Math.log10(peak / Math.max(...this.frameRms)) <= -TOP_DB) return
    const win = this.slice(Math.max(0, center - this.sr / 2), this.sr)
    const g = this.gain()
    for (let i = 0; i < win.length; i++) win[i] *= g
    const p = await this.model.prob(win)
    if (p >= this.model.cfg.threshold) this.onCough({ t: center / this.sr, p })
  }

  private async meter() {
    const t = this.total / this.sr
    const win = this.slice(this.total - this.sr, this.sr)
    let ss = 0
    for (const v of win) ss += v * v
    const level = Math.sqrt(ss / win.length)
    const g = this.gain()
    if (level * g <= SILENCE_RMS) {
      this.onScore({ t, p: 0, level })
      return
    }
    for (let i = 0; i < win.length; i++) win[i] *= g
    this.onScore({ t, p: await this.model.prob(win), level })
  }
}
