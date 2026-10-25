import { useEffect, useRef, useState } from 'react'
import { decodeFile, SR, startMic, type Mic } from './audio'
import { STOP_COUGH } from './events'
import { LungModel, type LungResult } from './lung'

const BASE = import.meta.env.BASE_URL
const BREATHS = 6
const BREATH_SECONDS = 4 // must match the model's window: 2 s in, 2 s out
const COUNTDOWN = 3

const NAMES: Record<string, string> = {
  normal: 'Normal',
  crackle: 'Crackles',
  wheeze: 'Wheeze',
  both: 'Crackles + wheeze',
}

type Phase =
  | { kind: 'idle' }
  | { kind: 'countdown'; left: number }
  | { kind: 'recording'; elapsed: number }
  | { kind: 'analyzing' }

interface Shown {
  source: string
  result: LungResult
}

export default function LungView() {
  const [model, setModel] = useState<LungModel | null>(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })
  const [shown, setShown] = useState<Shown | null>(null)
  const mic = useRef<Mic | null>(null)
  const wakeLock = useRef<WakeLockSentinel | null>(null)
  const timer = useRef(0)

  useEffect(() => {
    LungModel.load(BASE, setProgress).then(setModel, (e) =>
      setError(`Could not load the lung model: ${e}`),
    )
    return () => {
      cancelAnimationFrame(timer.current)
      mic.current?.stop()
    }
  }, [])

  async function start() {
    if (!model) return
    setError('')
    setShown(null)
    window.dispatchEvent(new Event(STOP_COUGH))
    const chunks: Float32Array[] = []
    let recording = false
    let got = 0
    const need = BREATHS * BREATH_SECONDS * SR
    try {
      mic.current = await startMic(BASE, (x) => {
        if (!recording || got >= need) return
        chunks.push(x)
        got += x.length
      })
    } catch (e) {
      setError(
        window.isSecureContext
          ? `Microphone not available: ${e}`
          : 'The microphone only works over HTTPS (or on localhost).',
      )
      return
    }
    wakeLock.current = (await navigator.wakeLock?.request('screen').catch(() => null)) ?? null

    // countdown while the phone is placed on the chest; that audio is thrown away
    const t0 = performance.now()
    const tick = () => {
      const s = (performance.now() - t0) / 1000
      if (s < COUNTDOWN) {
        setPhase({ kind: 'countdown', left: Math.ceil(COUNTDOWN - s) })
      } else if (got < need) {
        recording = true
        setPhase({ kind: 'recording', elapsed: s - COUNTDOWN })
      } else {
        finish(chunks, need)
        return
      }
      timer.current = requestAnimationFrame(tick)
    }
    tick()
  }

  async function finish(chunks: Float32Array[], need: number) {
    mic.current?.stop()
    mic.current = null
    void wakeLock.current?.release()
    navigator.vibrate?.(80)
    setPhase({ kind: 'analyzing' })
    const y = new Float32Array(need)
    let off = 0
    for (const c of chunks) {
      y.set(c.subarray(0, Math.min(c.length, need - off)), off)
      off += c.length
      if (off >= need) break
    }
    // the pacer makes each 4 s window one breath
    const starts = Array.from({ length: BREATHS }, (_, i) => i * BREATH_SECONDS * SR)
    try {
      setShown({ source: 'your chest', result: await model!.analyze(y, starts) })
    } catch (e) {
      setError(`Analysis failed: ${e}`)
    }
    setPhase({ kind: 'idle' })
  }

  function cancel() {
    cancelAnimationFrame(timer.current)
    mic.current?.stop()
    mic.current = null
    void wakeLock.current?.release()
    setPhase({ kind: 'idle' })
  }

  async function testFile(file: File) {
    if (!model) return
    setError('')
    setShown(null)
    setPhase({ kind: 'analyzing' })
    try {
      const y = await decodeFile(file)
      setShown({ source: file.name, result: await model.analyze(y) })
    } catch (e) {
      setError(`Could not read ${file.name}: ${e}`)
    }
    setPhase({ kind: 'idle' })
  }

  const busy = phase.kind !== 'idle'

  return (
    <>
      <section className="card">
        <h2>How to do a lung check</h2>
        <ol className="steps">
          <li>Sit still in a quiet room.</li>
          <li>Press the bottom of the phone (where the mic is) firmly on bare skin, just below the collarbone.</li>
          <li>Breathe slowly and deeply through your mouth, following the circle: in as it grows, out as it shrinks.</li>
        </ol>
      </section>

      <section className="card live">
        <Pacer phase={phase} />
        {!model ? (
          <>
            <div className="level" title="Download progress">
              <div style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <button className="btn" disabled>
              Loading lung model… {Math.round(progress * 100)}%
            </button>
          </>
        ) : phase.kind === 'countdown' || phase.kind === 'recording' ? (
          <button className="btn stop" onClick={cancel}>
            Cancel
          </button>
        ) : (
          <button className="btn" onClick={start} disabled={busy}>
            {phase.kind === 'analyzing' ? 'Analyzing…' : `Start (${BREATHS} breaths, ${BREATHS * BREATH_SECONDS} s)`}
          </button>
        )}
        {error && <p className="error">{error}</p>}
      </section>

      {shown && <Results shown={shown} />}

      <section className="card">
        <h2>Test with a recording</h2>
        <p className="muted">Pick a lung sound recording (for example from ICBHI). It is checked in 4 s pieces.</p>
        <input
          type="file"
          accept="audio/*"
          disabled={!model || busy}
          onChange={(e) => e.target.files?.[0] && testFile(e.target.files[0])}
        />
      </section>

      <footer className="muted">
        Prototype, not a medical device. The model was trained on stethoscope recordings, and a phone
        mic is not a stethoscope, so results are rough. Talk to a doctor about any symptoms.
      </footer>
    </>
  )
}

function Pacer({ phase }: { phase: Phase }) {
  let scale = 0.55
  let label = 'Ready'
  let sub = ''
  if (phase.kind === 'countdown') {
    label = String(phase.left)
    sub = 'place the phone on your chest'
  } else if (phase.kind === 'recording') {
    const breath = Math.min(BREATHS, Math.floor(phase.elapsed / BREATH_SECONDS) + 1)
    const inBreath = (phase.elapsed % BREATH_SECONDS) / BREATH_SECONDS // 0..1
    const inhale = inBreath < 0.5
    // smooth grow for the first half, shrink for the second
    scale = 0.55 + 0.45 * Math.sin(Math.PI * inBreath)
    label = inhale ? 'Breathe in' : 'Breathe out'
    sub = `breath ${breath} of ${BREATHS}`
  } else if (phase.kind === 'analyzing') {
    label = 'Analyzing'
    sub = 'listening for crackles and wheezes'
  }
  return (
    <div className="pacer" aria-live="polite">
      <div className={`ring ${phase.kind === 'recording' ? 'on' : ''}`} style={{ transform: `scale(${scale})` }} />
      <div className="pacer-text">
        <span className="pacer-label">{label}</span>
        {sub && <span className="unit">{sub}</span>}
      </div>
    </div>
  )
}

function Results({ shown }: { shown: Shown }) {
  const { breaths, counts, tooQuiet } = shown.result
  const n = breaths.length
  const crackles = counts.crackle + counts.both
  const wheezes = counts.wheeze + counts.both
  let headline: string
  if (n === 0) headline = 'The recording was too short to check.'
  else if (crackles === 0 && wheezes === 0) headline = `All ${n} breaths sounded normal.`
  else {
    const parts = []
    if (wheezes) parts.push(`wheezing in ${wheezes} of ${n}`)
    if (crackles) parts.push(`crackles in ${crackles} of ${n}`)
    headline = `Possible ${parts.join(' and ')} breaths.`
  }
  return (
    <section className="card">
      <h2>Result: {shown.source}</h2>
      {tooQuiet && (
        <p className="error left">
          The recording was very quiet. Press the mic firmly on bare skin and try again.
        </p>
      )}
      <p className="headline">{headline}</p>
      <ul className="log full">
        {breaths.map((b, i) => {
          const p = Math.max(...b.probs)
          return (
            <li key={i}>
              <span>
                {n > BREATHS ? `${b.t.toFixed(0)} s` : `Breath ${i + 1}`}
              </span>
              <span>
                <span className={`chip ${b.label}`}>{NAMES[b.label]}</span>{' '}
                <span className="muted">{Math.round(p * 100)}%</span>
              </span>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
