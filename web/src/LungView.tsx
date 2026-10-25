import { useEffect, useRef, useState } from 'react'
import { decodeFile, SR, startMic, type Mic } from './audio'
import { ChestDiagram } from './ChestDiagram'
import { SPOTS, type Side, type Spot, type SpotState } from './spots'
import { STOP_COUGH } from './events'
import { LungModel, type LungResult } from './lung'

const BASE = import.meta.env.BASE_URL
const BREATHS = 6
const BREATH_SECONDS = 4 // must match the model's window: 2 s in, 2 s out
const ABNORMAL_BREATHS = 2 // a spot is flagged when at least this many breaths sound abnormal

const NAMES: Record<string, string> = {
  normal: 'Normal',
  crackle: 'Crackles',
  wheeze: 'Wheeze',
  both: 'Crackles + wheeze',
}

const PLANS = {
  four: ['front-left', 'front-right', 'back-left', 'back-right'],
  two: ['front-left', 'front-right'],
}

type Phase =
  | { kind: 'idle' }
  | { kind: 'countdown'; left: number }
  | { kind: 'recording'; elapsed: number }
  | { kind: 'analyzing' }

const spotById = (id: string) => SPOTS.find((s) => s.id === id)!

function abnormalBreaths(r: LungResult) {
  return r.breaths.filter((b) => b.label !== 'normal').length
}

function spotState(r: LungResult | undefined): SpotState {
  if (!r) return 'todo'
  return abnormalBreaths(r) >= ABNORMAL_BREATHS ? 'abnormal' : 'normal'
}

function headline({ breaths, counts }: LungResult) {
  const n = breaths.length
  const crackles = counts.crackle + counts.both
  const wheezes = counts.wheeze + counts.both
  if (n === 0) return 'The recording was too short to check.'
  if (crackles === 0 && wheezes === 0) return `All ${n} breaths sounded normal.`
  const parts = []
  if (wheezes) parts.push(`wheezing in ${wheezes} of ${n}`)
  if (crackles) parts.push(`crackles in ${crackles} of ${n}`)
  return `Possible ${parts.join(' and ')} breaths.`
}

/** One short line per spot for the summary list, e.g. "3 crackles · 1 wheeze". */
function shortLine({ counts }: LungResult) {
  const crackles = counts.crackle + counts.both
  const wheezes = counts.wheeze + counts.both
  if (!crackles && !wheezes) return 'Normal'
  const parts = []
  if (crackles) parts.push(`${crackles} crackle${crackles > 1 ? 's' : ''}`)
  if (wheezes) parts.push(`${wheezes} wheeze${wheezes > 1 ? 's' : ''}`)
  return parts.join(' · ')
}

/** The spots of one side of the plan, numbered in plan order, with their state. */
function sideSpots(plan: Spot[], side: Side, results: Record<string, LungResult>, current?: Spot) {
  return plan
    .map((s, i) => ({ s, i }))
    .filter(({ s }) => s.side === side)
    .map(({ s, i }) => ({
      spot: s,
      n: i + 1,
      state: (s.id === current?.id && !results[s.id] ? 'current' : spotState(results[s.id])) as SpotState,
    }))
}

export default function LungView() {
  const [model, setModel] = useState<LungModel | null>(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })
  const [plan, setPlan] = useState<Spot[] | null>(null)
  const [step, setStep] = useState(0)
  const [results, setResults] = useState<Record<string, LungResult>>({})
  const [fileResult, setFileResult] = useState<{ name: string; result: LungResult } | null>(null)
  const [fileBusy, setFileBusy] = useState(false)
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

  const done = plan !== null && plan.every((s) => results[s.id])
  const spot = plan && !done ? plan[step] : null
  const spotResult = spot ? results[spot.id] : undefined
  const active = phase.kind !== 'idle' && !fileBusy

  // every new screen starts at the top, so nothing important is below the fold
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [plan, step, done, spotResult])

  // the recording view covers the screen; keep the page behind it from scrolling
  useEffect(() => {
    document.body.style.overflow = active ? 'hidden' : ''
    return () => {
      document.body.style.overflow = ''
    }
  }, [active])

  function choose(ids: string[]) {
    setPlan(ids.map(spotById))
    setStep(0)
    setResults({})
    setError('')
  }

  function startOver() {
    cancel()
    setPlan(null)
    setResults({})
    setStep(0)
  }

  function redo(s: Spot) {
    setResults((prev) => {
      const next = { ...prev }
      delete next[s.id]
      return next
    })
  }

  async function record(s: Spot) {
    if (!model) return
    setError('')
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

    // time to place the phone (more for the back, where a helper holds it); that audio is dropped
    const countdown = s.side === 'back' ? 5 : 3
    const t0 = performance.now()
    const tick = () => {
      const sec = (performance.now() - t0) / 1000
      if (sec < countdown) {
        setPhase({ kind: 'countdown', left: Math.ceil(countdown - sec) })
      } else if (got < need) {
        recording = true
        setPhase({ kind: 'recording', elapsed: sec - countdown })
      } else {
        void finish(s, chunks, need)
        return
      }
      timer.current = requestAnimationFrame(tick)
    }
    tick()
  }

  async function finish(s: Spot, chunks: Float32Array[], need: number) {
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
      const r = await model!.analyze(y, starts)
      setResults((prev) => ({ ...prev, [s.id]: r }))
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
    setFileResult(null)
    setFileBusy(true)
    try {
      const y = await decodeFile(file)
      setFileResult({ name: file.name, result: await model.analyze(y) })
    } catch (e) {
      setError(`Could not read ${file.name}: ${e}`)
    }
    setFileBusy(false)
  }

  return (
    <>
      {!plan && (
        <section className="card">
          <h2>Lung check</h2>
          <p className="muted">
            The phone listens to your breathing at a few spots on the chest, like a doctor with a
            stethoscope: {BREATHS} slow breaths at each spot.
          </p>
          <ul className="steps">
            <li>Sit still in a quiet room, with bare skin where the phone goes.</li>
            <li>Press the bottom edge of the phone (where the mic is) firmly on the skin.</li>
            <li>Breathe slowly through your mouth, following the circle.</li>
          </ul>
          <div className="choices">
            <button className="btn" onClick={() => choose(PLANS.four)} disabled={!model}>
              4 spots, front and back
              <span className="btn-sub">recommended · needs a helper for the back · about 2½ min</span>
            </button>
            <button className="btn secondary" onClick={() => choose(PLANS.two)} disabled={!model}>
              2 spots, front only
              <span className="btn-sub">on your own · about 1 min</span>
            </button>
          </div>
          {!model && <LoadBar progress={progress} />}
          {error && <p className="error">{error}</p>}
        </section>
      )}

      {plan && spot && (
        <section className="card step">
          <div className="row">
            <Stepper plan={plan} step={step} results={results} />
            <button className="link" onClick={startOver}>
              Start over
            </button>
          </div>
          <h2 className="step-title">
            {spot.name}
            <span className="muted"> · step {step + 1} of {plan.length}</span>
          </h2>

          {!spotResult ? (
            <>
              <ChestDiagram side={spot.side} spots={sideSpots(plan, spot.side, results, spot)} />
              <p className="tip">
                <b>Where:</b> {spot.tip}
              </p>
              <button className="btn" onClick={() => record(spot)} disabled={active}>
                Start · {BREATHS} breaths
              </button>
            </>
          ) : (
            <>
              <SpotResult result={spotResult} />
              <div className="choices">
                <button className="btn" onClick={() => setStep(step + 1)}>
                  {step + 1 < plan.length ? `Next: ${plan[step + 1].name}` : 'See summary'}
                </button>
                <button className="btn secondary" onClick={() => redo(spot)}>
                  Redo this spot
                </button>
              </div>
            </>
          )}
          {error && <p className="error">{error}</p>}
        </section>
      )}

      {plan && done && <Summary plan={plan} results={results} onAgain={startOver} />}

      <details className="card more">
        <summary>Test with a recording</summary>
        <p className="muted">Pick a lung sound recording (for example from ICBHI). It is checked in 4 s pieces.</p>
        <input
          type="file"
          accept="audio/*"
          disabled={!model || fileBusy || active}
          onChange={(e) => e.target.files?.[0] && testFile(e.target.files[0])}
        />
        {fileBusy && <p className="muted">Analyzing…</p>}
        {fileResult && (
          <div className="result">
            <p className="headline">{fileResult.name}</p>
            <SpotResult result={fileResult.result} />
          </div>
        )}
      </details>

      <footer className="muted">
        Prototype, not a medical device. The model was trained on stethoscope recordings, and a phone
        mic is not a stethoscope, so results are rough. Talk to a doctor about any symptoms.
      </footer>

      {active && spot && plan && (
        <Session phase={phase} spot={spot} plan={plan} step={step} results={results} onCancel={cancel} />
      )}
    </>
  )
}

function LoadBar({ progress }: { progress: number }) {
  return (
    <>
      <div className="level" title="Download progress">
        <div style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      <p className="muted center">Loading lung model… {Math.round(progress * 100)}%</p>
    </>
  )
}

/** Small numbered dots for the steps: done ones colored by result, the current one ringed. */
function Stepper({ plan, step, results }: { plan: Spot[]; step: number; results: Record<string, LungResult> }) {
  return (
    <ol className="stepper" aria-label="Progress">
      {plan.map((s, i) => (
        <li
          key={s.id}
          className={`${spotState(results[s.id])} ${i === step ? 'now' : ''}`}
          title={s.name}
        >
          {i + 1}
        </li>
      ))}
    </ol>
  )
}

/** Full-screen view while the countdown, breathing and analysis run: no scrolling. */
function Session({
  phase,
  spot,
  plan,
  step,
  results,
  onCancel,
}: {
  phase: Phase
  spot: Spot
  plan: Spot[]
  step: number
  results: Record<string, LungResult>
  onCancel: () => void
}) {
  let scale = 0.6
  let label = ''
  let sub = ''
  let breath = 0 // breaths finished
  let inBreath = 0
  if (phase.kind === 'countdown') {
    label = String(phase.left)
    sub = spot.side === 'back' ? 'Helper: place the phone on the back' : 'Place the phone on your chest'
  } else if (phase.kind === 'recording') {
    breath = Math.min(BREATHS, Math.floor(phase.elapsed / BREATH_SECONDS))
    inBreath = (phase.elapsed % BREATH_SECONDS) / BREATH_SECONDS // 0..1
    scale = 0.6 + 0.4 * Math.sin(Math.PI * inBreath) // grow, then shrink
    label = inBreath < 0.5 ? 'Breathe in' : 'Breathe out'
    sub = 'slowly, through your mouth'
  } else if (phase.kind === 'analyzing') {
    breath = BREATHS
    label = 'Analyzing'
    sub = 'listening for crackles and wheezes'
  }
  return (
    <div className="session" role="dialog" aria-modal="true" aria-label={`Recording ${spot.name}`}>
      <div className="session-top">
        <div className="session-spot">
          <ChestDiagram side={spot.side} spots={sideSpots(plan, spot.side, results, spot)} caption={false} />
        </div>
        <div>
          <div className="session-name">{spot.name}</div>
          <div className="muted">
            Step {step + 1} of {plan.length}
          </div>
        </div>
      </div>

      <div className="session-pacer" aria-live="polite">
        <div
          className={`ring ${phase.kind === 'recording' ? 'on' : ''} ${phase.kind === 'analyzing' ? 'spin' : ''}`}
          style={{ transform: `scale(${scale})` }}
        />
        <div className="pacer-text">
          <span className="pacer-label">{label}</span>
          <span className="unit">{sub}</span>
        </div>
      </div>

      <div className="breath-dots" aria-label={`Breath ${Math.min(breath + 1, BREATHS)} of ${BREATHS}`}>
        {Array.from({ length: BREATHS }, (_, i) => (
          <span
            key={i}
            className={i < breath ? 'done' : i === breath && phase.kind === 'recording' ? 'now' : ''}
          />
        ))}
      </div>

      {phase.kind !== 'analyzing' && (
        <button className="btn secondary session-cancel" onClick={onCancel}>
          Cancel
        </button>
      )}
    </div>
  )
}

function SpotResult({ result }: { result: LungResult }) {
  const n = result.breaths.length
  return (
    <>
      {result.tooQuiet && (
        <p className="error left">
          The recording was very quiet. Press the mic firmly on bare skin and redo this spot.
        </p>
      )}
      <p className="headline">{headline(result)}</p>
      <ol className="breath-grid">
        {result.breaths.map((b, i) => (
          <li key={i} className={b.label}>
            <span className="muted">{n > BREATHS ? `${b.t.toFixed(0)} s` : `Breath ${i + 1}`}</span>
            <span className="breath-label">{NAMES[b.label]}</span>
            <span className="muted">{Math.round(Math.max(...b.probs) * 100)}%</span>
          </li>
        ))}
      </ol>
    </>
  )
}

function Summary({
  plan,
  results,
  onAgain,
}: {
  plan: Spot[]
  results: Record<string, LungResult>
  onAgain: () => void
}) {
  const sides = (['front', 'back'] as Side[]).filter((side) => plan.some((s) => s.side === side))
  const flagged = plan.filter((s) => spotState(results[s.id]) === 'abnormal')

  // left vs right at the same height: one side flagged and the other not
  const differ = sides.filter((side) => {
    const l = results[`${side}-left`]
    const r = results[`${side}-right`]
    return l && r && spotState(l) !== spotState(r)
  })

  return (
    <section className="card">
      <div className="row">
        <h2>Summary</h2>
        <button className="link" onClick={onAgain}>
          Check again
        </button>
      </div>
      <p className="headline">
        {flagged.length === 0
          ? `All ${plan.length} spots sounded mostly normal.`
          : `Possible abnormal sounds at ${flagged.map((s) => s.name.toLowerCase()).join(', ')}.`}
      </p>
      <div className="diagrams">
        {sides.map((side) => (
          <ChestDiagram key={side} side={side} spots={sideSpots(plan, side, results)} />
        ))}
      </div>
      <div className="legend">
        <span>
          <span className="dot normal" /> mostly normal
        </span>
        <span>
          <span className="dot abnormal" /> {ABNORMAL_BREATHS}+ of {BREATHS} breaths abnormal
        </span>
      </div>
      <ul className="log full spot-list">
        {plan.map((s, i) => (
          <li key={s.id}>
            <span>
              <span className={`dot ${spotState(results[s.id])}`} /> {i + 1}. {s.name}
            </span>
            <span className="muted">{shortLine(results[s.id])}</span>
          </li>
        ))}
      </ul>
      {differ.length > 0 && (
        <p className="tip">
          <b>Left and right sound different at the {differ.join(' and ')}.</b> One-sided sounds are
          worth mentioning to a doctor.
        </p>
      )}
    </section>
  )
}
