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

export default function LungView() {
  const [model, setModel] = useState<LungModel | null>(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })
  const [plan, setPlan] = useState<Spot[] | null>(null)
  const [step, setStep] = useState(0)
  const [results, setResults] = useState<Record<string, LungResult>>({})
  const [fileResult, setFileResult] = useState<{ name: string; result: LungResult } | null>(null)
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

  async function record(spot: Spot) {
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
    const countdown = spot.side === 'back' ? 5 : 3
    const t0 = performance.now()
    const tick = () => {
      const s = (performance.now() - t0) / 1000
      if (s < countdown) {
        setPhase({ kind: 'countdown', left: Math.ceil(countdown - s) })
      } else if (got < need) {
        recording = true
        setPhase({ kind: 'recording', elapsed: s - countdown })
      } else {
        void finish(spot, chunks, need)
        return
      }
      timer.current = requestAnimationFrame(tick)
    }
    tick()
  }

  async function finish(spot: Spot, chunks: Float32Array[], need: number) {
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
      setResults((prev) => ({ ...prev, [spot.id]: r }))
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
    setPhase({ kind: 'analyzing' })
    try {
      const y = await decodeFile(file)
      setFileResult({ name: file.name, result: await model.analyze(y) })
    } catch (e) {
      setError(`Could not read ${file.name}: ${e}`)
    }
    setPhase({ kind: 'idle' })
  }

  const busy = phase.kind !== 'idle'
  const done = plan !== null && plan.every((s) => results[s.id])
  const spot = plan && !done ? plan[step] : null
  const spotResult = spot ? results[spot.id] : undefined

  return (
    <>
      {!plan && (
        <section className="card">
          <h2>Lung check</h2>
          <p className="muted">
            The phone listens to your breathing at a few spots on the chest, like a doctor with a
            stethoscope. Each spot takes {BREATHS} slow breaths ({BREATHS * BREATH_SECONDS} s).
          </p>
          <ul className="steps">
            <li>Sit still in a quiet room, with bare skin where the phone goes.</li>
            <li>Press the bottom edge of the phone (where the mic is) firmly on the skin.</li>
            <li>Breathe slowly and deeply through your mouth, following the circle.</li>
          </ul>
          <div className="choices">
            <button className="btn" onClick={() => choose(PLANS.four)} disabled={!model}>
              4 spots, front and back
              <span className="btn-sub">recommended, needs a helper for the back</span>
            </button>
            <button className="btn secondary" onClick={() => choose(PLANS.two)} disabled={!model}>
              2 spots, front only
              <span className="btn-sub">if you are on your own</span>
            </button>
          </div>
          {!model && <LoadBar progress={progress} />}
          {error && <p className="error">{error}</p>}
        </section>
      )}

      {spot && (
        <>
          <section className="card">
            <div className="row">
              <h2>
                Step {step + 1} of {plan!.length}: {spot.name}
              </h2>
              <button className="link" onClick={startOver} disabled={busy}>
                Start over
              </button>
            </div>
            <ChestDiagram
              side={spot.side}
              spots={plan!
                .map((s, i) => ({ s, i }))
                .filter(({ s }) => s.side === spot.side)
                .map(({ s, i }) => ({
                  spot: s,
                  n: i + 1,
                  state: s.id === spot.id && !results[s.id] ? 'current' : spotState(results[s.id]),
                }))}
            />
            <p className="tip">
              <b>Where:</b> {spot.tip}
            </p>
          </section>

          {!spotResult ? (
            <section className="card live">
              <Pacer phase={phase} back={spot.side === 'back'} />
              {phase.kind === 'countdown' || phase.kind === 'recording' ? (
                <button className="btn stop" onClick={cancel}>
                  Cancel
                </button>
              ) : (
                <button className="btn" onClick={() => record(spot)} disabled={busy}>
                  {phase.kind === 'analyzing' ? 'Analyzing…' : `Start ${spot.name.toLowerCase()}`}
                </button>
              )}
              {error && <p className="error">{error}</p>}
            </section>
          ) : (
            <section className="card">
              <SpotResult spot={spot} result={spotResult} />
              <div className="choices">
                <button className="btn" onClick={() => setStep(step + 1)}>
                  {step + 1 < plan!.length ? `Next: ${plan![step + 1].name}` : 'See summary'}
                </button>
                <button
                  className="btn secondary"
                  onClick={() =>
                    setResults((prev) => {
                      const { [spot.id]: _, ...rest } = prev
                      return rest
                    })
                  }
                >
                  Redo {spot.name.toLowerCase()}
                </button>
              </div>
            </section>
          )}
        </>
      )}

      {done && <Summary plan={plan!} results={results} onAgain={startOver} />}

      <section className="card">
        <h2>Test with a recording</h2>
        <p className="muted">Pick a lung sound recording (for example from ICBHI). It is checked in 4 s pieces.</p>
        <input
          type="file"
          accept="audio/*"
          disabled={!model || busy}
          onChange={(e) => e.target.files?.[0] && testFile(e.target.files[0])}
        />
        {fileResult && (
          <div className="result">
            <SpotResult name={fileResult.name} result={fileResult.result} />
          </div>
        )}
      </section>

      <footer className="muted">
        Prototype, not a medical device. The model was trained on stethoscope recordings, and a phone
        mic is not a stethoscope, so results are rough. Talk to a doctor about any symptoms.
      </footer>
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

function Pacer({ phase, back }: { phase: Phase; back: boolean }) {
  let scale = 0.55
  let label = 'Ready'
  let sub = ''
  if (phase.kind === 'countdown') {
    label = String(phase.left)
    sub = back ? 'helper: place the phone on the back' : 'place the phone on your chest'
  } else if (phase.kind === 'recording') {
    const breath = Math.min(BREATHS, Math.floor(phase.elapsed / BREATH_SECONDS) + 1)
    const inBreath = (phase.elapsed % BREATH_SECONDS) / BREATH_SECONDS // 0..1
    scale = 0.55 + 0.45 * Math.sin(Math.PI * inBreath) // grow, then shrink
    label = inBreath < 0.5 ? 'Breathe in' : 'Breathe out'
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

function SpotResult({ spot, name, result }: { spot?: Spot; name?: string; result: LungResult }) {
  const n = result.breaths.length
  return (
    <>
      <h2>Result: {spot?.name ?? name}</h2>
      {result.tooQuiet && (
        <p className="error left">
          The recording was very quiet. Press the mic firmly on bare skin and try again.
        </p>
      )}
      <p className="headline">{headline(result)}</p>
      <ul className="log full">
        {result.breaths.map((b, i) => (
          <li key={i}>
            <span>{n > BREATHS ? `${b.t.toFixed(0)} s` : `Breath ${i + 1}`}</span>
            <span>
              <span className={`chip ${b.label}`}>{NAMES[b.label]}</span>{' '}
              <span className="muted">{Math.round(Math.max(...b.probs) * 100)}%</span>
            </span>
          </li>
        ))}
      </ul>
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
          <ChestDiagram
            key={side}
            side={side}
            spots={plan
              .map((s, i) => ({ s, i }))
              .filter(({ s }) => s.side === side)
              .map(({ s, i }) => ({ spot: s, n: i + 1, state: spotState(results[s.id]) }))}
          />
        ))}
      </div>
      <div className="legend">
        <span className="dot normal" /> mostly normal <span className="dot abnormal" /> {ABNORMAL_BREATHS}+ breaths
        with crackles or wheezes
      </div>
      <ul className="log full">
        {plan.map((s, i) => (
          <li key={s.id}>
            <span>
              {i + 1}. {s.name}
            </span>
            <span className="muted">{headline(results[s.id])}</span>
          </li>
        ))}
      </ul>
      {differ.map((side) => (
        <p key={side} className="tip">
          <b>Left and right sound different at the {side}.</b> One-sided sounds are worth mentioning
          to a doctor.
        </p>
      ))}
    </section>
  )
}
