// Big-screen view for a computer (open the app with #display). Shows a QR code;
// the phone that scans it mirrors its cough count and lung check here, live.
import { useEffect, useRef, useState } from 'react'
import QRCode from 'qrcode'
import { ChestDiagram } from './ChestDiagram'
import type { CoughState, LungState, ScreenState } from './feed'
import { Link, newPairing, pairingUrl, type LinkStatus, type Pairing } from './link'
import { SPOTS, type Side, type SpotState } from './spots'

const SCREEN_KEY = 'nafas-screen'
const OFFLINE_AFTER_MS = 15000 // the phone sends at least every 5 s while connected
const THRESHOLD = 0.83 // cough model threshold, for the live bar
const BREATHS = 6
const BREATH_SECONDS = 4
const MINUTES = 15

function loadPairing(): Pairing {
  try {
    const saved = sessionStorage.getItem(SCREEN_KEY)
    if (saved) return JSON.parse(saved) as Pairing
  } catch {
    /* fall through to a new pairing */
  }
  return savePairing(newPairing())
}

function savePairing(p: Pairing) {
  try {
    sessionStorage.setItem(SCREEN_KEY, JSON.stringify(p))
  } catch {
    /* private mode: a reload makes a new QR code */
  }
  return p
}

const spotById = (id: string) => SPOTS.find((s) => s.id === id)!

function spotLine(labels: string[]) {
  const crackles = labels.filter((l) => l === 'crackle' || l === 'both').length
  const wheezes = labels.filter((l) => l === 'wheeze' || l === 'both').length
  if (!crackles && !wheezes) return 'Normal'
  const parts = []
  if (crackles) parts.push(`${crackles} crackle${crackles > 1 ? 's' : ''}`)
  if (wheezes) parts.push(`${wheezes} wheeze${wheezes > 1 ? 's' : ''}`)
  return parts.join(' · ')
}

export default function DisplayView() {
  const [pairing, setPairing] = useState(loadPairing)
  const [qr, setQr] = useState('')
  const [status, setStatus] = useState<LinkStatus>('connecting')
  const [state, setState] = useState<ScreenState | null>(null)
  const [receivedAt, setReceivedAt] = useState(0)
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    void QRCode.toDataURL(pairingUrl(pairing), { margin: 1, width: 480, errorCorrectionLevel: 'M' }).then(setQr)
    const link = new Link(pairing, 'screen')
    link.onStatus = setStatus
    link.onMessage = (d) => {
      if ((d as ScreenState)?.v !== 1) return
      setState(d as ScreenState)
      setReceivedAt(Date.now())
    }
    return () => link.close()
  }, [pairing])

  // tick: smoothly while the breathing circle runs, otherwise once a second
  const live = state?.lung.phase.kind === 'recording' || state?.lung.phase.kind === 'countdown'
  useEffect(() => {
    if (live) {
      let raf = 0
      const loop = () => {
        setNow(Date.now())
        raf = requestAnimationFrame(loop)
      }
      raf = requestAnimationFrame(loop)
      return () => cancelAnimationFrame(raf)
    }
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [live])

  function pairAnother() {
    setState(null)
    setReceivedAt(0)
    setPairing(savePairing(newPairing()))
  }

  const phoneOnline = state !== null && now - receivedAt < OFFLINE_AFTER_MS

  if (!state) {
    return (
      <div className="display">
        <div className="d-pair">
          <h1>
            Nafas <span className="muted">live screen</span>
          </h1>
          {qr ? <img className="d-qr" src={qr} alt="QR code to pair a phone" /> : <div className="d-qr" />}
          <ol className="d-steps">
            <li>Open the camera on the phone and scan this code.</li>
            <li>Tap the link: the Nafas app opens, paired with this screen.</li>
            <li>Use the app as usual. Coughs and lung checks appear here live.</li>
          </ol>
          <p className="muted">
            {status === 'connected' ? 'Ready, waiting for the phone…' : 'Connecting…'} Only numbers and
            labels are sent, end-to-end encrypted. No audio ever leaves the phone.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="display">
      <header className="d-top">
        <h1>
          Nafas <span className="muted">live</span>
        </h1>
        <span className={`pair-badge ${phoneOnline ? 'connected' : 'offline'}`}>
          <span className="pair-dot" />
          {phoneOnline
            ? 'Phone connected'
            : `Phone offline · last update ${Math.round((now - receivedAt) / 1000)} s ago`}
        </span>
        <button className="link" onClick={pairAnother}>
          Pair another phone
        </button>
      </header>
      <div className="d-grid">
        <section className={`d-card ${state.tab === 'cough' ? 'active' : ''}`}>
          <CoughPanel cough={state.cough} now={now} />
        </section>
        <section className={`d-card ${state.tab === 'lung' ? 'active' : ''}`}>
          <LungPanel lung={state.lung} now={now} receivedAt={receivedAt} />
        </section>
      </div>
    </div>
  )
}

function CoughPanel({ cough, now }: { cough: CoughState; now: number }) {
  // pulse the number when a new cough arrives
  const prev = useRef(cough.count)
  const [bump, setBump] = useState(false)
  useEffect(() => {
    if (cough.count > prev.current) {
      setBump(true)
      const id = setTimeout(() => setBump(false), 700)
      prev.current = cough.count
      return () => clearTimeout(id)
    }
    prev.current = cough.count
  }, [cough.count])

  const perMinute = Array.from({ length: MINUTES }, (_, i) => {
    const end = now - (MINUTES - 1 - i) * 60000
    return cough.times.filter((t) => t > end - 60000 && t <= end).length
  })
  const maxPerMinute = Math.max(3, ...perMinute)
  const last = cough.times.length ? cough.times[cough.times.length - 1] : null
  const recent = cough.times.slice(-5).reverse()
  const recentProbs = cough.probs.slice(-5).reverse()

  return (
    <>
      <div className="d-head">
        <h2>Coughs</h2>
        <span className={`chip ${cough.listening ? 'on' : ''}`}>{cough.listening ? 'Listening' : 'Not listening'}</span>
      </div>
      <div className={`d-count ${bump ? 'bump' : ''}`}>{cough.count}</div>
      <p className="muted d-sub">
        {last ? `Last cough at ${new Date(last).toLocaleTimeString()}` : 'No coughs yet'}
      </p>

      <div className="d-label">
        <span>Cough probability now</span>
        <b>{cough.listening ? `${Math.round(cough.p * 100)}%` : '–'}</b>
      </div>
      <div className="d-meter">
        <div
          className={cough.p >= THRESHOLD ? 'hot' : ''}
          style={{ width: `${cough.listening ? Math.round(cough.p * 100) : 0}%` }}
        />
        <span className="d-thr" style={{ left: `${THRESHOLD * 100}%` }} title="Threshold" />
      </div>

      <div className="d-label">
        <span>Coughs per minute, last {MINUTES} min</span>
      </div>
      <svg className="d-bars" viewBox={`0 0 ${MINUTES * 10} 60`} preserveAspectRatio="none" role="img" aria-label="Coughs per minute">
        {perMinute.map((n, i) => (
          <rect key={i} x={i * 10 + 1} width="8" y={60 - (n / maxPerMinute) * 56} height={Math.max((n / maxPerMinute) * 56, 0.8)} />
        ))}
      </svg>

      {recent.length > 0 && (
        <ul className="log d-recent">
          {recent.map((t, i) => (
            <li key={t}>
              <span>{new Date(t).toLocaleTimeString()}</span>
              <span className="muted">{Math.round(recentProbs[i] * 100)}% sure</span>
            </li>
          ))}
        </ul>
      )}
    </>
  )
}

function LungPanel({ lung, now, receivedAt }: { lung: LungState; now: number; receivedAt: number }) {
  if (!lung.plan) {
    return (
      <>
        <div className="d-head">
          <h2>Lung check</h2>
        </div>
        <p className="muted d-empty">Not started on the phone yet.</p>
      </>
    )
  }
  const plan = lung.plan.map(spotById)
  const current = !lung.done ? plan[lung.step] : null
  const sides = (['front', 'back'] as Side[]).filter((side) => plan.some((s) => s.side === side))
  const stateOf = (id: string): SpotState => {
    const r = lung.results[id]
    if (r) return r.flagged ? 'abnormal' : 'normal'
    return current?.id === id ? 'current' : 'todo'
  }
  const flagged = plan.filter((s) => lung.results[s.id]?.flagged)

  // Time since the phone stamped this update. Phones and computers sync their clocks
  // over the internet, so this undoes the network delay; only if the two clocks are
  // clearly far apart, fall back to when the message arrived.
  const phase = lung.phase
  let since = 'at' in phase ? (now - phase.at) / 1000 : 0
  if (since < -2 || since > 60) since = (now - receivedAt) / 1000
  const breathing = phase.kind === 'recording' || phase.kind === 'countdown'

  return (
    <>
      <div className="d-head">
        <h2>Lung check</h2>
        <span className="chip">
          {lung.done ? 'Done' : `Step ${lung.step + 1} of ${plan.length}${current ? `: ${current.name}` : ''}`}
        </span>
      </div>

      {breathing && current ? (
        <Mirror phase={phase} since={since} spot={current.name} />
      ) : (
        <>
          <div className="d-diagrams">
            {sides.map((side) => (
              <ChestDiagram
                key={side}
                side={side}
                spots={plan
                  .map((s, i) => ({ s, i }))
                  .filter(({ s }) => s.side === side)
                  .map(({ s, i }) => ({ spot: s, n: i + 1, state: stateOf(s.id) }))}
              />
            ))}
          </div>
          {phase.kind === 'analyzing' && <p className="headline center">Analyzing {current?.name.toLowerCase()}…</p>}
          {lung.done && (
            <p className="headline center">
              {flagged.length === 0
                ? `All ${plan.length} spots sounded mostly normal.`
                : `Possible abnormal sounds at ${flagged.map((s) => s.name.toLowerCase()).join(', ')}.`}
            </p>
          )}
        </>
      )}

      <ul className="log full spot-list">
        {plan.map((s, i) => {
          const r = lung.results[s.id]
          return (
            <li key={s.id}>
              <span>
                <span className={`dot ${stateOf(s.id) === 'current' ? '' : stateOf(s.id)}`} /> {i + 1}. {s.name}
              </span>
              <span className="muted">{r ? spotLine(r.labels) : s.id === current?.id ? 'in progress' : 'not yet'}</span>
            </li>
          )
        })}
      </ul>
    </>
  )
}

/** The phone's breathing circle, mirrored and moved on locally between updates. */
function Mirror({ phase, since, spot }: { phase: ScreenState['lung']['phase']; since: number; spot: string }) {
  // Seconds into the breathing on the phone's timeline (negative = still counting down).
  // Countdown and recording updates are both stamped with the phone's clock, so they
  // describe the same timeline: a late update can't move the circle backwards.
  let t = -Infinity
  if (phase.kind === 'countdown') t = since - phase.left
  else if (phase.kind === 'recording') t = phase.elapsed + since

  const total = BREATHS * BREATH_SECONDS
  let scale = 0.6
  let label = ''
  let sub = ''
  let done = 0
  let breathingNow = false
  if (t < 0) {
    label = String(Math.max(1, Math.ceil(-t)))
    sub = `Place the phone: ${spot.toLowerCase()}`
  } else if (t < total) {
    const inBreath = (t % BREATH_SECONDS) / BREATH_SECONDS
    done = Math.floor(t / BREATH_SECONDS)
    scale = 0.6 + 0.4 * Math.sin(Math.PI * inBreath)
    label = inBreath < 0.5 ? 'Breathe in' : 'Breathe out'
    sub = `${spot} · breath ${done + 1} of ${BREATHS}`
    breathingNow = true
  } else {
    done = BREATHS
    label = 'Analyzing'
    sub = spot
  }
  return (
    <div className="d-mirror">
      <div className="session-pacer">
        <div className={`ring ${breathingNow ? 'on' : ''}`} style={{ transform: `scale(${scale})` }} />
        <div className="pacer-text">
          <span className="pacer-label">{label}</span>
          <span className="unit">{sub}</span>
        </div>
      </div>
      <div className="breath-dots">
        {Array.from({ length: BREATHS }, (_, i) => (
          <span key={i} className={i < done ? 'done' : i === done && breathingNow ? 'now' : ''} />
        ))}
      </div>
    </div>
  )
}
