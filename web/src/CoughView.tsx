import { useEffect, useRef, useState } from 'react'
import { decodeFile, startMic, type Mic } from './audio'
import { Detector, HOP, MIC_WARMUP, Model, type CoughEvent, type Score } from './detector'
import { STOP_COUGH } from './events'

const BASE = import.meta.env.BASE_URL
const HISTORY = Math.round(30 / (HOP / 16000)) // 30 s of scores

interface LoggedCough {
  at: Date
  p: number
}

interface FileResult {
  name: string
  seconds: number
  scores: Score[]
  coughs: CoughEvent[]
}

export default function CoughView() {
  const [model, setModel] = useState<Model | null>(null)
  const [error, setError] = useState('')
  const [listening, setListening] = useState(false)
  const [score, setScore] = useState<Score | null>(null)
  const [history, setHistory] = useState<number[]>([])
  const [coughs, setCoughs] = useState<LoggedCough[]>([])
  const [flash, setFlash] = useState(false)
  const [fileResult, setFileResult] = useState<FileResult | null>(null)
  const [fileBusy, setFileBusy] = useState(false)
  const mic = useRef<Mic | null>(null)
  const wakeLock = useRef<WakeLockSentinel | null>(null)

  useEffect(() => {
    Model.load(BASE).then(setModel, (e) => setError(`Could not load the model: ${e}`))
    return () => mic.current?.stop()
  }, [])

  // the lung check needs the mic to itself (and chest sounds aren't coughs)
  useEffect(() => {
    const onStop = () => mic.current && stop()
    window.addEventListener(STOP_COUGH, onStop)
    return () => window.removeEventListener(STOP_COUGH, onStop)
  }, [])

  async function start() {
    if (!model) return
    setError('')
    const det = new Detector(model, MIC_WARMUP)
    det.onScore = (s) => {
      setScore(s)
      setHistory((h) => [...h.slice(-(HISTORY - 1)), s.p])
    }
    det.onCough = (e) => {
      setCoughs((c) => [{ at: new Date(), p: e.p }, ...c])
      setFlash(true)
      setTimeout(() => setFlash(false), 600)
      navigator.vibrate?.(120)
    }
    try {
      mic.current = await startMic(BASE, (x) => det.push(x))
      setListening(true)
      wakeLock.current = (await navigator.wakeLock?.request('screen').catch(() => null)) ?? null
    } catch (e) {
      setError(
        window.isSecureContext
          ? `Microphone not available: ${e}`
          : 'The microphone only works over HTTPS (or on localhost).',
      )
    }
  }

  function stop() {
    mic.current?.stop()
    mic.current = null
    void wakeLock.current?.release()
    wakeLock.current = null
    setListening(false)
    setScore(null)
  }

  async function testFile(file: File) {
    if (!model) return
    setFileBusy(true)
    setFileResult(null)
    try {
      const y = await decodeFile(file)
      const det = new Detector(model)
      const found: CoughEvent[] = []
      det.onCough = (e) => found.push(e)
      const scores = await det.run(y)
      setFileResult({ name: file.name, seconds: y.length / 16000, scores, coughs: found })
    } catch (e) {
      setError(`Could not read ${file.name}: ${e}`)
    } finally {
      setFileBusy(false)
    }
  }

  const threshold = model?.cfg.threshold ?? 0.83
  const p = score?.p ?? 0
  const isCough = p >= threshold

  return (
    <>
      <section className={`card live ${flash ? 'flash' : ''}`}>
        <div className={`dial ${isCough ? 'hot' : ''}`} aria-live="polite">
          <span className="pct">{listening ? Math.round(p * 100) : '–'}</span>
          <span className="unit">{listening ? (isCough ? 'cough' : '% cough') : 'not listening'}</span>
        </div>
        <Level value={score?.level ?? 0} active={listening} />
        {listening ? (
          <button className="btn stop" onClick={stop}>
            Stop
          </button>
        ) : (
          <button className="btn" onClick={start} disabled={!model}>
            {model ? 'Start listening' : 'Loading model…'}
          </button>
        )}
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <h2>Last 30 seconds</h2>
        <Strip values={history} threshold={threshold} length={HISTORY} />
      </section>

      <section className="card">
        <div className="row">
          <h2>Coughs</h2>
          <span className="count">{coughs.length}</span>
        </div>
        {coughs.length === 0 ? (
          <p className="muted">No coughs yet.</p>
        ) : (
          <ul className="log">
            {coughs.slice(0, 50).map((c, i) => (
              <li key={coughs.length - i}>
                <span>{c.at.toLocaleTimeString()}</span>
                <span className="muted">{Math.round(c.p * 100)}% sure</span>
              </li>
            ))}
          </ul>
        )}
        {coughs.length > 0 && (
          <button className="link" onClick={() => setCoughs([])}>
            Clear
          </button>
        )}
      </section>

      <section className="card">
        <h2>Test with a recording</h2>
        <p className="muted">Pick an audio file (wav, mp3, m4a, webm) to see where the coughs are.</p>
        <input
          type="file"
          accept="audio/*,video/webm"
          disabled={!model || fileBusy}
          onChange={(e) => e.target.files?.[0] && testFile(e.target.files[0])}
        />
        {fileBusy && <p className="muted">Analyzing…</p>}
        {fileResult && (
          <div className="result">
            <p>
              <b>{fileResult.coughs.length}</b> cough{fileResult.coughs.length === 1 ? '' : 's'} in{' '}
              {fileResult.name} ({fileResult.seconds.toFixed(1)} s)
              {fileResult.coughs.length > 0 &&
                ': at ' + fileResult.coughs.map((c) => `${c.t.toFixed(1)} s`).join(', ')}
            </p>
            <Strip
              values={fileResult.scores.map((s) => s.p)}
              threshold={threshold}
              length={fileResult.scores.length}
            />
          </div>
        )}
      </section>

      <footer className="muted">
        Runs entirely on this phone. Audio never leaves the device. Threshold {threshold.toFixed(2)}.
      </footer>
    </>
  )
}

function Level({ value, active }: { value: number; active: boolean }) {
  // RMS 0.001 .. 0.3 mapped to 0..100% on a log scale
  const pct = active ? Math.max(0, Math.min(100, ((Math.log10(value + 1e-6) + 3) / 2.5) * 100)) : 0
  return (
    <div className="level" title="Microphone level">
      <div style={{ width: `${pct}%` }} />
    </div>
  )
}

function Strip({ values, threshold, length }: { values: number[]; threshold: number; length: number }) {
  const n = Math.max(length, 1)
  const w = 100 / n
  return (
    <svg className="strip" viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label="Cough probability over time">
      <line x1="0" x2="100" y1={40 - threshold * 40} y2={40 - threshold * 40} className="thr" />
      {values.map((v, i) => (
        <rect
          key={i}
          x={(n - values.length + i) * w}
          width={w * 0.8}
          y={40 - v * 40}
          height={Math.max(v * 40, 0.4)}
          className={v >= threshold ? 'bar hot' : 'bar'}
        />
      ))}
    </svg>
  )
}
