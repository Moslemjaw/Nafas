// What the phone shows, mirrored to a paired computer screen (see link.ts).
// The cough and lung views call feed.update(); sends are batched to 5 per second.
import { useSyncExternalStore } from 'react'
import { Link, type LinkStatus, type Pairing } from './link'

export interface CoughState {
  listening: boolean
  p: number // live cough probability
  level: number // mic level (RMS)
  count: number
  times: number[] // when each cough happened (epoch ms), newest last
  probs: number[] // matching confidence for each cough
}

export type LungPhase =
  | { kind: 'idle' }
  | { kind: 'countdown'; left: number; at: number } // left: seconds, not rounded
  | { kind: 'recording'; elapsed: number; at: number }
  | { kind: 'analyzing' }

export interface SpotState {
  labels: string[] // one per breath
  probs: number[]
  flagged: boolean
}

export interface LungState {
  plan: string[] | null // spot ids, in order
  step: number
  phase: LungPhase
  results: Record<string, SpotState>
  done: boolean
}

export interface ScreenState {
  v: 1
  sentAt: number // phone clock, to undo network delay on the screen
  tab: 'cough' | 'lung'
  cough: CoughState
  lung: LungState
}

const SEND_EVERY_MS = 500 // public brokers slow down under load; the screen fills the gaps
const HEARTBEAT_MS = 5000 // so the screen can tell "idle" from "phone gone"
const MAX_COUGHS = 500

class Feed {
  private link: Link | null = null
  private state: ScreenState = {
    v: 1,
    sentAt: 0,
    tab: 'cough',
    cough: { listening: false, p: 0, level: 0, count: 0, times: [], probs: [] },
    lung: { plan: null, step: 0, phase: { kind: 'idle' }, results: {}, done: false },
  }
  private timer = 0
  private heartbeat = 0
  private status: LinkStatus | 'off' = 'off'
  private listeners = new Set<() => void>()

  connect(p: Pairing) {
    this.link?.close()
    this.link = new Link(p, 'phone')
    this.link.onStatus = (s) => {
      this.setStatus(s)
      if (s === 'connected') this.flush()
    }
    clearInterval(this.heartbeat)
    this.heartbeat = window.setInterval(() => this.flush(), HEARTBEAT_MS)
  }

  disconnect() {
    clearInterval(this.heartbeat)
    this.link?.close(true)
    this.link = null
    this.setStatus('off')
  }

  update(change: { tab?: ScreenState['tab']; cough?: Partial<CoughState>; lung?: Partial<LungState> }) {
    if (change.tab) this.state.tab = change.tab
    if (change.cough) this.state.cough = { ...this.state.cough, ...change.cough }
    if (change.lung) this.state.lung = { ...this.state.lung, ...change.lung }
    if (this.link && !this.timer) this.timer = window.setTimeout(() => this.flush(), SEND_EVERY_MS)
  }

  /** Record one detected cough. */
  cough(p: number) {
    const c = this.state.cough
    this.update({
      cough: {
        count: c.count + 1,
        times: [...c.times, Date.now()].slice(-MAX_COUGHS),
        probs: [...c.probs, p].slice(-MAX_COUGHS),
      },
    })
  }

  private flush() {
    this.timer = 0
    if (!this.link) return
    this.state.sentAt = Date.now()
    void this.link.send(this.state)
  }

  private setStatus(s: LinkStatus | 'off') {
    this.status = s
    this.listeners.forEach((l) => l())
  }

  subscribe = (l: () => void) => {
    this.listeners.add(l)
    return () => this.listeners.delete(l)
  }

  getStatus = () => this.status
}

export const feed = new Feed()

/** Connection status for the phone's "on screen" badge. */
export function useFeedStatus() {
  return useSyncExternalStore(feed.subscribe, feed.getStatus)
}
