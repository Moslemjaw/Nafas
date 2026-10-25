// Phone -> computer screen link over MQTT, end-to-end encrypted.
//
// The screen makes a random pairing id + AES key and shows them in a QR code
// (in the URL #fragment, which browsers never send to a server). The phone
// publishes its state to nafas/<id>/state, encrypted with AES-GCM, and the
// screen decrypts it. The broker only ever sees ciphertext, and no audio is
// ever sent: only counts, probabilities and labels.
//
// Today this goes through a public broker; in the full system it would be the
// Raspberry Pi's Mosquitto, which is the same protocol (set ?broker=wss://...).
import type { MqttClient } from 'mqtt'

const DEFAULT_BROKER = 'wss://broker.hivemq.com:8884/mqtt'
const PAIR_KEY = 'nafas-pair'

export interface Pairing {
  id: string
  key: string // base64url, 128-bit AES key
}

const b64url = (b: Uint8Array) =>
  btoa(String.fromCharCode(...b)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
const unb64url = (s: string) =>
  Uint8Array.from(atob(s.replace(/-/g, '+').replace(/_/g, '/')), (c) => c.charCodeAt(0))

export function newPairing(): Pairing {
  return {
    id: b64url(crypto.getRandomValues(new Uint8Array(9))),
    key: b64url(crypto.getRandomValues(new Uint8Array(16))),
  }
}

/** The link the screen shows as a QR code. */
export function pairingUrl(p: Pairing): string {
  const u = new URL(location.href)
  u.hash = `pair=${p.id}.${p.key}`
  return u.toString()
}

/** A pairing passed in the URL (#pair=id.key), else the one saved this session. */
export function pairingFromUrl(): Pairing | null {
  const m = location.hash.match(/pair=([\w-]+)\.([\w-]+)/)
  if (m) {
    const p = { id: m[1], key: m[2] }
    try {
      sessionStorage.setItem(PAIR_KEY, JSON.stringify(p))
    } catch {
      /* private mode: pairing just won't survive a reload */
    }
    history.replaceState(null, '', location.pathname + location.search) // keep the key out of view
    return p
  }
  try {
    const saved = sessionStorage.getItem(PAIR_KEY)
    return saved ? (JSON.parse(saved) as Pairing) : null
  } catch {
    return null
  }
}

export function forgetPairing() {
  try {
    sessionStorage.removeItem(PAIR_KEY)
  } catch {
    /* nothing saved */
  }
}

function brokerUrl() {
  return new URLSearchParams(location.search).get('broker') ?? DEFAULT_BROKER
}

export type LinkStatus = 'connecting' | 'connected' | 'offline'

export class Link {
  // mqtt.js is loaded only when a link is made, so the normal app stays small
  private client: Promise<MqttClient>
  private key: Promise<CryptoKey>
  private topic: string
  private closed = false
  onStatus: (s: LinkStatus) => void = () => {}
  onMessage: (data: unknown) => void = () => {}

  constructor(p: Pairing, role: 'phone' | 'screen') {
    this.topic = `nafas/${p.id}/state`
    this.key = crypto.subtle.importKey('raw', unb64url(p.key), 'AES-GCM', false, ['encrypt', 'decrypt'])
    this.client = import('mqtt').then(({ default: mqtt }) => {
      const c = mqtt.connect(brokerUrl(), {
        clientId: `nafas-${role}-${b64url(crypto.getRandomValues(new Uint8Array(6)))}`,
        reconnectPeriod: 2000,
        connectTimeout: 15000,
      })
      c.on('connect', () => {
        this.onStatus('connected')
        if (role === 'screen') c.subscribe(this.topic)
      })
      c.on('reconnect', () => this.onStatus('connecting'))
      c.on('offline', () => this.onStatus('offline'))
      c.on('message', (_t, payload) => void this.receive(payload))
      if (this.closed) c.end()
      return c
    })
    queueMicrotask(() => this.onStatus('connecting'))
  }

  /** Send the latest state. Retained, so a screen that (re)connects gets it at once. */
  async send(data: unknown) {
    const iv = crypto.getRandomValues(new Uint8Array(12))
    const plain = new TextEncoder().encode(JSON.stringify(data))
    const ct = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, await this.key, plain))
    const msg = new Uint8Array(iv.length + ct.length)
    msg.set(iv)
    msg.set(ct, iv.length)
    const c = await this.client
    if (c.connected) c.publish(this.topic, b64url(msg), { qos: 0, retain: true })
  }

  private async receive(payload: Uint8Array) {
    try {
      const msg = unb64url(new TextDecoder().decode(payload))
      const plain = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv: msg.slice(0, 12) },
        await this.key,
        msg.slice(12),
      )
      this.onMessage(JSON.parse(new TextDecoder().decode(plain)))
    } catch {
      /* not ours, or tampered with: AES-GCM rejects it */
    }
  }

  /** Clear the retained state on the broker, then disconnect. */
  close(clear = false) {
    this.closed = true
    void this.client.then((c) => {
      if (clear && c.connected) c.publish(this.topic, '', { retain: true })
      c.end()
    })
  }
}
