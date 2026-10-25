// Microphone and file input, both delivered as 16 kHz mono Float32Array.

export const SR = 16000

export interface Mic {
  stop(): void
  rate: number // the AudioContext's real rate (16000 unless the browser refused)
}

/** Linear-interpolation resampler for browsers that won't open a 16 kHz AudioContext. */
class Resampler {
  private step: number
  private pos = 0
  private prev = 0

  constructor(from: number) {
    this.step = from / SR
  }

  push(x: Float32Array): Float32Array {
    const out: number[] = []
    while (this.pos < x.length) {
      const i = Math.floor(this.pos)
      const a = i === 0 ? this.prev : x[i - 1]
      const f = this.pos - i
      out.push(a + (x[i] - a) * f)
      this.pos += this.step
    }
    this.pos -= x.length
    this.prev = x[x.length - 1]
    return Float32Array.from(out)
  }
}

export async function startMic(base: string, onAudio: (x: Float32Array) => void): Promise<Mic> {
  const stream = await navigator.mediaDevices.getUserMedia({
    // phone "voice" processing is tuned to remove exactly the sounds we want to hear
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  })
  let ctx: AudioContext
  try {
    ctx = new AudioContext({ sampleRate: SR })
  } catch {
    ctx = new AudioContext()
  }
  let source: MediaStreamAudioSourceNode
  try {
    source = ctx.createMediaStreamSource(stream)
  } catch {
    // Firefox can't connect a mic to a context at a different rate: fall back to native
    await ctx.close()
    ctx = new AudioContext()
    source = ctx.createMediaStreamSource(stream)
  }
  await ctx.audioWorklet.addModule(`${base}capture-worklet.js`)
  const node = new AudioWorkletNode(ctx, 'capture')
  const rs = ctx.sampleRate === SR ? null : new Resampler(ctx.sampleRate)
  node.port.onmessage = (e: MessageEvent<Float32Array>) => onAudio(rs ? rs.push(e.data) : e.data)
  source.connect(node)
  if (ctx.state === 'suspended') await ctx.resume()

  return {
    rate: ctx.sampleRate,
    stop() {
      node.port.onmessage = null
      source.disconnect()
      stream.getTracks().forEach((t) => t.stop())
      void ctx.close()
    },
  }
}

/** Decode any audio file the browser understands to 16 kHz mono. */
export async function decodeFile(file: File): Promise<Float32Array> {
  const bytes = await file.arrayBuffer()
  const tmp = new AudioContext()
  const decoded = await tmp.decodeAudioData(bytes)
  await tmp.close()
  const len = Math.ceil(decoded.duration * SR)
  const off = new OfflineAudioContext(1, len, SR)
  const src = off.createBufferSource()
  src.buffer = decoded
  src.connect(off.destination)
  src.start()
  return (await off.startRendering()).getChannelData(0)
}
