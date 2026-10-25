// Runs on the audio thread: forwards mono mic samples to the page in ~0.1 s chunks.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super()
    this.chunk = new Float32Array(Math.round(sampleRate / 10))
    this.n = 0
  }

  process(inputs) {
    const ch = inputs[0][0]
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.chunk[this.n++] = ch[i]
        if (this.n === this.chunk.length) {
          this.port.postMessage(this.chunk.slice())
          this.n = 0
        }
      }
    }
    return true
  }
}

registerProcessor('capture', Capture)
