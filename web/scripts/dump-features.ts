// Parity check helper: computes features.ts log-mels for some dataset clips so
// scripts/check_features.py can compare them with the Python features.npy.
//   node --experimental-strip-types scripts/dump-features.ts <dataset> <config.json> <out.json> <row>...
// e.g. ../cough_dataset public/model/cough_config.json   or   ../lung_dataset public/model/lung_config.json
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { LogMel, type MelConfig } from '../src/features.ts'

const [dataDir, cfgPath, outPath, ...rows] = process.argv.slice(2)
const cfg: MelConfig & { sr: number; seconds?: number } = JSON.parse(readFileSync(cfgPath, 'utf8'))
const lm = new LogMel(cfg)
const files = readFileSync(join(dataDir, 'labels.csv'), 'utf8').trim().split('\n').slice(1)
  .map((l) => l.split(',')[0])

function readWav16(path: string): Float32Array {
  const b = readFileSync(path)
  let p = 12
  while (b.toString('ascii', p, p + 4) !== 'data') p += 8 + b.readUInt32LE(p + 4)
  const n = b.readUInt32LE(p + 4) / 2
  const y = new Float32Array(n)
  for (let i = 0; i < n; i++) y[i] = b.readInt16LE(p + 8 + 2 * i) / 32768
  return y
}

/** train_lung.py fit_cycle: repeat a short cycle, or take the middle of a long one. */
function fitCycle(y: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n)
  if (y.length < n) for (let i = 0; i < n; i++) out[i] = y[i % y.length]
  else out.set(y.subarray((y.length - n) >> 1, ((y.length - n) >> 1) + n))
  return out
}

const out: Record<string, number[]> = {}
for (const r of rows) {
  let y = readWav16(join(dataDir, files[Number(r)]))
  if (cfg.seconds) y = fitCycle(y, cfg.seconds * cfg.sr)
  out[r] = Array.from(lm.logmel(y))
}
writeFileSync(outPath, JSON.stringify(out))
console.log(`wrote ${rows.length} clips to ${outPath}`)
