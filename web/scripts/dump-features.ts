// Parity check helper: computes features.ts log-mels for some dataset clips so
// scripts/check_features.py can compare them with train.py's features.npy.
//   node --experimental-strip-types scripts/dump-features.ts <cough_dataset> <out.json> <row>...
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { LogMel, type CoughConfig } from '../src/features.ts'

const [dataDir, outPath, ...rows] = process.argv.slice(2)
const cfg: CoughConfig = JSON.parse(readFileSync('public/model/cough_config.json', 'utf8'))
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

const out: Record<string, number[]> = {}
for (const r of rows) out[r] = Array.from(lm.logmel(readWav16(join(dataDir, files[Number(r)]))))
writeFileSync(outPath, JSON.stringify(out))
console.log(`wrote ${rows.length} clips to ${outPath}`)
