# Nafas web app (prototype)

A React + Vite + TypeScript PWA that detects coughs from the phone microphone.
The model runs **in the browser** (onnxruntime-web), so audio never leaves the phone.
In the full Nafas design the model runs on the Raspberry Pi; this is the prototype step.

**Live:** https://moslemjaw.github.io/Nafas/

## Run

```bash
npm install
npm run dev        # http://localhost:5173 (the mic works on localhost)
npm run build      # production build in dist/
npm run preview    # serve dist/ on http://localhost:4173
```

A phone needs **HTTPS** to use the mic. The easiest way is GitHub Pages: the
`.github/workflows/pages.yml` workflow builds and deploys this folder on every push to
`main` (in the repo settings, set **Pages → Source** to **GitHub Actions**). Open the
page on the phone, then use **Add to Home Screen** to install it. After the first
load it works offline.

## How it works

| File | What it does |
|---|---|
| `src/audio.ts` | Mic capture at 16 kHz through an AudioWorklet (`public/capture-worklet.js`), with echo cancellation, noise suppression and auto-gain **off**. Also decodes audio files for the test mode. |
| `src/features.ts` | Log-mel spectrogram, matching `train.py` exactly (checked: max difference 0.004, probabilities within 0.001) |
| `src/detector.ts` | Finds loud events the same way `cough_prep.py` cut the training clips, scores a 1 s window centered on each one, and counts it as a cough above the threshold. A sliding window every 0.25 s drives the live meter. |
| `src/App.tsx` | UI: live meter, 30 s graph, cough log, file test |
| `public/model/` | `cough.onnx` + `cough_config.json`, written by `python train.py web` |

Because the model only sees normalized audio, each window is scaled by the loudest
sound in the last 10 s, which is the live version of the per-recording normalization
used in training. Each event is confirmed 1.5 s later, so counts appear with a short delay.

## Checking the features against Python

```bash
node --experimental-strip-types scripts/dump-features.ts ../cough_dataset feats.json 12912 94638
python scripts/check_features.py ../cough_dataset feats.json
```
