# Nafas web app (prototype)

A React + Vite + TypeScript PWA with two tabs:

- **Coughs:** listens through the phone mic and counts coughs live.
- **Lung check:** you press the phone's mic on the chest and breathe along with an
  on-screen circle for 6 breaths. Each breath is classified as normal, crackles, wheeze
  or both by the ICBHI-trained lung model.

Both models run **in the browser** (onnxruntime-web), so audio never leaves the phone.
In the full Nafas design the model runs on the Raspberry Pi; this is the prototype step.

**Live:** https://moslemjaw.github.io/Nafas/

## Run

```bash
npm install
npm run dev        # http://localhost:5173 (the mic works on localhost)
npm run build      # production build in dist/
npm run preview    # serve dist/ on http://localhost:4173
```

A phone needs **HTTPS** to use the mic, which GitHub Pages provides. On every push to
`main` that changes `web/`, the `.github/workflows/pages.yml` workflow builds this
folder and publishes it to the `gh-pages` branch, and Pages serves that branch. Open
the page on the phone, then use **Add to Home Screen** to install it. After the first
load it works offline.

## How it works

| File | What it does |
|---|---|
| `src/audio.ts` | Mic capture at 16 kHz through an AudioWorklet (`public/capture-worklet.js`), with echo cancellation, noise suppression and auto-gain **off**. Also decodes audio files for the test mode. |
| `src/features.ts` | Log-mel spectrogram, matching `train.py` exactly (checked: max difference 0.004, probabilities within 0.001) |
| `src/detector.ts` | Finds loud events the same way `cough_prep.py` cut the training clips, scores a 1 s window centered on each one, and counts it as a cough above the threshold. A sliding window every 0.25 s drives the live meter. |
| `src/CoughView.tsx` | Cough tab: live meter, 30 s graph, cough log, file test |
| `src/lung.ts` | Lung analysis, done the way the model was trained: peak-normalize the recording, one 4 s window per breath, log-mel, subtract each band's mean over the recording, then the model |
| `src/LungView.tsx` | Lung tab: breathing pacer (2 s in, 2 s out, so each 4 s window is one breath), per-breath results, file test |
| `src/features.ts` | Shared log-mel code: a DFT for the cough model's 400-point window, and an FFT for the lung model's 1024-point window |
| `public/model/` | `cough.*` from `python train.py web`, and `lung.*` from `python train_lung.py web` |

The lung model is 45 MB, so it isn't part of the install download. It downloads the
first time Lung check is opened and is cached for offline use after that.

Checked in the browser: the 6-breath analysis gives the same label as Python on all 24
test breaths (4 recordings built from ICBHI test-set cycles), in about 1 s on a laptop.

Because the model only sees normalized audio, each window is scaled by the loudest
sound in the last 10 s, which is the live version of the per-recording normalization
used in training. Each event is confirmed 1.5 s later, so counts appear with a short delay.

## Checking the features against Python

```bash
node --experimental-strip-types scripts/dump-features.ts ../cough_dataset public/model/cough_config.json feats.json 12912 94638
python scripts/check_features.py ../cough_dataset public/model/cough_config.json public/model/cough.onnx feats.json

node --experimental-strip-types scripts/dump-features.ts ../lung_dataset public/model/lung_config.json feats.json 36 181 2659
python scripts/check_features.py ../lung_dataset public/model/lung_config.json public/model/lung.onnx feats.json
```
