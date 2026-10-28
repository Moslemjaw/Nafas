# Nafas web app (prototype)

This is the phone prototype for Nafas. In the final system the models run on the
Raspberry Pi, but the vest isn't ready yet, so we put them in a web app (PWA) to test
them on a real phone mic. Both models run in the browser with onnxruntime-web, so no
audio leaves the phone.

Live: https://nafas-lemon.vercel.app (backup: https://moslemjaw.github.io/Nafas/)

The app has two tabs:

- **Coughs**: listens through the mic and counts coughs live.
- **Lung check**: a step-by-step check at 4 spots (front left/right under the
  collarbones, back left/right under the shoulder blades, needs a helper) or 2 front
  spots if you're alone. Each step shows where to put the phone on a chest drawing,
  then you breathe with a circle on the screen for 6 breaths. Each breath goes through
  the lung model (normal, crackles, wheeze or both). At the end each spot gets a color
  (flagged if 2 or more of its 6 breaths sound abnormal) and the app says if left and
  right sound different.

## Showing it on a computer screen

Open the app on a computer with `#display` at the end of the link
(https://nafas-lemon.vercel.app/#display, `#device`, `#screen` and `#tv` also work).
It shows a QR code. Scan it with the phone and the app opens already paired. The
computer then shows a big cough counter, the cough probability, coughs per minute and
the lung check (including the breathing circle, in step with the phone).

How it works:

- The phone sends its state over MQTT, which is the same protocol we plan to use with
  Mosquitto on the Pi. For now it uses a public broker (`broker.hivemq.com`). You can
  add `?broker=wss://...` to the link on both devices to use a different one.
- Only numbers and labels are sent, never audio, and they are encrypted (AES-GCM). The
  key is in the `#` part of the QR link, which the browser never sends to any server,
  so the broker can't read anything.
- The public broker adds a few seconds of delay. For the breathing circle the screen
  uses the phone's timestamps to catch up, so it stays in step. With a broker on the
  Pi the delay would be a few milliseconds.
- The QR code points to the same site the screen is opened on, so a Vercel screen pairs
  with the Vercel app.

## Running it

```bash
npm install
npm run dev        # http://localhost:5173 (the mic works on localhost)
npm run build      # build into dist/
npm run preview    # serve dist/ on http://localhost:4173
```

Phones only allow the mic over HTTPS, so for phone testing we use the hosted versions.
Both update on every push to `main`:

- Vercel builds this folder (Root Directory `web`) for https://nafas-lemon.vercel.app.
- `.github/workflows/pages.yml` builds it and pushes it to the `gh-pages` branch for
  GitHub Pages. That branch also has a `web/vercel.json` so Vercel doesn't try to
  deploy it.

On the phone you can use "Add to Home Screen" to install it. After the first load it
also works offline.

## Code

| File | What it does |
|---|---|
| `src/audio.ts` | mic capture at 16 kHz with an AudioWorklet (`public/capture-worklet.js`). We turn off echo cancellation, noise suppression and auto gain because they change the sound. Also decodes audio files for testing. |
| `src/features.ts` | log-mel spectrogram, same as in Python (max difference 0.004). Uses a DFT for the cough model's 400-point window and an FFT for the lung model's 1024-point window. |
| `src/detector.ts` | finds loud events the same way `cough_prep.py` made the training clips, scores 1 s around each one, and counts it if it's above the threshold. A sliding window every 0.25 s drives the live meter. |
| `src/CoughView.tsx` | the Coughs tab |
| `src/lung.ts` | lung analysis, same steps as training: normalize, one 4 s window per breath, log-mel, subtract each band's mean, run the model |
| `src/LungView.tsx` | the Lung check tab (steps, breathing pacer, results, file test) |
| `src/ChestDiagram.tsx`, `src/spots.ts` | the chest drawings and where each spot is |
| `src/link.ts` | pairing and the encrypted MQTT connection |
| `src/feed.ts` | what the phone sends to the screen (at most twice a second, plus a heartbeat every 5 s) |
| `src/DisplayView.tsx` | the computer screen view |
| `public/model/` | the models, copied here by `python train.py web` and `python train_lung.py web` |

Some things we had to deal with:

- The lung model is 45 MB, so it's not part of the install. It downloads the first
  time you open Lung check and is cached after that.
- In training every recording was peak-normalized, but live there is no "whole
  recording". So we scale each window by the loudest sound in the last 10 s.
- Each cough is confirmed 1.5 s later (to compare it with what came after), so the
  count shows up with a small delay.

We checked the lung analysis against Python on 24 test breaths from ICBHI and got the
same label for all of them. It takes about 1 s on a laptop.

## Checking the features against Python

```bash
node --experimental-strip-types scripts/dump-features.ts ../cough_dataset public/model/cough_config.json feats.json 12912 94638
python scripts/check_features.py ../cough_dataset public/model/cough_config.json public/model/cough.onnx feats.json

node --experimental-strip-types scripts/dump-features.ts ../lung_dataset public/model/lung_config.json feats.json 36 181 2659
python scripts/check_features.py ../lung_dataset public/model/lung_config.json public/model/lung.onnx feats.json
```
