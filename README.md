# Nafas

**Nafas** (نَفَس, "breath") is a capstone project at Abdullah Al-Salem University: a
smart vest that monitors the breathing of people with asthma at home, with all the
analysis done locally for privacy. This repo holds its AI part and a phone prototype:

- **Cough detection:** a small CNN listens to 1-second windows of audio and decides
  **cough** or **not cough**.
- **Lung sounds:** a ResNet18 listens to one breath from a chest mic and says
  **normal**, **crackles**, **wheeze** or **both**.
- **Phone app:** a web app (PWA) that runs both models on the phone, with a guided
  lung check and an optional live view on a computer screen.

> Nafas does not diagnose. It flags warning signs and leaves the decision to a doctor.

**Try it:** https://nafas-lemon.vercel.app ·
**Computer screen:** https://nafas-lemon.vercel.app/#display ·
**Backup:** https://moslemjaw.github.io/Nafas/

## Results at a glance

| Model | Trained on | Test set | Result | Size |
|---|---|---|---|---|
| Cough detector | COUGHVID, Coswara, ESC-50 | 66,324 clips from unseen people | **F1 0.954** (precision 0.960, recall 0.948); F1 0.894 with background noise as loud as the cough | 294k parameters, 1.2 MB |
| Lung sounds | ICBHI 2017 | Official split: 2,756 breaths from 49 unseen patients | **ICBHI score 0.537** (Sp 0.703, Se 0.371); normal vs abnormal 0.608 | 11.2M parameters, 44.7 MB |

Both models were trained on other people's microphones (phones and stethoscopes), so
they still need fine-tuning on recordings from the Nafas vest. First real-phone test:
a lung check on a team member gave **normal** for both lungs.

## Datasets

All public and labelled. None are stored in this repo (together they take tens of GB):
download them and point the scripts at them.

| Dataset | What it is | How we use it | Used |
|---|---|---|---|
| [COUGHVID](https://zenodo.org/records/7024894) (v3) | ~34,000 crowdsourced phone recordings of coughs | Coughs (only recordings the dataset scores ≥ 0.8 as a cough) | 80,835 cough clips from 18,544 recordings |
| [Coswara](https://github.com/iiscleap/Coswara-Data) | 2,747 people recording coughs, breathing, counting and vowels on a phone | Coughs; breathing and speech as "not cough" | 25,437 cough + 327,682 not-cough clips from 2,662 people |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 2,000 five-second clips of everyday sounds in 50 classes | The "coughing" class as cough; the other 49 as "not cough" | 102 cough + 15,561 not-cough clips |
| [ICBHI 2017](https://bhichallenge.med.auth.gr/) | 920 stethoscope recordings from 126 patients; doctors marked every breath | Lung sounds: normal, crackles, wheeze, both | 6,781 breaths |

Not used yet: **HF_Lung** (more lung sounds) and **our own vest recordings**, which
matter most.

## Repo layout

| Path | What it is |
|---|---|
| `cough_prep.py` | Cuts the cough datasets into labelled 1 s clips |
| `train.py` | Cough model: features, training, evaluation, ONNX and web export |
| `lung_prep.py` | Cuts ICBHI recordings into labelled breaths |
| `train_lung.py` | Lung model: features, training (small CNN or ResNet18), ONNX and web export |
| `web/` | The phone app (React + Vite + TypeScript); see [web/README.md](web/README.md) |
| `.github/workflows/pages.yml` | Builds the app and publishes it to GitHub Pages (the backup host) |

Kept out of git: the datasets, the generated clips and features (`cough_dataset/`,
`lung_dataset/`), the ICBHI test sounds (`test_sounds/`) and the data workbook
(`Nafas_Data.xlsx`), which contains patient-level ICBHI data.

## Setup

```bash
pip install librosa soundfile pandas scikit-learn tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime
```

[ffmpeg](https://ffmpeg.org/) must be on `PATH` to decode COUGHVID's `.webm` / `.ogg` files.

---

## Cough model (`cough_prep.py`, `train.py`)

```
COUGHVID + Coswara + ESC-50
        │  cough_prep.py
        ▼
1 s, 16 kHz mono clips + labels.csv  (split by person)
        │  train.py features
        ▼
log-mel spectrograms  (64 bands × 101 frames)
        │  train.py train
        ▼
model.pt / model.onnx + results.txt
```

```bash
# 1. Build the clip dataset
python cough_prep.py --coughvid coughvid --coswara Coswara-Data/Extracted_data --esc50 ESC-50 --out cough_dataset

# 2. Compute log-mel features once (writes cough_dataset/features.npy, ~5.8 GB)
python train.py features --data cough_dataset

# 3. Train (the final model uses 25 epochs with noise mixing)
python train.py train --data cough_dataset --epochs 25 --noise 0.5

# Re-export ONNX from model.pt, or copy the model into the phone app
python train.py export --data cough_dataset
python train.py web --data cough_dataset
```

### Clip preparation

- All audio is resampled to 16 kHz mono and peak-normalized.
- **Cough recordings:** loud regions are found with an energy-based splitter
  (`librosa.effects.split`, 30 dB below peak, events shorter than 100 ms ignored),
  and a 1 s clip is centered on each one.
- **Non-cough recordings:** a sliding 1 s window with 0.5 s hop, skipping
  near-silent windows.
- **Splitting:** 70 / 15 / 15 train / val / test, **grouped by person** (Coswara,
  COUGHVID) or **by source recording** (ESC-50). No person appears in two splits, so
  the test score measures performance on unseen people rather than memorized voices.

Resulting dataset: **449,617 clips**.

| Split | cough | not cough |
|---|---|---|
| train | 74,369 | 238,536 |
| val | 16,102 | 54,286 |
| test | 15,903 | 50,421 |

### Features, model and training

- **Features:** log-mel spectrogram: 25 ms window (`n_fft` 400), 10 ms hop, 64 mel
  bands from 50 Hz to 8 kHz, `log(mel_power + 1e-6)`, giving 64 × 101. Each band is
  normalized with the training set's mean and standard deviation.
- **Model:** four convolution blocks (16 → 32 → 64 → 128 channels; each is
  2 × [3×3 conv, batch norm, ReLU] then 2×2 max pooling), global average pooling,
  dropout 0.3 and a single output. **293,841 parameters.**
- **Loss:** binary cross-entropy, with the cough class weighted about 3.2× to offset
  the 1 : 3.2 class imbalance.
- **Optimizer:** AdamW with a one-cycle learning-rate schedule (max 3e-3), batch size 256.
- **Augmentation:**
  - SpecAugment-style: random gain (about ±4 dB), time shift up to ±100 ms, one
    frequency mask and one time mask.
  - **Background-noise mixing** (`--noise 0.5`): half of the training clips get a
    random not-cough training clip added at a random 0–20 dB SNR (on mel power:
    `log(exp(x) + g·exp(n))`). A cough mixed with noise is still a cough.
- **Model selection:** after each epoch, the threshold that maximizes F1 on val is
  chosen, and the epoch with the best val F1 is kept. The test set is used once, at the end.

### Results

Held-out test set: 66,324 clips from people not seen in training.

| | Round 1 | **Round 2 (final)** |
|---|---|---|
| Training | 10 epochs, no noise mixing | 25 epochs, noise mixing on 50% of clips |
| Threshold (chosen on val) | 0.79 | 0.83 |
| F1 | 0.953 | **0.954** |
| Precision | 0.955 | **0.960** |
| Recall | **0.951** | 0.948 |
| ROC AUC | 0.996 | 0.996 |
| False alarms | 706 | **621** |
| Missed coughs | **787** | 833 |

Per dataset (round 2):

| Dataset | Clips | Precision | Recall | Notes |
|---|---|---|---|---|
| COUGHVID | 12,145 | 1.000 | 0.953 | All coughs, so precision is trivially 1.0 |
| Coswara | 51,959 | 0.854 | 0.929 | 596 false alarms out of ~48k breathing / speech clips (round 1: 682) |
| ESC-50 | 2,220 | 0.306 | 0.917 | 25 false alarms out of 2,208 everyday-sound clips, about 1.1% (round 1: 24) |

With background noise (each test clip mixed with a random not-cough **test** clip at a
fixed SNR; each model with its own threshold):

| Noise level | Round 1 F1 | **Round 2 F1** | Round 1 recall | **Round 2 recall** |
|---|---|---|---|---|
| Clean | 0.953 | **0.954** | 0.951 | 0.948 |
| 20 dB SNR | 0.912 | **0.948** | 0.870 | **0.943** |
| 10 dB SNR | 0.845 | **0.937** | 0.760 | **0.926** |
| 5 dB SNR | 0.774 | **0.922** | 0.657 | **0.899** |
| 0 dB SNR | 0.665 | **0.894** | 0.518 | **0.849** |

Noise mixing is what makes the model usable in real rooms: with the background as
loud as the cough, round 1 misses about half the coughs, and round 2 still catches 85%.

### Limitations

- **Everyday-sound false alarms:** about 1% of non-cough everyday sounds are still
  flagged as coughs, and noise mixing didn't change that. ESC-50 is only about 3% of the
  training data, so more varied negative audio (for example FSD50K or AudioSet) is the
  likely fix.
- **Recording devices:** all training audio was recorded on phones. The vest's mics will
  sound different, so a few hundred recordings from them should be used for fine-tuning.
- **No medical use:** the model detects *that* someone coughed, not *why*.

### Using the model

Files in `cough_dataset/` (not in git): `model.pt` (weights, per-band `mean` / `std`,
`threshold`), `model.onnx` (same network; outputs match PyTorch to about 3e-6),
`results.txt`, and `model_r1.*` / `model_r2.*` copies of each round.

To run it on a 1 s, 16 kHz mono window `y`:

1. `x = log(melspectrogram(y, sr=16000, n_fft=400, hop_length=160, n_mels=64, fmin=50, fmax=8000) + 1e-6)`, shape 64 × 101.
2. `x = (x - mean) / std`, using the per-band values in `model.pt`.
3. Feed it as `[1, 1, 64, 101]` (input `logmel`). The output `logit` means **cough** when `sigmoid(logit) ≥ 0.83`.

For a continuous stream, slide the 1 s window with a 0.5 s hop.

---

## Lung model (`lung_prep.py`, `train_lung.py`)

Classifies each breath heard by a chest mic as **normal, crackles, wheeze or both**.

```bash
python lung_prep.py --icbhi ICBHI_final_database --split ICBHI_challenge_train_test.txt --out lung_dataset
python train_lung.py features --data lung_dataset
python train_lung.py train --data lung_dataset --arch resnet18 --recnorm --mixup 0.4 --epochs 30 --lr 1e-3
python train_lung.py web --data lung_dataset   # copy the model into the phone app
```

- **Breaths:** 6,781 breaths cut from the recordings at 16 kHz. Each is repeat-padded or
  center-cropped to 4 s (90% of breaths are shorter than 4.1 s). The chest position
  (trachea, anterior, posterior or lateral; left or right) is kept for "which lung" later.
- **Split:** the **official ICBHI patient-wise split**, so the test set (2,756 breaths,
  49 patients) is the one papers report on. The official split puts 2 patients
  (156, 218) on both sides, so their *train* breaths are dropped and the test set stays
  unchanged. Val is 16 of the train patients.
- **Features:** log-mel spectrogram, 64 bands from 50 Hz to 2 kHz (where lung sounds
  are), 64 ms window (fine frequency detail for tonal wheezes), 10 ms hop → 64 × 401.
- **Per-recording normalization:** each recording's average spectrum is subtracted. That
  removes most of the stethoscope's frequency "coloring" but keeps the crackles and
  wheezes. On the vest this becomes a running average per mic.
- **Model:** either the cough CNN with 4 outputs (`--arch cnn`, 294k parameters) or an
  ImageNet-pretrained ResNet18 (`--arch resnet18`, 11.2M parameters). For ResNet18 the
  log-mel is stretched to 128 bands and copied to 3 channels. Both use square-root class weights.
- **Score (the ICBHI metric):** Sp = normal breaths called normal; Se = abnormal breaths
  called the exact right class; Score = (Se + Sp) / 2.

### Results (official test set)

| | Round 1 | Round 2 | **Round 3 (final)** |
|---|---|---|---|
| Model | small CNN | small CNN | **ResNet18, ImageNet-pretrained** |
| Training | 60 epochs | 40 epochs, per-recording norm, mixup 0.4 | 30 epochs, per-recording norm, mixup 0.4, lr 1e-3 |
| Se | 0.373 | **0.412** | 0.371 |
| Sp | 0.508 | 0.636 | **0.703** |
| **ICBHI score** | 0.440 | 0.524 | **0.537** |
| Normal vs abnormal score | 0.543 | 0.598 | **0.608** |
| Size (ONNX) | 1.2 MB | 1.2 MB | 44.7 MB |

Per-class recall:

| | normal | crackles | wheeze | both |
|---|---|---|---|---|
| Round 2 (CNN) | 0.64 | 0.46 | **0.48** | 0.04 |
| Round 3 (ResNet18) | **0.70** | 0.46 | 0.32 | **0.14** |

For comparison, published results on this split are roughly 0.50–0.55 for small CNNs
trained from scratch and 0.60–0.65 for the best (mostly pretrained) models.

Pretraining helped, but less than hoped: +0.013 overall, from a single run each, and
validation (16 patients) is noisy enough that a gap this small could partly be chance.
The clearer gain is robustness to the stethoscope (below). The trade-off is that it
catches fewer wheezes than the small CNN. ResNet18 runs in about 10 ms per breath on a
laptop CPU, which is fine for a Raspberry Pi 5.

### Why the stethoscope matters

In ICBHI the stethoscope type is tangled up with the labels:

| Stethoscope | Train breaths | Test breaths | Score, round 2 (CNN) | Score, round 3 (ResNet18) |
|---|---|---|---|---|
| AKG C417L | 2,109 | 1,836 | **0.578** | 0.535 |
| Meditron | 783 (73% normal) | 459 | 0.527 | **0.607** |
| Littmann 3200 | **28** (mostly wheeze) | 461 (mostly normal) | 0.374 | **0.448** |

The model barely saw a Littmann 3200 in training, and what it saw was mostly wheezes, so
the small CNN calls most Littmann 3200 test breaths abnormal (Sp 0.09). ResNet18 handles
it better (Sp 0.57), but it's still the weakest device. The vest uses one mic type of its
own, so this mix won't exist there, but it shows the model must be fine-tuned on vest
recordings before its output means much.

### Limitations

- **"Both" (crackles + wheeze together) is rarely detected** (recall 0.14). It's the
  rarest class, with 203 training breaths.
- **Small dataset:** 61 training patients. Validation scores swing a lot between epochs,
  so the chosen checkpoint is partly luck.
- **Not a diagnosis:** it flags sounds a doctor would listen for.

---

## Phone app (`web/`)

A React + Vite + TypeScript PWA that runs both models **in the phone's browser**
(onnxruntime-web). Audio never leaves the phone. Full details: [web/README.md](web/README.md).

- **Coughs tab:** live cough counting through the phone mic, with a probability meter,
  a 30-second graph and a cough log.
- **Lung check tab:** a guided check at 4 spots (front and back, with a helper) or
  2 front spots. Each step shows a chest diagram with where to place the phone, then a
  full-screen breathing circle paces 6 breaths (2 s in, 2 s out, so each 4 s window is
  one breath). Each breath is classified, and the summary colors each spot and notes
  when left and right sound different.
- **Computer screen:** open the app with `#display` (or `#device`, `#screen`, `#tv`) on
  a computer and scan its QR code with the phone. The computer then shows the cough
  count, live probability and lung check live. Only numbers and labels are sent,
  over MQTT (the protocol planned for the Pi), end-to-end encrypted with a key that
  exists only in the QR code.
- **Hosting:** Vercel (main) and GitHub Pages (backup) both rebuild on every push to `main`.

How it was checked in the browser, with recordings played in place of the mic:

- **Same results as Python:** the TypeScript log-mel matches the training code (max
  difference 0.004), and the app's lung labels matched Python on all 24 test breaths.
- **Cough counting:** on full recordings from 3 test-split Coswara people, 36 of 37
  loud cough events were counted, with no false alarms on breathing, counting, sneezing,
  laughing or knocking. The one miss was two coughs 0.65 s apart counted as one.
- **Computer screen:** phone and screen showed the same cough count and lung results,
  and the screen's breathing circle stayed in step with the phone's.

---

## Next steps

- Record our own coughs and breaths with the vest mics, and fine-tune both models on them.
- Move detection from the phone to the Raspberry Pi (onnxruntime), on live audio from
  the ESP32 vest mics and the phone. In the Nafas design the ESP32 only collects audio,
  and all the AI runs on the Pi.
- Lung model: add HF_Lung as extra training data, and average several training runs.
- Cough model: add more varied everyday sounds to cut false alarms.
