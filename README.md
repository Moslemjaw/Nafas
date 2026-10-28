# Nafas

Nafas (نَفَس, "breath") is our capstone project at Abdullah Al-Salem University. The
idea is a smart vest that keeps track of the breathing of people with asthma at home.
It has microphones for coughs and for the lungs, and all the analysis happens on a
Raspberry Pi inside the system, so no audio has to go to a server.

This repo is the AI side of the project and a phone prototype we built to test it
before the vest hardware is ready. So far we have:

- a cough detector (small CNN) that listens to 1-second windows and says cough / not cough
- a lung sound model (ResNet18) that listens to one breath and says normal, crackles,
  wheeze or both
- a web app that runs both models on a phone, with a guided lung check and a live view
  on a computer screen

Nafas does not diagnose anything. It flags warning signs and the decision stays with
a doctor.

Try the app: https://nafas-lemon.vercel.app (computer screen view: add `#display` to
the link). Backup copy: https://moslemjaw.github.io/Nafas/

## Team

- Moslem Jawich
- Rayan Alsousou

Supervisors: Dr. Khaled Alanezi and Abdullah Karar

## Where we are now

| Model | Trained on | Tested on | Result | Size |
|---|---|---|---|---|
| Cough detector | COUGHVID, Coswara, ESC-50 | 66,324 clips from people the model never saw | F1 0.954 (0.894 with background noise as loud as the cough) | 294k params, 1.2 MB |
| Lung sounds | ICBHI 2017 | official test split, 2,756 breaths from 49 patients | ICBHI score 0.537 | 11.2M params, 44.7 MB |

Both models learned from other people's phones and stethoscopes, not from our vest
mics, so we still need to record our own data and fine-tune on it. Our first real test
on the phone (a lung check on one of us) came out normal for both lungs, which is what
we expected.

## Datasets

All the datasets are public and already labelled. They are too big for GitHub (tens of
GB together), so they are not in the repo. [SETUP.md](SETUP.md) has the download links
and the steps to set everything up on another computer.

| Dataset | What it is | What we used it for | How much we used |
|---|---|---|---|
| [COUGHVID](https://zenodo.org/records/7024894) v3 | ~34,000 phone recordings of coughs | coughs (only recordings scored ≥ 0.8 as a cough) | 80,835 cough clips |
| [Coswara](https://github.com/iiscleap/Coswara-Data) | 2,747 people recording coughs, breathing, counting and vowels | coughs, plus breathing and speech as "not cough" | 25,437 cough + 327,682 not-cough clips |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 2,000 clips of everyday sounds in 50 classes | the coughing class as cough, the rest as not cough | 102 cough + 15,561 not-cough clips |
| [ICBHI 2017](https://bhichallenge.med.auth.gr/) | 920 stethoscope recordings from 126 patients, every breath labelled by doctors | lung sounds | 6,781 breaths |

We haven't used HF_Lung yet, and we haven't recorded anything with the vest yet. The
vest recordings are the most important missing piece.

## Files

| File | What it does |
|---|---|
| `cough_prep.py` | cuts the cough datasets into labelled 1 s clips |
| `train.py` | cough model: features, training, testing, ONNX export, copy to the app |
| `lung_prep.py` | cuts the ICBHI recordings into labelled breaths |
| `train_lung.py` | lung model: same steps, with a choice of small CNN or ResNet18 |
| `web/` | the phone app (React + Vite + TypeScript), see [web/README.md](web/README.md) |
| `.github/workflows/pages.yml` | builds the app and puts it on GitHub Pages |

Not in git: the datasets, the clips and features the scripts make (`cough_dataset/`,
`lung_dataset/`), our ICBHI test sounds (`test_sounds/`) and our data workbook
(`Nafas_Data.xlsx`, it has patient-level ICBHI data).

## Cough model

Steps (full commands and timings are in SETUP.md):

```bash
python cough_prep.py --coughvid coughvid --coswara Coswara-Data/Extracted_data --esc50 ESC-50 --out cough_dataset
python train.py features --data cough_dataset
python train.py train --data cough_dataset --epochs 25 --noise 0.5
python train.py web --data cough_dataset
```

### Making the clips

Everything is converted to 16 kHz mono. For cough recordings we find the loud parts
with `librosa.effects.split` (30 dB below the peak, at least 100 ms) and take a 1 s clip
around each one. For everything else we slide a 1 s window with a 0.5 s step and skip
the silent parts.

We split the data 70/15/15 by person (or by recording for ESC-50), so nobody appears
in both training and testing. Otherwise the test score would partly measure how well
the model remembers voices. In total we got 449,617 clips:

| Split | cough | not cough |
|---|---|---|
| train | 74,369 | 238,536 |
| val | 16,102 | 54,286 |
| test | 15,903 | 50,421 |

### The model

The input is a log-mel spectrogram (25 ms window, 10 ms hop, 64 mel bands from 50 Hz
to 8 kHz), which gives a 64 × 101 image per clip. The network is 4 conv blocks
(16, 32, 64, 128 channels) with batch norm and max pooling, then global average
pooling and one output. 293,841 parameters in total.

Training details:

- binary cross-entropy, with coughs weighted ~3.2× because there are 3.2× more
  non-cough clips
- AdamW, one-cycle learning rate (max 3e-3), batch size 256
- augmentation: random gain, small time shifts, one frequency mask and one time mask
- background noise mixing (round 2 only): half the training clips get a random
  non-cough clip added at 0 to 20 dB SNR
- after every epoch we pick the threshold with the best F1 on val and keep the best
  epoch. The test set is only used once at the end.

### Results

We trained two rounds. Round 1 was 10 epochs without noise mixing. It did well on
clean audio but we were worried about real rooms, so round 2 added noise mixing and
trained for 25 epochs.

| | Round 1 | Round 2 (final) |
|---|---|---|
| threshold (from val) | 0.79 | 0.83 |
| F1 | 0.953 | 0.954 |
| precision | 0.955 | 0.960 |
| recall | 0.951 | 0.948 |
| ROC AUC | 0.996 | 0.996 |
| false alarms | 706 | 621 |
| missed coughs | 787 | 833 |

On clean audio the two rounds are almost the same. The difference shows up when we add
noise to the test clips:

| Noise level | Round 1 F1 | Round 2 F1 | Round 1 recall | Round 2 recall |
|---|---|---|---|---|
| clean | 0.953 | 0.954 | 0.951 | 0.948 |
| 20 dB SNR | 0.912 | 0.948 | 0.870 | 0.943 |
| 10 dB SNR | 0.845 | 0.937 | 0.760 | 0.926 |
| 5 dB SNR | 0.774 | 0.922 | 0.657 | 0.899 |
| 0 dB SNR | 0.665 | 0.894 | 0.518 | 0.849 |

When the noise is as loud as the cough, round 1 misses about half of the coughs and
round 2 still gets 85% of them.

By dataset (round 2): COUGHVID recall 0.953, Coswara precision 0.854 / recall 0.929,
and on ESC-50 about 1.1% of everyday sounds were wrongly called coughs (25 out of
2,208).

### Problems we still have

- Around 1% of everyday sounds still trigger it, and noise mixing didn't help with
  that. ESC-50 is only ~3% of our training data, so we think adding more varied sounds
  (FSD50K or AudioSet) would fix it.
- All the training audio comes from phones. The vest mics will sound different, so we
  need a few hundred of our own recordings for fine-tuning.
- It only knows that someone coughed, not why.

### Using the trained model

`cough_dataset/model.pt` has the weights, the per-band mean and std, and the
threshold. `model.onnx` is the same network (outputs match PyTorch to about 3e-6).
To score a 1 s, 16 kHz window `y`:

1. `x = log(melspectrogram(y, sr=16000, n_fft=400, hop_length=160, n_mels=64, fmin=50, fmax=8000) + 1e-6)`
2. `x = (x - mean) / std`
3. run it as `[1, 1, 64, 101]`; it's a cough if `sigmoid(logit) >= 0.83`

## Lung model

This one classifies each breath as normal, crackles, wheeze or both.

```bash
python lung_prep.py --icbhi ICBHI_final_database --split ICBHI_challenge_train_test.txt --out lung_dataset
python train_lung.py features --data lung_dataset
python train_lung.py train --data lung_dataset --arch resnet18 --recnorm --mixup 0.4 --epochs 30 --lr 1e-3
python train_lung.py web --data lung_dataset
```

How we set it up:

- Each breath is padded (by repeating it) or cropped to 4 s. 90% of breaths are
  shorter than 4.1 s. We also keep the chest position so we can compare left and right.
- We used the official ICBHI patient split so our test score can be compared with
  papers. Two patients (156 and 218) are on both sides of the official split, so we
  dropped their training breaths and left the test set as it is.
- The log-mel here goes from 50 Hz to 2 kHz (where lung sounds are) with a longer 64 ms
  window, because wheezes are tonal and need finer frequency detail. That gives 64 × 401.
- For each recording we subtract its average spectrum. This removes a lot of the
  stethoscope's own sound but keeps the crackles and wheezes. On the vest this would be
  a running average for each mic.
- The ICBHI score is (Se + Sp) / 2, where Sp is normal breaths called normal and Se is
  abnormal breaths given the exact right class.

### Results

We tried three rounds:

| | Round 1 | Round 2 | Round 3 (final) |
|---|---|---|---|
| model | small CNN | small CNN | ResNet18 pretrained on ImageNet |
| training | 60 epochs | 40 epochs, per-recording norm, mixup 0.4 | 30 epochs, per-recording norm, mixup 0.4 |
| Se | 0.373 | 0.412 | 0.371 |
| Sp | 0.508 | 0.636 | 0.703 |
| ICBHI score | 0.440 | 0.524 | 0.537 |
| normal vs abnormal | 0.543 | 0.598 | 0.608 |
| size | 1.2 MB | 1.2 MB | 44.7 MB |

Recall per class:

| | normal | crackles | wheeze | both |
|---|---|---|---|---|
| round 2 (CNN) | 0.64 | 0.46 | 0.48 | 0.04 |
| round 3 (ResNet18) | 0.70 | 0.46 | 0.32 | 0.14 |

From what we read, papers on this split get around 0.50 to 0.55 with small CNNs
trained from scratch, and 0.60 to 0.65 for the best models (mostly pretrained). So we
are in the normal range but not near the top.

The pretrained model only gained 0.013, and we only ran each setup once, so part of
that could be luck. Where it clearly helped was with the different stethoscopes (see
below). On the other hand it catches fewer wheezes than the small CNN. It takes about
10 ms per breath on a laptop CPU, which should be fine on a Raspberry Pi 5.

### The stethoscope problem

The biggest problem we found is that in ICBHI the type of stethoscope is mixed up with
the labels:

| Stethoscope | train breaths | test breaths | score, round 2 | score, round 3 |
|---|---|---|---|---|
| AKG C417L | 2,109 | 1,836 | 0.578 | 0.535 |
| Meditron | 783 (73% normal) | 459 | 0.527 | 0.607 |
| Littmann 3200 | 28 (mostly wheeze) | 461 (mostly normal) | 0.374 | 0.448 |

The model hardly saw the Littmann 3200 in training, and what it saw was mostly wheezes,
so the small CNN called most Littmann test breaths abnormal. ResNet18 did better but
it's still the worst device. Our vest will only have one mic type so this exact
problem won't happen there, but it shows the model has to be fine-tuned on vest
recordings before we can trust it.

Other limits: "both" is rarely detected (it's the rarest class, 203 training breaths),
and with only 61 training patients the validation score jumps around a lot between
epochs.

## Phone app

The app is a React + Vite + TypeScript PWA. Both models run inside the phone's
browser with onnxruntime-web, so the audio stays on the phone. More details are in
[web/README.md](web/README.md).

- Coughs tab: counts coughs live from the mic, with a probability meter, a 30 s graph
  and a list of coughs.
- Lung check tab: a guided check at 4 spots (front and back, needs a helper) or 2 front
  spots. Each step shows a chest drawing with where to put the phone, then a breathing
  circle guides you through 6 breaths (2 s in, 2 s out, so every 4 s is one breath).
  At the end it colors each spot and tells you if left and right sound different.
- Computer screen: open the app with `#display` on a computer and scan the QR code with
  the phone. The computer then shows the cough count and the lung check live. We used
  MQTT for this because it's what we plan to use on the Pi. Only numbers and labels
  are sent, and they are encrypted with a key that is only inside the QR code.
- It's hosted on Vercel, with GitHub Pages as a backup. Both update when we push to
  `main`.

How we tested it (playing recordings instead of using the mic):

- the TypeScript log-mel matches the Python one (max difference 0.004), and the app gave
  the same lung labels as Python on all 24 test breaths
- on full recordings of 3 Coswara people from the test split, it counted 36 of 37
  coughs, with no false alarms on breathing, counting, sneezing, laughing or knocking.
  The one it missed was two coughs 0.65 s apart that it counted as one.
- the phone and the computer screen showed the same counts and results

## Next steps

- Record coughs and breaths with the vest mics and fine-tune both models on them.
- Move the models from the phone to the Raspberry Pi, using audio from the ESP32 mics.
  In our design the ESP32 only collects the audio and the Pi does all the AI.
- Lung model: add HF_Lung and average a few training runs.
- Cough model: add more everyday sounds to reduce false alarms.
