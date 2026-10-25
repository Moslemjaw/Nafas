# Nafas

Cough detection from audio (Capstone 1). A small CNN listens to 1-second windows of
audio and decides **cough** or **not cough**. It is trained on three public datasets
and is small enough (~294k parameters, 1.2 MB) to run on a phone or a microcontroller.

## Pipeline

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

## Datasets

| Dataset | What it contributes | Label |
|---|---|---|
| [COUGHVID](https://zenodo.org/records/7024894) (v3) | ~34k crowdsourced phone cough recordings | cough (only recordings with `cough_detected` ≥ 0.8 are used) |
| [Coswara](https://github.com/iiscleap/Coswara-Data) | 2,747 participants: heavy/shallow coughs, plus breathing, counting and vowels | coughs → cough, everything else → not cough |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 2,000 environmental sound clips in 50 classes | `coughing` → cough, other 49 classes → not cough |

The datasets are not in this repo (about 30 GB). Download them and point the script at them.

## Setup

```bash
pip install librosa soundfile pandas scikit-learn tqdm
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime
```

[ffmpeg](https://ffmpeg.org/) must be on `PATH` to decode COUGHVID's `.webm` / `.ogg` files.

## Usage

```bash
# 1. Build the clip dataset
python cough_prep.py --coughvid coughvid --coswara Coswara-Data/Extracted_data --esc50 ESC-50 --out cough_dataset

# 2. Compute log-mel features once (writes cough_dataset/features.npy, ~5.8 GB)
python train.py features --data cough_dataset

# 3. Train (the final model uses 25 epochs with noise mixing)
python train.py train --data cough_dataset --epochs 25 --noise 0.5

# Re-export ONNX from model.pt if needed
python train.py export --data cough_dataset
```

## How it works

### 1. Clip preparation (`cough_prep.py`)

- All audio is resampled to 16 kHz mono and peak-normalized.
- **Cough recordings:** loud regions are found with an energy-based splitter
  (`librosa.effects.split`, 30 dB below peak, events shorter than 100 ms ignored),
  and a 1 s clip is centered on each one.
- **Non-cough recordings:** a sliding 1 s window with 0.5 s hop, skipping
  near-silent windows.
- **Splitting:** 70 / 15 / 15 train / val / test, **grouped by person** (Coswara,
  COUGHVID) or **by source recording** (ESC-50). No person appears in two
  splits, so the test score measures performance on unseen people rather than
  memorized voices.

Resulting dataset: **449,617 clips**.

| Split | cough | not cough |
|---|---|---|
| train | 74,369 | 238,536 |
| val | 16,102 | 54,286 |
| test | 15,903 | 50,421 |

| Source | cough | not cough |
|---|---|---|
| COUGHVID | 80,835 | 0 |
| Coswara | 25,437 | 327,682 |
| ESC-50 | 102 | 15,561 |

### 2. Features

Each clip becomes a log-mel spectrogram: 25 ms window (`n_fft` 400), 10 ms hop,
64 mel bands from 50 Hz to 8 kHz, `log(mel_power + 1e-6)`, giving a 64 × 101 image.
Each mel band is then normalized with a mean and standard deviation measured on the training set.

### 3. Model

Four convolution blocks (16 → 32 → 64 → 128 channels; each block is
2 × [3×3 conv, batch norm, ReLU] followed by 2×2 max pooling), then global average
pooling, dropout 0.3 and a single output. **293,841 parameters.**

### 4. Training

- **Loss:** binary cross-entropy, with the cough class weighted by about 3.2× to
  offset the 1 : 3.2 class imbalance.
- **Optimizer:** AdamW with a one-cycle learning-rate schedule (max 3e-3), batch size 256.
- **Augmentation:**
  - SpecAugment-style: random gain (about ±4 dB), time shift up to ±100 ms, and
    one frequency mask plus one time mask.
  - **Background-noise mixing** (`--noise 0.5`): half of the training clips get a
    random not-cough training clip (environmental sound, breathing or speech) added
    at a random 0–20 dB SNR. Mixing is done on mel power: `log(exp(x) + g·exp(n))`.
    The label stays the same: a cough mixed with noise is still a cough.
- **Model selection:** after each epoch, the decision threshold that maximizes F1 on
  val is chosen, and the epoch with the best val F1 is kept. The test set is only
  used once, at the end.

## Results

All numbers are on the **held-out test set**: 66,324 clips from people not seen in training.

### Round 1 vs. round 2

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

### Per source (round 2)

| Source | Clips | Precision | Recall | Notes |
|---|---|---|---|---|
| COUGHVID | 12,145 | 1.000 | 0.953 | All coughs, so precision is trivially 1.0 |
| Coswara | 51,959 | 0.854 | 0.929 | 596 false alarms out of ~48k breathing / speech clips (round 1: 682) |
| ESC-50 | 2,220 | 0.306 | 0.917 | 25 false alarms out of 2,208 environmental clips, about 1.1% (round 1: 24) |

### Robustness to background noise

Each test clip was mixed with a random not-cough **test** clip at a fixed SNR.
Both models were run with their own thresholds.

| Noise level | Round 1 F1 | **Round 2 F1** | Round 1 recall | **Round 2 recall** |
|---|---|---|---|---|
| Clean | 0.953 | **0.954** | 0.951 | 0.948 |
| 20 dB SNR | 0.912 | **0.948** | 0.870 | **0.943** |
| 10 dB SNR | 0.845 | **0.937** | 0.760 | **0.926** |
| 5 dB SNR | 0.774 | **0.922** | 0.657 | **0.899** |
| 0 dB SNR | 0.665 | **0.894** | 0.518 | **0.849** |

Noise mixing is what makes the model usable in real rooms. At 0 dB SNR, where the
background is as loud as the cough, round 1 misses about half the coughs, and round 2
still catches 85%.

### Limitations

- **Environmental false alarms:** about 1% of non-cough environmental sounds are still
  flagged as coughs, and noise mixing did not change this. ESC-50 is only about 3% of
  the training data, so more varied negative audio (for example FSD50K or AudioSet) is
  the likely fix.
- **Recording devices:** all training audio was recorded on phones. A different
  microphone (for example an I2S MEMS mic on an ESP32) will sound different, and a few
  hundred recordings from the target device should be used for fine-tuning.
- **No medical use:** the model detects *that* someone coughed, not *why*. It has not
  been validated for any medical purpose.

## Using the model

Files in `cough_dataset/` (not in git):

| File | Contents |
|---|---|
| `model.pt` | PyTorch weights, per-band `mean` / `std`, `threshold`, feature settings |
| `model.onnx` | Same network for onnxruntime / mobile (outputs match PyTorch to about 3e-6) |
| `results.txt` | Test report |
| `model_r1.*`, `model_r2.*`, `results_r*.txt` | Copies of each training round |

To run inference on a 1 s, 16 kHz mono window `y`:

1. `x = log(melspectrogram(y, sr=16000, n_fft=400, hop_length=160, n_mels=64, fmin=50, fmax=8000) + 1e-6)`, shape 64 × 101.
2. `x = (x - mean) / std`, using the per-band values stored in `model.pt`.
3. Feed it as shape `[1, 1, 64, 101]` (input name `logmel`). The output `logit` means **cough** when `sigmoid(logit) ≥ 0.83`.

For a continuous stream, slide the 1 s window with a 0.5 s hop.

## Next steps

- Run the model on the Raspberry Pi (onnxruntime) on live audio streamed from the
  ESP32 vest mics and the phone. In the Nafas design the ESP32 only collects audio,
  and all the AI runs on the Pi.
- Fine-tune on our own vest recordings.
- Add more varied negative audio to cut environmental false alarms.
