# Setting up Nafas on a new computer

These are the steps we followed to set up the project, written so a teammate can do
the same on another computer. The code is on GitHub, but the datasets and tools have
to be downloaded separately.

> If you only want to use the app you don't need any of this. Just open
> https://nafas-lemon.vercel.app on a phone (or `/#display` on a computer). The trained
> models are already in the repo, in `web/public/model/`.

## 1. Install the tools

| What | Why | Where |
|---|---|---|
| **Git** | Get the code | https://git-scm.com (comes with Git Bash, used for the commands below) |
| **Python 3.10** | The training scripts | https://python.org (we used 3.10.5) |
| **Node.js 22** | The phone app in `web/` | https://nodejs.org |
| **ffmpeg** | Reads COUGHVID's `.webm` and `.ogg` files | Windows: `winget install Gyan.FFmpeg` (open a new terminal afterwards) |
| **NVIDIA GPU** (optional) | Training is about 10× faster (we used a GTX 1660 Ti) | Install the latest NVIDIA driver |

> Keep the laptop plugged in while training. We learned this the hard way: on battery
> Windows slows the GPU down to about a seventh of its speed, and a long run can die
> when the battery does.

Python packages:

```bash
pip install librosa soundfile pandas scikit-learn tqdm openpyxl
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime
```

(No NVIDIA GPU? Install `torch` and `torchvision` without the `--index-url` part. It
works, just slower.)

## 2. Get the code

```bash
git clone https://github.com/Moslemjaw/Nafas.git
cd Nafas
cd web && npm install && cd ..
```

## 3. Download the datasets

Put them all **inside the `Nafas` folder**, named exactly as below (they're already in
`.gitignore`, so git won't try to upload them). You need about **60 GB** of free space:
the downloads plus the clips and features the scripts make.

```
Nafas/
├── ESC-50/                    ← audio/ and meta/ inside
├── coughvid/                  ← ~68,000 .webm/.ogg/.wav/.json files + metadata_compiled.csv
├── Coswara-Data/
│   └── Extracted_data/        ← one folder per date, one folder per person inside
├── ICBHI_final_database/      ← 920 .wav + 920 .txt
└── _downloads/
    └── icbhi/
        ├── ICBHI_challenge_train_test.txt
        └── ICBHI_Challenge_diagnosis.txt
```

| Dataset | Size | Download | Notes |
|---|---|---|---|
| **ESC-50** | 0.6 GB | https://github.com/karolpiczak/ESC-50 → **Code → Download ZIP** | Unzip, then rename `ESC-50-master` to `ESC-50` |
| **COUGHVID v3** | 2.3 GB | https://zenodo.org/records/7024894 → `public_dataset_v3.zip` | Download it **in a browser** (Zenodo blocks some download tools). Move the files from the zip's `coughvid_20211012` folder into `coughvid/`. If Windows' own unzip complains, use 7-Zip |
| **Coswara** | ~13 GB to download, ~25 GB on disk | `git clone --depth 1 https://github.com/iiscleap/Coswara-Data.git _downloads/Coswara-Data` | The audio is in split archives; unpack them with the command below |
| **ICBHI 2017** | 2 GB | https://bhichallenge.med.auth.gr/ → `ICBHI_final_database.zip`, plus `ICBHI_challenge_train_test.txt` and `ICBHI_Challenge_diagnosis.txt` | The site's security certificate had expired when we downloaded, so the browser may warn. The server is slow (~0.3 MB/s, over an hour); a download manager that uses several connections is much faster. Unzip so you get `ICBHI_final_database/` |

**Unpacking Coswara** (in Git Bash, from the `Nafas` folder; takes about 15 minutes):

```bash
mkdir -p Coswara-Data/Extracted_data
cd _downloads/Coswara-Data
for d in 2*/; do d=${d%/}; cat $d/$d.tar.gz.* | tar -xzf - -C ../../Coswara-Data/Extracted_data; done
cd ../..
```

When it's done, `Coswara-Data/Extracted_data` should have 43 date folders and about
2,747 person folders inside them.

## 4. Build the data and train the models

From the `Nafas` folder. Times are with a GTX 1660 Ti, plugged in.

**Cough model**

```bash
python cough_prep.py --coughvid coughvid --coswara Coswara-Data/Extracted_data --esc50 ESC-50 --out cough_dataset   # ~1 h
python train.py features --data cough_dataset                                                                       # ~10 min, writes a 5.8 GB file
python train.py train --data cough_dataset --epochs 25 --noise 0.5                                                  # ~1.5 h
python train.py web --data cough_dataset                                                                            # copies the model into the app
```

Expected result (`cough_dataset/results.txt`): F1 about 0.935 on the test set. This test set
also has the hard loud-voice clips, so it is lower than the 0.948 in the README (which is
on the original test set).

**Lung model**

```bash
python lung_prep.py --icbhi ICBHI_final_database --split _downloads/icbhi/ICBHI_challenge_train_test.txt --out lung_dataset   # ~2 min
python train_lung.py features --data lung_dataset                                                                             # ~1 min
python train_lung.py train --data lung_dataset --arch resnet18 --recnorm --mixup 0.4 --epochs 30 --lr 1e-3                    # ~10 min
python train_lung.py web --data lung_dataset                                                                                  # copies the model into the app
```

Expected result (`lung_dataset/results.txt`): ICBHI score about 0.54. Small differences
between runs are normal, since the dataset is small.

## 5. Run the app on your computer

```bash
cd web
npm run dev
```

Open http://localhost:5173 (the microphone works on localhost). Add `#display` to the
address for the computer-screen view.

## What isn't on GitHub

These stay on the original computer. Ask for a copy if you need them, instead of
retraining:

| File | What it is |
|---|---|
| `cough_dataset/model.pt`, `lung_dataset/model.pt` | The trained PyTorch models (the app uses the `.onnx` copies, which *are* on GitHub) |
| `Nafas_Data.xlsx` | The project data workbook (it contains patient-level ICBHI data) |
| `test_sounds/` | ICBHI wheeze, crackle and normal clips for testing the app through a speaker |

## Problems we ran into (and how we fixed them)

| Problem | Fix |
|---|---|
| COUGHVID files are skipped or fail to load | ffmpeg isn't installed, or the terminal was opened before installing it |
| Zenodo answers "403 Forbidden" | Download COUGHVID in a browser instead of a script |
| Training is very slow, or the GPU shows ~300 MHz | The laptop is on battery: plug it in |
| `lung_prep.py` stops with a recording-name error | Use the `ICBHI_challenge_train_test.txt` from the ICBHI site (the script already handles its one misnamed recording) |
| The app shows an old version | Press Ctrl + Shift + R, or close and reopen it: it caches itself for offline use |
