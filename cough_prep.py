"""
Nafas - cough dataset prep (Capstone 1)

Builds 1-second, 16 kHz mono clips labeled cough / not_cough from:
  - COUGHVID  (phone coughs)              -> cough
  - Coswara   (phone coughs + breathing,  -> cough / not_cough
               counting, vowels)
  - ESC-50    (environmental noise)       -> not_cough ("coughing" class -> cough)

Then splits train/val/test BY PERSON / SOURCE FILE so no speaker appears in two splits.

Usage:
  pip install librosa soundfile pandas scikit-learn tqdm
  (ffmpeg must be installed to read COUGHVID .webm/.ogg files)

  python cough_prep.py --coughvid path/to/coughvid --coswara path/to/Coswara-Data/Extracted_data \
                       --esc50 path/to/ESC-50 --out cough_dataset

Any dataset path can be left out.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

SR = 16000
WIN = SR            # 1 s clips
HOP = SR // 2       # 0.5 s hop for non-cough windows
SILENCE_RMS = 0.005 # drop near-silent windows
COUGHVID_MIN_CONF = 0.8  # COUGHVID "cough_detected" threshold

COSWARA_COUGH = ["cough-heavy", "cough-shallow"]
COSWARA_OTHER = ["breathing-deep", "breathing-shallow", "counting-normal",
                 "counting-fast", "vowel-a", "vowel-e", "vowel-o"]


# ---------- audio helpers ----------
def load(path: Path):
    """Load any file as 16 kHz mono float32. Uses ffmpeg for webm/ogg."""
    if path.suffix.lower() in {".webm", ".ogg"}:
        cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le",
               "-ac", "1", "-ar", str(SR), "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        y = np.frombuffer(raw, dtype=np.float32)
    else:
        y, _ = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        return None
    peak = np.max(np.abs(y))
    return y / peak * 0.9 if peak > 0 else None


def fit(seg):
    """Pad or crop to exactly 1 s."""
    if len(seg) >= WIN:
        return seg[:WIN]
    return np.pad(seg, (0, WIN - len(seg)))


def cough_events(y):
    """Find loud regions (coughs) and return a 1 s clip centered on each."""
    clips = []
    for start, end in librosa.effects.split(y, top_db=30):
        if end - start < SR * 0.1:      # ignore clicks < 100 ms
            continue
        center = (start + end) // 2
        s = max(0, center - WIN // 2)
        clips.append(fit(y[s:s + WIN]))
    return clips


def windows(y):
    """Sliding 1 s windows, skipping silence."""
    clips = []
    for s in range(0, max(1, len(y) - WIN + 1), HOP):
        seg = fit(y[s:s + WIN])
        if np.sqrt(np.mean(seg ** 2)) > SILENCE_RMS:
            clips.append(seg)
    return clips


# ---------- dataset readers ----------
def read_coughvid(root: Path):
    meta = pd.read_csv(root / "metadata_compiled.csv")
    meta = meta[meta["cough_detected"] >= COUGHVID_MIN_CONF]
    for uuid in tqdm(meta["uuid"], desc="COUGHVID"):
        f = next((root / f"{uuid}{ext}" for ext in (".webm", ".ogg", ".wav")
                  if (root / f"{uuid}{ext}").exists()), None)
        if f is None:
            continue
        y = load(f)
        if y is None:
            continue
        for c in cough_events(y):
            yield c, "cough", "coughvid", f"cv_{uuid}"


def read_coswara(root: Path):
    for person in tqdm(sorted(p for p in root.glob("*/*") if p.is_dir()), desc="Coswara"):
        group = f"cw_{person.name}"
        for name in COSWARA_COUGH + COSWARA_OTHER:
            f = person / f"{name}.wav"
            if not f.exists():
                continue
            y = load(f)
            if y is None:
                continue
            if name in COSWARA_COUGH:
                for c in cough_events(y):
                    yield c, "cough", "coswara", group
            else:
                for c in windows(y):
                    yield c, "not_cough", "coswara", group


def read_esc50(root: Path):
    meta = pd.read_csv(root / "meta" / "esc50.csv")
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="ESC-50"):
        y = load(root / "audio" / r["filename"])
        if y is None:
            continue
        label = "cough" if r["category"] == "coughing" else "not_cough"
        clips = cough_events(y) if label == "cough" else windows(y)
        for c in clips:
            yield c, label, "esc50", f"esc_{r['src_file']}"


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coughvid", type=Path)
    ap.add_argument("--coswara", type=Path)
    ap.add_argument("--esc50", type=Path)
    ap.add_argument("--out", type=Path, default=Path("cough_dataset"))
    a = ap.parse_args()

    readers = []
    if a.coughvid: readers.append(read_coughvid(a.coughvid))
    if a.coswara:  readers.append(read_coswara(a.coswara))
    if a.esc50:    readers.append(read_esc50(a.esc50))
    if not readers:
        ap.error("give at least one dataset path")

    rows, n = [], 0
    for reader in readers:
        for clip, label, source, group in reader:
            rel = Path(label) / f"{source}_{n:07d}.wav"
            (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(a.out / rel, clip, SR, subtype="PCM_16")
            rows.append({"file": str(rel), "label": label,
                         "source": source, "group": group})
            n += 1

    df = pd.DataFrame(rows)

    # split by person/source file: 70 / 15 / 15
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr, rest = next(gss.split(df, groups=df["group"]))
    df.loc[df.index[tr], "split"] = "train"
    rest_df = df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    va, te = next(gss2.split(rest_df, groups=rest_df["group"]))
    df.loc[rest_df.index[va], "split"] = "val"
    df.loc[rest_df.index[te], "split"] = "test"

    df.to_csv(a.out / "labels.csv", index=False)
    print("\nClips per split and label:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\nClips per source:")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
