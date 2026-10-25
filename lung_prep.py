"""
Nafas - lung sound dataset prep (Capstone 1)

Cuts ICBHI 2017 recordings into respiratory cycles labeled
normal / crackle / wheeze / both, using the doctors' cycle annotations.

Each recording is named {patient}_{index}_{location}_{mode}_{device}, for example
101_1b1_Al_sc_Meditron: patient 101, anterior left chest, single channel, Meditron
stethoscope. The location is kept so a later model can say which lung.

Split: the official ICBHI train/test split (by patient). 20% of the official-train
patients are held out as val, so no patient appears in two splits.

Usage:
  python lung_prep.py --icbhi path/to/ICBHI_final_database \
                      --split ICBHI_challenge_train_test.txt --out lung_dataset
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

SR = 16000           # same rate as the cough pipeline; recordings are 4-44.1 kHz
LABELS = ["normal", "crackle", "wheeze", "both"]
LOCATIONS = {"Tc": "trachea", "Al": "anterior left", "Ar": "anterior right",
             "Pl": "posterior left", "Pr": "posterior right",
             "Ll": "lateral left", "Lr": "lateral right"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icbhi", type=Path, required=True, help="folder with the .wav and .txt files")
    ap.add_argument("--split", type=Path, required=True, help="ICBHI_challenge_train_test.txt")
    ap.add_argument("--out", type=Path, default=Path("lung_dataset"))
    a = ap.parse_args()

    official = dict(line.split() for line in a.split.read_text().splitlines() if line.strip())
    # the split file names one recording with a different device (226_1b1_Pl_sc_Meditron
    # vs LittC2SE in the data), so fall back to matching without the device name
    by_prefix = {k.rsplit("_", 1)[0]: v for k, v in official.items()}
    rows = []
    for wav in tqdm(sorted(a.icbhi.glob("*.wav")), desc="ICBHI"):
        rec = wav.stem
        patient, _, loc, mode, device = rec.split("_")
        ann = pd.read_csv(wav.with_suffix(".txt"), sep=r"\s+", header=None,
                          names=["start", "end", "crackle", "wheeze"])
        y, _ = librosa.load(wav, sr=SR, mono=True)
        peak = np.abs(y).max()
        if peak == 0:
            continue
        y = y / peak * 0.9          # per recording, like cough_prep.py
        for i, c in enumerate(ann.itertuples()):
            seg = y[int(c.start * SR):int(c.end * SR)]
            if len(seg) < SR // 10:  # a few annotations run past the end of the audio
                continue
            label = LABELS[int(c.crackle) + 2 * int(c.wheeze)]
            rel = Path("cycles") / f"{rec}_{i:02d}.wav"
            (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(a.out / rel, seg, SR, subtype="PCM_16")
            rows.append({"file": str(rel), "label": label, "crackle": int(c.crackle),
                         "wheeze": int(c.wheeze), "patient": int(patient), "recording": rec,
                         "location": LOCATIONS.get(loc, loc), "mode": mode, "device": device,
                         "seconds": round(len(seg) / SR, 3),
                         "split": official.get(rec) or by_prefix[rec.rsplit("_", 1)[0]]})

    df = pd.DataFrame(rows)
    # The official split has a few patients on both sides. Keep the official test set
    # unchanged (so scores compare with papers) and drop their train recordings instead.
    both = set(df[df.split == "train"].patient) & set(df[df.split == "test"].patient)
    if both:
        drop = (df.split == "train") & df.patient.isin(both)
        print(f"patients in both official splits: {sorted(both)}; "
              f"dropping their {drop.sum()} train cycles")
        df = df[~drop].reset_index(drop=True)

    tr = df[df.split == "train"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, va = next(gss.split(tr, groups=tr["patient"]))
    df.loc[tr.index[va], "split"] = "val"

    df.to_csv(a.out / "labels.csv", index=False)
    print(f"\n{len(df)} cycles from {df.recording.nunique()} recordings, "
          f"{df.patient.nunique()} patients")
    print("\nCycles per split and label:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0)[LABELS])
    print("\nPatients per split:", df.groupby("split").patient.nunique().to_dict())
    print(f"\nCycle length: median {df.seconds.median():.1f} s, "
          f"90% under {df.seconds.quantile(0.9):.1f} s")


if __name__ == "__main__":
    main()
