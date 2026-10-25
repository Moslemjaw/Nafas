"""
Nafas - lung sound classifier training (Capstone 1)

Classifies each respiratory cycle from lung_prep.py as normal / crackle / wheeze / both,
with either the small cough CNN (train.py, 4 outputs) or an ImageNet-pretrained ResNet18.

Usage:
  pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

  python train_lung.py features --data lung_dataset   # once: log-mels -> features.npy
  python train_lung.py train    --data lung_dataset --arch resnet18 --recnorm \
                                --mixup 0.4 --epochs 30 --lr 1e-3   # the final model
  python train_lung.py web      --data lung_dataset   # copy model + mel config into web/public/model

Scored the ICBHI 2017 way:
  Sp    = normal cycles classified normal
  Se    = abnormal cycles classified as exactly the right abnormal class
  Score = (Se + Sp) / 2

Outputs (in --data): features.npy, model.pt, model.onnx, lung_config.json, results.txt
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from train import build_model

SR = 16000
SECONDS = 4          # every cycle is cut or repeat-padded to this length
N_FFT = 1024         # 64 ms: finer frequency detail for tonal wheezes
HOP = 160            # 10 ms
N_MELS = 64
FMIN, FMAX = 50, 2000  # lung sounds live below 2 kHz; also hides device bandwidth
FRAMES = SECONDS * SR // HOP + 1
LABELS = ["normal", "crackle", "wheeze", "both"]


# ---------- features ----------
def fit_cycle(y):
    """Repeat a short cycle until it fills SECONDS, or take the middle of a long one."""
    n = SECONDS * SR
    if len(y) < n:
        y = np.tile(y, n // len(y) + 1)[:n]
    else:
        s = (len(y) - n) // 2
        y = y[s:s + n]
    return y


def logmel(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return np.log(m + 1e-6).astype(np.float16)


def _feat(path):
    y, _ = sf.read(path, dtype="float32")
    return logmel(fit_cycle(y))


def make_features(data: Path):
    df = pd.read_csv(data / "labels.csv")
    out = np.lib.format.open_memmap(data / "features.npy", mode="w+",
                                    dtype=np.float16, shape=(len(df), N_MELS, FRAMES))
    paths = [str(data / f) for f in df["file"]]
    with Pool() as pool:
        for i, x in enumerate(tqdm(pool.imap(_feat, paths, chunksize=32),
                                   total=len(paths), desc="log-mel")):
            out[i] = x
    out.flush()
    print(f"wrote {data / 'features.npy'}  {out.shape}")


# ---------- training ----------
def augment(x):
    """Random gain, cyclic time shift (cycles are repeat-padded), freq + time masks."""
    import torch
    b = x.shape[0]
    x = x + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 2.0
    x = torch.roll(x, int(torch.randint(0, FRAMES, (1,))), dims=3)
    f0 = torch.randint(0, N_MELS - 8, (b,), device=x.device)
    t0 = torch.randint(0, FRAMES - 40, (b,), device=x.device)
    fi = torch.arange(N_MELS, device=x.device)[None, :]
    ti = torch.arange(FRAMES, device=x.device)[None, :]
    fmask = (fi >= f0[:, None]) & (fi < f0[:, None] + torch.randint(0, 9, (b, 1), device=x.device))
    tmask = (ti >= t0[:, None]) & (ti < t0[:, None] + torch.randint(0, 41, (b, 1), device=x.device))
    return x.masked_fill(fmask[:, None, :, None] | tmask[:, None, None, :], 0.0)


def icbhi_scores(y, pred):
    cm = confusion_matrix(y, pred, labels=range(4))
    sp = cm[0, 0] / max(cm[0].sum(), 1)
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / max(cm[1:].sum(), 1)
    # 2-class view: is anything abnormal there at all?
    se2 = cm[1:, 1:].sum() / max(cm[1:].sum(), 1)
    return dict(se=se, sp=sp, score=(se + sp) / 2, se_any=se2, cm=cm)


def build_lung_model(arch, pretrained=True):
    """'cnn': the small cough CNN (294k params). 'resnet18': ImageNet-pretrained
    ResNet18 (11M params); the 1-channel log-mel is stretched to 128 bands and copied
    to 3 channels so the pretrained filters fit."""
    import torch.nn as nn
    if arch == "cnn":
        return build_model(n_out=4)
    import torchvision

    class ResNetLung(nn.Module):
        def __init__(self):
            super().__init__()
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = torchvision.models.resnet18(weights=w)
            self.net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 4))

        def forward(self, x):
            x = nn.functional.interpolate(x, scale_factor=(2, 1), mode="bilinear",
                                          align_corners=False)
            return self.net(x.expand(-1, 3, -1, -1))

    return ResNetLung()


def recording_norm(X, recordings):
    """Subtract each recording's average log-mel spectrum (over all its cycles and
    frames), per band. A stethoscope's frequency response is a constant offset in
    log-mel, so this removes most of it; crackles and wheezes come and go within a
    recording, so they survive. On the vest this becomes a running mean per mic."""
    X = X.astype(np.float32)
    for rec in np.unique(recordings):
        k = recordings == rec
        X[k] -= X[k].mean(axis=(0, 2), keepdims=True)
    return X


def predict(model, X, idx, mean, std, device, bs=256):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            x = torch.from_numpy(X[idx[b:b + bs]].astype(np.float32)).to(device)
            out.append(torch.softmax(model(((x - mean) / std)[:, None]), 1).cpu().numpy())
    return np.concatenate(out)


def train(data: Path, epochs: int, bs: int, lr: float, recnorm: bool, mixup: float, arch: str):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    df = pd.read_csv(data / "labels.csv")
    X = np.load(data / "features.npy")          # small enough for RAM
    if recnorm:
        X = recording_norm(X, df["recording"].to_numpy())
    print(f"recording normalization: {recnorm}, mixup alpha: {mixup}")
    y_all = df["label"].map(LABELS.index).to_numpy()
    split = {s: np.flatnonzero(df["split"] == s) for s in ("train", "val", "test")}
    for s, i in split.items():
        counts = np.bincount(y_all[i], minlength=4)
        print(f"{s:5s} {len(i):5d} cycles  " + "  ".join(f"{l} {c}" for l, c in zip(LABELS, counts)))

    xs = X[split["train"]].astype(np.float32)
    mean = torch.tensor(xs.mean(axis=(0, 2))[:, None], device=device)
    std = torch.tensor(xs.std(axis=(0, 2))[:, None] + 1e-5, device=device)

    counts = np.bincount(y_all[split["train"]], minlength=4)
    weights = torch.tensor(np.sqrt(counts.sum() / (4 * counts)), dtype=torch.float32, device=device)
    print("class weights:", dict(zip(LABELS, weights.cpu().numpy().round(2))))

    model = build_lung_model(arch).to(device)
    print(f"model: {arch}, {sum(p.numel() for p in model.parameters()):,} params")
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps = epochs * ((len(split["train"]) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        idx = np.random.permutation(split["train"])
        tot = 0.0
        for b in range(0, len(idx), bs):
            sel = idx[b:b + bs]
            x = torch.from_numpy(X[sel].astype(np.float32)).to(device)
            x = augment(((x - mean) / std)[:, None])
            yb = torch.from_numpy(y_all[sel]).to(device)
            if mixup > 0:
                lam = float(np.random.beta(mixup, mixup))
                perm = torch.randperm(len(sel), device=device)
                out = model(lam * x + (1 - lam) * x[perm])
                loss = lam * loss_fn(out, yb) + (1 - lam) * loss_fn(out, yb[perm])
            else:
                loss = loss_fn(model(x), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(sel)
        pv = predict(model, X, split["val"], mean, std, device)
        m = icbhi_scores(y_all[split["val"]], pv.argmax(1))
        mark = ""
        if m["score"] > best:
            best = m["score"]
            mark = "  *"
            torch.save({"model": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                        "recnorm": recnorm, "arch": arch}, data / "model.pt")
        if ep % 5 == 0 or mark:
            print(f"epoch {ep:3d}  loss {tot / len(idx):.4f}  val Se {m['se']:.3f}  "
                  f"Sp {m['sp']:.3f}  Score {m['score']:.3f}{mark}")

    # ---------- test ----------
    ck = torch.load(data / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    te = split["test"]
    pt = predict(model, X, te, mean, std, device)
    yt, pred = y_all[te], pt.argmax(1)
    m = icbhi_scores(yt, pred)
    lines = [f"ICBHI 2017 official test split: {len(te)} cycles, "
             f"{df.iloc[te].patient.nunique()} patients", "",
             f"4-class   Se {m['se']:.3f}  Sp {m['sp']:.3f}  Score {m['score']:.3f}",
             f"2-class   normal vs abnormal: Se {m['se_any']:.3f}  Sp {m['sp']:.3f}  "
             f"Score {(m['se_any'] + m['sp']) / 2:.3f}", "",
             "Confusion matrix (rows = true, cols = predicted):",
             "          " + "".join(f"{l:>9s}" for l in LABELS)]
    for l, row in zip(LABELS, m["cm"]):
        lines.append(f"{l:9s} " + "".join(f"{v:9d}" for v in row) +
                     f"   recall {row[LABELS.index(l)] / max(row.sum(), 1):.3f}")
    lines += ["", "By stethoscope:"]
    dev = df["device"].to_numpy()[te]
    for d in sorted(set(dev)):
        k = dev == d
        md = icbhi_scores(yt[k], pred[k])
        lines.append(f"{d:10s} n={k.sum():5d}  Se {md['se']:.3f}  Sp {md['sp']:.3f}  "
                     f"Score {md['score']:.3f}")
    report = "\n".join(lines)
    print("\n" + report)
    (data / "results.txt").write_text(report + "\n")
    export(data)


def export(data: Path):
    import torch
    ck = torch.load(data / "model.pt", map_location="cpu", weights_only=True)
    model = build_lung_model(ck.get("arch", "cnn"), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 1, N_MELS, FRAMES), str(data / "model.onnx"),
                      input_names=["logmel"], output_names=["logits"],
                      dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}})
    cfg = {"sr": SR, "seconds": SECONDS, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS,
           "fmin": FMIN, "fmax": FMAX, "frames": FRAMES, "log_eps": 1e-6, "labels": LABELS,
           "recording_norm": bool(ck.get("recnorm", False)), "arch": ck.get("arch", "cnn"),
           "mean": ck["mean"].squeeze(1).tolist(), "std": ck["std"].squeeze(1).tolist()}
    (data / "lung_config.json").write_text(json.dumps(cfg))
    print(f"exported {data / 'model.onnx'} and {data / 'lung_config.json'}")


def export_web(data: Path, out: Path):
    """lung.onnx + the mel filterbank and normalization for web/src/lung.ts."""
    import shutil
    cfg = json.loads((data / "lung_config.json").read_text())
    cfg["mel_fb"] = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                                        fmin=FMIN, fmax=FMAX).tolist()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(data / "model.onnx", out / "lung.onnx")
    (out / "lung_config.json").write_text(json.dumps(cfg))
    print(f"wrote {out / 'lung.onnx'} and {out / 'lung_config.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "train", "export", "web"])
    ap.add_argument("--data", type=Path, default=Path("lung_dataset"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--recnorm", action="store_true",
                    help="subtract each recording's mean spectrum (removes stethoscope color)")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup alpha, 0 = off")
    ap.add_argument("--arch", choices=["cnn", "resnet18"], default="cnn")
    ap.add_argument("--web-out", type=Path, default=Path("web/public/model"))
    a = ap.parse_args()
    if a.cmd == "features":
        make_features(a.data)
    elif a.cmd == "export":
        export(a.data)
    elif a.cmd == "web":
        export_web(a.data, a.web_out)
    else:
        train(a.data, a.epochs, a.bs, a.lr, a.recnorm, a.mixup, a.arch)


if __name__ == "__main__":
    main()
