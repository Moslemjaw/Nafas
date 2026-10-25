"""
Nafas - cough detector training (Capstone 1)

Trains a small CNN on log-mel spectrograms of the 1 s clips made by cough_prep.py.

Usage:
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install onnx onnxruntime

  python train.py features --data cough_dataset     # once: log-mels -> features.npy
  python train.py train    --data cough_dataset     # train, pick best on val, report test
  python train.py export   --data cough_dataset     # (re)export model.onnx from model.pt

Outputs (in --data):
  features.npy   float16 [N, 64, 101] log-mel spectrograms, row order = labels.csv
  model.pt       best weights + normalization stats + threshold
  model.onnx     same model for mobile / onnxruntime
  results.txt    test metrics, overall and per source
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from tqdm import tqdm

SR = 16000
N_FFT = 400        # 25 ms window
HOP = 160          # 10 ms hop
N_MELS = 64
FRAMES = 101       # 1 s at 10 ms hop, centered


# ---------- features ----------
def logmel(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=50, fmax=8000)
    return np.log(m + 1e-6).astype(np.float16)


def _feat(path):
    y, _ = sf.read(path, dtype="float32")
    return logmel(y)


def make_features(data: Path):
    df = pd.read_csv(data / "labels.csv")
    out = np.lib.format.open_memmap(data / "features.npy", mode="w+",
                                    dtype=np.float16, shape=(len(df), N_MELS, FRAMES))
    paths = [str(data / f) for f in df["file"]]
    with Pool() as pool:
        for i, x in enumerate(tqdm(pool.imap(_feat, paths, chunksize=256),
                                   total=len(paths), desc="log-mel")):
            out[i] = x
    out.flush()
    print(f"wrote {data / 'features.npy'}  {out.shape}")


# ---------- model ----------
def build_model():
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))

    return nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(128, 1))


def augment(x):
    """SpecAugment-style: random gain, time shift, one freq mask, one time mask."""
    import torch
    b = x.shape[0]
    x = x + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 2.0   # +-1 natural log = ~+-4 dB
    x = torch.roll(x, int(torch.randint(-10, 11, (1,))), dims=3)
    f0 = torch.randint(0, N_MELS - 8, (b,), device=x.device)
    t0 = torch.randint(0, FRAMES - 15, (b,), device=x.device)
    fi = torch.arange(N_MELS, device=x.device)[None, :]
    ti = torch.arange(FRAMES, device=x.device)[None, :]
    fmask = (fi >= f0[:, None]) & (fi < f0[:, None] + torch.randint(0, 9, (b, 1), device=x.device))
    tmask = (ti >= t0[:, None]) & (ti < t0[:, None] + torch.randint(0, 16, (b, 1), device=x.device))
    mask = fmask[:, None, :, None] | tmask[:, None, None, :]
    return x.masked_fill(mask, 0.0)


def batches(idx, bs, shuffle):
    if shuffle:
        idx = np.random.permutation(idx)
    for s in range(0, len(idx), bs):
        yield np.sort(idx[s:s + bs])   # sorted reads are faster on the memmap


def predict(model, X, idx, mean, std, device, bs=1024):
    import torch
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    pos = 0
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            sel = idx[b:b + bs]
            x = torch.from_numpy(X[sel].astype(np.float32)).to(device)
            x = ((x - mean) / std)[:, None]
            out[pos:pos + len(sel)] = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            pos += len(sel)
    return out


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), auc=auc,
                tp=tp, fp=fp, fn=fn, tn=tn)


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])])


# ---------- train ----------
def train(data: Path, epochs: int, bs: int, lr: float):
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    df = pd.read_csv(data / "labels.csv")
    X = np.load(data / "features.npy", mmap_mode="r")
    y_all = (df["label"] == "cough").to_numpy().astype(np.float32)
    split = {s: np.flatnonzero(df["split"] == s) for s in ("train", "val", "test")}
    for s, i in split.items():
        print(f"{s:5s} {len(i):7d} clips, {y_all[i].mean():.1%} cough")

    # per-mel-band normalization from a train sample
    samp = np.sort(np.random.default_rng(0).choice(split["train"], 20000, replace=False))
    xs = X[samp].astype(np.float32)
    mean = torch.tensor(xs.mean(axis=(0, 2))[:, None], device=device)
    std = torch.tensor(xs.std(axis=(0, 2))[:, None] + 1e-5, device=device)

    model = build_model().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    pos_weight = torch.tensor((1 - y_all[split["train"]].mean()) / y_all[split["train"]].mean(),
                              device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(split["train"]) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    best_f1, yv = -1.0, y_all[split["val"]]
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for sel in tqdm(batches(split["train"], bs, True), total=steps // epochs,
                        desc=f"epoch {ep}/{epochs}", leave=False):
            x = torch.from_numpy(X[sel].astype(np.float32)).to(device)
            x = augment(((x - mean) / std)[:, None])
            yb = torch.from_numpy(y_all[sel]).to(device)
            loss = loss_fn(model(x).squeeze(1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(sel)
            n += len(sel)

        pv = predict(model, X, split["val"], mean, std, device)
        thr = best_threshold(yv, pv)
        m = metrics(yv, pv, thr)
        print(f"epoch {ep:2d}  loss {tot / n:.4f}  val F1 {m['f1']:.4f}  "
              f"P {m['precision']:.4f}  R {m['recall']:.4f}  AUC {m['auc']:.4f}  thr {thr:.2f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"model": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                        "threshold": thr, "sr": SR, "n_fft": N_FFT, "hop": HOP,
                        "n_mels": N_MELS}, data / "model.pt")

    # ---------- test ----------
    ck = torch.load(data / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    thr = ck["threshold"]
    te = split["test"]
    pt = predict(model, X, te, mean, std, device)
    yt = y_all[te]

    lines = [f"Test set ({len(te)} clips), threshold {thr:.2f} (chosen on val)", ""]
    m = metrics(yt, pt, thr)
    lines.append("overall   F1 {f1:.4f}  P {precision:.4f}  R {recall:.4f}  AUC {auc:.4f}  "
                 "TP {tp} FP {fp} FN {fn} TN {tn}".format(**m))
    lines.append("")
    src = df["source"].to_numpy()[te]
    for s in sorted(set(src)):
        k = src == s
        ms = metrics(yt[k], pt[k], thr)
        lines.append(f"{s:9s} n={k.sum():6d}  P {ms['precision']:.4f}  R {ms['recall']:.4f}  "
                     f"TP {ms['tp']} FP {ms['fp']} FN {ms['fn']} TN {ms['tn']}")
    report = "\n".join(lines)
    print("\n" + report)
    (data / "results.txt").write_text(report + "\n")

    try:
        export(data)
    except Exception as e:
        print("ONNX export skipped:", e)


# ---------- export ----------
def export(data: Path):
    """model.pt -> model.onnx. Input: normalized log-mel [B,1,64,101]; output: logit."""
    import torch
    ck = torch.load(data / "model.pt", map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(ck["model"])
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 1, N_MELS, FRAMES), str(data / "model.onnx"),
                      input_names=["logmel"], output_names=["logit"],
                      dynamic_axes={"logmel": {0: "batch"}, "logit": {0: "batch"}})
    print(f"exported {data / 'model.onnx'}  (threshold {ck['threshold']:.2f} on sigmoid(logit))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "train", "export"])
    ap.add_argument("--data", type=Path, default=Path("cough_dataset"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    if a.cmd == "features":
        make_features(a.data)
    elif a.cmd == "export":
        export(a.data)
    else:
        train(a.data, a.epochs, a.bs, a.lr)


if __name__ == "__main__":
    main()
