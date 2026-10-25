"""Compare web/src/features.ts against the Python features.npy, and the model's output on each.

  python scripts/check_features.py ../cough_dataset public/model/cough_config.json public/model/cough.onnx feats.json
  python scripts/check_features.py ../lung_dataset  public/model/lung_config.json  public/model/lung.onnx  feats.json

Lung models were trained with per-recording normalization; this check skips it on both
sides, so only the features and the network are compared.
"""
import json
import sys

import numpy as np
import onnxruntime as ort

data, cfg_path, model_path, js_path = sys.argv[1:5]
X = np.load(f"{data}/features.npy", mmap_mode="r")
cfg = json.load(open(cfg_path))
mean = np.array(cfg["mean"], dtype=np.float32)[:, None]
std = np.array(cfg["std"], dtype=np.float32)[:, None]
sess = ort.InferenceSession(model_path)
name = sess.get_inputs()[0].name


def probs(x):
    z = sess.run(None, {name: ((x - mean) / std)[None, None]})[0][0]
    if z.size == 1:
        return np.array([1 / (1 + np.exp(-z[0]))])
    e = np.exp(z - z.max())
    return e / e.sum()


js = json.load(open(js_path))
worst_feat, worst_prob = 0.0, 0.0
for row, v in js.items():
    a = np.array(v, dtype=np.float32).reshape(cfg["n_mels"], cfg["frames"])
    b = X[int(row)].astype(np.float32)
    keep = b > -13.0                      # ignore the log(1e-6) silence floor
    d = float(np.abs(a - b)[keep].max()) if keep.any() else 0.0
    pa, pb = probs(a), probs(b)
    worst_feat, worst_prob = max(worst_feat, d), max(worst_prob, float(np.abs(pa - pb).max()))
    print(f"row {row:>7s}  max |dlogmel| {d:.4f}   p(js) {np.round(pa, 3)}  p(py) {np.round(pb, 3)}")
print(f"\nworst log-mel diff {worst_feat:.4f}, worst probability diff {worst_prob:.4f}")
