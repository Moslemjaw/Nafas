"""Compare web/src/features.ts against train.py's features.npy, and the model's output on each.

  python scripts/check_features.py ../cough_dataset feats.json
"""
import json
import sys

import numpy as np
import onnxruntime as ort

data, js_path = sys.argv[1], sys.argv[2]
X = np.load(f"{data}/features.npy", mmap_mode="r")
cfg = json.load(open("public/model/cough_config.json"))
mean = np.array(cfg["mean"], dtype=np.float32)[:, None]
std = np.array(cfg["std"], dtype=np.float32)[:, None]
sess = ort.InferenceSession("public/model/cough.onnx")
prob = lambda x: 1 / (1 + np.exp(-sess.run(None, {"logmel": ((x - mean) / std)[None, None]})[0][0, 0]))

js = json.load(open(js_path))
worst_feat, worst_prob = 0.0, 0.0
for row, v in js.items():
    a = np.array(v, dtype=np.float32).reshape(cfg["n_mels"], cfg["frames"])
    b = X[int(row)].astype(np.float32)
    keep = b > -13.0                      # ignore the log(1e-6) silence floor
    d = float(np.abs(a - b)[keep].max()) if keep.any() else 0.0
    pa, pb = prob(a), prob(b)
    worst_feat, worst_prob = max(worst_feat, d), max(worst_prob, abs(pa - pb))
    print(f"row {row:>7s}  max |dlogmel| {d:.4f}   p(js) {pa:.4f}  p(py) {pb:.4f}")
print(f"\nworst log-mel diff {worst_feat:.4f}, worst probability diff {worst_prob:.4f}")
