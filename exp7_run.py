import os
import time
import gzip
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
HERE = os.path.dirname(os.path.abspath(__file__))
for f in FILES:
    p = os.path.join(HERE, f + ".gz")
    if not os.path.exists(p):
        print("Downloading", f)
        urllib.request.urlretrieve(BASE + f + ".gz", p)
def load_img(name, n):
    with gzip.open(os.path.join(HERE, name + ".gz"), "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(n, 784).astype(np.float32) / 255.0
def load_lbl(name, n):
    with gzip.open(os.path.join(HERE, name + ".gz"), "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)[:n]
X_all = load_img("train-images-idx3-ubyte", 60000)
y_all = load_lbl("train-labels-idx1-ubyte", 60000)
Xt_all = load_img("t10k-images-idx3-ubyte", 10000)
yt_all = load_lbl("t10k-labels-idx1-ubyte", 10000)
print("Full: train", X_all.shape, " test", Xt_all.shape)
rng = np.random.default_rng(42)
tri = rng.permutation(60000)[:6000]
tei = rng.permutation(10000)[:1000]
Xtr, ytr = X_all[tri], y_all[tri]
Xte, yte = Xt_all[tei], yt_all[tei]
print("Used: train", Xtr.shape, " test", Xte.shape)
t0 = time.perf_counter()
D = np.empty((len(Xte), len(Xtr)), dtype=np.float32)
for i in range(0, len(Xte), 100):
    D[i:i + 100] = ((Xte[i:i + 100, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
print(f"Distance matrix {D.shape} in {time.perf_counter() - t0:.1f}s")
acc = []
for K in [1, 3, 5, 7, 10, 15]:
    t1 = time.perf_counter()
    nn = np.argpartition(D, K - 1, axis=1)[:, :K]
    votes = ytr[nn]
    pred = np.array([np.bincount(r, minlength=10).argmax() for r in votes])
    a = float((pred == yte).mean())
    acc.append(a)
    print(f"K = {K:2d}  accuracy = {a:.4f}  vote time = {time.perf_counter() - t1:.2f}s")
print(f"Best K = {[1, 3, 5, 7, 10, 15][int(np.argmax(acc))]} with accuracy {max(acc):.4f}")
plt.figure(figsize=(7, 4.5))
plt.plot([1, 3, 5, 7, 10, 15], acc, marker="o", linewidth=2)
plt.xlabel("K")
plt.ylabel("Test accuracy")
plt.title("KNN on Fashion MNIST: Accuracy vs K")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "exp7_acc.png"), dpi=150)
print("saved exp7_acc.png")
