import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)
print("Digits:", X.shape)
print("Classes:", np.bincount(y).tolist())
Ks, inertias, silhouettes = [], [], []
for k in [2, 5, 8, 10, 12, 15]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    Ks.append(k)
    inertias.append(km.inertia_)
    silhouettes.append(sil)
    print(f"K = {k:2d}  inertia = {km.inertia_:.1f}  silhouette = {sil:.4f}")
best = Ks[silhouettes.index(max(silhouettes))]
print(f"Best K by silhouette = {best}")
km = KMeans(n_clusters=best, n_init=10, random_state=42)
labels = km.fit_predict(X)
print("Cluster sizes:", np.bincount(labels).tolist())
plt.figure(figsize=(7, 4.5))
plt.plot(Ks, inertias, marker="o", linewidth=2)
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("K-means on Digits: Elbow Method")
plt.grid(True)
plt.tight_layout()
plt.savefig("exp10_elbow.png", dpi=150)
plt.figure(figsize=(7, 4.5))
plt.plot(Ks, silhouettes, marker="s", linewidth=2)
plt.xlabel("K")
plt.ylabel("Silhouette score")
plt.title("K-means on Digits: Silhouette vs K")
plt.grid(True)
plt.tight_layout()
plt.savefig("exp10_sil.png", dpi=150)
