from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load and preprocess Digits dataset
X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)
print("Digits:", X.shape)

# K-means with various K, evaluate inertia + silhouette
Ks, inertias, silhouettes = [], [], []
for k in [2, 5, 8, 10, 12, 15]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    Ks.append(k); inertias.append(km.inertia_); silhouettes.append(sil)
    print(f"K = {k:2d}  inertia = {km.inertia_:.1f}  silhouette = {sil:.4f}")

print(f"Best K by silhouette = {Ks[silhouettes.index(max(silhouettes))]}")

# Elbow + silhouette plot
plt.figure(figsize=(7, 4.5))
plt.plot(Ks, inertias, marker="o", linewidth=2, label="Inertia")
plt.xlabel("K"); plt.ylabel("Inertia"); plt.grid(True)
plt.title("K-means on Digits: Elbow Method"); plt.tight_layout()
plt.savefig("exp9_elbow.png", dpi=150)

plt.figure(figsize=(7, 4.5))
plt.plot(Ks, silhouettes, marker="s", linewidth=2, label="Silhouette")
plt.xlabel("K"); plt.ylabel("Silhouette score"); plt.grid(True)
plt.title("K-means on Digits: Silhouette vs K"); plt.tight_layout()
plt.savefig("exp9_sil.png", dpi=150)
print("saved exp9_elbow.png, exp9_sil.png")
