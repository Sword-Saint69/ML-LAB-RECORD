import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print("Samples:", X.shape, "Classes:", data.target_names.tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", len(X_train), " Test:", len(X_test))
print("K  acc prec rec f1")
cm_best, best_a, best_k, acc = None, -1.0, 0, []
for K in [3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=K, p=2)
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    a = accuracy_score(y_test, p)
    acc.append(a)
    print(f"{K}  {a:.4f} {precision_score(y_test, p):.4f} {recall_score(y_test, p):.4f} {f1_score(y_test, p):.4f}")
    if a > best_a:
        best_a, best_k = a, K
        cm_best = confusion_matrix(y_test, p)
print("Confusion matrix at best K:")
print(cm_best)
print(f"Best K = {best_k}")
plt.figure(figsize=(7, 4.5))
plt.plot([3, 5, 7, 9], acc, marker="o", linewidth=2)
plt.xlabel("K")
plt.ylabel("Test accuracy")
plt.title("KNN on Breast Cancer: Accuracy vs K")
plt.grid(True)
plt.tight_layout()
plt.savefig("exp5_knn_acc.png", dpi=150)
