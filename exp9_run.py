import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Samples:", X.shape, "Train:", len(X_train), " Test:", len(X_test))
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
models = [
    ("MLE", LogisticRegression(C=np.inf, max_iter=5000)),
    ("MAP L2", LogisticRegression(C=1.0, max_iter=5000)),
    ("MAP L1", LogisticRegression(l1_ratio=1, C=1.0, solver="saga", max_iter=5000)),
]
print("Model     acc    prec   rec    f1     nonzero")
for name, m in models:
    m.fit(X_train_s, y_train)
    p = m.predict(X_test_s)
    nz = int(np.sum(m.coef_[0] != 0))
    print(f"{name:8s} {accuracy_score(y_test, p):.4f} {precision_score(y_test, p):.4f} {recall_score(y_test, p):.4f} {f1_score(y_test, p):.4f} {nz}")
