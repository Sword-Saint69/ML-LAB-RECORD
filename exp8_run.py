import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
data = load_diabetes()
X = data.data
y = data.target
print("Samples:", X.shape)
print("Features:", " ".join(data.feature_names))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", len(X_train), " Test:", len(X_test))
models = [("Linear", LinearRegression()), ("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=1.0))]
print("Model             MSE       R2")
for name, m in models:
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    mse = mean_squared_error(y_test, p)
    r2 = r2_score(y_test, p)
    print(f"{name:14s} {mse:9.2f} {r2:9.4f}")
    if name == "Linear":
        py = p
    if name == "Ridge":
        pr = p
    if name == "Lasso":
        pl = p
plt.figure(figsize=(6.5, 4.5))
plt.scatter(y_test, py, s=18, label="Linear")
plt.scatter(y_test, pr, s=18, label="Ridge")
plt.scatter(y_test, pl, s=18, label="Lasso")
lo, hi = y_test.min(), y_test.max()
plt.plot([lo, hi], [lo, hi], "k--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Diabetes: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("exp8_pred.png", dpi=150)
pos = np.arange(len(data.feature_names))
plt.figure(figsize=(7, 4.5))
for (name, m), off in zip(models, [-0.26, 0.0, 0.26]):
    plt.barh(pos + off, m.coef_, height=0.26, label=name)
plt.yticks(pos, data.feature_names)
plt.axvline(0, color="k", linewidth=0.8)
plt.xlabel("Coefficient")
plt.title("Coefficient Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("exp8_coef.png", dpi=150)
print("Lasso zeroed", int(np.sum(models[2][1].coef_ == 0)), "of 10 coefficients")
