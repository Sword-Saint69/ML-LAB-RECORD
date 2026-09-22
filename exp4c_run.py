import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
print("Shape:", df.shape)
print("Missing:", int(df.isnull().sum().sum()))
X = df[["AveRooms"]]
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Train:", len(X_train), " Test:", len(X_test))
model = LinearRegression()
model.fit(X_train, y_train)
print(f"m = {model.coef_[0]:.6f} b = {model.intercept_:.4f}")
p = model.predict(X_test)
print(f"MSE={mean_squared_error(y_test, p):.4f} "
      f"R2={r2_score(y_test, p):.4f} RMSE={np.sqrt(mean_squared_error(y_test, p)):.4f}")
plt.figure(figsize=(7, 4.5))
plt.scatter(X_test, y_test, s=6, alpha=0.25, label="Test data")
xs = np.linspace(0, 15, 200)
plt.plot(xs, model.coef_[0] * xs + model.intercept_,
         linewidth=2, label="Sklearn line")
plt.xlim(0, 15)
plt.xlabel("AveRooms")
plt.ylabel("MedHouseVal ($100k)")
plt.title("Simple Linear Regression with Scikit-learn")
plt.legend()
plt.tight_layout()
plt.savefig("exp4c_fit.png", dpi=150)
