import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
ZP = "online_retail.zip"
if not os.path.exists(ZP):
    print("Downloading Online Retail")
    urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/352/online+retail.zip", ZP)
with zipfile.ZipFile(ZP) as z:
    name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
    df = pd.read_excel(z.open(name), engine="openpyxl")
print("Rows:", len(df))
df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
print("Clean rows:", len(df))
df["Total"] = df["Quantity"] * df["UnitPrice"]
maxd = df["InvoiceDate"].max()
g = df.groupby("CustomerID")
cust = pd.DataFrame({
    "Recency": (maxd - g["InvoiceDate"].max()).dt.days,
    "Frequency": g["InvoiceNo"].nunique(),
    "Monetary": g["Total"].sum(),
    "AvgBasket": g["Total"].mean(),
    "Items": g["StockCode"].nunique(),
})
print("Customers:", len(cust))
m = cust["Monetary"].values
y = np.where(m <= np.percentile(m, 33.3), 0, np.where(m <= np.percentile(m, 66.7), 1, 2))
X = cust[["Recency", "Frequency", "AvgBasket", "Items"]]
print("Classes:", np.bincount(y).tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", len(X_train), " Test:", len(X_test))
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)
p = model.predict(X_test)
print(f"DecisionTree: acc={accuracy_score(y_test, p):.4f} depth={model.get_depth()} leaves={model.get_n_leaves()}")
print("Importance:", " ".join(f"{f}={v:.3f}" for f, v in zip(X.columns, model.feature_importances_)))
plt.figure(figsize=(12, 7))
plot_tree(model, feature_names=list(X.columns), class_names=["Low", "Mid", "High"], filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (entropy) for Customer Segments")
plt.tight_layout()
plt.savefig("exp6_tree.png", dpi=150)
plt.figure(figsize=(7, 4))
plt.barh(list(X.columns), model.feature_importances_)
plt.xlabel("Importance")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.savefig("exp6_imp.png", dpi=150)
