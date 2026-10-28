import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv("diabetes.csv")
print("Dataset:", df.shape)
data = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
print("Train:", len(X_train), " Test:", len(X_test))
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
p = model.predict(X_test)
print("Without scaling:")
print(f"acc={accuracy_score(y_test, p):.4f} prec={precision_score(y_test, p):.4f} rec={recall_score(y_test, p):.4f} f1={f1_score(y_test, p):.4f}")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model_s = LogisticRegression(max_iter=1000)
model_s.fit(X_train_s, y_train)
ps = model_s.predict(X_test_s)
print("With scaling:")
print(f"acc={accuracy_score(y_test, ps):.4f} prec={precision_score(y_test, ps):.4f} rec={recall_score(y_test, ps):.4f} f1={f1_score(y_test, ps):.4f}")
diff = accuracy_score(y_test, ps) - accuracy_score(y_test, p)
print(f"Accuracy changed by {diff:+.4f} after scaling")
