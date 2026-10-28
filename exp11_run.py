import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("winequality-red.csv", sep=";")
print("Dataset:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicates:", int(df.duplicated().sum()))
X = df.drop(columns=["quality"])
y = df["quality"] - 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", len(X_train), " Test:", len(X_test))
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype("float32")
X_test_s = scaler.transform(X_test).astype("float32")
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
model = keras.Sequential([
    layers.Input(shape=(11,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(6, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train_s, y_train, epochs=150, batch_size=100, validation_data=(X_test_s, y_test), verbose=0)
loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"Train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {acc:.4f}")
plt.figure(figsize=(7, 4.5))
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP on Wine Quality: Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("exp11_acc.png", dpi=150)
