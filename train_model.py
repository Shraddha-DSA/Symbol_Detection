import os
import cv2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout,
    RandomRotation, RandomZoom, RandomTranslation, Input
)
from keras.callbacks import EarlyStopping


# ================== CONFIG ==================
BASE_DIR = r"C:\Users\Shraddha\OneDrive\Desktop\symbol_detection\r2_boxes"
CSV_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "real_fake_model.h5")

IMG_SIZE = 64
BATCH_SIZE = 4        # small batch for small dataset
EPOCHS = 25

os.makedirs(MODEL_DIR, exist_ok=True)


# ================== LOAD DATA ==================
print("📥 Loading dataset...")

df = pd.read_csv(CSV_PATH)

X = []
y = []

for _, row in df.iterrows():

    relative_path = row["image_path"].replace("\\", "/")
    img_path = os.path.join(BASE_DIR, relative_path)

    label = int(row["remark"])  # 1 = REAL, 0 = FAKE

    if not os.path.exists(img_path):
        print(f"⚠ Missing image: {img_path}")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠ Cannot read image: {img_path}")
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    X.append(img)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

print("✅ Dataset loaded")
print("Images shape:", X.shape)
print("Labels shape:", y.shape)
print("REAL:", np.sum(y == 1), "FAKE:", np.sum(y == 0))


# ================== TRAIN / TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================== CLASS WEIGHTS ==================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)


# ================== DATA AUGMENTATION ==================
data_augmentation = tf.keras.Sequential(
    [
        RandomRotation(0.05),
        RandomZoom(0.1),
        RandomTranslation(0.1, 0.1),
    ],
    name="data_augmentation"
)


# ================== MODEL ==================
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    data_augmentation,

    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.4),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()


# ================== EARLY STOPPING ==================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ================== TRAIN ==================
print("🚀 Training started...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    class_weight=class_weights
)


# ================== EVALUATION ==================
print("\n📊 Evaluating model...")

y_pred = (model.predict(X_test).ravel() > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ================== SAVE MODEL ==================
model.save(MODEL_PATH)
print(f"\n✅ Model saved at: {MODEL_PATH}")
