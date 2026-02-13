# ==============================
# Face / No-Face Trainer (TinyML)
# ==============================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ------------------------------
# Config
# ------------------------------

IMG_SIZE = 96
DATASET_DIR = "dataset"

EPOCHS = 20
BATCH_SIZE = 32

# ------------------------------
# Load Dataset
# ------------------------------

def load_images(folder, label):
    data = []

    path = os.path.join(DATASET_DIR, folder)

    for file in os.listdir(path):

        img_path = os.path.join(path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        # Normalize
        norm = resized / 255.0

        data.append((norm, label))

    return data


print("[INFO] Loading dataset...")

face = load_images("face", 1)
noface = load_images("noface", 0)

dataset = face + noface

np.random.shuffle(dataset)

X = np.array([x[0] for x in dataset]).reshape(-1,96,96,1)
y = np.array([x[1] for x in dataset])

print("Images:", len(X))


# ------------------------------
# Train / Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


# ------------------------------
# Model
# ------------------------------

print("[INFO] Building model...")

model = models.Sequential([

    layers.Input(shape=(96,96,1)),

    layers.Conv2D(8, 3, activation="relu"),
    layers.MaxPool2D(),

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPool2D(),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),

    layers.Flatten(),

    layers.Dense(32, activation="relu"),

    layers.Dense(1, activation="sigmoid")

])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ------------------------------
# Train
# ------------------------------

print("[INFO] Training...")

model.fit(
    X_train, y_train,
    validation_data=(X_test,y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# ------------------------------
# Evaluate
# ------------------------------

loss, acc = model.evaluate(X_test,y_test)

print("Accuracy:", acc)


# ------------------------------
# Quantize for Arduino
# ------------------------------

print("[INFO] Converting to TFLite...")

def rep_data():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = rep_data

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8


tflite_model = converter.convert()


# ------------------------------
# Save
# ------------------------------

with open("face_model.tflite","wb") as f:
    f.write(tflite_model)


print("[DONE] face_model.tflite created")


# ------------------------------
# Convert to Header
# ------------------------------

print("[INFO] Creating header...")

os.system("xxd -i face_model.tflite > face_model.h")

print("[DONE] face_model.h ready")
