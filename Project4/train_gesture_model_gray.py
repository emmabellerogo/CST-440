"""
=============================================================================
Hand Gesture Detection – Training Script  (v4 — Crop-First Pipeline)
Target Hardware : Arduino Nano BLE 33
Gestures        : thumbs_up (like), thumbs_down (dislike), rock_on
=============================================================================
WHY THIS BEATS v3 (62%):

The YOLOX notebook achieved high accuracy because it NEVER showed the model
a full background image. YOLOX first localises the hand with a bounding box,
then classifies only what's inside that box.

Your v2/v3 fed the entire 64x64 frame to the classifier. At that resolution,
the hand might occupy only a fraction of the frame — the rest is desk, shirt,
and background noise the model had to ignore.

THIS SCRIPT replicates the crop-first approach:
  1. Load HaGRID images WITH their JSON bounding box annotations
  2. Crop each image to the hand bounding box (+ small padding)
  3. Resize the CROP (not the full frame) to 64x64
  4. Train MobileNetV2 on these clean hand crops

Result: the model sees almost exclusively hand pixels, matching what the
YOLOX notebook's classifier saw. Background noise is eliminated at the
data level rather than hoping the model learns to ignore it.

FOR THE CAMERA (camera_test.py):
  MediaPipe detects the hand bounding box in real time → crop → classify.
  This mirrors the YOLOX detect-then-classify pipeline exactly.

=============================================================================
FOLDER LAYOUT EXPECTED:
  hagrid_120k/
    train_val_like/         *.jpg
    train_val_dislike/      *.jpg
    train_val_rock/         *.jpg
  ann_train_val/
    like.json
    dislike.json
    rock.json

  train_val_background/     *.jpg   (webcam captures, no hand)

Set HAGRID_ROOT below to point to your hagrid_120k parent folder.
=============================================================================
"""

import json
import pathlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, applications, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import cv2

# ---------------------------------------------------------------------------
# 1. GPU
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs available: {gpus}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------
# Your disk layout (from terminal path shown in error output):
#
#   hagrid-sample-120k-384p/
#     ann_train_val/          ← annotation JSONs live here
#       like.json
#       dislike.json
#       rock.json
#     hagrid_120k/            ← you run the script from here
#       train_val_like/
#       train_val_dislike/
#       train_val_rock/
#       train_val_background/
#       train_gesture_model.py   ← THIS FILE
#
# HAGRID_ROOT points one level UP from where this script sits so it can
# reach both hagrid_120k/ (images) and ann_train_val/ (annotations).

HAGRID_ROOT  = pathlib.Path(__file__).resolve().parent.parent
# Images live inside hagrid_120k/ under HAGRID_ROOT
IMG_ROOT     = HAGRID_ROOT / "hagrid_120k"

IMG_SIZE     = 96        # matches Arduino OV7670 capture resolution
CHANNELS     = 1        # grayscale -- matches Arduino camera output
INPUT_SHAPE  = (IMG_SIZE, IMG_SIZE, CHANNELS)
BATCH_SIZE   = 32
EPOCHS_HEAD  = 20
EPOCHS_TUNE  = 30
LR_HEAD      = 1e-3
LR_TUNE      = 1e-5
DROPOUT      = 0.4
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
SEED         = 42
BBOX_PADDING = 0.10

CLASS_NAMES  = ["thumbs_up", "thumbs_down", "rock_on", "background"]
NUM_CLASSES  = len(CLASS_NAMES)

# Maps class name → (image subfolder under IMG_ROOT, annotation JSON under HAGRID_ROOT)
GESTURE_CONFIGS = {
    "thumbs_up":   ("train_val_like",         "ann_train_val/like.json"),
    "thumbs_down": ("train_val_dislike",       "ann_train_val/dislike.json"),
    "rock_on":     ("train_val_rock",          "ann_train_val/rock.json"),
    "background":  ("train_val_background",    None),
}

print(f"\n  HAGRID_ROOT : {HAGRID_ROOT}")
print(f"  IMG_ROOT    : {IMG_ROOT}")
# Sanity-check paths so you see immediately if something is wrong
for cls, (img_sub, ann_sub) in GESTURE_CONFIGS.items():
    img_p = IMG_ROOT / img_sub
    ann_p = (HAGRID_ROOT / ann_sub) if ann_sub else None
    img_ok = "OK" if img_p.exists() else "MISSING"
    ann_ok = ("OK" if ann_p.exists() else "MISSING") if ann_p else "n/a"
    print(f"  {cls:>12}  images={img_ok} ({img_p.name})  ann={ann_ok}")

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# 3. Crop-based data loader
#    Key insight: crop to hand bounding box BEFORE resizing.
#    This is what gives the YOLOX notebook its accuracy advantage.
# ---------------------------------------------------------------------------

def load_annotated_crops(class_name, img_folder, ann_json, img_size, padding):
    """
    Load hand crops using HaGRID bounding box annotations.
    Each annotation entry has: bboxes [[x, y, w, h] normalised], labels [...]
    We only keep boxes whose label matches the gesture class.
    """
    folder  = IMG_ROOT / img_folder
    ann_map = {}   # image_id -> list of [x,y,w,h] in normalised coords

    if ann_json is not None:
        ann_path = HAGRID_ROOT / ann_json
        if ann_path.exists():
            with open(ann_path) as f:
                ann_data = json.load(f)
            # Map short gesture name (like/dislike/rock) from filename
            gesture_key = pathlib.Path(ann_json).stem   # e.g. "like"
            for img_id, record in ann_data.items():
                boxes = []
                for bbox, lbl in zip(record["bboxes"], record["labels"]):
                    # Keep boxes labelled as the main gesture
                    # (skip "no_gesture" secondary annotations)
                    if lbl != "no_gesture":
                        boxes.append(bbox)
                if boxes:
                    ann_map[img_id] = boxes
        else:
            print(f"  WARNING: Annotation file not found: {ann_path}")

    images = (list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png")))

    crops  = []
    for img_path in images:
        img_id = img_path.stem
        img    = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale — matches Arduino
        h, w = img.shape[:2]                          # shape is (H, W) now

        if ann_json is None:
            # Background: no crop, just resize full frame
            crop = cv2.resize(img, (img_size, img_size))
            crop = np.expand_dims(crop, axis=-1)      # (H,W) → (H,W,1)
            crops.append(crop)
        elif img_id in ann_map:
            for bbox in ann_map[img_id]:
                # HaGRID bbox format: [x_center_norm, y_center_norm, w_norm, h_norm]
                # Actually stored as [x_topleft_norm, y_topleft_norm, w_norm, h_norm]
                bx, by, bw, bh = bbox
                # Convert normalised → pixel
                x1 = int(bx * w)
                y1 = int(by * h)
                x2 = int((bx + bw) * w)
                y2 = int((by + bh) * h)
                # Add padding
                pad_x = int((x2 - x1) * padding)
                pad_y = int((y2 - y1) * padding)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (img_size, img_size))
                crop = np.expand_dims(crop, axis=-1)  # (H,W) → (H,W,1)
                crops.append(crop)

    print(f"  [{class_name:>12}]  {len(crops):>5} crops")
    return crops


def load_all_data():
    X, y = [], []
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for class_name, (img_folder, ann_json) in GESTURE_CONFIGS.items():
        crops = load_annotated_crops(
            class_name, img_folder, ann_json, IMG_SIZE, BBOX_PADDING
        )
        for crop in crops:
            X.append(crop.astype(np.float32) / 255.0)
            y.append(label_map[class_name])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


print("\n[1/8] Loading dataset with bounding-box crops...")
X, y = load_all_data()
print(f"\n  Total : {len(X)} crops  |  Shape : {X.shape}")

print("\n  Label distribution:")
for i, name in enumerate(CLASS_NAMES):
    count = int(np.sum(y == i))
    bar   = "=" * (count // 200)
    print(f"    {i} {name:>12} : {count:>5}  {bar}")

if len(np.unique(y)) < NUM_CLASSES:
    print("\n  ERROR: Not all classes found. Check HAGRID_ROOT and folder names.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 4. Split
# ---------------------------------------------------------------------------
print("\n[2/8] Splitting (70 / 15 / 15)...")
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv,
    test_size=VAL_SPLIT / (1 - TEST_SPLIT),
    stratify=y_tv, random_state=SEED
)
print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

# ---------------------------------------------------------------------------
# 5. Augmentation — label-safe only
#    Rotation is NOW safe because we're looking at a cropped hand, not a
#    full frame where thumb direction encodes the class. A cropped thumbs_up
#    rotated slightly is still recognisably a thumbs_up.
#    Keep rotation small (±15°) so it doesn't become ambiguous.
# ---------------------------------------------------------------------------

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    # Zoom via random crop + resize
    crop_size = tf.random.uniform([], int(IMG_SIZE * 0.85), IMG_SIZE, dtype=tf.int32)
    offset_h  = tf.random.uniform([], 0, IMG_SIZE - crop_size + 1, dtype=tf.int32)
    offset_w  = tf.random.uniform([], 0, IMG_SIZE - crop_size + 1, dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_size, crop_size)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # Brightness and contrast work on grayscale; saturation does NOT (needs 3ch)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train), seed=SEED)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ---------------------------------------------------------------------------
# 6. Model — Custom CNN for grayscale input
#
# MobileNetV2 requires exactly 3 input channels and ImageNet RGB weights.
# Since we are training on grayscale (1 channel) to match the Arduino, we
# use a custom Tiny CNN instead.  It is still deeper than v2 and uses
# GlobalAveragePooling (not Flatten) per ILab Q1 requirements.
#
# The crop-first pipeline (hand region only) means the model sees clean
# hand pixels, so a well-designed small CNN can achieve high accuracy
# without needing a pretrained RGB backbone.
# ---------------------------------------------------------------------------

def build_model():
    inp = tf.keras.Input(shape=INPUT_SHAPE, name="image_input")  # (96,96,1)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)          # → 48x48

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)          # → 24x24

    # Block 3 — depthwise separable (MobileNet-style, fewer multiply-adds)
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 1, use_bias=False)(x)   # pointwise
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)          # → 12x12

    # Block 4 — depthwise separable
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Head — GlobalAveragePooling NOT Flatten (ILab Q1)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dropout(DROPOUT)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inp, out, name="GestureNet_Grayscale96")
    # No separate base_model since this is not transfer learning
    return model, None


print("\n[3/8] Building model...")
model, base_model = build_model()
model.summary()

# ---------------------------------------------------------------------------
# 7. Phase 1 — frozen base, train head
# ---------------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_HEAD),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

ckpt_cb = callbacks.ModelCheckpoint(
    "best_gesture_model.keras", monitor="val_accuracy",
    save_best_only=True, verbose=1
)

print("\n[4/8] Training...")
h1 = model.fit(
    train_ds, epochs=EPOCHS_HEAD + EPOCHS_TUNE, validation_data=val_ds,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-6, verbose=1),
        ckpt_cb,
    ]
)
print(f"  Best val_acc: {max(h1.history['val_accuracy'])*100:.2f}%")

# Phase 2 block removed — custom CNN trains in one pass (no frozen base to unfreeze)
h2 = h1   # alias so the curves section below still works unchanged

# ---------------------------------------------------------------------------
# 9. Curves
# ---------------------------------------------------------------------------
acc  = h1.history["accuracy"]     + h2.history["accuracy"]
val  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"]         + h2.history["loss"]
vloss= h1.history["val_loss"]     + h2.history["val_loss"]
pb   = len(h1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(acc, label="Train"); axes[0].plot(val, label="Val")
axes[0].axvline(pb, color="gray", linestyle=":", label="Unfreeze")
axes[0].axhline(0.80, color="red", linestyle="--", label="80% target")
axes[0].set_title("Accuracy"); axes[0].legend()
axes[1].plot(loss, label="Train"); axes[1].plot(vloss, label="Val")
axes[1].axvline(pb, color="gray", linestyle=":")
axes[1].set_title("Loss"); axes[1].legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.show()

# ---------------------------------------------------------------------------
# 10. Evaluation
# ---------------------------------------------------------------------------
print("\n[6/8] Evaluating...")
loss_val, acc_val = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test accuracy : {acc_val*100:.2f}%  "
      f"({'PASS' if acc_val >= 0.80 else 'FAIL'})")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set(xticks=np.arange(NUM_CLASSES), yticks=np.arange(NUM_CLASSES),
       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
       title="Confusion Matrix", ylabel="True", xlabel="Predicted")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
thresh = cm.max() / 2
for i, j in itertools.product(range(NUM_CLASSES), range(NUM_CLASSES)):
    ax.text(j, i, cm[i, j], ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.show()

# ---------------------------------------------------------------------------
# 11. TFLite export
# ---------------------------------------------------------------------------
print("\n[7/8] Converting to TFLite...")

c32 = tf.lite.TFLiteConverter.from_keras_model(model)
t32 = c32.convert()
with open("gesture_model_f32.tflite", "wb") as f: f.write(t32)
print(f"  f32 : {len(t32)/1024:.1f} KB")

def rep_ds():
    for i in np.random.choice(len(X_train), min(300, len(X_train)), replace=False):
        yield [X_train[i:i+1].astype(np.float32)]

c8 = tf.lite.TFLiteConverter.from_keras_model(model)
c8.optimizations             = [tf.lite.Optimize.DEFAULT]
c8.representative_dataset    = rep_ds
c8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c8.inference_input_type      = tf.int8
c8.inference_output_type     = tf.int8
t8 = c8.convert()
with open("gesture_model_int8.tflite", "wb") as f: f.write(t8)
print(f"  int8: {len(t8)/1024:.1f} KB")

# INT8 accuracy check
interp = tf.lite.Interpreter(model_content=t8)
interp.allocate_tensors()
id_ = interp.get_input_details()[0]; od_ = interp.get_output_details()[0]
sc, zp = id_["quantization"]
correct = sum(
    1 for i in range(len(X_test))
    if (interp.set_tensor(id_["index"], (X_test[i:i+1]/sc+zp).astype(np.int8)),
        interp.invoke(),
        np.argmax(interp.get_tensor(od_["index"])) == y_test[i])[2]
)
q_acc = correct / len(X_test)
print(f"  INT8 accuracy : {q_acc*100:.2f}%  (drop: {(acc_val-q_acc)*100:.2f}%)")
print(f"  Model flash   : {len(t8)/1024:.1f} KB  (limit ~1024 KB)")

# ---------------------------------------------------------------------------
# 12. C header
# ---------------------------------------------------------------------------
print("\n[8/8] Writing C header...")
hex_vals = ", ".join(f"0x{b:02x}" for b in t8)
classes_str = ", ".join(f"{n}({i})" for i, n in enumerate(CLASS_NAMES))
with open("gesture_model.h", "w") as f:
    f.write(f"""\
// Auto-generated — Arduino Nano BLE 33 / TFLite Micro
// IMPORTANT: Camera must detect hand, crop, then resize to {IMG_SIZE}x{IMG_SIZE} RGB
//            before running inference — same as training pipeline.
// Gestures : {classes_str}
// Input    : {IMG_SIZE}x{IMG_SIZE} RGB INT8
// Size     : {len(t8)} bytes ({len(t8)/1024:.1f} KB)

#ifndef GESTURE_MODEL_H
#define GESTURE_MODEL_H
#include <stdint.h>
const unsigned char gesture_model[] = {{
  {hex_vals}
}};
const unsigned int gesture_model_len = {len(t8)};
#endif
""")
print("  Saved: gesture_model.h")

print(f"""
Done! Test accuracy: {acc_val*100:.2f}%

KEY: The camera_test.py script MUST use MediaPipe to detect the hand
bounding box and crop before running inference — otherwise accuracy
will drop back to ~62% because of background noise.
""")
