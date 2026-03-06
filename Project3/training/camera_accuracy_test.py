#!/usr/bin/env python3
# ============================================================
# camera_accuracy_test.py
# Live camera accuracy benchmark for the face-detection TFLite model.
#
# Usage (from Project3/training/):
#   python camera_accuracy_test.py
#   python camera_accuracy_test.py --model face_model.tflite --camera 0
#
# How it works:
#   The webcam feed runs live with real-time model inference displayed
#   on screen.  YOU provide the ground truth by pressing a key for each
#   frame you want to score:
#
#     F  — current frame is a TRUE FACE    (label 1)
#     N  — current frame is TRUE NO FACE   (label 0)
#     S  — skip this frame (not scored)
#     Q  — quit and print the accuracy report
#
#   Only frames you explicitly label (F or N) are included in the
#   accuracy calculation, so you control the quality of the test set.
#
# Output:
#   • Live console tally every 10 labels
#   • Terminal report: accuracy, confusion matrix, precision/recall/F1
#   • Saved PNG: camera_accuracy_report.png  (same folder as this script)
# ============================================================

import os
import sys
import argparse
import time
from pathlib import Path

# ── Suppress noisy logs before importing cv2 / tf ────────────
os.environ["QT_QPA_PLATFORM"]      = "xcb"
os.environ["QT_QPA_FONTDIR"]       = "/usr/share/fonts"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"]     = "SILENT"

import cv2
import numpy as np
import tensorflow as tf

# ── Optional: richer terminal table ──────────────────────────
try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")          # headless — no second window needed
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================
# CLI arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Live camera accuracy test for face-detection TFLite model."
)
parser.add_argument(
    "--model",
    type=str,
    default="face_model.tflite",
    help="Path to the INT8 TFLite model file (default: face_model.tflite).",
)
parser.add_argument(
    "--camera",
    type=int,
    default=0,
    help="OpenCV camera index (default: 0).",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Decision threshold for face vs no-face (default: 0.5).",
)
parser.add_argument(
    "--save",
    type=str,
    default="camera_accuracy_report.png",
    help="Filename for the saved report PNG (default: camera_accuracy_report.png).",
)
args = parser.parse_args()

# ============================================================
# Load TFLite model
# ============================================================

model_path = Path(args.model)
if not model_path.exists():
    # Try one level up (running from repo root)
    model_path = Path(__file__).parent / args.model
if not model_path.exists():
    print(f"[ERROR] Model not found: {args.model}")
    print("        Run the notebook first to generate face_model.tflite,")
    print("        then run this script from Project3/training/.")
    sys.exit(1)

print(f"[INFO] Loading model: {model_path}")
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
IMG_SIZE       = input_details["shape"][1]          # 96
THRESHOLD      = args.threshold

print(f"  Input  shape : {input_details['shape']}  dtype: {input_details['dtype'].__name__}")
print(f"  Output dtype : {output_details['dtype'].__name__}")
print(f"  Decision threshold : {THRESHOLD}")

# ============================================================
# Haar cascade (used only for the optional face-crop preview)
# ============================================================

cascade_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade  = cv2.CascadeClassifier(cascade_path)

# ============================================================
# Inference helper
# ============================================================

def infer(gray_96x96: np.ndarray) -> float:
    """
    Run a single inference on a 96×96 grayscale crop.
    Handles both uint8 (INT8 model) and float32 models.
    Returns probability in [0, 1].
    """
    if input_details["dtype"] == np.uint8:
        inp = gray_96x96.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.uint8)
    else:
        inp = (gray_96x96 / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

    interpreter.set_tensor(input_details["index"], inp)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details["index"])

    sc, zp = output_details["quantization"]
    if output_details["dtype"] == np.uint8:
        prob = float(np.clip((int(raw[0][0]) - zp) * sc, 0.0, 1.0))
    else:
        prob = float(np.clip(raw[0][0], 0.0, 1.0))
    return prob

# ============================================================
# Accuracy tracker
# ============================================================

y_true: list[int] = []   # ground-truth labels provided by the user
y_pred: list[int] = []   # model predictions
y_prob: list[float] = [] # raw model probabilities

def print_tally(y_true, y_pred):
    total    = len(y_true)
    correct  = sum(t == p for t, p in zip(y_true, y_pred))
    n_face   = sum(t == 1 for t in y_true)
    n_noface = sum(t == 0 for t in y_true)
    print(f"  Labeled so far: {total}  |  face={n_face}  noface={n_noface}  "
          f"|  accuracy={correct/total*100:.1f}%")

# ============================================================
# Overlay helpers
# ============================================================

FONT  = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 220, 0)
RED   = (0, 0, 220)
GRAY  = (160, 160, 160)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def draw_hud(frame, prob, label, n_labeled, accuracy_pct):
    h, w = frame.shape[:2]

    # ── Inference result banner ─────────────────────────────
    color = GREEN if label == "FACE" else RED
    cv2.rectangle(frame, (0, 0), (w, 45), BLACK, -1)
    cv2.putText(frame, f"{label}   p={prob:.3f}", (10, 32),
                FONT, 1.0, color, 2, cv2.LINE_AA)

    # ── Score tally (bottom left) ───────────────────────────
    cv2.rectangle(frame, (0, h - 50), (w, h), BLACK, -1)
    acc_str = f"{accuracy_pct:.1f}%" if n_labeled > 0 else "---"
    cv2.putText(frame,
                f"Labeled: {n_labeled}   Live accuracy: {acc_str}",
                (10, h - 18), FONT, 0.7, WHITE, 1, cv2.LINE_AA)

    # ── Key hints (top right) ────────────────────────────────
    hints = "[F]=face  [N]=noface  [S]=skip  [Q]=quit"
    (tw, _), _ = cv2.getTextSize(hints, FONT, 0.55, 1)
    cv2.putText(frame, hints, (w - tw - 10, 32),
                FONT, 0.55, GRAY, 1, cv2.LINE_AA)

    return frame

# ============================================================
# Webcam loop
# ============================================================

print()
print("=" * 56)
print(" LIVE CAMERA ACCURACY TEST")
print("=" * 56)
print("  F  — label frame as FACE    (ground truth = 1)")
print("  N  — label frame as NO FACE (ground truth = 0)")
print("  S  — skip this frame")
print("  Q  — quit and show accuracy report")
print("=" * 56)
print()

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"[ERROR] Cannot open camera index {args.camera}.")
    sys.exit(1)

# Warm up camera
for _ in range(5):
    cap.read()

WINDOW = "Camera Accuracy Test — F=face  N=noface  S=skip  Q=quit"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 900, 680)

last_prob   = 0.0
last_label  = "NO FACE"
flash_until = 0.0    # timestamp until which to flash a feedback color
flash_color = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read camera frame.")
        break

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Always infer on the centre 96×96 crop (matches main.cpp) ──
    fh, fw = gray_full.shape
    cy, cx = fh // 2, fw // 2
    half   = IMG_SIZE // 2
    crop   = gray_full[max(0, cy-half):cy+half, max(0, cx-half):cx+half]
    crop   = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))    # ensure exact size

    prob        = infer(crop)
    last_prob   = prob
    last_label  = "FACE" if prob >= THRESHOLD else "NO FACE"

    # ── Draw center-crop box on frame ─────────────────────────
    box_color = GREEN if prob >= THRESHOLD else RED
    cv2.rectangle(frame,
                  (cx - half, cy - half),
                  (cx + half, cy + half),
                  box_color, 2)

    # ── Flash feedback (briefly tint frame after a label key) ─
    now = time.time()
    if flash_color and now < flash_until:
        overlay         = frame.copy()
        overlay[:, :] = flash_color
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # ── Accuracy so far ──────────────────────────────────────
    n_labeled    = len(y_true)
    correct      = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy_pct = (correct / n_labeled * 100) if n_labeled > 0 else 0.0

    draw_hud(frame, prob, last_label, n_labeled, accuracy_pct)

    cv2.imshow(WINDOW, frame)

    # ── Key handling ─────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break

    elif key == ord("f") or key == ord("F"):
        pred = 1 if prob >= THRESHOLD else 0
        y_true.append(1)
        y_pred.append(pred)
        y_prob.append(prob)
        flash_color = (0, 180, 0)   # green tint
        flash_until = time.time() + 0.25
        correct_str = "✓" if pred == 1 else "✗"
        print(f"  [F] frame {n_labeled+1:3d}  prob={prob:.3f}  pred={last_label}  {correct_str}")
        if (n_labeled + 1) % 10 == 0:
            print_tally(y_true, y_pred)

    elif key == ord("n") or key == ord("N"):
        pred = 1 if prob >= THRESHOLD else 0
        y_true.append(0)
        y_pred.append(pred)
        y_prob.append(prob)
        flash_color = (0, 0, 180)   # red tint
        flash_until = time.time() + 0.25
        correct_str = "✓" if pred == 0 else "✗"
        print(f"  [N] frame {n_labeled+1:3d}  prob={prob:.3f}  pred={last_label}  {correct_str}")
        if (n_labeled + 1) % 10 == 0:
            print_tally(y_true, y_pred)

    # S / space — explicit skip (no action needed, just continues)

cap.release()
cv2.destroyAllWindows()

# ============================================================
# Accuracy Report
# ============================================================

print()
print("=" * 56)
print(" ACCURACY REPORT")
print("=" * 56)

if len(y_true) == 0:
    print("[WARN] No frames were labeled — nothing to report.")
    sys.exit(0)

y_true_arr = np.array(y_true)
y_pred_arr = np.array(y_pred)
y_prob_arr = np.array(y_prob)

total   = len(y_true_arr)
correct = int((y_true_arr == y_pred_arr).sum())
acc     = correct / total

n_face   = int((y_true_arr == 1).sum())
n_noface = int((y_true_arr == 0).sum())

print(f"  Total labeled frames : {total}")
print(f"    Face    (label=1)  : {n_face}")
print(f"    No Face (label=0)  : {n_noface}")
print(f"  Correct predictions  : {correct}")
print(f"  Overall accuracy     : {acc*100:.2f}%")
print()

if HAS_SKLEARN:
    print(classification_report(
        y_true_arr, y_pred_arr,
        target_names=["no face", "face"],
        zero_division=0,
    ))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"            pred:noface  pred:face")
    print(f"  act:noface   {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  act:face     {cm[1,0]:5d}       {cm[1,1]:5d}")
else:
    # Manual confusion matrix
    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    print(f"  TP (face  → face)    : {tp}")
    print(f"  TN (noface→ noface)  : {tn}")
    print(f"  FP (noface→ face)    : {fp}")
    print(f"  FN (face  → noface)  : {fn}")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"  Precision            : {prec:.3f}")
    print(f"  Recall               : {rec:.3f}")
    print(f"  F1                   : {f1:.3f}")

print("=" * 56)

# ============================================================
# Save PNG report (requires matplotlib)
# ============================================================

if HAS_MPL and HAS_SKLEARN:
    save_path = Path(__file__).parent / args.save

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Camera Accuracy Report — {total} frames  |  Accuracy: {acc*100:.1f}%",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: confusion matrix ──────────────────────────
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["no face", "face"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title("Confusion Matrix")

    # ── Panel 2: precision / recall / F1 per class ─────────
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=[0, 1], zero_division=0
    )
    x, w = np.arange(2), 0.22
    axes[1].bar(x - w,   prec, w, label="Precision",  color="steelblue",  edgecolor="white")
    axes[1].bar(x,       rec,  w, label="Recall",     color="darkorange", edgecolor="white")
    axes[1].bar(x + w,   f1,   w, label="F1",         color="#55A868",    edgecolor="white")
    axes[1].axhline(0.80, color="black", ls="--", lw=1.2, label="80% target")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["no face", "face"])
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Precision / Recall / F1")
    axes[1].legend(fontsize=8)
    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        axes[1].text(i - w, p + 0.02, f"{p:.2f}", ha="center", fontsize=7)
        axes[1].text(i,     r + 0.02, f"{r:.2f}", ha="center", fontsize=7)
        axes[1].text(i + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=7)

    # ── Panel 3: confidence histogram ─────────────────────
    face_p   = y_prob_arr[y_true_arr == 1]
    noface_p = y_prob_arr[y_true_arr == 0]
    if len(face_p) > 0:
        axes[2].hist(face_p,   bins=15, alpha=0.7, color="steelblue",
                     label="Actual: face",    edgecolor="white")
    if len(noface_p) > 0:
        axes[2].hist(noface_p, bins=15, alpha=0.7, color="#C44E52",
                     label="Actual: no face", edgecolor="white")
    axes[2].axvline(THRESHOLD, color="black", ls="--", lw=1.5,
                    label=f"Threshold ({THRESHOLD})")
    axes[2].set_xlabel("P(face)")
    axes[2].set_ylabel("Frame count")
    axes[2].set_title("Confidence Distribution")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[SAVED] Report PNG → {save_path}")
elif not HAS_MPL:
    print("[INFO] matplotlib not found — PNG report skipped.")
