# ==============================
# Face / No-Face Webcam Tester
# ==============================
# Matches the preprocessing from train.py:
#   - Grayscale
#   - Resize to 96x96
#   - Input dtype: float32, normalized to [0, 1]
#
# Uses OpenCV Haar cascade to crop detected face regions first,
# so the model input matches the tight face crops it was trained on.

import os
os.environ["QT_QPA_PLATFORM"]      = "xcb"            # force X11 (fixes Wayland plugin error)
os.environ["QT_QPA_FONTDIR"]       = "/usr/share/fonts" # point Qt to system fonts
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"               # suppress TF info/warning logs
os.environ["OPENCV_LOG_LEVEL"]     = "SILENT"           # suppress OpenCV logs

import cv2
import numpy as np
import tensorflow as tf
import serial
import serial.tools.list_ports
import argparse

# ------------------------------
# Args
# ------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default=None,
                    help="Serial port of the Nano (e.g. /dev/ttyACM0). "
                         "Omit to run Python-only mode.")
parser.add_argument("--baud", type=int, default=115200)
args = parser.parse_args()

# ------------------------------
# Config
# ------------------------------

MODEL_PATH = "face_model.tflite"  # INT8 quantized — same model used on Arduino
IMG_SIZE = 96
THRESHOLD = 0.5      # above = face, below = no face

# ------------------------------
# Load TFLite Model
# ------------------------------

print("[INFO] Loading model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale, in_zero_point   = input_details[0]["quantization"]
out_scale, out_zero_point = output_details[0]["quantization"]

print(f"  Input  shape : {input_details[0]['shape']}")
print(f"  Input  dtype : {input_details[0]['dtype']}")
print(f"  Output dtype : {output_details[0]['dtype']}")

# ------------------------------
# Optional: Nano serial connection
# ------------------------------

nano = None
if args.port:
    try:
        nano = serial.Serial(args.port, args.baud, timeout=5)
        print(f"[INFO] Connected to Nano on {args.port} @ {args.baud} baud")
    except Exception as e:
        print(f"[WARN] Could not open serial port: {e}. Running Python-only mode.")
else:
    ports = [p.device for p in serial.tools.list_ports.comports()]
    print(f"[INFO] No --port given. Python-only mode. Available ports: {ports or 'none'}")

def send_to_nano(crop_uint8: np.ndarray) -> str | None:
    """Send a 96x96 uint8 crop to the Nano and return its prediction string."""
    if nano is None:
        return None
    try:
        # Wait for READY signal
        line = nano.readline().decode(errors="ignore").strip()
        if line != "READY":
            return None
        # Send header + raw bytes
        nano.write(b"IMG:")
        nano.write(crop_uint8.tobytes())
        nano.flush()
        # Read prediction
        result = nano.readline().decode(errors="ignore").strip()
        return result
    except Exception as e:
        print(f"[WARN] Serial error: {e}")
        return None

# ------------------------------
# Haar Cascade (finds face regions to crop)
# ------------------------------

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
print(f"[INFO] Loaded Haar cascade from {cascade_path}")

# ------------------------------
# Webcam Loop
# ------------------------------

WINDOW = "Face Detector (TFLite) — press Q to quit"

print("[INFO] Starting webcam. Press 'q' to quit.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit(1)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 800, 600)

frame_num = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Detect face regions with Haar cascade ──────────────────
    faces = face_cascade.detectMultiScale(
        gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        # No face found by Haar — label the full frame
        cv2.putText(frame, "NO FACE DETECTED  0.0%",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            # Crop the detected face region
            face_crop = gray_full[y:y+h, x:x+w]

            # ── Preprocess crop (mirrors train.py) ─────────────
            resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            # INT8 model: in_scale≈1/255, in_zero_point=0
            # so quantized = pixel/255 / scale = pixel → pass raw uint8 directly
            inp = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.uint8)

            # ── Inference ──────────────────────────────────────
            interpreter.set_tensor(input_details[0]["index"], inp)
            interpreter.invoke()
            raw_out = interpreter.get_tensor(output_details[0]["index"])
            # Dequantize: probability = (raw_uint8 - zero_point) * scale
            out_scale, out_zero_point = output_details[0]["quantization"]
            probability = float(np.clip((int(raw_out[0][0]) - out_zero_point) * out_scale, 0.0, 1.0))

            frame_num += 1
            if frame_num <= 10:
                print(f"[DEBUG] frame={frame_num}  prob={probability:.4f}")

            label  = "FACE"     if probability >= THRESHOLD else "NO FACE"
            color  = (0, 255, 0) if probability >= THRESHOLD else (0, 0, 255)
            conf_pct = probability * 100

            # Draw bounding box around detected region
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Show 96x96 model input as thumbnail inside the box
            preview = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            px, py = x, max(0, y - 100)
            frame[py:py+96, px:px+96] = cv2.resize(preview, (96, 96))

            # Label above bounding box
            cv2.putText(frame, f"{label}  {conf_pct:.1f}%",
                        (x, y - 10 if y > 30 else y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow(WINDOW, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
