"""
=============================================================================
camera_test.py  — MediaPipe + MobileNetV2 Gesture Detection  (v3)
=============================================================================
Mirrors the YOLOX notebook's detect-then-classify pipeline:
  1. MediaPipe finds the hand and gives us a bounding box
  2. We crop to that box (+ padding)
  3. We resize the crop to 64x64 and run the classifier

This is why the YOLOX notebook got high accuracy — the classifier only
ever sees hand pixels, never background. We replicate that here without
needing to train a detector ourselves.

Install:
  pip install mediapipe opencv-python tensorflow numpy

Run:
  python camera_test.py

Controls:
  Q — quit
  S — save screenshot
=============================================================================
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque

# ---------------------------------------------------------------------------
# Config — must match training v4
# ---------------------------------------------------------------------------
MODEL_PATH    = "gesture_model_f32.tflite"
IMG_SIZE      = 64
CLASS_NAMES   = ["thumbs_up", "thumbs_down", "rock_on", "background"]
CONF_THRESH   = 0.80
COOLDOWN_SEC  = 1.5
SMOOTH_FRAMES = 8       # temporal smoothing window
BBOX_PADDING  = 0.15    # expand MediaPipe bbox before cropping

COLOURS = {
    "thumbs_up":   (0,   200,  50),
    "thumbs_down": (0,    50, 220),
    "rock_on":     (220, 150,   0),
    "background":  (120, 120, 120),
}

# ---------------------------------------------------------------------------
# Load TFLite model
# ---------------------------------------------------------------------------
print(f"[INFO] Loading model: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_d = interpreter.get_input_details()[0]
out_d = interpreter.get_output_details()[0]
print(f"  Input  : {inp_d['shape']}  dtype={inp_d['dtype'].__name__}")
print(f"  Output : {out_d['shape']}")

# ---------------------------------------------------------------------------
# MediaPipe hand detector — compatible with MediaPipe 0.10+
# The old mp.solutions.hands API was removed in 0.10.
# We use the new vision Tasks API instead.
# ---------------------------------------------------------------------------
print("[INFO] Initialising MediaPipe hands...")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import urllib.request
import os

# Download the hand landmark model if not already present
MODEL_FILE = "hand_landmarker.task"
if not os.path.exists(MODEL_FILE):
    print("  Downloading hand_landmarker.task (~8 MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_FILE)
    print("  Downloaded.")

# Build options for live camera (VIDEO mode processes frames sequentially)
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
hand_detector = HandLandmarker.create_from_options(options)
print("  MediaPipe ready (Tasks API).")

# Connection pairs for drawing the hand skeleton manually
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (5,9),(9,10),(10,11),(11,12),   # middle
    (9,13),(13,14),(14,15),(15,16), # ring
    (13,17),(17,18),(18,19),(19,20),# pinky
    (0,17),                         # palm base
]


def draw_hand_skeleton(frame, landmarks_list, h, w):
    """Draw skeleton using pixel coordinates from normalised landmarks."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 180, 80), 1)


def get_hand_crop(frame_rgb, landmarks_list):
    """
    Convert landmark list to bounding box, add padding, return crop + pixel bbox.
    landmarks_list is a list of NormalizedLandmark objects from the Tasks API.
    """
    h, w = frame_rgb.shape[:2]
    xs = [lm.x for lm in landmarks_list]
    ys = [lm.y for lm in landmarks_list]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_x = (x_max - x_min) * BBOX_PADDING
    pad_y = (y_max - y_min) * BBOX_PADDING
    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(1.0, x_max + pad_x)
    y_max = min(1.0, y_max + pad_y)

    px1 = int(x_min * w); py1 = int(y_min * h)
    px2 = int(x_max * w); py2 = int(y_max * h)

    if px2 <= px1 or py2 <= py1:
        return None, None

    crop = frame_rgb[py1:py2, px1:px2]
    return crop, (px1, py1, px2, py2)


def preprocess_crop(crop):
    """Resize crop to 64x64 RGB float32 — matches training pipeline exactly."""
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    normed  = resized.astype(np.float32) / 255.0
    return normed.reshape(1, IMG_SIZE, IMG_SIZE, 3)


def run_inference(crop):
    inp = preprocess_crop(crop)
    interpreter.set_tensor(inp_d["index"], inp)
    interpreter.invoke()
    return interpreter.get_tensor(out_d["index"])[0].copy()


def draw_overlay(frame, smoothed_probs, bbox, state, last_detection):
    h, w = frame.shape[:2]
    label  = CLASS_NAMES[int(np.argmax(smoothed_probs))]
    conf   = float(np.max(smoothed_probs))
    colour = COLOURS.get(label, (255, 255, 255))
    active = label != "background" and conf >= CONF_THRESH

    # Draw hand bounding box if we have one
    if bbox is not None:
        px1, py1, px2, py2 = bbox
        box_colour = colour if active else (150, 150, 150)
        cv2.rectangle(frame, (px1, py1), (px2, py2), box_colour, 2)
        # Show crop preview in top-left corner
        crop_h = py2 - py1; crop_w = px2 - px1
        if crop_h > 0 and crop_w > 0:
            thumb = cv2.resize(frame[py1:py2, px1:px2], (80, 80))
            frame[70:150, 5:85] = thumb
            cv2.rectangle(frame, (5, 70), (85, 150), (200,200,200), 1)
            cv2.putText(frame, "model sees", (5, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160,160,160), 1)

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 65), (25, 25, 25), -1)

    if state == "COOLDOWN":
        rem = max(0.0, COOLDOWN_SEC - (time.time() - last_detection))
        cv2.putText(frame, f"COOLDOWN  {rem:.1f}s", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
    elif bbox is None:
        cv2.putText(frame, "No hand detected", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    elif not active:
        cv2.putText(frame, f"Uncertain ({conf*100:.0f}%)", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 160, 160), 2)
    else:
        text = f"{label.replace('_', ' ').upper()}  {conf*100:.0f}%"
        cv2.putText(frame, text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

    # State badge
    sc = {"WAITING": (70,70,70), "DETECTING": (0,170,0), "COOLDOWN": (0,130,240)}
    cv2.rectangle(frame, (w-145, 8), (w-5, 58), sc.get(state,(70,70,70)), -1)
    cv2.putText(frame, state, (w-138, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # Confidence bars
    bar_h, bar_gap = 22, 5
    panel_h = (bar_h + bar_gap) * len(CLASS_NAMES) + bar_gap
    py = h - panel_h - 5
    cv2.rectangle(frame, (0, py-5), (285, h), (18,18,18), -1)
    for i, (cname, prob) in enumerate(zip(CLASS_NAMES, smoothed_probs)):
        y   = py + i * (bar_h + bar_gap) + bar_gap
        col = COLOURS.get(cname, (160,160,160))
        cv2.rectangle(frame, (5, y), (205, y+bar_h), (55,55,55), -1)
        cv2.rectangle(frame, (5, y), (5+int(200*prob), y+bar_h), col, -1)
        cv2.putText(frame, f"{cname[:11]} {prob*100:.0f}%",
                    (210, y+bar_h-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (210,210,210), 1)

    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def open_camera():
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[INFO] Camera opened at index {idx}")
            return cap
    return None


def main():
    cap = open_camera()
    if cap is None:
        print("[ERROR] No camera. Check System Settings > Privacy > Camera.")
        return

    print(f"\n[INFO] Q = quit | S = screenshot")
    print(f"[INFO] Smoothing: {SMOOTH_FRAMES}f | Threshold: {CONF_THRESH*100:.0f}%\n")
    print("[INFO] Show your hand to the camera to begin.\n")

    prob_buffer    = deque(maxlen=SMOOTH_FRAMES)
    background_vec = np.array([0, 0, 0, 1], dtype=np.float32)

    state          = "WAITING"
    last_detection = 0.0
    frame_count    = 0
    fail_count     = 0
    t_start        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 10:
                print("[WARN] Camera lost — reconnecting...")
                cap.release(); time.sleep(1.0); cap = open_camera()
                fail_count = 0
                if cap is None: break
            continue

        fail_count  = 0
        frame_count += 1
        now         = time.time()

        # --- MediaPipe hand detection (Tasks API) ---
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tasks API requires a timestamp in milliseconds
        timestamp_ms = int(now * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = hand_detector.detect_for_video(mp_image, timestamp_ms)

        crop, bbox = None, None
        if result.hand_landmarks:
            landmarks_list = result.hand_landmarks[0]   # list of NormalizedLandmark
            h_frame, w_frame = frame.shape[:2]
            draw_hand_skeleton(frame, landmarks_list, h_frame, w_frame)
            crop, bbox = get_hand_crop(rgb, landmarks_list)

        # --- Inference (only if hand found) ---
        if crop is not None and crop.size > 0:
            probs = run_inference(crop)
        else:
            probs = background_vec.copy()

        prob_buffer.append(probs)
        smoothed = np.mean(prob_buffer, axis=0)

        s_label = CLASS_NAMES[int(np.argmax(smoothed))]
        s_conf  = float(np.max(smoothed))

        # --- State machine ---
        if state == "COOLDOWN":
            if now - last_detection >= COOLDOWN_SEC:
                state = "WAITING"
                prob_buffer.clear()

        if state == "WAITING":
            if s_label != "background" and s_conf >= CONF_THRESH and bbox is not None:
                state = "DETECTING"
                last_detection = now
                print(f"  Detected: {s_label:>12}  ({s_conf*100:.1f}%)")

        elif state == "DETECTING":
            state = "COOLDOWN"
            last_detection = now

        # --- Draw ---
        display = draw_overlay(frame.copy(), smoothed, bbox, state, last_detection)

        fps = frame_count / (now - t_start + 1e-9)
        cv2.putText(display, f"{fps:.0f} fps",
                    (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90,90,90), 1)

        cv2.imshow("Gesture Detection — Q quit | S screenshot", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(now)}.png"
            cv2.imwrite(fname, display)
            print(f"  Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()  # Tasks API cleanup
    elapsed = time.time() - t_start
    print(f"\n[DONE] {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.0f} fps avg)")


if __name__ == "__main__":
    main()
