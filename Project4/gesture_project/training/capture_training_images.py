#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


CLASS_KEYS = {
    ord("1"): "like",
    ord("2"): "dislike",
    ord("3"): "peace",
    ord("4"): "background",
    ord("b"): "background",
    ord("B"): "background",
}

CLASS_NAMES = ["like", "dislike", "peace", "background"]

CAM_WIDTH = 160
CAM_HEIGHT = 120
IMG_W = 64
IMG_H = 64
CROP_X = (CAM_WIDTH - IMG_W) // 2
CROP_Y = (CAM_HEIGHT - IMG_H) // 2


def firmware_preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    luma = ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
    crop = luma[CROP_Y:CROP_Y + IMG_H, CROP_X:CROP_X + IMG_W]
    return crop


def ensure_dirs(root: Path) -> dict[str, Path]:
    out = {}
    for name in CLASS_NAMES:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        out[name] = p
    return out


def next_filename(folder: Path, prefix: str = "img") -> Path:
    idx = 1
    while True:
        candidate = folder / f"{prefix}_{idx:06d}.jpg"
        if not candidate.exists():
            return candidate
        idx += 1


def draw_hud(frame, active_label: str, counts: dict[str, int], mode_text: str):
    counts_text = "Counts -> " + " ".join(f"{name}:{counts[name]}" for name in CLASS_NAMES)
    lines = [
        f"Active label: {active_label}",
        counts_text,
        mode_text,
        "Keys: [1]=like [2]=dislike [3]=peace [4]/[b]=background [space]=capture [u]=undo [q]=quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 240, 40), 2, cv2.LINE_AA)
        y += 28


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture webcam images into class folders for training")
    parser.add_argument("--output-root", default="raw_dataset", help="Output dataset root (creates like/dislike/peace/background)")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--mirror", action="store_true", help="Mirror preview horizontally")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality 0-100")
    parser.add_argument("--save-mode", choices=["full", "crop", "both"], default="crop",
                        help="full=original frame, crop=firmware 64x64 crop, both=save both")
    args = parser.parse_args()

    root = Path(args.output_root).expanduser().resolve()
    class_dirs = ensure_dirs(root)

    counts = {k: len(list(v.glob("*.jpg"))) for k, v in class_dirs.items()}
    active_label = "like"
    history: list[Path] = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    print(f"Saving images under: {root}")
    print("Press 1/2/3/4 (or b) to select label, space to capture, u to undo, q to quit")

    last_flash_until = 0.0
    flash_label = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        # Keep a clean copy for disk writes. HUD is only for preview.
        clean_frame = frame.copy()
        firmware_crop = firmware_preprocess(clean_frame)

        # Draw the firmware crop rectangle on the live preview for alignment.
        h, w = frame.shape[:2]
        sx = w / CAM_WIDTH
        sy = h / CAM_HEIGHT
        x0 = int(CROP_X * sx)
        y0 = int(CROP_Y * sy)
        x1 = int((CROP_X + IMG_W) * sx)
        y1 = int((CROP_Y + IMG_H) * sy)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        mode_text = "Ready"
        now = time.time()
        if now < last_flash_until:
            mode_text = f"Captured -> {flash_label}"

        draw_hud(frame, active_label, counts, mode_text)
        cv2.imshow("Capture Training Images", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in CLASS_KEYS:
            active_label = CLASS_KEYS[key]
            continue

        if key == ord(" "):
            out_path = next_filename(class_dirs[active_label])
            ok_write = True
            qparams = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(0, min(100, args.jpeg_quality)))]

            if args.save_mode in {"full", "both"}:
                ok_write = ok_write and cv2.imwrite(str(out_path), clean_frame, qparams)

            if args.save_mode in {"crop", "both"}:
                crop_path = out_path if args.save_mode == "crop" else out_path.with_name(out_path.stem + "_crop.jpg")
                ok_write = ok_write and cv2.imwrite(str(crop_path), firmware_crop, qparams)

            if ok_write:
                history.append(out_path)
                counts[active_label] += 1
                flash_label = f"{active_label}: {out_path.name}"
                last_flash_until = time.time() + 0.8
                print(f"Saved {out_path}")
            else:
                print(f"Failed to save {out_path}")
            continue

        if key == ord("u"):
            if history:
                p = history.pop()
                label = p.parent.name
                try:
                    p.unlink(missing_ok=True)
                    if label in counts and counts[label] > 0:
                        counts[label] -= 1
                    flash_label = f"Undo -> {label}: {p.name}"
                    last_flash_until = time.time() + 0.8
                    print(f"Deleted {p}")
                except OSError as ex:
                    print(f"Could not delete {p}: {ex}")
            else:
                print("Undo history is empty")
            continue

        if key == ord("q"):
            break

        crop_preview = cv2.resize(firmware_crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Firmware Crop 64x64", crop_preview)

    cap.release()
    cv2.destroyAllWindows()

    print("Done.")
    for cls in CLASS_NAMES:
        print(f"{cls}: {counts[cls]}")


if __name__ == "__main__":
    main()
