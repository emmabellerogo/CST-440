#!/usr/bin/env python3
import argparse
import re
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


CLASS_NAMES = ["like", "dislike", "peace", "background"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

CAM_WIDTH = 160
CAM_HEIGHT = 120
IMG_W = 64
IMG_H = 64
CROP_X = (CAM_WIDTH - IMG_W) // 2
CROP_Y = (CAM_HEIGHT - IMG_H) // 2


def preprocess_firmware_like(frame_bgr: np.ndarray) -> np.ndarray:
    # Accept images captured as full frames or as pre-cropped 64x64 images.
    if frame_bgr.ndim == 2:
        if frame_bgr.shape[0] == IMG_H and frame_bgr.shape[1] == IMG_W:
            return frame_bgr.astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

    if frame_bgr.shape[0] == IMG_H and frame_bgr.shape[1] == IMG_W:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = rgb[:, :, 0].astype(np.uint16)
        g = rgb[:, :, 1].astype(np.uint16)
        b = rgb[:, :, 2].astype(np.uint16)
        return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)

    resized = cv2.resize(frame_bgr, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    luma = ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)

    crop = luma[CROP_Y:CROP_Y + IMG_H, CROP_X:CROP_X + IMG_W]
    return crop


def infer_session_id(path: Path) -> str:
    stem = path.stem
    # Preferred capture format: sYYYYMMDD_HHMMSS_<label>_######
    m = re.match(r"^(s\d{8}_\d{6})_.*_\d{6}(?:_crop)?$", stem)
    if m:
        return m.group(1)

    # Fallback: strip trailing numeric frame suffix if present.
    m = re.match(r"^(.+?)_\d{6}(?:_crop)?$", stem)
    if m:
        return m.group(1)

    return f"{path.parent.name}_{stem}"


def load_dataset(data_root: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    items: list[tuple[Path, int]] = []

    for name in CLASS_NAMES:
        cls_dir = data_root / name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                items.append((p, CLASS_TO_INDEX[name]))

    if not items:
        raise ValueError(f"No images found under {data_root}")

    rng = np.random.default_rng(seed)
    rng.shuffle(items)

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    s_list: list[str] = []

    for p, y in items:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        x_list.append(preprocess_firmware_like(img))
        y_list.append(y)
        s_list.append(infer_session_id(p))

    if not x_list:
        raise ValueError("Could not decode any images from dataset")

    x = np.asarray(x_list, dtype=np.uint8)[..., None]
    y = np.asarray(y_list, dtype=np.int32)
    sessions = np.asarray(s_list, dtype=object)
    return x, y, sessions


def stratified_group_split(
    x: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    val_fraction: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for c in range(len(CLASS_NAMES)):
        idx = np.nonzero(y == c)[0]
        if len(idx) <= 1:
            train_idx.extend(idx.tolist())
            continue

        sess_to_indices: dict[str, list[int]] = {}
        for i in idx.tolist():
            sess_to_indices.setdefault(str(sessions[i]), []).append(int(i))

        sess_ids = list(sess_to_indices.keys())
        rng.shuffle(sess_ids)

        target_val = max(1, int(len(idx) * val_fraction))
        class_val: list[int] = []
        class_train: list[int] = []

        # If this class only has one capture session, fall back to a plain
        # per-class shuffled split so validation is still populated.
        if len(sess_ids) <= 1:
            shuffled_idx = idx.copy()
            rng.shuffle(shuffled_idx)
            n_val = min(target_val, len(shuffled_idx) - 1)
            class_val.extend(shuffled_idx[:n_val].tolist())
            class_train.extend(shuffled_idx[n_val:].tolist())
            train_idx.extend(class_train)
            val_idx.extend(class_val)
            continue

        # Keep sessions intact where possible to reduce burst leakage.
        for sid in sess_ids:
            sidx = sess_to_indices[sid]
            if len(class_val) < target_val and len(sess_ids) > 1:
                class_val.extend(sidx)
            else:
                class_train.extend(sidx)

        # Safety: always keep at least one sample in train.
        if not class_train and class_val:
            class_train.append(class_val.pop())

        train_idx.extend(class_train)
        val_idx.extend(class_val)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_idx = np.asarray(train_idx, dtype=np.int32)
    val_idx = np.asarray(val_idx, dtype=np.int32)

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def class_counts(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=len(CLASS_NAMES))


def session_counts_per_class(y: np.ndarray, sessions: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, name in enumerate(CLASS_NAMES):
        idx = np.nonzero(y == i)[0]
        if len(idx) == 0:
            out[name] = 0
            continue
        out[name] = len(set(str(sessions[j]) for j in idx.tolist()))
    return out


def class_weights(y: np.ndarray) -> dict[int, float]:
    counts = class_counts(y)
    total = float(np.sum(counts))
    n_classes = float(len(CLASS_NAMES))
    weights: dict[int, float] = {}
    for i, c in enumerate(counts):
        weights[i] = total / (n_classes * float(c)) if c > 0 else 1.0
    return weights


def build_model(learning_rate: float, label_smoothing: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_H, IMG_W, 1), name="input")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    x = tf.keras.layers.Conv2D(12, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(24, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="probs")(x)

    model = tf.keras.Model(inputs, outputs, name="gesture_raw_pixel_firmware")
    try:
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing)
    except TypeError:
        # Older TF/Keras builds do not support label_smoothing on sparse CE.
        if label_smoothing > 0.0:
            print(
                "Warning: This TensorFlow build does not support --label-smoothing with "
                "SparseCategoricalCrossentropy. Continuing with label_smoothing=0.0."
            )
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss_obj,
        metrics=["accuracy"],
    )
    return model


def make_train_ds(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    use_augmentation: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)

    def _aug(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.image.random_brightness(img, max_delta=18.0)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
        img = tf.image.random_crop(img, size=[IMG_H, IMG_W, 1])
        # Mild sensor-like noise and occasional blur improve robustness to live camera frames.
        img = img + tf.random.normal(tf.shape(img), mean=0.0, stddev=4.0)
        if tf.random.uniform(()) < 0.25:
            img = tf.nn.avg_pool2d(tf.expand_dims(img, 0), ksize=3, strides=1, padding="SAME")[0]
        img = tf.clip_by_value(img, 0.0, 255.0)
        return tf.cast(img, tf.uint8), label

    if use_augmentation:
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_eval_ds(x: np.ndarray, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def balanced_representative_subset(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_samples: int = 400,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_classes = len(CLASS_NAMES)
    per_class = max(1, max_samples // n_classes)

    chosen: list[int] = []
    for c in range(n_classes):
        idx = np.nonzero(y == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        take = min(len(idx), per_class)
        chosen.extend(idx[:take].tolist())

    if len(chosen) < max_samples:
        rest = np.setdiff1d(np.arange(len(y), dtype=np.int32), np.asarray(chosen, dtype=np.int32), assume_unique=False)
        rng.shuffle(rest)
        chosen.extend(rest[: max_samples - len(chosen)].tolist())

    rng.shuffle(chosen)
    return x[np.asarray(chosen, dtype=np.int32)]


def convert_int8(model: tf.keras.Model, out_path: Path, rep_images: np.ndarray) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        take = min(len(rep_images), 400)
        for i in range(take):
            yield [rep_images[i:i + 1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflm = converter.convert()
    out_path.write_bytes(tflm)


def run_quality_gates(
    x: np.ndarray,
    y: np.ndarray,
    min_blur_p10: float,
    min_like_peace_distance: float,
) -> None:
    blur_by_class: dict[int, list[float]] = {i: [] for i in range(len(CLASS_NAMES))}
    vectors: dict[int, list[np.ndarray]] = {i: [] for i in range(len(CLASS_NAMES))}

    for i in range(len(x)):
        cls = int(y[i])
        gray = x[i, :, :, 0]
        blur_by_class[cls].append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        vectors[cls].append(gray.astype(np.float32).reshape(-1) / 255.0)

    for i, name in enumerate(CLASS_NAMES):
        vals = blur_by_class[i]
        if not vals:
            raise ValueError(f"No samples present for class {name}")
        blur_p10 = float(np.percentile(vals, 10))
        if blur_p10 < min_blur_p10:
            raise ValueError(
                f"Quality gate failed: {name} blur p10={blur_p10:.2f} < required {min_blur_p10:.2f}. "
                "Remove blurry samples or recapture sharper data."
            )

    like_idx = CLASS_TO_INDEX.get("like")
    peace_idx = CLASS_TO_INDEX.get("peace")
    if like_idx is not None and peace_idx is not None:
        like_cent = np.mean(np.stack(vectors[like_idx]), axis=0)
        peace_cent = np.mean(np.stack(vectors[peace_idx]), axis=0)
        dist = float(np.linalg.norm(like_cent - peace_cent))
        if dist < min_like_peace_distance:
            raise ValueError(
                f"Quality gate failed: like__peace centroid distance={dist:.2f} < required {min_like_peace_distance:.2f}. "
                "Collect more distinctive peace/like poses before training."
            )


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def evaluate_tflite_int8(
    model_path: Path,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    preds: list[int] = []
    max_probs: list[float] = []

    for i in range(len(x)):
        x_u8 = x[i].astype(np.float32)

        if in_scale and in_scale < 0.02:
            x_f = x_u8 / 255.0
        else:
            x_f = x_u8

        q = np.round(x_f / in_scale + in_zp)
        q = np.clip(q, -128, 127).astype(np.int8)[None, ...]

        interpreter.set_tensor(in_det["index"], q)
        interpreter.invoke()

        out_q = interpreter.get_tensor(out_det["index"])[0]
        if out_det["dtype"] == np.int8:
            logits = (out_q.astype(np.float32) - out_zp) * out_scale
        else:
            logits = out_q.astype(np.float32)

        probs = _softmax(logits)
        pred = int(np.argmax(probs))

        preds.append(pred)
        max_probs.append(float(np.max(probs)))

    preds_arr = np.asarray(preds, dtype=np.int32)
    max_probs_arr = np.asarray(max_probs, dtype=np.float32)
    acc = float(np.mean(preds_arr == y))
    mean_max_prob = float(np.mean(max_probs))

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int32)
    for t, p in zip(y, preds_arr):
        cm[int(t), int(p)] += 1

    return acc, mean_max_prob, cm, preds_arr, max_probs_arr


def classification_metrics_from_cm(cm: np.ndarray) -> dict[str, object]:
    n = cm.shape[0]
    per_class: list[dict[str, float | int | str]] = []

    total = int(np.sum(cm))
    correct = int(np.trace(cm))
    accuracy = (correct / total) if total > 0 else 0.0

    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f1 = 0.0

    for i in range(n):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        support = int(np.sum(cm[i, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support

        per_class.append(
            {
                "class_name": CLASS_NAMES[i],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )

    n_float = float(n) if n > 0 else 1.0
    total_float = float(total) if total > 0 else 1.0

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p / n_float),
        "macro_recall": float(macro_r / n_float),
        "macro_f1": float(macro_f1 / n_float),
        "weighted_precision": float(weighted_p / total_float),
        "weighted_recall": float(weighted_r / total_float),
        "weighted_f1": float(weighted_f1 / total_float),
        "per_class": per_class,
    }


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    row_sums = np.sum(cm, axis=1, keepdims=True).astype(np.float32)
    row_sums[row_sums == 0] = 1.0
    return cm.astype(np.float32) / row_sums


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str], fmt: str) -> None:
    header = ["true\\pred", *labels]
    lines = [",".join(header)]
    for i, label in enumerate(labels):
        row_vals = [format(float(v), fmt) for v in matrix[i]]
        lines.append(",".join([label, *row_vals]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_int8_eval_markdown(
    out_path: Path,
    int8_val_acc: float,
    int8_mean_conf: float,
    metrics: dict[str, object],
    cm: np.ndarray,
    cm_norm: np.ndarray,
    pred_distribution: list[int],
) -> None:
    lines: list[str] = []
    lines.append("# Int8 Evaluation Report")
    lines.append("")
    lines.append(f"- int8_val_acc: {int8_val_acc:.5f}")
    lines.append(f"- int8_mean_conf: {int8_mean_conf:.5f}")
    lines.append(f"- macro_f1: {float(metrics['macro_f1']):.5f}")
    lines.append(f"- weighted_f1: {float(metrics['weighted_f1']):.5f}")
    lines.append("")
    lines.append("## Per-Class Metrics")
    lines.append("")
    lines.append("| class | precision | recall | f1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in metrics["per_class"]:
        lines.append(
            "| "
            f"{row['class_name']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['f1']):.4f} | {int(row['support'])} |"
        )
    lines.append("")
    lines.append("## Prediction Distribution")
    lines.append("")
    for i, label in enumerate(CLASS_NAMES):
        lines.append(f"- {label}: {pred_distribution[i]}")
    lines.append("")
    lines.append("## Confusion Matrix (Counts)")
    lines.append("")
    lines.append("Rows=true, cols=pred")
    lines.append("```")
    lines.append(np.array2string(cm))
    lines.append("```")
    lines.append("")
    lines.append("## Confusion Matrix (Row-Normalized)")
    lines.append("")
    lines.append("Rows sum to 1.0")
    lines.append("```")
    lines.append(np.array2string(cm_norm, formatter={"float_kind": lambda x: f"{x:.3f}"}))
    lines.append("```")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_header_from_tflite(tflite_path: Path, header_path: Path) -> None:
    data = tflite_path.read_bytes()
    lines = [
        "#ifndef GESTURE_MODEL_H",
        "#define GESTURE_MODEL_H",
        "",
        "unsigned char gesture_model[] = {",
    ]

    for i, b in enumerate(data):
        if i % 12 == 0:
            lines.append("  ")
        lines[-1] += f"0x{b:02x}, "

    lines.extend([
        "};",
        f"unsigned int gesture_model_len = {len(data)};",
        "",
        "#endif  // GESTURE_MODEL_H",
    ])

    header_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_data_root = script_dir / "raw_dataset"
    default_output_dir = script_dir / "model_artifacts_firmware"
    default_firmware_src = script_dir.parent / "src"

    parser = argparse.ArgumentParser(description="Train a raw-pixel gesture model aligned to firmware preprocessing")
    parser.add_argument("--data-root", type=Path, default=default_data_root,
                        help="Folder containing like/, dislike/, peace/, background/")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Sparse categorical label smoothing in [0, 1)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=100,
                        help="Minimum images required per class before training")
    parser.add_argument("--min-total-images", type=int, default=360,
                        help="Minimum total dataset size required before training")
    parser.add_argument("--min-sessions-per-class", type=int, default=2,
                        help="Minimum distinct capture sessions required per class")
    parser.add_argument("--min-train-per-class", type=int, default=80,
                        help="Minimum train split images per class required after stratified split")
    parser.add_argument("--min-val-per-class", type=int, default=20,
                        help="Minimum validation split images per class required after stratified split")
    parser.add_argument("--min-val-acc", type=float, default=0.55,
                        help="Minimum int8 validation accuracy required to consider export deployable")
    parser.add_argument("--min-blur-p10", type=float, default=8.0,
                        help="Minimum 10th-percentile Laplacian variance required per class")
    parser.add_argument("--min-like-peace-distance", type=float, default=6.0,
                        help="Minimum centroid L2 distance required between like and peace")
    parser.add_argument("--disable-quality-gates", action="store_true",
                        help="Skip dataset quality gates before training")
    parser.add_argument("--weighting-mode", choices=["auto", "on", "off"], default="auto",
                        help="Class-weight strategy: auto applies only when imbalance is high")
    parser.add_argument("--weighting-auto-ratio", type=float, default=1.5,
                        help="Apply class weights in auto mode when max/min class count exceeds this ratio")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation for a deterministic baseline run")
    parser.add_argument("--export-firmware", action="store_true",
                        help="Copy tflite and regenerate gesture_model.h in firmware src directory")
    parser.add_argument("--firmware-src", type=Path, default=default_firmware_src,
                        help="Path to PlatformIO firmware src directory")
    args = parser.parse_args()

    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError("--label-smoothing must be in [0, 1)")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    x, y, sessions = load_dataset(args.data_root, args.seed)
    counts = class_counts(y)
    print("Raw class counts:", counts.tolist())

    total_images = int(np.sum(counts))
    if total_images < args.min_total_images:
        raise ValueError(
            f"Insufficient total images: {total_images}. "
            f"Need at least {args.min_total_images} across all classes."
        )

    too_small = [CLASS_NAMES[i] for i, c in enumerate(counts) if c < args.min_per_class]
    if too_small:
        raise ValueError(
            "Insufficient images for classes: "
            f"{too_small}. Current counts={counts.tolist()}, required min={args.min_per_class}. "
            "Capture more images, especially for underrepresented classes."
        )

    session_counts = session_counts_per_class(y, sessions)
    low_sessions = [
        f"{name}={count}"
        for name, count in session_counts.items()
        if count < args.min_sessions_per_class
    ]
    if low_sessions:
        raise ValueError(
            "Insufficient capture-session diversity: "
            f"{', '.join(low_sessions)}. Required min sessions per class={args.min_sessions_per_class}. "
            "Capture at least 2+ separate sessions per class (different times/lighting/positioning), "
            "or intentionally override with --min-sessions-per-class 1 for a baseline run."
        )

    if not args.disable_quality_gates:
        run_quality_gates(x, y, args.min_blur_p10, args.min_like_peace_distance)

    x_train, y_train, x_val, y_val = stratified_group_split(x, y, sessions, args.val_fraction, args.seed)

    train_counts = class_counts(y_train)
    val_counts = class_counts(y_val)

    low_train = [
        f"{CLASS_NAMES[i]}={int(c)}"
        for i, c in enumerate(train_counts)
        if c < args.min_train_per_class
    ]
    if low_train:
        raise ValueError(
            "Insufficient train split images per class: "
            f"{', '.join(low_train)}. Required min train per class={args.min_train_per_class}. "
            "Capture more data or reduce --val-fraction."
        )

    low_val = [
        f"{CLASS_NAMES[i]}={int(c)}"
        for i, c in enumerate(val_counts)
        if c < args.min_val_per_class
    ]
    if low_val:
        raise ValueError(
            "Insufficient validation split images per class: "
            f"{', '.join(low_val)}. Required min val per class={args.min_val_per_class}. "
            "Capture more data or reduce --min-val-per-class."
        )

    print("Train:", x_train.shape, "Val:", x_val.shape)
    print("Train class counts:", train_counts.tolist())
    print("Val class counts:", val_counts.tolist())

    nonzero_train = train_counts[train_counts > 0]
    imbalance_ratio = float(np.max(nonzero_train) / np.min(nonzero_train)) if len(nonzero_train) else 1.0
    use_class_weights = args.weighting_mode == "on" or (
        args.weighting_mode == "auto" and imbalance_ratio >= args.weighting_auto_ratio
    )
    cw = class_weights(y_train) if use_class_weights else None
    print(f"Class weighting: {'enabled' if use_class_weights else 'disabled'} (imbalance ratio={imbalance_ratio:.2f})")

    model = build_model(args.learning_rate, args.label_smoothing)
    train_ds = make_train_ds(
        x_train,
        y_train,
        args.batch_size,
        args.seed,
        use_augmentation=not args.no_augment,
    )
    val_ds = make_eval_ds(x_val, y_val, args.batch_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=cw,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Final val loss: {val_loss:.5f}")
    print(f"Final val acc : {val_acc:.5f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = args.output_dir / "hagrid_firmware_congruent.keras"
    int8_path = args.output_dir / "hagrid_firmware_congruent_int8.tflite"
    labels_path = args.output_dir / "labels.json"

    model.save(keras_path)
    rep_images = balanced_representative_subset(x_train, y_train, seed=args.seed, max_samples=400)
    convert_int8(model, int8_path, rep_images)
    labels_path.write_text(json.dumps(CLASS_NAMES), encoding="utf-8")

    int8_val_acc, int8_mean_conf, int8_cm, int8_preds, int8_max_probs = evaluate_tflite_int8(int8_path, x_val, y_val)
    int8_metrics = classification_metrics_from_cm(int8_cm)
    int8_cm_norm = normalize_confusion_matrix(int8_cm)
    pred_distribution = class_counts(int8_preds).tolist()
    print(f"Int8 val acc  : {int8_val_acc:.5f}")
    print(f"Int8 mean conf: {int8_mean_conf:.5f}")
    print(f"Int8 macro F1 : {float(int8_metrics['macro_f1']):.5f}")
    print(f"Int8 wtd F1   : {float(int8_metrics['weighted_f1']):.5f}")
    print("Int8 per-class metrics:")
    for row in int8_metrics["per_class"]:
        print(
            f"  {row['class_name']:>10s} "
            f"P={float(row['precision']):.3f} "
            f"R={float(row['recall']):.3f} "
            f"F1={float(row['f1']):.3f} "
            f"N={int(row['support'])}"
        )
    print("Int8 confusion matrix (rows=true, cols=pred):")
    print(int8_cm)
    print("Int8 confusion matrix normalized (rows=true, cols=pred):")
    print(np.array2string(int8_cm_norm, formatter={"float_kind": lambda x: f"{x:.3f}"}))

    metrics_path = args.output_dir / "int8_eval.json"
    cm_csv_path = args.output_dir / "int8_confusion_matrix.csv"
    cm_norm_csv_path = args.output_dir / "int8_confusion_matrix_normalized.csv"
    report_md_path = args.output_dir / "int8_eval_report.md"

    write_matrix_csv(cm_csv_path, int8_cm, CLASS_NAMES, ".0f")
    write_matrix_csv(cm_norm_csv_path, int8_cm_norm, CLASS_NAMES, ".6f")
    write_int8_eval_markdown(
        report_md_path,
        int8_val_acc,
        int8_mean_conf,
        int8_metrics,
        int8_cm,
        int8_cm_norm,
        pred_distribution,
    )

    metrics_path.write_text(
        json.dumps(
            {
                "class_names": CLASS_NAMES,
                "int8_val_acc": int8_val_acc,
                "int8_mean_conf": int8_mean_conf,
                "int8_pred_distribution": pred_distribution,
                "int8_max_prob_summary": {
                    "mean": float(np.mean(int8_max_probs)),
                    "p10": float(np.percentile(int8_max_probs, 10)),
                    "p50": float(np.percentile(int8_max_probs, 50)),
                    "p90": float(np.percentile(int8_max_probs, 90)),
                },
                "int8_confusion_matrix": int8_cm.tolist(),
                "int8_confusion_matrix_normalized": int8_cm_norm.tolist(),
                "int8_classification_metrics": int8_metrics,
                "int8_artifacts": {
                    "confusion_matrix_csv": str(cm_csv_path),
                    "confusion_matrix_normalized_csv": str(cm_norm_csv_path),
                    "report_markdown": str(report_md_path),
                },
                "min_val_acc_required": args.min_val_acc,
                "is_deployable": int8_val_acc >= args.min_val_acc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {keras_path}")
    print(f"Saved: {int8_path} ({int8_path.stat().st_size} bytes)")
    print(f"Saved: {labels_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {cm_csv_path}")
    print(f"Saved: {cm_norm_csv_path}")
    print(f"Saved: {report_md_path}")

    if int8_val_acc < args.min_val_acc:
        raise RuntimeError(
            f"Int8 validation accuracy {int8_val_acc:.3f} is below required {args.min_val_acc:.3f}. "
            "Model appears non-deployable; capture more data and retrain."
        )

    if args.export_firmware:
        firmware_src = args.firmware_src.resolve()
        firmware_src.mkdir(parents=True, exist_ok=True)

        firmware_tflite = firmware_src / "hagrid_tiny_64_gray_int8.tflite"
        firmware_header = firmware_src / "gesture_model.h"

        firmware_tflite.write_bytes(int8_path.read_bytes())
        write_header_from_tflite(firmware_tflite, firmware_header)

        print(f"Exported firmware model: {firmware_tflite}")
        print(f"Exported firmware header: {firmware_header}")


if __name__ == "__main__":
    main()
