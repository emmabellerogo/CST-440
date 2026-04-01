#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


CLASS_NAMES = ["like", "dislike", "palm"]
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


def load_dataset(data_root: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
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

    for p, y in items:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        x_list.append(preprocess_firmware_like(img))
        y_list.append(y)

    if not x_list:
        raise ValueError("Could not decode any images from dataset")

    x = np.asarray(x_list, dtype=np.uint8)[..., None]
    y = np.asarray(y_list, dtype=np.int32)
    return x, y


def stratified_split(x: np.ndarray, y: np.ndarray, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for c in range(len(CLASS_NAMES)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction)) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_idx = np.asarray(train_idx, dtype=np.int32)
    val_idx = np.asarray(val_idx, dtype=np.int32)

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def class_weights(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    total = float(np.sum(counts))
    n_classes = float(len(CLASS_NAMES))
    weights: dict[int, float] = {}
    for i, c in enumerate(counts):
        weights[i] = total / (n_classes * float(c)) if c > 0 else 1.0
    return weights


def build_model() -> tf.keras.Model:
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
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
        img = tf.image.random_saturation(tf.image.grayscale_to_rgb(img), lower=0.95, upper=1.05)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
        img = tf.image.random_crop(img, size=[IMG_H, IMG_W, 1])
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def evaluate_tflite_int8(model_path: Path, x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
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
    acc = float(np.mean(preds_arr == y))
    mean_max_prob = float(np.mean(max_probs))

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int32)
    for t, p in zip(y, preds_arr):
        cm[int(t), int(p)] += 1

    return acc, mean_max_prob, cm


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
                        help="Folder containing like/, dislike/, palm/")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=100,
                        help="Minimum images required per class before training")
    parser.add_argument("--min-total-images", type=int, default=360,
                        help="Minimum total dataset size required before training")
    parser.add_argument("--min-train-per-class", type=int, default=80,
                        help="Minimum train split images per class required after stratified split")
    parser.add_argument("--min-val-per-class", type=int, default=20,
                        help="Minimum validation split images per class required after stratified split")
    parser.add_argument("--min-val-acc", type=float, default=0.55,
                        help="Minimum int8 validation accuracy required to consider export deployable")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation for a deterministic baseline run")
    parser.add_argument("--export-firmware", action="store_true",
                        help="Copy tflite and regenerate gesture_model.h in firmware src directory")
    parser.add_argument("--firmware-src", type=Path, default=default_firmware_src,
                        help="Path to PlatformIO firmware src directory")
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    x, y = load_dataset(args.data_root, args.seed)
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
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

    x_train, y_train, x_val, y_val = stratified_split(x, y, args.val_fraction, args.seed)

    train_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    val_counts = np.bincount(y_val, minlength=len(CLASS_NAMES))

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

    model = build_model()
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
        class_weight=class_weights(y_train),
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
    convert_int8(model, int8_path, x_train)
    labels_path.write_text(json.dumps(CLASS_NAMES), encoding="utf-8")

    int8_val_acc, int8_mean_conf, int8_cm = evaluate_tflite_int8(int8_path, x_val, y_val)
    print(f"Int8 val acc  : {int8_val_acc:.5f}")
    print(f"Int8 mean conf: {int8_mean_conf:.5f}")
    print("Int8 confusion matrix (rows=true, cols=pred):")
    print(int8_cm)

    metrics_path = args.output_dir / "int8_eval.json"
    metrics_path.write_text(
        json.dumps(
            {
                "class_names": CLASS_NAMES,
                "int8_val_acc": int8_val_acc,
                "int8_mean_conf": int8_mean_conf,
                "int8_confusion_matrix": int8_cm.tolist(),
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
