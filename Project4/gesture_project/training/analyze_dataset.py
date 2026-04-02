#!/usr/bin/env python3
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_NAMES = ["like", "dislike", "peace", "background"]
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CAM_WIDTH = 160
CAM_HEIGHT = 120
IMG_W = 64
IMG_H = 64
CROP_X = (CAM_WIDTH - IMG_W) // 2
CROP_Y = (CAM_HEIGHT - IMG_H) // 2


@dataclass
class ImageRecord:
    path: Path
    class_name: str
    gray: np.ndarray
    digest: str
    dhash: int


def preprocess_firmware_like(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        if frame_bgr.shape[:2] == (IMG_H, IMG_W):
            return frame_bgr.astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

    if frame_bgr.shape[:2] == (IMG_H, IMG_W):
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

    return luma[CROP_Y:CROP_Y + IMG_H, CROP_X:CROP_X + IMG_W]


def image_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(np.sum(hist), 1.0)
    nz = prob[prob > 0]
    return float(-(nz * np.log2(nz)).sum())


def dhash_64(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def class_summary(records: list[ImageRecord]) -> dict[str, Any]:
    means = []
    stds = []
    p01 = []
    p99 = []
    blur = []
    entropy = []
    clipped_low = []
    clipped_high = []

    for rec in records:
        arr = rec.gray.astype(np.float32)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))
        p01.append(float(np.percentile(arr, 1)))
        p99.append(float(np.percentile(arr, 99)))
        blur.append(float(cv2.Laplacian(rec.gray, cv2.CV_32F).var()))
        entropy.append(image_entropy(rec.gray))
        clipped_low.append(float(np.mean(arr <= 3)))
        clipped_high.append(float(np.mean(arr >= 252)))

    return {
        "count": len(records),
        "mean_luma": float(np.mean(means)),
        "mean_luma_p10": float(np.percentile(means, 10)),
        "mean_luma_p90": float(np.percentile(means, 90)),
        "std_luma": float(np.mean(stds)),
        "std_luma_p10": float(np.percentile(stds, 10)),
        "std_luma_p90": float(np.percentile(stds, 90)),
        "p01_luma": float(np.mean(p01)),
        "p99_luma": float(np.mean(p99)),
        "blur_laplacian_mean": float(np.mean(blur)),
        "blur_laplacian_p10": float(np.percentile(blur, 10)),
        "blur_laplacian_p90": float(np.percentile(blur, 90)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_p10": float(np.percentile(entropy, 10)),
        "entropy_p90": float(np.percentile(entropy, 90)),
        "clipped_dark_ratio_mean": float(np.mean(clipped_low)),
        "clipped_bright_ratio_mean": float(np.mean(clipped_high)),
    }


def nearest_centroid_confusion(vectors: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    centroids = {k: np.mean(v, axis=0) for k, v in vectors.items()}
    labels = list(vectors.keys())
    confusion = {a: dict.fromkeys(labels, 0) for a in labels}

    for true_cls, x in vectors.items():
        for row in x:
            best_cls = labels[0]
            best_dist = float(np.linalg.norm(row - centroids[best_cls]))
            for label in labels[1:]:
                d = float(np.linalg.norm(row - centroids[label]))
                if d < best_dist:
                    best_dist = d
                    best_cls = label
            confusion[true_cls][best_cls] += 1

    return confusion


def centroid_distances(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    labels = list(vectors.keys())
    centroids = {k: np.mean(v, axis=0) for k, v in vectors.items()}
    out: dict[str, float] = {}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = labels[i]
            b = labels[j]
            d = float(np.linalg.norm(centroids[a] - centroids[b]))
            out[f"{a}__{b}"] = d

    return out


def duplicate_stats(records: list[ImageRecord]) -> dict[str, Any]:
    by_digest: dict[str, list[ImageRecord]] = {}
    by_dhash: dict[int, list[ImageRecord]] = {}

    for rec in records:
        by_digest.setdefault(rec.digest, []).append(rec)
        by_dhash.setdefault(rec.dhash, []).append(rec)

    exact_groups = [g for g in by_digest.values() if len(g) > 1]
    phash_groups = [g for g in by_dhash.values() if len(g) > 1]

    exact_cross_class = 0
    for g in exact_groups:
        classes = {x.class_name for x in g}
        if len(classes) > 1:
            exact_cross_class += 1

    return {
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_images": int(sum(len(g) for g in exact_groups)),
        "exact_cross_class_groups": exact_cross_class,
        "dhash_collision_groups": len(phash_groups),
        "dhash_collision_images": int(sum(len(g) for g in phash_groups)),
        "sample_exact_groups": [
            {
                "classes": sorted({x.class_name for x in g}),
                "files": [str(x.path) for x in g[:6]],
            }
            for g in exact_groups[:10]
        ],
    }


def generate_recommendations(report: dict[str, Any]) -> list[str]:
    recs: list[str] = []

    counts = report["counts"]
    non_zero = [v for v in counts.values() if v > 0]
    if non_zero:
        ratio = max(non_zero) / min(non_zero)
        if ratio > 2.0:
            recs.append(
                f"Class imbalance is high (max/min={ratio:.2f}). Add samples to underrepresented classes."
            )

    stats = report["class_stats"]
    for c, s in stats.items():
        if s["count"] < 150:
            recs.append(f"{c}: only {s['count']} images. Target at least 200+ per class.")
        if s["blur_laplacian_p10"] < 10.0:
            recs.append(
                f"{c}: many low-detail frames (blur p10={s['blur_laplacian_p10']:.2f}). Remove very blurry samples."
            )
        if s["entropy_p10"] < 3.0:
            recs.append(
                f"{c}: low-information tail (entropy p10={s['entropy_p10']:.2f}). Add more varied lighting/backgrounds."
            )

    dups = report["duplicates"]
    if dups["exact_cross_class_groups"] > 0:
        recs.append(
            f"Found {dups['exact_cross_class_groups']} exact duplicate groups across classes. Fix mislabeled copies."
        )
    if dups["exact_duplicate_groups"] > 20:
        recs.append(
            f"Found many exact duplicates ({dups['exact_duplicate_groups']} groups). Capture more natural variation."
        )

    dists = report["centroid_distances_l2"]
    if dists:
        close_pair = min(dists.items(), key=lambda kv: kv[1])
        if close_pair[1] < 5.5:
            recs.append(
                f"Closest class centroids are {close_pair[0]} (L2={close_pair[1]:.2f}). Add more distinctive poses for those classes."
            )

    confusion = report["nearest_centroid_confusion"]
    if "background" in confusion:
        bg = confusion["background"]
        bg_total = sum(bg.values())
        bg_correct = bg.get("background", 0)
        bg_error_rate = 1.0 - (bg_correct / max(bg_total, 1))
        if bg_error_rate > 0.20:
            recs.append(
                f"Background confusion is high ({bg_error_rate * 100:.1f}% to gesture classes). Add harder negative/background samples."
            )

    if not recs:
        recs.append("Dataset quality looks reasonable. Proceed with training and validate with confusion matrix.")

    return recs


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Dataset Analysis Report")
    lines.append("")
    lines.append(f"Dataset root: `{report['dataset_root']}`")
    lines.append("")

    lines.append("## Class Counts")
    lines.append("")
    for c, n in report["counts"].items():
        lines.append(f"- {c}: {n}")
    lines.append("")

    lines.append("## Class Statistics")
    lines.append("")
    for c, s in report["class_stats"].items():
        lines.append(f"### {c}")
        lines.append(f"- count: {s['count']}")
        lines.append(f"- mean_luma: {s['mean_luma']:.2f}")
        lines.append(f"- std_luma: {s['std_luma']:.2f}")
        lines.append(f"- blur_laplacian_mean: {s['blur_laplacian_mean']:.2f}")
        lines.append(f"- blur_laplacian_p10: {s['blur_laplacian_p10']:.2f}")
        lines.append(f"- entropy_mean: {s['entropy_mean']:.2f}")
        lines.append("")

    lines.append("## Centroid Distances (L2)")
    lines.append("")
    for k, v in sorted(report["centroid_distances_l2"].items(), key=lambda kv: kv[1]):
        lines.append(f"- {k}: {v:.3f}")
    lines.append("")

    lines.append("## Duplicate Summary")
    lines.append("")
    d = report["duplicates"]
    lines.append(f"- exact_duplicate_groups: {d['exact_duplicate_groups']}")
    lines.append(f"- exact_duplicate_images: {d['exact_duplicate_images']}")
    lines.append(f"- exact_cross_class_groups: {d['exact_cross_class_groups']}")
    lines.append(f"- dhash_collision_groups: {d['dhash_collision_groups']}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for rec in report["recommendations"]:
        lines.append(f"- {rec}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze gesture dataset quality and class separability")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"), help="Folder containing class subfolders")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_analysis"), help="Where to write report files")
    parser.add_argument("--class-names", nargs="+", default=CLASS_NAMES, help="Class folder names to analyze")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in args.class_names if not (data_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing class folders under {data_root}: {missing}")

    records: list[ImageRecord] = []
    decode_fail: dict[str, int] = dict.fromkeys(args.class_names, 0)

    for class_name in args.class_names:
        for p in sorted((data_root / class_name).rglob("*")):
            if p.suffix.lower() not in VALID_SUFFIXES:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                decode_fail[class_name] += 1
                continue
            gray = preprocess_firmware_like(img)
            digest = hashlib.sha1(gray.tobytes()).hexdigest()
            records.append(
                ImageRecord(
                    path=p,
                    class_name=class_name,
                    gray=gray,
                    digest=digest,
                    dhash=dhash_64(gray),
                )
            )

    if not records:
        raise ValueError(f"No decodable images found in {data_root}")

    by_class: dict[str, list[ImageRecord]] = {name: [] for name in args.class_names}
    for rec in records:
        by_class[rec.class_name].append(rec)

    class_stats: dict[str, Any] = {}
    vectors: dict[str, np.ndarray] = {}

    for class_name, items in by_class.items():
        if not items:
            class_stats[class_name] = {"count": 0, "decode_fail": decode_fail[class_name]}
            continue
        class_stats[class_name] = class_summary(items)
        class_stats[class_name]["decode_fail"] = decode_fail[class_name]
        vectors[class_name] = np.stack([(x.gray.astype(np.float32) / 255.0).reshape(-1) for x in items])

    counts = {k: int(v["count"]) for k, v in class_stats.items()}

    report: dict[str, Any] = {
        "dataset_root": str(data_root),
        "counts": counts,
        "class_stats": class_stats,
        "duplicates": duplicate_stats(records),
        "centroid_distances_l2": centroid_distances(vectors),
        "nearest_centroid_confusion": nearest_centroid_confusion(vectors),
    }
    report["recommendations"] = generate_recommendations(report)

    json_path = output_dir / "analysis_report.json"
    md_path = output_dir / "analysis_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"Dataset analyzed: {data_root}")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {md_path}")
    print("Recommendations:")
    for idx, rec in enumerate(report["recommendations"], start=1):
        print(f"  {idx}. {rec}")


if __name__ == "__main__":
    main()
