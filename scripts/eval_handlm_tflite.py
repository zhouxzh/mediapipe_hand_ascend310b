#!/usr/bin/env python3
"""Evaluate hand landmark TFLite models against manually corrected COCO keypoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from hand_pipeline.inference import TfliteModel


DEFAULT_MODELS = [
    ("task_full", "models/tflite/mediapipe_task_hand_landmark_full.tflite"),
    ("legacy_full", "models/tflite/mediapipe_legacy_0_10_14_hand_landmark_full.tflite"),
    ("legacy_lite", "models/tflite/mediapipe_legacy_0_10_14_hand_landmark_lite.tflite"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/handlm_datasets")
    parser.add_argument("--annotations", default="")
    parser.add_argument("--output-dir", default="runs/baseline/handlm_manual_gt")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Landmark model spec. Use key=path or just path. Defaults to task_full, legacy_full, legacy_lite.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--save-vis", type=int, default=0)
    return parser.parse_args()


def parse_model_specs(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return [(key, Path(path)) for key, path in DEFAULT_MODELS]
    specs: list[tuple[str, Path]] = []
    for value in values:
        if "=" in value:
            key, raw_path = value.split("=", 1)
            specs.append((key.strip(), Path(raw_path.strip())))
        else:
            path = Path(value)
            specs.append((path.stem, path))
    return specs


def load_coco(data_root: Path, annotations_path: Path, max_images: int) -> list[dict[str, Any]]:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data.get("images", [])}
    items: list[dict[str, Any]] = []
    for ann in data.get("annotations", []):
        image = images.get(int(ann["image_id"]))
        if not image:
            continue
        keypoints = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        if keypoints.shape[0] != 21:
            continue
        image_path = data_root / image["file_name"]
        items.append(
            {
                "annotation_id": int(ann["id"]),
                "image_id": int(image["id"]),
                "image_name": str(image["file_name"]),
                "image_path": image_path,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "bbox": np.array(ann.get("bbox", [0, 0, image["width"], image["height"]]), dtype=np.float32),
                "handedness": ann.get("handedness", ""),
                "score": float(ann.get("score", 1.0)),
                "keypoints": keypoints,
            }
        )
    items.sort(key=lambda item: (item["image_id"], item["annotation_id"]))
    if max_images > 0:
        items = items[:max_images]
    return items


def preprocess_image(image: np.ndarray, input_size: int) -> np.ndarray:
    if image.shape[0] != input_size or image.shape[1] != input_size:
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb[None])


def pick_landmark_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    landmarks = None
    world = None
    one_value: list[np.ndarray] = []
    for value in outputs:
        arr = np.asarray(value)
        if arr.size == 63 and landmarks is None:
            landmarks = arr.reshape(21, 3)
        elif arr.size == 63:
            world = arr.reshape(21, 3)
        elif arr.size == 1:
            one_value.append(arr.reshape(-1))
    if landmarks is None:
        raise ValueError(f"Could not find 63-value landmark output: {[x.shape for x in outputs]}")
    hand_score = float(one_value[0][0]) if len(one_value) >= 1 else math.nan
    handedness = float(one_value[1][0]) if len(one_value) >= 2 else math.nan
    return landmarks.astype(np.float32), hand_score, handedness, None if world is None else world.astype(np.float32)


def landmarks_to_pixels(landmarks: np.ndarray, input_size: int) -> np.ndarray:
    points = landmarks[:, :2].astype(np.float32).copy()
    if float(np.nanmax(np.abs(points))) <= 2.0:
        points *= input_size
    return points


def bbox_norm(bbox_xywh: np.ndarray) -> float:
    width = max(float(bbox_xywh[2]), 1.0)
    height = max(float(bbox_xywh[3]), 1.0)
    return max(math.sqrt(width * height), 1.0)


def keypoint_bbox(points: np.ndarray, visible: np.ndarray) -> np.ndarray:
    pts = points[visible]
    if pts.size == 0:
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return np.array([x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32)


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p95": math.nan,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
    }


def metric_summary(errors: list[float], norm_errors: list[float], prefix: str) -> dict[str, float]:
    if not errors:
        return {
            f"{prefix}_mean_px": math.nan,
            f"{prefix}_median_px": math.nan,
            f"{prefix}_p95_px": math.nan,
            f"{prefix}_max_px": math.nan,
            f"{prefix}_nme": math.nan,
            f"{prefix}_pck@0.01": math.nan,
            f"{prefix}_pck@0.02": math.nan,
            f"{prefix}_pck@0.05": math.nan,
            f"{prefix}_pck@0.10": math.nan,
        }
    err = np.array(errors, dtype=np.float32)
    nerr = np.array(norm_errors, dtype=np.float32)
    return {
        f"{prefix}_mean_px": float(np.mean(err)),
        f"{prefix}_median_px": float(np.median(err)),
        f"{prefix}_p95_px": float(np.percentile(err, 95)),
        f"{prefix}_max_px": float(np.max(err)),
        f"{prefix}_nme": float(np.mean(nerr)),
        f"{prefix}_pck@0.01": float(np.mean(nerr <= 0.01)),
        f"{prefix}_pck@0.02": float(np.mean(nerr <= 0.02)),
        f"{prefix}_pck@0.05": float(np.mean(nerr <= 0.05)),
        f"{prefix}_pck@0.10": float(np.mean(nerr <= 0.10)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def draw_points(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    canvas = image.copy()
    for a, b in edges:
        cv2.line(canvas, tuple(np.round(gt[a]).astype(int)), tuple(np.round(gt[b]).astype(int)), (0, 200, 255), 2)
        cv2.line(canvas, tuple(np.round(pred[a]).astype(int)), tuple(np.round(pred[b]).astype(int)), (255, 80, 0), 1)
    for x, y in gt:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (0, 200, 255), -1)
    for x, y in pred:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (255, 80, 0), -1)
    return canvas


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Hand Landmark Manual GT Evaluation",
        "",
        f"- data: `{summary['data']}`",
        f"- annotations: `{summary['annotations']}`",
        f"- images: `{summary['images']}`",
        f"- hands: `{summary['hands']}`",
        f"- visible points: `{summary['visible_points']}`",
        "",
        "| Model | Mean px | Median px | P95 px | NME | PCK@0.05 | PCK@0.10 | infer ms | total ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["models"]:
        lines.append(
            f"| `{item['key']}` | {fmt(item['gt_mean_px'])} | {fmt(item['gt_median_px'])} | "
            f"{fmt(item['gt_p95_px'])} | {fmt(item['gt_nme'])} | {fmt(item['gt_pck@0.05'])} | "
            f"{fmt(item['gt_pck@0.10'])} | {fmt(item['infer_ms_mean'])} | {fmt(item['total_ms_mean'])} |"
        )
    lines += [
        "",
        "The dataset is treated as manually corrected 21-point ground truth. Metrics are computed in the 224x224 crop coordinate system.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_model(
    key: str,
    model_path: Path,
    items: list[dict[str, Any]],
    input_size: int,
    num_threads: int,
    output_dir: Path,
    save_vis: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model = TfliteModel(model_path, num_threads=num_threads)
    point_errors: list[float] = []
    norm_errors: list[float] = []
    keypoint_norm_errors: list[float] = []
    read_ms: list[float] = []
    pre_ms: list[float] = []
    infer_ms: list[float] = []
    post_ms: list[float] = []
    total_ms: list[float] = []
    hand_scores: list[float] = []
    handedness_values: list[float] = []
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    vis_dir = output_dir / "vis" / key
    saved_vis = 0
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        total_start = time.perf_counter()
        read_start = time.perf_counter()
        image = cv2.imread(str(item["image_path"]))
        read_ms.append((time.perf_counter() - read_start) * 1000.0)
        if image is None:
            raise ValueError(f"Failed to read image: {item['image_path']}")

        pre_start = time.perf_counter()
        tensor = preprocess_image(image, input_size)
        pre_ms.append((time.perf_counter() - pre_start) * 1000.0)

        infer_start = time.perf_counter()
        outputs = model(tensor)
        infer_ms.append((time.perf_counter() - infer_start) * 1000.0)

        post_start = time.perf_counter()
        raw_landmarks, hand_score, handedness, _world = pick_landmark_outputs(outputs)
        pred_points = landmarks_to_pixels(raw_landmarks, input_size=input_size)
        post_ms.append((time.perf_counter() - post_start) * 1000.0)

        gt_all = item["keypoints"]
        visible = gt_all[:, 2] > 0
        gt_points = gt_all[:, :2].astype(np.float32)
        err = np.linalg.norm(pred_points[visible] - gt_points[visible], axis=1)
        norm = bbox_norm(item["bbox"])
        kp_norm = bbox_norm(keypoint_bbox(gt_points, visible))
        point_errors.extend(float(x) for x in err)
        norm_errors.extend(float(x / norm) for x in err)
        keypoint_norm_errors.extend(float(x / kp_norm) for x in err)
        if not math.isnan(hand_score):
            hand_scores.append(hand_score)
        if not math.isnan(handedness):
            handedness_values.append(handedness)
        total_ms.append((time.perf_counter() - total_start) * 1000.0)

        row = {
            "model": key,
            "image": item["image_name"],
            "annotation_id": item["annotation_id"],
            "handedness_gt": item["handedness"],
            "visible_points": int(np.sum(visible)),
            "mean_px": float(np.mean(err)) if err.size else math.nan,
            "max_px": float(np.max(err)) if err.size else math.nan,
            "nme_bbox": float(np.mean(err / norm)) if err.size else math.nan,
            "nme_keypoint_bbox": float(np.mean(err / kp_norm)) if err.size else math.nan,
            "hand_score": hand_score,
            "handedness_raw": handedness,
        }
        rows.append(row)
        predictions.append(
            {
                **row,
                "pred21": pred_points.astype(float).tolist(),
                "gt21": gt_points.astype(float).tolist(),
            }
        )
        if save_vis and saved_vis < save_vis:
            canvas = draw_points(image, gt_points, pred_points)
            cv2.imwrite(str(vis_dir / item["image_name"]), canvas)
            saved_vis += 1

    metrics = metric_summary(point_errors, norm_errors, "gt")
    kp_metrics = metric_summary(point_errors, keypoint_norm_errors, "gt_keypoint_box")
    result = {
        "key": key,
        "model": str(model_path),
        "hands": len(items),
        "visible_points": len(point_errors),
        **metrics,
        "gt_keypoint_box_nme": kp_metrics["gt_keypoint_box_nme"],
        "gt_keypoint_box_pck@0.05": kp_metrics["gt_keypoint_box_pck@0.05"],
        "gt_keypoint_box_pck@0.10": kp_metrics["gt_keypoint_box_pck@0.10"],
        **summarize(read_ms, "read_ms"),
        **summarize(pre_ms, "preprocess_ms"),
        **summarize(infer_ms, "infer_ms"),
        **summarize(post_ms, "post_ms"),
        **summarize(total_ms, "total_ms"),
        "fps": 1000.0 / float(np.mean(total_ms)) if total_ms else math.nan,
        "hand_score_mean": float(np.mean(hand_scores)) if hand_scores else math.nan,
        "handedness_raw_mean": float(np.mean(handedness_values)) if handedness_values else math.nan,
        "visualizations": saved_vis,
    }
    return result, rows, predictions


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    annotations_path = Path(args.annotations) if args.annotations else data_root / "annotations.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_coco(data_root, annotations_path, args.max_images)
    if not items:
        raise ValueError(f"No valid hand landmark annotations found in {annotations_path}")

    model_specs = parse_model_specs(args.model)
    model_results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for key, model_path in model_specs:
        result, rows, predictions = evaluate_model(
            key=key,
            model_path=model_path,
            items=items,
            input_size=args.input_size,
            num_threads=args.num_threads,
            output_dir=output_dir,
            save_vis=args.save_vis,
        )
        model_results.append(result)
        all_rows.extend(rows)
        all_predictions.extend(predictions)

    summary = {
        "task": "eval_handlm_tflite_manual_gt",
        "data": str(data_root),
        "annotations": str(annotations_path),
        "images": len({item["image_id"] for item in items}),
        "hands": len(items),
        "visible_points": int(sum(np.sum(item["keypoints"][:, 2] > 0) for item in items)),
        "input_size": args.input_size,
        "models": model_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.json").write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "per_image_metrics.csv", all_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
