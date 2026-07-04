#!/usr/bin/env python3
"""Evaluate landmark TFLite using legacy MediaPipe graph exported rects."""

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
from hand_pipeline.roi import crop_from_normalized_rect
from hand_pipeline.roi import landmarks_to_original
from hand_pipeline.roi import preprocess_landmark_tflite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-predictions",
        default="runs/eval_legacy_graph_test/legacy_hand_predictions.json",
    )
    parser.add_argument(
        "--landmark",
        default="models/tflite/mediapipe_legacy_0_10_14_hand_landmark_full.tflite",
    )
    parser.add_argument("--rect-field", choices=("hand_rect_from_palm", "hand_rect_from_landmarks"), default="hand_rect_from_palm")
    parser.add_argument("--output-dir", default="runs/eval_legacy_rect_tflite")
    parser.add_argument("--max-hands", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


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


def norm_from_box(box: np.ndarray) -> float:
    w = max(float(box[2] - box[0]), 1.0)
    h = max(float(box[3] - box[1]), 1.0)
    return max(math.sqrt(w * h), 1.0)


def metric_summary(errors: list[float], norm_errors: list[float]) -> dict[str, float]:
    err = np.array(errors, dtype=np.float32)
    nerr = np.array(norm_errors, dtype=np.float32)
    return {
        "mean_px": float(np.mean(err)),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(np.max(err)),
        "nme": float(np.mean(nerr)),
        "pck@0.01": float(np.mean(nerr <= 0.01)),
        "pck@0.02": float(np.mean(nerr <= 0.02)),
        "pck@0.05": float(np.mean(nerr <= 0.05)),
        "pck@0.10": float(np.mean(nerr <= 0.10)),
    }


def summarize_times(values: list[float], prefix: str) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        f"{prefix}_mean_ms": float(np.mean(arr)),
        f"{prefix}_median_ms": float(np.median(arr)),
        f"{prefix}_p95_ms": float(np.percentile(arr, 95)),
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


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Legacy Rect + Landmark TFLite Evaluation",
        "",
        "## Settings",
        "",
        f"- legacy predictions: `{summary['legacy_predictions']}`",
        f"- landmark: `{summary['landmark']}`",
        f"- rect field: `{summary['rect_field']}`",
        f"- hands: `{summary['hands']}`",
        "",
        "## Landmark Error",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| mean px | {fmt(summary['mean_px'])} |",
        f"| median px | {fmt(summary['median_px'])} |",
        f"| P95 px | {fmt(summary['p95_px'])} |",
        f"| max px | {fmt(summary['max_px'])} |",
        f"| NME | {fmt(summary['nme'])} |",
        f"| PCK@0.01 | {fmt(summary['pck@0.01'])} |",
        f"| PCK@0.02 | {fmt(summary['pck@0.02'])} |",
        f"| PCK@0.05 | {fmt(summary['pck@0.05'])} |",
        f"| PCK@0.10 | {fmt(summary['pck@0.10'])} |",
        "",
        "## Notes",
        "",
        "- This check uses NormalizedRect exported from the legacy graph as the ROI source.",
        "- If this error is near zero, remaining two-stage error is mainly palm-to-rect geometry.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(args.legacy_predictions).read_text(encoding="utf-8"))
    if args.max_hands:
        items = items[: args.max_hands]
    landmark = TfliteModel(args.landmark, num_threads=args.num_threads)
    cache: dict[Path, np.ndarray] = {}
    errors: list[float] = []
    norm_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    roi_ms: list[float] = []
    infer_ms: list[float] = []
    post_ms: list[float] = []
    used = 0
    for item in items:
        rect = item.get(args.rect_field)
        if not rect:
            continue
        image_path = Path(item["image"])
        image = cache.get(image_path)
        if image is None:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            cache[image_path] = image
        roi_start = time.perf_counter()
        crop, inverse = crop_from_normalized_rect(image, rect, input_size=224)
        tensor = preprocess_landmark_tflite(crop)
        roi_ms.append((time.perf_counter() - roi_start) * 1000.0)
        infer_start = time.perf_counter()
        outputs = landmark(tensor)
        infer_ms.append((time.perf_counter() - infer_start) * 1000.0)
        post_start = time.perf_counter()
        landmarks, hand_score, handedness, _world = pick_landmark_outputs(outputs)
        hand21 = landmarks_to_original(landmarks, inverse, input_size=224, coord_scale="auto")
        expected = np.array(item["hand21"], dtype=np.float32)
        err = np.linalg.norm(hand21 - expected, axis=1)
        norm = norm_from_box(np.array(item["box"], dtype=np.float32))
        post_ms.append((time.perf_counter() - post_start) * 1000.0)
        errors.extend(float(x) for x in err)
        norm_errors.extend(float(x / norm) for x in err)
        rows.append(
            {
                "image": image_path.name,
                "hand_index": item["hand_index"],
                "mean_px": float(np.mean(err)),
                "max_px": float(np.max(err)),
                "nme": float(np.mean(err / norm)),
                "hand_score": hand_score,
                "handedness": handedness,
            }
        )
        predictions.append(
            {
                "image": str(image_path),
                "hand_index": item["hand_index"],
                "hand_score": hand_score,
                "handedness": handedness,
                "hand21": hand21.astype(float).tolist(),
                "rect": rect,
            }
        )
        used += 1
    summary = {
        "task": "eval_legacy_rect_tflite",
        "legacy_predictions": args.legacy_predictions,
        "landmark": args.landmark,
        "rect_field": args.rect_field,
        "hands": used,
        **metric_summary(errors, norm_errors),
        **summarize_times(roi_ms, "roi"),
        **summarize_times(infer_ms, "landmark"),
        **summarize_times(post_ms, "post"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "matches.csv", rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

