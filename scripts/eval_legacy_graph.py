#!/usr/bin/env python3
"""Evaluate the legacy MediaPipe hand landmark graph and export middle streams."""

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
import mediapipe as mp
import numpy as np
from mediapipe.python.solution_base import SolutionBase

from hand_pipeline.eval import PalmPrediction
from hand_pipeline.eval import box_iou
from hand_pipeline.eval import detection_metrics
from hand_pipeline.eval import list_images
from hand_pipeline.eval import load_targets


GRAPH_PATH = "mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb"
GRAPH_OUTPUTS = [
    "multi_hand_landmarks",
    "multi_hand_world_landmarks",
    "multi_handedness",
    "palm_detections",
    "hand_rects_from_landmarks",
    "hand_rects_from_palm_detections",
]
HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/palm_datasets")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="runs/eval_legacy_graph_test")
    parser.add_argument("--current-mediapipe", default="runs/mediapipe_baseline_vs_om/mediapipe_predictions.json")
    parser.add_argument(
        "--two-stage",
        default="runs/eval_two_stage_tflite_vs_mediapipe_test/predictions.json",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--save-vis", type=int, default=16)
    return parser.parse_args()


def detection_to_palm(
    detection: Any,
    image_id: int,
    image_name: str,
    width: int,
    height: int,
    palm_index: int,
) -> tuple[dict[str, Any], PalmPrediction]:
    location = detection.location_data
    bbox = location.relative_bounding_box
    box = np.array(
        [
            bbox.xmin * width,
            bbox.ymin * height,
            (bbox.xmin + bbox.width) * width,
            (bbox.ymin + bbox.height) * height,
        ],
        dtype=np.float32,
    )
    keypoints = np.array([[kp.x * width, kp.y * height] for kp in location.relative_keypoints], dtype=np.float32)
    score = float(detection.score[0]) if detection.score else 0.0
    row = {
        "image_id": image_id,
        "image": image_name,
        "palm_index": palm_index,
        "score": score,
        "box": box.astype(float).tolist(),
        "palm7": keypoints.astype(float).tolist(),
    }
    pred = PalmPrediction(image_id=image_id, image_name=image_name, score=score, box=box, palm7=keypoints)
    return row, pred


def normalized_landmarks_to_points(landmarks: Any, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([[lm.x * width, lm.y * height] for lm in landmarks.landmark], dtype=np.float32)
    xyz = np.array([[lm.x * width, lm.y * height, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    return xy, xyz


def world_landmarks_to_list(world_landmarks: Any | None, hand_index: int) -> list[list[float]] | None:
    if world_landmarks is None or hand_index >= len(world_landmarks):
        return None
    return [[float(lm.x), float(lm.y), float(lm.z)] for lm in world_landmarks[hand_index].landmark]


def bbox_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    x1 = float(np.clip(np.min(points[:, 0]), 0, width))
    y1 = float(np.clip(np.min(points[:, 1]), 0, height))
    x2 = float(np.clip(np.max(points[:, 0]), 0, width))
    y2 = float(np.clip(np.max(points[:, 1]), 0, height))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def rect_to_dict(rect: Any | None, width: int, height: int) -> dict[str, float] | None:
    if rect is None:
        return None
    return {
        "x_center": float(rect.x_center),
        "y_center": float(rect.y_center),
        "width": float(rect.width),
        "height": float(rect.height),
        "rotation": float(rect.rotation),
        "x_center_px": float(rect.x_center * width),
        "y_center_px": float(rect.y_center * height),
        "width_px": float(rect.width * width),
        "height_px": float(rect.height * height),
    }


def handedness_to_dict(items: Any | None, hand_index: int) -> dict[str, Any]:
    if items is None or hand_index >= len(items) or not items[hand_index].classification:
        return {"label": "", "score": math.nan, "index": -1}
    classification = items[hand_index].classification[0]
    return {
        "label": str(classification.label),
        "score": float(classification.score),
        "index": int(classification.index),
    }


def norm_from_box(box: np.ndarray) -> float:
    w = max(float(box[2] - box[0]), 1.0)
    h = max(float(box[3] - box[1]), 1.0)
    return max(math.sqrt(w * h), 1.0)


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


def load_reference_predictions(path: Path, image_names: set[str]) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not item.get("hand21") or not item.get("box"):
            continue
        image_name = Path(str(item.get("image", "")).replace("\\", "/")).name
        if image_name not in image_names:
            continue
        grouped.setdefault(image_name, []).append(item)
    return grouped


def match_hand21(
    legacy_hands: list[dict[str, Any]],
    refs_by_image: dict[str, list[dict[str, Any]]],
    match_iou: float,
    prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[float] = []
    norm_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    used_total = 0
    matched = 0
    ref_total = sum(len(v) for v in refs_by_image.values())
    for pred in legacy_hands:
        image_name = Path(str(pred["image"]).replace("\\", "/")).name
        refs = refs_by_image.get(image_name, [])
        used = {idx for idx, ref in enumerate(refs) if ref.get(f"__used_{prefix}")}
        available = [idx for idx in range(len(refs)) if idx not in used]
        best_idx = None
        best_iou = 0.0
        if available:
            pred_box = np.array(pred["box"], dtype=np.float32)
            ref_boxes = np.stack([np.array(refs[idx]["box"], dtype=np.float32) for idx in available], axis=0)
            ious = box_iou(pred_box, ref_boxes)
            best_pos = int(np.argmax(ious))
            best_iou = float(ious[best_pos])
            if best_iou >= match_iou:
                best_idx = available[best_pos]
        if best_idx is None:
            rows.append(
                {
                    "image": image_name,
                    "legacy_hand_index": pred["hand_index"],
                    "reference_index": "",
                    "match_iou": best_iou,
                    "mean_px": math.nan,
                    "max_px": math.nan,
                    "nme": math.nan,
                }
            )
            continue
        refs[best_idx][f"__used_{prefix}"] = True
        used_total += 1
        matched += 1
        ref = refs[best_idx]
        pred_points = np.array(pred["hand21"], dtype=np.float32)
        ref_points = np.array(ref["hand21"], dtype=np.float32)
        err = np.linalg.norm(pred_points - ref_points, axis=1)
        norm = norm_from_box(np.array(ref["box"], dtype=np.float32))
        errors.extend(float(x) for x in err)
        norm_errors.extend(float(x / norm) for x in err)
        rows.append(
            {
                "image": image_name,
                "legacy_hand_index": pred["hand_index"],
                "reference_index": best_idx,
                "match_iou": best_iou,
                "mean_px": float(np.mean(err)),
                "max_px": float(np.max(err)),
                "nme": float(np.mean(err / norm)),
            }
        )

    for refs in refs_by_image.values():
        for ref in refs:
            ref.pop(f"__used_{prefix}", None)

    summary = {
        f"{prefix}_reference_hands": ref_total,
        f"{prefix}_matched_hands": matched,
        f"{prefix}_legacy_unmatched_hands": len(legacy_hands) - matched,
        f"{prefix}_reference_unmatched_hands": ref_total - used_total,
        **metric_summary(errors, norm_errors, prefix),
    }
    return summary, rows


def summarize_times(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean_ms": math.nan,
            f"{prefix}_median_ms": math.nan,
            f"{prefix}_p95_ms": math.nan,
        }
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


def draw_legacy(image: np.ndarray, hands: list[dict[str, Any]], palms: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for palm in palms:
        box = np.array(palm["box"], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(x)) for x in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        for x, y in np.array(palm["palm7"], dtype=np.float32):
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (0, 180, 255), -1)
    for hand in hands:
        points = np.array(hand["hand21"], dtype=np.float32)
        for a, b in HAND_EDGES:
            cv2.line(canvas, tuple(np.round(points[a]).astype(int)), tuple(np.round(points[b]).astype(int)), (255, 80, 0), 1)
        for x, y in points:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (255, 80, 0), -1)
    return canvas


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    palm = summary["legacy_palm_detection"]
    current = summary.get("compare_current_mediapipe", {})
    two_stage = summary.get("compare_two_stage_tflite", {})
    lines = [
        "# Legacy MediaPipe Graph Evaluation",
        "",
        "## Settings",
        "",
        f"- environment: `mediapipe_legacy`",
        f"- MediaPipe version: `{summary['mediapipe_version']}`",
        f"- legacy graph: `{summary['graph_path']}`",
        f"- split: `{summary['split']}`",
        f"- images: `{summary['images']}`",
        f"- max hands: `{summary['max_hands']}`",
        f"- model_complexity: `{summary['model_complexity']}`",
        "",
        "This script runs the legacy `hand_landmark_tracking_cpu.binarypb` graph and exports these graph streams:",
        "",
        "- `palm_detections`",
        "- `hand_rects_from_palm_detections`",
        "- `hand_rects_from_landmarks`",
        "- `multi_hand_landmarks`",
        "",
        "## Palm Detection",
        "",
        "| Predictions | TP | FP | FN | Precision | Recall | AP@0.50 | AP@0.75 | mAP@0.50:0.95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {palm['predictions']} | {palm['tp']} | {palm['fp']} | {palm['fn']} | "
            f"{fmt(palm['precision'])} | {fmt(palm['recall'])} | {fmt(palm['ap@0.50'])} | "
            f"{fmt(palm['ap@0.75'])} | {fmt(palm['map@0.50:0.95'])} |"
        ),
        "",
        "## Landmark Alignment",
        "",
        "| Reference | Reference hands | Matched hands | Legacy unmatched | Reference unmatched | Mean px | Median px | P95 px | NME | PCK@0.05 | PCK@0.10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if current:
        lines.append(
            f"| current MediaPipe Tasks baseline | {current.get('current_reference_hands', 0)} | "
            f"{current.get('current_matched_hands', 0)} | {current.get('current_legacy_unmatched_hands', 0)} | "
            f"{current.get('current_reference_unmatched_hands', 0)} | {fmt(current.get('current_mean_px', math.nan))} | "
            f"{fmt(current.get('current_median_px', math.nan))} | {fmt(current.get('current_p95_px', math.nan))} | "
            f"{fmt(current.get('current_nme', math.nan), 6)} | {fmt(current.get('current_pck@0.05', math.nan))} | "
            f"{fmt(current.get('current_pck@0.10', math.nan))} |"
        )
    if two_stage:
        lines.append(
            f"| current two-stage TFLite pipeline | {two_stage.get('two_stage_reference_hands', 0)} | "
            f"{two_stage.get('two_stage_matched_hands', 0)} | {two_stage.get('two_stage_legacy_unmatched_hands', 0)} | "
            f"{two_stage.get('two_stage_reference_unmatched_hands', 0)} | {fmt(two_stage.get('two_stage_mean_px', math.nan))} | "
            f"{fmt(two_stage.get('two_stage_median_px', math.nan))} | {fmt(two_stage.get('two_stage_p95_px', math.nan))} | "
            f"{fmt(two_stage.get('two_stage_nme', math.nan), 6)} | {fmt(two_stage.get('two_stage_pck@0.05', math.nan))} | "
            f"{fmt(two_stage.get('two_stage_pck@0.10', math.nan))} |"
        )
    lines += [
        "",
        "## Timing",
        "",
        "| Stage | Mean ms | Median ms | P95 ms |",
        "| --- | ---: | ---: | ---: |",
        f"| read and color conversion | {fmt(summary['preprocess_mean_ms'])} | {fmt(summary['preprocess_median_ms'])} | {fmt(summary['preprocess_p95_ms'])} |",
        f"| legacy graph process | {fmt(summary['graph_mean_ms'])} | {fmt(summary['graph_median_ms'])} | {fmt(summary['graph_p95_ms'])} |",
        f"| end to end | {fmt(summary['total_mean_ms'])} | {fmt(summary['total_median_ms'])} | {fmt(summary['total_p95_ms'])} |",
        f"| FPS | {fmt(summary['fps'], 2)} | | |",
        "",
        "## Notes",
        "",
        "- The legacy wheel can run the official graph and export palm detections and ROI streams.",
        "- The legacy weights are not necessarily identical to the current Tasks weights.",
        "- Use this output as graph-calculator reference for `palm_detections`, `hand_rect`, and rotated ROI details.",
        "",
        "## Artifacts",
        "",
        "- `summary.json`",
        "- `legacy_hand_predictions.json`",
        "- `legacy_palm_predictions.json`",
        "- `timings.csv`",
        "- `matches_current_mediapipe.csv`",
        "- `matches_two_stage_tflite.csv`",
        "- `vis/`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_graph(args: argparse.Namespace) -> SolutionBase:
    return SolutionBase(
        binary_graph_path=GRAPH_PATH,
        side_inputs={
            "model_complexity": args.model_complexity,
            "num_hands": args.max_hands,
            "use_prev_landmarks": False,
        },
        calculator_params={
            "palmdetectioncpu__TensorsToDetectionsCalculator.min_score_thresh": args.min_detection_confidence,
            "handlandmarkcpu__ThresholdingCalculator.threshold": args.min_tracking_confidence,
        },
        outputs=GRAPH_OUTPUTS,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data)
    image_paths = list_images(data_root / args.split / "images")
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    image_names = {p.name for p in image_paths}
    targets_by_image = load_targets(data_root, args.split)
    if args.max_images:
        targets_by_image = {k: v for k, v in targets_by_image.items() if k in image_names}

    hand_predictions: list[dict[str, Any]] = []
    palm_predictions_json: list[dict[str, Any]] = []
    palm_predictions_eval: list[PalmPrediction] = []
    timing_rows: list[dict[str, Any]] = []
    preprocess_ms: list[float] = []
    graph_ms: list[float] = []
    total_ms: list[float] = []
    saved_vis = 0

    graph = make_graph(args)
    try:
        for image_id, image_path in enumerate(image_paths):
            total_start = time.perf_counter()
            pre_start = time.perf_counter()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            height, width = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pre_ms = (time.perf_counter() - pre_start) * 1000.0

            graph_start = time.perf_counter()
            result = graph.process({"image": rgb})
            run_ms = (time.perf_counter() - graph_start) * 1000.0

            image_palms: list[dict[str, Any]] = []
            for palm_index, detection in enumerate(result.palm_detections or []):
                palm_json, palm_pred = detection_to_palm(detection, image_id, image_path.name, width, height, palm_index)
                rects = result.hand_rects_from_palm_detections or []
                if palm_index < len(rects):
                    palm_json["hand_rect_from_palm"] = rect_to_dict(rects[palm_index], width, height)
                palm_predictions_json.append(palm_json)
                palm_predictions_eval.append(palm_pred)
                image_palms.append(palm_json)

            image_hands: list[dict[str, Any]] = []
            for hand_index, landmarks in enumerate(result.multi_hand_landmarks or []):
                xy, xyz = normalized_landmarks_to_points(landmarks, width, height)
                rect_from_landmarks = None
                rects_from_landmarks = result.hand_rects_from_landmarks or []
                if hand_index < len(rects_from_landmarks):
                    rect_from_landmarks = rect_to_dict(rects_from_landmarks[hand_index], width, height)
                rect_from_palm = None
                rects_from_palm = result.hand_rects_from_palm_detections or []
                if hand_index < len(rects_from_palm):
                    rect_from_palm = rect_to_dict(rects_from_palm[hand_index], width, height)
                hand = {
                    "image_id": image_id,
                    "image": str(image_path),
                    "hand_index": hand_index,
                    "box": bbox_from_points(xy, width, height).astype(float).tolist(),
                    "hand21": xy.astype(float).tolist(),
                    "hand21_xyz": xyz.astype(float).tolist(),
                    "world_landmarks": world_landmarks_to_list(result.multi_hand_world_landmarks, hand_index),
                    "handedness": handedness_to_dict(result.multi_handedness, hand_index),
                    "hand_rect_from_palm": rect_from_palm,
                    "hand_rect_from_landmarks": rect_from_landmarks,
                }
                hand_predictions.append(hand)
                image_hands.append(hand)

            one_total_ms = (time.perf_counter() - total_start) * 1000.0
            preprocess_ms.append(pre_ms)
            graph_ms.append(run_ms)
            total_ms.append(one_total_ms)
            timing_rows.append(
                {
                    "image": image_path.name,
                    "palms": len(image_palms),
                    "hands": len(image_hands),
                    "preprocess_ms": pre_ms,
                    "graph_ms": run_ms,
                    "total_ms": one_total_ms,
                }
            )

            if args.save_vis and saved_vis < args.save_vis and (image_hands or image_palms):
                cv2.imwrite(str(vis_dir / f"{image_path.stem}.jpg"), draw_legacy(image, image_hands, image_palms))
                saved_vis += 1
    finally:
        graph.close()

    palm_metrics = detection_metrics(palm_predictions_eval, targets_by_image, args.min_detection_confidence, 0.50)

    current_refs = load_reference_predictions(Path(args.current_mediapipe), image_names)
    two_stage_refs = load_reference_predictions(Path(args.two_stage), image_names)
    current_summary, current_rows = match_hand21(hand_predictions, current_refs, args.match_iou, "current")
    two_stage_summary, two_stage_rows = match_hand21(hand_predictions, two_stage_refs, args.match_iou, "two_stage")

    summary = {
        "task": "eval_legacy_mediapipe_graph",
        "mediapipe_version": mp.__version__,
        "graph_path": GRAPH_PATH,
        "data": str(data_root),
        "split": args.split,
        "images": len(image_paths),
        "max_hands": args.max_hands,
        "model_complexity": args.model_complexity,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "match_iou": args.match_iou,
        "legacy_hands": len(hand_predictions),
        "legacy_palms": len(palm_predictions_json),
        "legacy_palm_detection": palm_metrics,
        "compare_current_mediapipe": current_summary,
        "compare_two_stage_tflite": two_stage_summary,
        **summarize_times(preprocess_ms, "preprocess"),
        **summarize_times(graph_ms, "graph"),
        **summarize_times(total_ms, "total"),
        "fps": 1000.0 / max(float(np.mean(total_ms)), 1e-9) if total_ms else math.nan,
        "visualizations": saved_vis,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "legacy_hand_predictions.json").write_text(
        json.dumps(hand_predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "legacy_palm_predictions.json").write_text(
        json.dumps(palm_predictions_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "timings.csv", timing_rows)
    write_csv(output_dir / "matches_current_mediapipe.csv", current_rows)
    write_csv(output_dir / "matches_two_stage_tflite.csv", two_stage_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

