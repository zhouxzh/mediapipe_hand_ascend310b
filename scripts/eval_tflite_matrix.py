#!/usr/bin/env python3
"""Evaluate MediaPipe TFLite detector/landmark version matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import weighted_nms
from hand_pipeline.eval import PalmPrediction
from hand_pipeline.eval import box_iou
from hand_pipeline.eval import detection_metrics
from hand_pipeline.eval import list_images
from hand_pipeline.eval import load_targets
from hand_pipeline.inference import TfliteModel
from hand_pipeline.preprocess import image_to_tensor
from hand_pipeline.roi import landmarks_to_original
from hand_pipeline.roi import make_hand_roi
from hand_pipeline.roi import preprocess_landmark_tflite


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


@dataclass(frozen=True)
class ModelSpec:
    key: str
    role: str
    version: str
    path: Path


@dataclass
class ComboState:
    predictions: list[dict[str, Any]]
    palm_predictions: list[PalmPrediction]
    det_pre_ms: list[float]
    det_infer_ms: list[float]
    det_decode_ms: list[float]
    roi_ms: list[float]
    landmark_ms: list[float]
    landmark_post_ms: list[float]
    total_ms: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/palm_datasets")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-dir", default="models/tflite")
    parser.add_argument("--output-dir", default="runs/eval_tflite_version_matrix_test")
    parser.add_argument("--reference-current", default="runs/mediapipe_baseline_vs_om/mediapipe_predictions.json")
    parser.add_argument(
        "--reference-legacy-full",
        default="runs/eval_legacy_graph_test/legacy_hand_predictions.json",
    )
    parser.add_argument(
        "--reference-legacy-lite",
        default="runs/eval_legacy_graph_lite_test/legacy_hand_predictions.json",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--save-vis", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def detector_specs(model_dir: Path) -> list[ModelSpec]:
    return [
        ModelSpec("task_full", "detector", "mediapipe_task", model_dir / "mediapipe_task_hand_detector_full.tflite"),
        ModelSpec(
            "legacy_full",
            "detector",
            "mediapipe_legacy_0_10_14",
            model_dir / "mediapipe_legacy_0_10_14_palm_detection_full.tflite",
        ),
        ModelSpec(
            "legacy_lite",
            "detector",
            "mediapipe_legacy_0_10_14",
            model_dir / "mediapipe_legacy_0_10_14_palm_detection_lite.tflite",
        ),
    ]


def landmark_specs(model_dir: Path) -> list[ModelSpec]:
    return [
        ModelSpec("task_full", "landmark", "mediapipe_task", model_dir / "mediapipe_task_hand_landmark_full.tflite"),
        ModelSpec(
            "legacy_full",
            "landmark",
            "mediapipe_legacy_0_10_14",
            model_dir / "mediapipe_legacy_0_10_14_hand_landmark_full.tflite",
        ),
        ModelSpec(
            "legacy_lite",
            "landmark",
            "mediapipe_legacy_0_10_14",
            model_dir / "mediapipe_legacy_0_10_14_hand_landmark_lite.tflite",
        ),
    ]


def combo_key(detector: ModelSpec, landmark: ModelSpec) -> str:
    return f"det_{detector.key}__lm_{landmark.key}"


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


def bbox_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    x1 = float(np.clip(np.min(points[:, 0]), 0, width))
    y1 = float(np.clip(np.min(points[:, 1]), 0, height))
    x2 = float(np.clip(np.max(points[:, 0]), 0, width))
    y2 = float(np.clip(np.max(points[:, 1]), 0, height))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


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


def load_reference(path: Path, image_names: set[str]) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not item.get("hand21"):
            continue
        image_name = Path(str(item.get("image", "")).replace("\\", "/")).name
        if image_name not in image_names:
            continue
        box = item.get("box")
        if not box:
            points = np.array(item["hand21"], dtype=np.float32)
            box = bbox_from_points(points, int(np.max(points[:, 0]) + 1), int(np.max(points[:, 1]) + 1)).tolist()
        copied = dict(item)
        copied["box"] = box
        grouped.setdefault(image_name, []).append(copied)
    return grouped


def match_predictions(
    predictions: list[dict[str, Any]],
    refs_by_image: dict[str, list[dict[str, Any]]],
    match_iou: float,
    prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[float] = []
    norm_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    used: dict[str, set[int]] = {name: set() for name in refs_by_image}
    matched = 0
    for pred in predictions:
        image_name = Path(str(pred["image"]).replace("\\", "/")).name
        refs = refs_by_image.get(image_name, [])
        available = [idx for idx in range(len(refs)) if idx not in used.setdefault(image_name, set())]
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
                    "hand_index": pred["hand_index"],
                    "reference_index": "",
                    "match_iou": best_iou,
                    "mean_px": math.nan,
                    "max_px": math.nan,
                    "nme": math.nan,
                    "score": pred.get("score", math.nan),
                    "hand_score": pred.get("hand_score", math.nan),
                }
            )
            continue
        used[image_name].add(best_idx)
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
                "hand_index": pred["hand_index"],
                "reference_index": best_idx,
                "match_iou": best_iou,
                "mean_px": float(np.mean(err)),
                "max_px": float(np.max(err)),
                "nme": float(np.mean(err / norm)),
                "score": pred.get("score", math.nan),
                "hand_score": pred.get("hand_score", math.nan),
            }
        )
    reference_hands = sum(len(v) for v in refs_by_image.values())
    summary = {
        f"{prefix}_reference_hands": reference_hands,
        f"{prefix}_matched_hands": matched,
        f"{prefix}_pred_unmatched_hands": len(predictions) - matched,
        f"{prefix}_reference_unmatched_hands": reference_hands - matched,
        **metric_summary(errors, norm_errors, prefix),
    }
    return summary, rows


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


def draw_predictions(image: np.ndarray, predictions: list[dict[str, Any]], refs: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for ref in refs:
        points = np.array(ref["hand21"], dtype=np.float32)
        for a, b in HAND_EDGES:
            cv2.line(canvas, tuple(np.round(points[a]).astype(int)), tuple(np.round(points[b]).astype(int)), (0, 180, 255), 2)
        for x, y in points:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (0, 180, 255), -1)
    for pred in predictions:
        box = np.array(pred["box"], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(x)) for x in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        points = np.array(pred["hand21"], dtype=np.float32)
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


def model_manifest(detectors: list[ModelSpec], landmarks: list[ModelSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in [*detectors, *landmarks]:
        rows.append(
            {
                "key": spec.key,
                "role": spec.role,
                "version": spec.version,
                "path": str(spec.path),
                "file": spec.path.name,
                "size_bytes": spec.path.stat().st_size if spec.path.exists() else 0,
                "sha256": sha256(spec.path) if spec.path.exists() else "",
            }
        )
    return rows


def empty_state() -> ComboState:
    return ComboState([], [], [], [], [], [], [], [], [])


def run_combo(
    args: argparse.Namespace,
    detector_spec: ModelSpec,
    landmark_spec: ModelSpec,
    image_paths: list[Path],
    targets_by_image: dict[str, Any],
    refs_by_name: dict[str, dict[str, list[dict[str, Any]]]],
    anchors: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    key = combo_key(detector_spec, landmark_spec)
    combo_dir = output_dir / key
    combo_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = combo_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    detector = TfliteModel(detector_spec.path, num_threads=args.num_threads)
    landmark = TfliteModel(landmark_spec.path, num_threads=args.num_threads)
    state = empty_state()
    timing_rows: list[dict[str, Any]] = []
    saved_vis = 0

    for image_id, image_path in enumerate(image_paths):
        total_start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        pre_start = time.perf_counter()
        tensor, letterbox = image_to_tensor(image, input_size=192)
        det_pre = (time.perf_counter() - pre_start) * 1000.0

        det_start = time.perf_counter()
        raw_boxes, raw_scores = detector(tensor)
        det_infer = (time.perf_counter() - det_start) * 1000.0

        decode_start = time.perf_counter()
        palms = decode_raw_palm(raw_boxes, raw_scores, anchors, letterbox, score_threshold=args.score_threshold)
        palms = weighted_nms(palms, iou_threshold=args.nms_iou, max_detections=args.max_det)
        palms = sorted(palms, key=lambda x: x.score, reverse=True)[: args.max_hands]
        det_decode = (time.perf_counter() - decode_start) * 1000.0

        image_predictions: list[dict[str, Any]] = []
        for hand_index, palm in enumerate(palms):
            state.palm_predictions.append(
                PalmPrediction(
                    image_id=image_id,
                    image_name=image_path.name,
                    score=float(palm.score),
                    box=palm.box.astype(np.float32),
                    palm7=palm.keypoints.astype(np.float32),
                )
            )
            roi_start = time.perf_counter()
            roi = make_hand_roi(image, palm.box, palm.keypoints)
            lm_tensor = preprocess_landmark_tflite(roi.crop)
            roi_ms = (time.perf_counter() - roi_start) * 1000.0

            lm_start = time.perf_counter()
            lm_outputs = landmark(lm_tensor)
            lm_infer = (time.perf_counter() - lm_start) * 1000.0

            post_start = time.perf_counter()
            lm_crop, hand_score, handedness, _world = pick_landmark_outputs(lm_outputs)
            hand21 = landmarks_to_original(lm_crop, roi.inverse, input_size=224, coord_scale="auto")
            hand_box = bbox_from_points(hand21, image.shape[1], image.shape[0])
            lm_post = (time.perf_counter() - post_start) * 1000.0
            pred = {
                "image_id": image_id,
                "image": str(image_path),
                "hand_index": hand_index,
                "score": float(palm.score),
                "hand_score": float(hand_score),
                "handedness": float(handedness),
                "box": hand_box.astype(float).tolist(),
                "palm_box": palm.box.astype(float).tolist(),
                "palm7": palm.keypoints.astype(float).tolist(),
                "hand21": hand21.astype(float).tolist(),
                "roi_center": roi.center.astype(float).tolist(),
                "roi_size": float(roi.size),
                "roi_rotation_rad": float(roi.rotation),
            }
            state.predictions.append(pred)
            image_predictions.append(pred)
            state.roi_ms.append(roi_ms)
            state.landmark_ms.append(lm_infer)
            state.landmark_post_ms.append(lm_post)

        total_ms = (time.perf_counter() - total_start) * 1000.0
        state.det_pre_ms.append(det_pre)
        state.det_infer_ms.append(det_infer)
        state.det_decode_ms.append(det_decode)
        state.total_ms.append(total_ms)
        timing_rows.append(
            {
                "image": image_path.name,
                "hands": len(image_predictions),
                "det_preprocess_ms": det_pre,
                "det_infer_ms": det_infer,
                "det_decode_ms": det_decode,
                "total_ms": total_ms,
            }
        )
        if args.save_vis and saved_vis < args.save_vis and image_predictions:
            current_refs = refs_by_name.get("current_tasks", {}).get(image_path.name, [])
            cv2.imwrite(str(vis_dir / f"{image_path.stem}.jpg"), draw_predictions(image, image_predictions, current_refs))
            saved_vis += 1

    palm = detection_metrics(state.palm_predictions, targets_by_image, args.score_threshold, 0.50)
    summary: dict[str, Any] = {
        "combo": key,
        "data": args.data,
        "split": args.split,
        "detector_key": detector_spec.key,
        "landmark_key": landmark_spec.key,
        "detector_version": detector_spec.version,
        "landmark_version": landmark_spec.version,
        "detector": str(detector_spec.path),
        "landmark": str(landmark_spec.path),
        "images": len(image_paths),
        "hands": len(state.predictions),
        "palm_predictions": len(state.palm_predictions),
        "palm_tp@0.50": palm["tp"],
        "palm_fp@0.50": palm["fp"],
        "palm_fn@0.50": palm["fn"],
        "palm_precision@0.50": palm["precision"],
        "palm_recall@0.50": palm["recall"],
        "palm_ap@0.50": palm["ap@0.50"],
        "palm_ap@0.75": palm["ap@0.75"],
        "palm_map@0.50:0.95": palm["map@0.50:0.95"],
        "palm_recall@0.10": palm["operating_iou_sweep"]["0.10"]["recall"],
        "visualizations": saved_vis,
        **summarize_times(state.det_pre_ms, "det_preprocess"),
        **summarize_times(state.det_infer_ms, "det_infer"),
        **summarize_times(state.det_decode_ms, "det_decode"),
        **summarize_times(state.roi_ms, "roi_preprocess"),
        **summarize_times(state.landmark_ms, "landmark_infer"),
        **summarize_times(state.landmark_post_ms, "landmark_post"),
        **summarize_times(state.total_ms, "total"),
    }
    summary["fps"] = 1000.0 / max(summary["total_mean_ms"], 1e-9)

    for ref_name, refs in refs_by_name.items():
        ref_summary, rows = match_predictions(state.predictions, refs, args.match_iou, ref_name)
        summary.update(ref_summary)
        write_csv(combo_dir / f"matches_{ref_name}.csv", rows)

    (combo_dir / "predictions.json").write_text(
        json.dumps(state.predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (combo_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(combo_dir / "timings.csv", timing_rows)
    return summary


def best_by(summaries: list[dict[str, Any]], key: str, reverse: bool = False) -> dict[str, Any] | None:
    values = [item for item in summaries if isinstance(item.get(key), (int, float)) and not math.isnan(float(item[key]))]
    if not values:
        return None
    return sorted(values, key=lambda x: float(x[key]), reverse=reverse)[0]


def write_report(
    path: Path,
    summaries: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    refs: dict[str, Any],
    data: str,
    split: str,
) -> None:
    lines = [
        "# MediaPipe TFLite Model Matrix",
        "",
        "## Goal",
        "",
        "This report evaluates detector/landmark combinations with the same Python two-stage pipeline.",
        "",
        "## Dataset",
        "",
        f"- data: `{data}`",
        f"- split: `{split}`",
        "",
        "## Models",
        "",
        "| key | role | version | file | size bytes | SHA256 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for item in manifest:
        lines.append(
            f"| `{item['key']}` | `{item['role']}` | `{item['version']}` | `{item['file']}` | "
            f"{item['size_bytes']} | `{item['sha256']}` |"
        )
    lines += [
        "",
        "## References",
        "",
        "| Reference | Path | Hands |",
        "| --- | --- | ---: |",
    ]
    for name, info in refs.items():
        lines.append(f"| `{name}` | `{info['path']}` | {info['hands']} |")
    lines += [
        "",
        "## Palm Detector",
        "",
        "| Combo | Detector | Landmark | Palm count | TP@0.50 | FP@0.50 | FN@0.50 | Precision | Recall | Recall@0.10 | mAP@0.50:0.95 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item['combo']}` | `{item['detector_key']}` | `{item['landmark_key']}` | {item['palm_predictions']} | "
            f"{item['palm_tp@0.50']} | {item['palm_fp@0.50']} | {item['palm_fn@0.50']} | "
            f"{fmt(item['palm_precision@0.50'])} | {fmt(item['palm_recall@0.50'])} | "
            f"{fmt(item['palm_recall@0.10'])} | {fmt(item['palm_map@0.50:0.95'])} |"
        )
    lines += [
        "",
        "## Landmark Consistency",
        "",
        "| Combo | Hands | Current Tasks mean px | Current PCK@0.05 | Legacy full mean px | Legacy full PCK@0.05 | Legacy lite mean px | Legacy lite PCK@0.05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item['combo']}` | {item['hands']} | "
            f"{fmt(item.get('current_tasks_mean_px', math.nan))} | {fmt(item.get('current_tasks_pck@0.05', math.nan))} | "
            f"{fmt(item.get('legacy_full_mean_px', math.nan))} | {fmt(item.get('legacy_full_pck@0.05', math.nan))} | "
            f"{fmt(item.get('legacy_lite_mean_px', math.nan))} | {fmt(item.get('legacy_lite_pck@0.05', math.nan))} |"
        )
    lines += [
        "",
        "## Timing",
        "",
        "| Combo | total_mean_ms | FPS | detector ms | landmark ms | ROI+post ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        roi_post = float(item["roi_preprocess_mean_ms"]) + float(item["landmark_post_mean_ms"])
        lines.append(
            f"| `{item['combo']}` | {fmt(item['total_mean_ms'])} | {fmt(item['fps'], 2)} | "
            f"{fmt(item['det_infer_mean_ms'])} | {fmt(item['landmark_infer_mean_ms'])} | {fmt(roi_post)} |"
        )

    lines += ["", "## Best Combinations", ""]
    for ref_name, label in (
        ("current_tasks", "current MediaPipe Tasks"),
        ("legacy_full", "legacy graph full"),
        ("legacy_lite", "legacy graph lite"),
    ):
        best = best_by(summaries, f"{ref_name}_mean_px")
        if best:
            lines.append(
                f"- Best for `{label}`: `{best['combo']}`, "
                f"mean `{fmt(best[f'{ref_name}_mean_px'])} px`, "
                f"PCK@0.05 `{fmt(best[f'{ref_name}_pck@0.05'])}`."
            )
    speed_best = best_by(summaries, "total_mean_ms")
    if speed_best:
        lines.append(f"- Fastest local CPU/LiteRT combo: `{speed_best['combo']}`, total `{fmt(speed_best['total_mean_ms'])} ms`.")

    lines += [
        "",
        "## Engineering Notes",
        "",
        "- Use one coherent detector + landmark version when claiming compatibility with a MediaPipe release.",
        "- Compare full/lite speed and accuracy only within the same run.",
        "- For Ascend 310B deployment, first convert and validate the selected full baseline, then evaluate lite or INT8.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detectors = detector_specs(model_dir)
    landmarks = landmark_specs(model_dir)
    for spec in [*detectors, *landmarks]:
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)

    image_paths = list_images(Path(args.data) / args.split / "images")
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    image_names = {path.name for path in image_paths}
    targets_by_image = load_targets(Path(args.data), args.split)
    if args.max_images:
        targets_by_image = {name: targets for name, targets in targets_by_image.items() if name in image_names}

    reference_paths = {
        "current_tasks": Path(args.reference_current),
        "legacy_full": Path(args.reference_legacy_full),
        "legacy_lite": Path(args.reference_legacy_lite),
    }
    refs_by_name = {name: load_reference(path, image_names) for name, path in reference_paths.items() if path.exists()}
    ref_info = {
        name: {"path": str(reference_paths[name]), "hands": sum(len(v) for v in refs.values())}
        for name, refs in refs_by_name.items()
    }
    anchors = generate_palm_anchors()
    manifest = model_manifest(detectors, landmarks)
    summaries: list[dict[str, Any]] = []
    for detector in detectors:
        for landmark in landmarks:
            print(f"Running {combo_key(detector, landmark)}", flush=True)
            summaries.append(run_combo(args, detector, landmark, image_paths, targets_by_image, refs_by_name, anchors, output_dir))

    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "summary.csv", summaries)
    write_report(output_dir / "deep_analysis.md", summaries, manifest, ref_info, args.data, args.split)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "data": args.data, "split": args.split, "combos": len(summaries), "references": ref_info},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

