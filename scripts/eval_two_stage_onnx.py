#!/usr/bin/env python3
"""Evaluate the two-stage MediaPipe hand pipeline with ONNX Runtime models."""

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
PARENT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import weighted_nms
from hand_pipeline.eval import PalmPrediction
from hand_pipeline.eval import detection_metrics
from hand_pipeline.eval import load_targets
from hand_pipeline.inference import OnnxModel
from hand_pipeline.preprocess import image_to_tensor
from hand_pipeline.roi import landmarks_to_original
from hand_pipeline.roi import make_hand_roi
from hand_pipeline.roi import preprocess_landmark_tflite

from scripts.eval_two_stage_tflite import HAND_EDGES
from scripts.eval_two_stage_tflite import draw_compare
from scripts.eval_two_stage_tflite import list_images
from scripts.eval_two_stage_tflite import match_to_official
from scripts.eval_two_stage_tflite import metric_summary
from scripts.eval_two_stage_tflite import norm_from_box
from scripts.eval_two_stage_tflite import pick_landmark_outputs
from scripts.eval_two_stage_tflite import summarize_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="", help="Palm dataset root. Defaults to data/palm_datasets or ../data/palm_datasets.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--detector", default="models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx")
    parser.add_argument("--landmark", default="models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx")
    parser.add_argument("--reference-tflite", default="runs/baseline/two_stage_vs_legacy_graph/predictions.json")
    parser.add_argument("--reference-legacy", default="runs/baseline/legacy_graph/legacy_hand_predictions.json")
    parser.add_argument("--reference-current", default="references/current_tasks/mediapipe_predictions.json")
    parser.add_argument("--output-dir", default="runs/onnx_two_stage/legacy_full")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--save-vis", type=int, default=0)
    parser.add_argument("--roi-scale", type=float, default=2.6)
    parser.add_argument("--shift-y", type=float, default=-0.5)
    parser.add_argument("--rotation-offset-degrees", type=float, default=0.0)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_dataset(path_text: str) -> Path:
    if path_text:
        return resolve_path(path_text)
    for candidate in (PROJECT_ROOT / "data" / "palm_datasets", PARENT_ROOT / "data" / "palm_datasets"):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Cannot find palm dataset. Tried data/palm_datasets and ../data/palm_datasets.")


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
        grouped.setdefault(image_name, []).append(item)
    return grouped


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


def compare_to_reference(
    predictions_by_image: dict[str, list[dict[str, Any]]],
    refs_by_image: dict[str, list[dict[str, Any]]],
    match_iou: float,
    prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    point_errors: list[float] = []
    norm_errors: list[float] = []
    match_rows: list[dict[str, Any]] = []
    matched = 0
    pred_unmatched = 0
    refs_total = sum(len(items) for items in refs_by_image.values())
    refs_used_total = 0
    for image_name, preds in predictions_by_image.items():
        refs = refs_by_image.get(image_name, [])
        used: set[int] = set()
        for pred in preds:
            ref_idx, iou = match_to_official(pred, refs, used, match_iou)
            if ref_idx is None:
                pred_unmatched += 1
                continue
            matched += 1
            ref = refs[ref_idx]
            pred_points = np.array(pred["hand21"], dtype=np.float32)
            ref_points = np.array(ref["hand21"], dtype=np.float32)
            err = np.linalg.norm(pred_points - ref_points, axis=1)
            norm = norm_from_box(np.array(ref["box"], dtype=np.float32))
            point_errors.extend(float(item) for item in err)
            norm_errors.extend(float(item / norm) for item in err)
            match_rows.append(
                {
                    "reference": prefix,
                    "image": image_name,
                    "hand_index": pred.get("hand_index", -1),
                    "reference_index": ref_idx,
                    "match_iou": iou,
                    "mean_px": float(np.mean(err)),
                    "max_px": float(np.max(err)),
                    "nme": float(np.mean(err / norm)),
                    "score": pred.get("score", math.nan),
                    "hand_score": pred.get("hand_score", math.nan),
                }
            )
        refs_used_total += len(used)
    metrics = metric_summary(point_errors, norm_errors, prefix)
    metrics.update(
        {
            f"{prefix}_reference_hands": refs_total,
            f"{prefix}_matched_hands": matched,
            f"{prefix}_pred_unmatched_hands": pred_unmatched,
            f"{prefix}_reference_unmatched_hands": refs_total - refs_used_total,
        }
    )
    return metrics, match_rows


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Two-stage ONNX Hand Pipeline Evaluation",
        "",
        "## Settings",
        "",
        f"- data: `{summary['data']}`",
        f"- split: `{summary['split']}`",
        f"- images: `{summary['images']}`",
        f"- detector: `{summary['detector']}`",
        f"- landmark: `{summary['landmark']}`",
        "",
        "## Palm Detector GT Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| precision | {fmt(summary['palm_precision'], 6)} |",
        f"| recall | {fmt(summary['palm_recall'], 6)} |",
        f"| AP@0.50 | {fmt(summary['palm_ap@0.50'], 6)} |",
        f"| mAP@0.50:0.95 | {fmt(summary['palm_map@0.50:0.95'], 6)} |",
        "",
        "## End-to-end Landmark Comparisons",
        "",
        "| Reference | Matched | Mean px | P95 px | NME | PCK@0.05 | PCK@0.10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prefix in ("vs_tflite", "vs_legacy", "vs_current"):
        if f"{prefix}_reference_hands" not in summary:
            continue
        lines.append(
            f"| `{prefix}` | {summary[f'{prefix}_matched_hands']} | "
            f"{fmt(summary[f'{prefix}_mean_px'])} | {fmt(summary[f'{prefix}_p95_px'])} | "
            f"{fmt(summary[f'{prefix}_nme'], 6)} | {fmt(summary[f'{prefix}_pck@0.05'])} | "
            f"{fmt(summary[f'{prefix}_pck@0.10'])} |"
        )
    lines.extend(
        [
            "",
            "## Timing",
            "",
            "| Stage | Mean ms | Median ms | P95 ms |",
            "| --- | ---: | ---: | ---: |",
            f"| detector preprocess | {fmt(summary['det_preprocess_mean_ms'])} | {fmt(summary['det_preprocess_median_ms'])} | {fmt(summary['det_preprocess_p95_ms'])} |",
            f"| detector inference | {fmt(summary['det_infer_mean_ms'])} | {fmt(summary['det_infer_median_ms'])} | {fmt(summary['det_infer_p95_ms'])} |",
            f"| detector decode | {fmt(summary['det_decode_mean_ms'])} | {fmt(summary['det_decode_median_ms'])} | {fmt(summary['det_decode_p95_ms'])} |",
            f"| ROI preprocess | {fmt(summary['roi_preprocess_mean_ms'])} | {fmt(summary['roi_preprocess_median_ms'])} | {fmt(summary['roi_preprocess_p95_ms'])} |",
            f"| landmark inference | {fmt(summary['landmark_infer_mean_ms'])} | {fmt(summary['landmark_infer_median_ms'])} | {fmt(summary['landmark_infer_p95_ms'])} |",
            f"| landmark postprocess | {fmt(summary['landmark_post_mean_ms'])} | {fmt(summary['landmark_post_median_ms'])} | {fmt(summary['landmark_post_p95_ms'])} |",
            f"| end to end | {fmt(summary['total_mean_ms'])} | {fmt(summary['total_median_ms'])} | {fmt(summary['total_p95_ms'])} |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = resolve_dataset(args.data)
    detector_path = resolve_path(args.detector)
    landmark_path = resolve_path(args.landmark)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(data_root / args.split / "images")
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    image_names = {path.name for path in image_paths}
    targets_by_image = load_targets(data_root, args.split)
    targets_by_image = {name: targets_by_image.get(name, []) for name in image_names}
    refs = {
        "vs_tflite": load_reference(resolve_path(args.reference_tflite), image_names),
        "vs_legacy": load_reference(resolve_path(args.reference_legacy), image_names),
        "vs_current": load_reference(resolve_path(args.reference_current), image_names),
    }

    detector = OnnxModel(detector_path)
    landmark = OnnxModel(landmark_path)
    anchors = generate_palm_anchors()
    palm_predictions: list[PalmPrediction] = []
    predictions: list[dict[str, Any]] = []
    predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    det_pre_ms: list[float] = []
    det_infer_ms: list[float] = []
    det_decode_ms: list[float] = []
    roi_ms: list[float] = []
    landmark_infer_ms: list[float] = []
    landmark_post_ms: list[float] = []
    total_ms: list[float] = []
    saved_vis = 0

    for image_id, image_path in enumerate(image_paths):
        total_start = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        pre_start = time.perf_counter()
        tensor, letterbox = image_to_tensor(image, input_size=192)
        det_pre_ms.append((time.perf_counter() - pre_start) * 1000.0)

        infer_start = time.perf_counter()
        raw_boxes, raw_scores = detector(tensor)
        det_infer_ms.append((time.perf_counter() - infer_start) * 1000.0)

        decode_start = time.perf_counter()
        palms = decode_raw_palm(raw_boxes, raw_scores, anchors, letterbox, score_threshold=args.score_threshold)
        palms = weighted_nms(palms, iou_threshold=args.nms_iou, max_detections=args.max_det)
        det_decode_ms.append((time.perf_counter() - decode_start) * 1000.0)

        for palm in palms:
            palm_predictions.append(
                PalmPrediction(
                    image_id=image_id,
                    image_name=image_path.name,
                    score=float(palm.score),
                    box=palm.box.copy(),
                    palm7=palm.keypoints.copy(),
                )
            )

        image_predictions: list[dict[str, Any]] = []
        for hand_index, palm in enumerate(sorted(palms, key=lambda item: item.score, reverse=True)[: args.max_hands]):
            roi_start = time.perf_counter()
            roi = make_hand_roi(
                image,
                palm.box,
                palm.keypoints,
                scale=args.roi_scale,
                shift_y=args.shift_y,
                rotation_offset_degrees=args.rotation_offset_degrees,
            )
            lm_tensor = preprocess_landmark_tflite(roi.crop)
            roi_ms.append((time.perf_counter() - roi_start) * 1000.0)

            lm_start = time.perf_counter()
            lm_outputs = landmark(lm_tensor)
            landmark_infer_ms.append((time.perf_counter() - lm_start) * 1000.0)

            post_start = time.perf_counter()
            lm_crop, hand_score, handedness, _world = pick_landmark_outputs(lm_outputs)
            if not math.isnan(hand_score) and hand_score < args.min_hand_score:
                landmark_post_ms.append((time.perf_counter() - post_start) * 1000.0)
                continue
            hand21 = landmarks_to_original(lm_crop, roi.inverse, input_size=224, coord_scale="auto")
            landmark_post_ms.append((time.perf_counter() - post_start) * 1000.0)

            pred = {
                "image_id": image_id,
                "image": str(image_path),
                "hand_index": hand_index,
                "score": float(palm.score),
                "hand_score": float(hand_score),
                "handedness": float(handedness),
                "box": palm.box.astype(float).tolist(),
                "palm7": palm.keypoints.astype(float).tolist(),
                "hand21": hand21.astype(float).tolist(),
                "roi_center": roi.center.astype(float).tolist(),
                "roi_size": float(roi.size),
                "roi_rotation_rad": float(roi.rotation),
            }
            predictions.append(pred)
            image_predictions.append(pred)
        predictions_by_image[image_path.name] = image_predictions
        total_ms.append((time.perf_counter() - total_start) * 1000.0)
        if args.save_vis and saved_vis < args.save_vis and image_predictions:
            canvas = draw_compare(image, image_predictions, refs["vs_tflite"].get(image_path.name, []))
            cv2.imwrite(str(vis_dir / f"{image_path.stem}.jpg"), canvas)
            saved_vis += 1

    palm = detection_metrics(palm_predictions, targets_by_image, conf=args.score_threshold, iou_threshold=0.5)
    summary: dict[str, Any] = {
        "task": "eval_two_stage_onnx",
        "data": str(data_root),
        "split": args.split,
        "images": len(image_paths),
        "detector": str(detector_path),
        "landmark": str(landmark_path),
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "max_hands": args.max_hands,
        "min_hand_score": args.min_hand_score,
        "two_stage_hands": len(predictions),
        "visualizations": saved_vis,
        "palm_predictions": len(palm_predictions),
        "palm_gt_targets": palm["gt_targets"],
        "palm_precision": palm["precision"],
        "palm_recall": palm["recall"],
        "palm_ap@0.50": palm["ap@0.50"],
        "palm_map@0.50:0.95": palm["map@0.50:0.95"],
        **summarize_times(det_pre_ms, "det_preprocess"),
        **summarize_times(det_infer_ms, "det_infer"),
        **summarize_times(det_decode_ms, "det_decode"),
        **summarize_times(roi_ms, "roi_preprocess"),
        **summarize_times(landmark_infer_ms, "landmark_infer"),
        **summarize_times(landmark_post_ms, "landmark_post"),
        **summarize_times(total_ms, "total"),
    }

    all_match_rows: list[dict[str, Any]] = []
    for prefix, refs_by_image in refs.items():
        if not refs_by_image:
            continue
        metrics, rows = compare_to_reference(predictions_by_image, refs_by_image, args.match_iou, prefix)
        summary.update(metrics)
        all_match_rows.extend(rows)

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "matches.csv", all_match_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
