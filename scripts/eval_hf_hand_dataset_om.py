#!/usr/bin/env python3
"""Evaluate Ascend OM hand models on the portable HaGRIDv2 MediaPipe dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pyarrow.parquet as pq

from hand_pipeline.eval import IOU_THRESHOLDS
from hand_pipeline.eval import PalmPrediction
from hand_pipeline.eval import PalmTarget
from hand_pipeline.eval import box_iou
from hand_pipeline.eval import detection_metrics
from hand_pipeline.two_stage import OmHandPipeline
from hand_pipeline.two_stage import OnnxHandPipeline
from hand_pipeline.visualization import draw_hand_predictions


DEFAULT_DATASET = "data/portable-hagridv2-mediapipe-hand/test-00000.parquet"
DEFAULT_OUTPUT_ROOT = "runs/hf_hand_dataset_om"

MODEL_SETS = {
    "full": {
        "om_detector": "models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om",
        "om_landmark": "models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om",
        "onnx_detector": "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx",
        "onnx_landmark": "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx",
        "required": True,
    },
    "lite": {
        "om_detector": "models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype_ascend310b4_singlethread.om",
        "om_landmark": "models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om",
        "onnx_detector": "models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx",
        "onnx_landmark": "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx",
        "required": False,
    },
}


@dataclass(frozen=True)
class HandTarget:
    image_id: int
    image_name: str
    target_index: int
    box: np.ndarray
    palm7: np.ndarray
    hand21: np.ndarray
    gesture_label: str


@dataclass(frozen=True)
class DatasetRow:
    image_id: int
    image_name: str
    gesture_label: str
    width: int
    height: int
    image_bgr: np.ndarray
    targets: list[HandTarget]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def summarize(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
    }


def summarize_ms(values: list[float], prefix: str) -> dict[str, float]:
    base = summarize(values, prefix)
    return {
        f"{prefix}_mean_ms": base[f"{prefix}_mean"],
        f"{prefix}_median_ms": base[f"{prefix}_median"],
        f"{prefix}_p95_ms": base[f"{prefix}_p95"],
        f"{prefix}_max_ms": base[f"{prefix}_max"],
    }


def normalize_model_set(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("--model-set cannot be empty")
    unknown = [name for name in names if name not in MODEL_SETS]
    if unknown:
        raise ValueError(f"Unknown model set(s): {unknown}. Available: {sorted(MODEL_SETS)}")
    return names


def xyxy_normalized_to_pixels(values: Any, width: int, height: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(4)
    scale = np.array([width, height, width, height], dtype=np.float32)
    out = arr * scale
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, float(width))
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, float(height))
    return out


def keypoints_normalized_to_pixels(values: Any, width: int, height: int, count: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(count, 2)
    arr[:, 0] *= width
    arr[:, 1] *= height
    arr[:, 0] = np.clip(arr[:, 0], 0.0, float(width))
    arr[:, 1] = np.clip(arr[:, 1], 0.0, float(height))
    return arr


def decode_image(image_value: dict[str, Any], image_name: str) -> np.ndarray:
    payload = image_value.get("bytes")
    if not payload:
        raise ValueError(f"Missing embedded JPEG bytes for {image_name}")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode embedded JPEG for {image_name}")
    return image


def parse_targets(row: dict[str, Any], image_id: int) -> list[HandTarget]:
    width = int(row["width"])
    height = int(row["height"])
    image_name = str(row["file_name"])
    gesture_label = str(row.get("gesture_label") or "")
    instances = json.loads(row["instances"] or "[]")
    targets: list[HandTarget] = []
    for target_index, instance in enumerate(instances):
        if not all(key in instance for key in ("palm_bbox_xyxy", "palm7_keypoints", "full21_keypoints")):
            continue
        targets.append(
            HandTarget(
                image_id=image_id,
                image_name=image_name,
                target_index=target_index,
                box=xyxy_normalized_to_pixels(instance["palm_bbox_xyxy"], width, height),
                palm7=keypoints_normalized_to_pixels(instance["palm7_keypoints"], width, height, 7),
                hand21=keypoints_normalized_to_pixels(instance["full21_keypoints"], width, height, 21),
                gesture_label=gesture_label,
            )
        )
    return targets


def iter_dataset_rows(dataset_path: Path, max_images: int = 0) -> Any:
    columns = ["image", "file_name", "width", "height", "instances", "gesture_label"]
    table = pq.read_table(dataset_path, columns=columns)
    rows = table.to_pylist()
    if max_images:
        rows = rows[:max_images]
    for image_id, row in enumerate(rows):
        image_name = str(row["file_name"])
        image = decode_image(row["image"], image_name)
        width = int(row["width"])
        height = int(row["height"])
        if image.shape[1] != width or image.shape[0] != height:
            raise ValueError(f"{image_name} shape {image.shape[1]}x{image.shape[0]} != metadata {width}x{height}")
        yield DatasetRow(
            image_id=image_id,
            image_name=image_name,
            gesture_label=str(row.get("gesture_label") or ""),
            width=width,
            height=height,
            image_bgr=image,
            targets=parse_targets(row, image_id),
        )


def targets_by_image(rows: list[DatasetRow]) -> dict[str, list[PalmTarget]]:
    result: dict[str, list[PalmTarget]] = {}
    for row in rows:
        result[row.image_name] = [
            PalmTarget(
                image_id=target.image_id,
                image_name=target.image_name,
                target_index=target.target_index,
                box=target.box,
                palm7=target.palm7,
            )
            for target in row.targets
        ]
    return result


def predictions_to_palm(predictions: list[dict[str, Any]], row: DatasetRow) -> list[PalmPrediction]:
    output: list[PalmPrediction] = []
    for pred in predictions:
        output.append(
            PalmPrediction(
                image_id=row.image_id,
                image_name=row.image_name,
                score=float(pred["score"]),
                box=np.asarray(pred["box"], dtype=np.float32),
                palm7=np.asarray(pred["palm7"], dtype=np.float32),
            )
        )
    return output


def match_targets_to_predictions(
    targets: list[HandTarget],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    if not targets and not predictions:
        return rows, 0, 0
    if not targets:
        return rows, 0, len(predictions)
    if not predictions:
        return rows, len(targets), 0

    pred_boxes = np.stack([np.asarray(pred["box"], dtype=np.float32) for pred in predictions], axis=0)
    used_pred: set[int] = set()
    ordered_targets = sorted(targets, key=lambda item: item.target_index)
    for target in ordered_targets:
        ious = box_iou(target.box.astype(np.float32), pred_boxes)
        order = np.argsort(ious)[::-1]
        matched = None
        for pos in order:
            pred_index = int(pos)
            if pred_index in used_pred:
                continue
            if float(ious[pred_index]) >= iou_threshold:
                matched = pred_index
                break
        if matched is None:
            continue
        used_pred.add(matched)
        pred = predictions[matched]
        pred_palm7 = np.asarray(pred["palm7"], dtype=np.float32)
        pred_hand21 = np.asarray(pred["hand21"], dtype=np.float32)
        palm7_err = np.linalg.norm(pred_palm7 - target.palm7.astype(np.float32), axis=1)
        hand21_err = np.linalg.norm(pred_hand21 - target.hand21.astype(np.float32), axis=1)
        box_abs = np.abs(np.asarray(pred["box"], dtype=np.float32) - target.box.astype(np.float32))
        norm = max(float(np.linalg.norm(target.box[[2, 3]] - target.box[[0, 1]])), 1e-6)
        rows.append(
            {
                "image_id": target.image_id,
                "image_name": target.image_name,
                "target_index": target.target_index,
                "pred_index": matched,
                "gesture_label": target.gesture_label,
                "match_iou": float(ious[matched]),
                "score": float(pred["score"]),
                "hand_score": float(pred.get("hand_score", math.nan)),
                "box_mean_abs_px": float(np.mean(box_abs)),
                "box_max_abs_px": float(np.max(box_abs)),
                "palm7_mean_px": float(np.mean(palm7_err)),
                "palm7_p95_px": float(np.percentile(palm7_err, 95)),
                "palm7_max_px": float(np.max(palm7_err)),
                "palm7_nme": float(np.mean(palm7_err) / norm),
                "palm7_pck_001": float(np.mean(palm7_err <= 0.01 * norm)),
                "palm7_pck_005": float(np.mean(palm7_err <= 0.05 * norm)),
                "full21_mean_px": float(np.mean(hand21_err)),
                "full21_p95_px": float(np.percentile(hand21_err, 95)),
                "full21_max_px": float(np.max(hand21_err)),
                "full21_nme": float(np.mean(hand21_err) / norm),
                "full21_pck_001": float(np.mean(hand21_err <= 0.01 * norm)),
                "full21_pck_005": float(np.mean(hand21_err <= 0.05 * norm)),
            }
        )
    return rows, len(targets) - len(rows), len(predictions) - len(used_pred)


def compare_prediction_sets(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    match_iou: float,
) -> tuple[list[dict[str, Any]], int, int]:
    by_image_right: dict[str, list[dict[str, Any]]] = {}
    for row in right_rows:
        by_image_right.setdefault(str(row["image_name"]), []).append(row)
    rows: list[dict[str, Any]] = []
    right_used: set[tuple[str, int]] = set()
    left_unmatched = 0
    for left in left_rows:
        candidates = by_image_right.get(str(left["image_name"]), [])
        if not candidates:
            left_unmatched += 1
            continue
        left_box = np.asarray(left["box"], dtype=np.float32)
        boxes = np.stack([np.asarray(item["box"], dtype=np.float32) for item in candidates], axis=0)
        ious = box_iou(left_box, boxes)
        order = np.argsort(ious)[::-1]
        matched = None
        for pos in order:
            idx = int(pos)
            key = (str(left["image_name"]), int(candidates[idx]["pred_index"]))
            if key in right_used:
                continue
            if float(ious[idx]) >= match_iou:
                matched = idx
                right_used.add(key)
                break
        if matched is None:
            left_unmatched += 1
            continue
        right = candidates[matched]
        box_abs = np.abs(np.asarray(left["box"], dtype=np.float32) - np.asarray(right["box"], dtype=np.float32))
        palm7_err = np.linalg.norm(
            np.asarray(left["palm7"], dtype=np.float32) - np.asarray(right["palm7"], dtype=np.float32),
            axis=1,
        )
        hand21_err = np.linalg.norm(
            np.asarray(left["hand21"], dtype=np.float32) - np.asarray(right["hand21"], dtype=np.float32),
            axis=1,
        )
        rows.append(
            {
                "image_name": left["image_name"],
                "left_index": left["pred_index"],
                "right_index": right["pred_index"],
                "match_iou": float(ious[matched]),
                "box_mean_abs_px": float(np.mean(box_abs)),
                "palm7_mean_px": float(np.mean(palm7_err)),
                "hand21_mean_px": float(np.mean(hand21_err)),
                "score_abs": abs(float(left["score"]) - float(right["score"])),
                "hand_score_abs": abs(float(left["hand_score"]) - float(right["hand_score"])),
            }
        )
    right_unmatched = len(right_rows) - len(right_used)
    return rows, left_unmatched, right_unmatched


def flat_prediction_rows(model_name: str, row: DatasetRow, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pred_index, pred in enumerate(predictions):
        rows.append(
            {
                "model": model_name,
                "image_id": row.image_id,
                "image_name": row.image_name,
                "pred_index": pred_index,
                "score": float(pred["score"]),
                "hand_score": float(pred.get("hand_score", math.nan)),
                "box": pred["box"],
                "palm7": pred["palm7"],
                "hand21": pred["hand21"],
            }
        )
    return rows


def pipeline_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "max_hands": args.max_hands,
        "min_hand_score": args.min_hand_score,
        "roi_scale": args.roi_scale,
        "shift_y": args.shift_y,
        "rotation_offset_degrees": args.rotation_offset_degrees,
    }


def make_om_pipeline(args: argparse.Namespace, spec: dict[str, Any]) -> OmHandPipeline:
    return OmHandPipeline(
        resolve_path(spec["om_detector"]),
        resolve_path(spec["om_landmark"]),
        device_id=args.device_id,
        reload_detector_each_frame=args.reload_detector_each_frame,
        finalize_on_release=False,
        **pipeline_kwargs(args),
    )


def make_onnx_pipeline(args: argparse.Namespace, spec: dict[str, Any]) -> OnnxHandPipeline:
    return OnnxHandPipeline(
        resolve_path(spec["onnx_detector"]),
        resolve_path(spec["onnx_landmark"]),
        **pipeline_kwargs(args),
    )


def check_model_files(model_sets: list[str], run_onnx: bool) -> None:
    missing: list[Path] = []
    for name in model_sets:
        spec = MODEL_SETS[name]
        for key in ("om_detector", "om_landmark"):
            path = resolve_path(spec[key])
            if not path.exists():
                missing.append(path)
        if run_onnx:
            for key in ("onnx_detector", "onnx_landmark"):
                path = resolve_path(spec[key])
                if not path.exists():
                    missing.append(path)
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing model file(s):\n{lines}")


def evaluate_backend(
    *,
    backend_name: str,
    pipeline: Any,
    dataset_rows: list[DatasetRow],
    targets_map: dict[str, list[PalmTarget]],
    args: argparse.Namespace,
    output_dir: Path,
    save_vis_prefix: str,
) -> dict[str, Any]:
    predictions: list[PalmPrediction] = []
    per_image_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    flat_predictions: list[dict[str, Any]] = []
    timing: dict[str, list[float]] = {
        "preprocess": [],
        "detector": [],
        "decode": [],
        "roi": [],
        "landmark": [],
        "post": [],
        "total": [],
    }
    saved_vis = 0
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(dataset_rows):
        result = pipeline.infer(row.image_bgr)
        predictions.extend(predictions_to_palm(result.predictions, row))
        flat_predictions.extend(flat_prediction_rows(backend_name, row, result.predictions))
        frame_matches, unmatched_gt, unmatched_pred = match_targets_to_predictions(
            row.targets,
            result.predictions,
            args.landmark_match_iou,
        )
        match_rows.extend(frame_matches)
        for key, source_key in (
            ("preprocess", "preprocess_ms"),
            ("detector", "detector_ms"),
            ("decode", "decode_ms"),
            ("roi", "roi_ms"),
            ("landmark", "landmark_ms"),
            ("post", "post_ms"),
            ("total", "total_ms"),
        ):
            timing[key].append(float(result.timings.get(source_key, math.nan)))
        per_image_rows.append(
            {
                "image_id": row.image_id,
                "image_name": row.image_name,
                "gesture_label": row.gesture_label,
                "width": row.width,
                "height": row.height,
                "gt_hands": len(row.targets),
                "pred_hands": len(result.predictions),
                "matched_hands": len(frame_matches),
                "unmatched_gt": unmatched_gt,
                "unmatched_pred": unmatched_pred,
                **{f"{key}_ms": float(result.timings.get(f"{key}_ms", math.nan)) for key in ("preprocess", "detector", "decode", "roi", "landmark", "post", "total")},
            }
        )
        if args.save_vis and saved_vis < args.save_vis and (result.predictions or row.targets):
            canvas = draw_hand_predictions(row.image_bgr, result.predictions)
            cv2.imwrite(str(vis_dir / f"{save_vis_prefix}_{index:06d}.jpg"), canvas)
            saved_vis += 1
        if args.progress_interval and (index + 1) % args.progress_interval == 0:
            print(f"[{backend_name}] {index + 1}/{len(dataset_rows)}", flush=True)

    detection = detection_metrics(predictions, targets_map, args.score_threshold, 0.50)
    matched_count = len(match_rows)
    gt_count = sum(len(row.targets) for row in dataset_rows)
    pred_count = len(predictions)
    unmatched_gt_total = sum(int(row["unmatched_gt"]) for row in per_image_rows)
    unmatched_pred_total = sum(int(row["unmatched_pred"]) for row in per_image_rows)
    summary: dict[str, Any] = {
        "backend": backend_name,
        "images": len(dataset_rows),
        "gt_hands": gt_count,
        "pred_hands": pred_count,
        "matched_hands": matched_count,
        "unmatched_gt": unmatched_gt_total,
        "unmatched_pred": unmatched_pred_total,
        "unmatched_gt_rate": unmatched_gt_total / max(gt_count, 1),
        "unmatched_pred_rate": unmatched_pred_total / max(pred_count, 1),
        "detection": detection,
        **summarize([float(row["palm7_mean_px"]) for row in match_rows], "palm7_mean_px"),
        **summarize([float(row["palm7_p95_px"]) for row in match_rows], "palm7_p95_px"),
        **summarize([float(row["palm7_max_px"]) for row in match_rows], "palm7_max_px"),
        **summarize([float(row["palm7_nme"]) for row in match_rows], "palm7_nme"),
        **summarize([float(row["palm7_pck_001"]) for row in match_rows], "palm7_pck_001"),
        **summarize([float(row["palm7_pck_005"]) for row in match_rows], "palm7_pck_005"),
        **summarize([float(row["full21_mean_px"]) for row in match_rows], "full21_mean_px"),
        **summarize([float(row["full21_p95_px"]) for row in match_rows], "full21_p95_px"),
        **summarize([float(row["full21_max_px"]) for row in match_rows], "full21_max_px"),
        **summarize([float(row["full21_nme"]) for row in match_rows], "full21_nme"),
        **summarize([float(row["full21_pck_001"]) for row in match_rows], "full21_pck_001"),
        **summarize([float(row["full21_pck_005"]) for row in match_rows], "full21_pck_005"),
    }
    for key, values in timing.items():
        summary.update(summarize_ms(values, key))

    write_csv(output_dir / "per_image.csv", per_image_rows)
    write_csv(output_dir / "matches.csv", match_rows)
    write_json(output_dir / "predictions.json", flat_predictions)
    write_json(output_dir / "summary.json", summary)
    write_backend_report(output_dir / "report.md", summary)
    return {
        "summary": summary,
        "predictions": flat_predictions,
    }


def full_passed(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    detection = summary["detection"]
    return bool(
        detection["recall"] >= args.min_recall
        and detection["ap@0.50"] >= args.min_ap50
        and summary["full21_mean_px_mean"] <= args.max_full21_mean_px
        and summary["full21_mean_px_p95"] <= args.max_full21_p95_px
        and summary["unmatched_gt_rate"] <= args.max_unmatched_gt_rate
    )


def add_threshold_verdict(summary: dict[str, Any], args: argparse.Namespace, enforce: bool) -> None:
    summary["thresholds"] = {
        "min_recall": args.min_recall,
        "min_ap50": args.min_ap50,
        "max_full21_mean_px": args.max_full21_mean_px,
        "max_full21_p95_px": args.max_full21_p95_px,
        "max_unmatched_gt_rate": args.max_unmatched_gt_rate,
        "enforced": enforce,
    }
    summary["passed"] = full_passed(summary, args) if enforce else None


def write_backend_report(path: Path, summary: dict[str, Any]) -> None:
    detection = summary["detection"]
    lines = [
        f"# {summary['backend']} Dataset Evaluation",
        "",
        "## Counts",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| images | {summary['images']} |",
        f"| GT hands | {summary['gt_hands']} |",
        f"| predicted hands | {summary['pred_hands']} |",
        f"| matched hands | {summary['matched_hands']} |",
        f"| unmatched GT rate | {fmt(summary['unmatched_gt_rate'], 6)} |",
        "",
        "## Palm Detection",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| precision | {fmt(detection['precision'], 6)} |",
        f"| recall | {fmt(detection['recall'], 6)} |",
        f"| miss_rate | {fmt(detection['miss_rate'], 6)} |",
        f"| AP@0.50 | {fmt(detection['ap@0.50'], 6)} |",
        f"| AP@0.75 | {fmt(detection['ap@0.75'], 6)} |",
        f"| mAP@0.50:0.95 | {fmt(detection['map@0.50:0.95'], 6)} |",
        "",
        "## Landmark Error",
        "",
        "| Metric | Mean | P95 | Max |",
        "| --- | ---: | ---: | ---: |",
        f"| palm7 mean px | {fmt(summary['palm7_mean_px_mean'])} | {fmt(summary['palm7_mean_px_p95'])} | {fmt(summary['palm7_mean_px_max'])} |",
        f"| full21 mean px | {fmt(summary['full21_mean_px_mean'])} | {fmt(summary['full21_mean_px_p95'])} | {fmt(summary['full21_mean_px_max'])} |",
        f"| full21 NME | {fmt(summary['full21_nme_mean'], 6)} | {fmt(summary['full21_nme_p95'], 6)} | {fmt(summary['full21_nme_max'], 6)} |",
        f"| full21 PCK@0.01 | {fmt(summary['full21_pck_001_mean'], 6)} | {fmt(summary['full21_pck_001_p95'], 6)} | {fmt(summary['full21_pck_001_max'], 6)} |",
        f"| full21 PCK@0.05 | {fmt(summary['full21_pck_005_mean'], 6)} | {fmt(summary['full21_pck_005_p95'], 6)} | {fmt(summary['full21_pck_005_max'], 6)} |",
        "",
        "## Timing",
        "",
        "| Stage | Mean ms | Median ms | P95 ms | Max ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("preprocess", "detector", "decode", "roi", "landmark", "post", "total"):
        lines.append(
            f"| {key} | {fmt(summary[f'{key}_mean_ms'])} | {fmt(summary[f'{key}_median_ms'])} | "
            f"{fmt(summary[f'{key}_p95_ms'])} | {fmt(summary[f'{key}_max_ms'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Portable HaGRIDv2 OM Dataset Evaluation",
        "",
        f"- dataset: `{report['dataset']}`",
        f"- images: `{report['images']}`",
        f"- output_dir: `{report['output_dir']}`",
        f"- overall_passed: `{report['overall_passed']}`",
        "",
        "## Model Sets",
        "",
        "| Model | Enforced | Passed | Recall | AP50 | Full21 mean px | Full21 P95 px | Total mean ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["model_results"]:
        summary = item["summary"]
        detection = summary["detection"]
        thresholds = summary["thresholds"]
        lines.append(
            f"| {item['model_set']} | {thresholds['enforced']} | {summary['passed']} | "
            f"{fmt(detection['recall'], 6)} | {fmt(detection['ap@0.50'], 6)} | "
            f"{fmt(summary['full21_mean_px_mean'])} | {fmt(summary['full21_mean_px_p95'])} | "
            f"{fmt(summary['total_mean_ms'])} |"
        )
    if report.get("onnx_comparisons"):
        lines.extend(
            [
                "",
                "## OM vs ONNX",
                "",
                "| Model | Matched | OM unmatched | ONNX unmatched | Hand21 mean px | Palm7 mean px | Box mean abs px |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in report["onnx_comparisons"]:
            summary = item["summary"]
            lines.append(
                f"| {item['model_set']} | {summary['matched']} | {summary['om_unmatched']} | {summary['onnx_unmatched']} | "
                f"{fmt(summary['hand21_mean_px_mean'])} | {fmt(summary['palm7_mean_px_mean'])} | "
                f"{fmt(summary['box_mean_abs_px_mean'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_comparison(rows: list[dict[str, Any]], om_unmatched: int, onnx_unmatched: int) -> dict[str, Any]:
    return {
        "matched": len(rows),
        "om_unmatched": om_unmatched,
        "onnx_unmatched": onnx_unmatched,
        **summarize([float(row["box_mean_abs_px"]) for row in rows], "box_mean_abs_px"),
        **summarize([float(row["palm7_mean_px"]) for row in rows], "palm7_mean_px"),
        **summarize([float(row["hand21_mean_px"]) for row in rows], "hand21_mean_px"),
        **summarize([float(row["score_abs"]) for row in rows], "score_abs"),
        **summarize([float(row["hand_score_abs"]) for row in rows], "hand_score_abs"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-set", default="full,lite")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=0)
    parser.add_argument("--run-onnx", action="store_true")
    parser.add_argument("--fail-on-lite", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--roi-scale", type=float, default=2.6)
    parser.add_argument("--shift-y", type=float, default=-0.5)
    parser.add_argument("--rotation-offset-degrees", type=float, default=0.0)
    parser.add_argument("--landmark-match-iou", type=float, default=0.10)
    parser.add_argument("--reload-detector-each-frame", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-ap50", type=float, default=0.95)
    parser.add_argument("--max-full21-mean-px", type=float, default=2.0)
    parser.add_argument("--max-full21-p95-px", type=float, default=5.0)
    parser.add_argument("--max-unmatched-gt-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {dataset_path}")
    model_sets = normalize_model_set(args.model_set)
    check_model_files(model_sets, args.run_onnx)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_path(args.output_root) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset_rows = list(iter_dataset_rows(dataset_path, max_images=args.max_images))
    if not dataset_rows:
        raise ValueError(f"No rows loaded from {dataset_path}")
    targets_map = targets_by_image(dataset_rows)
    write_json(
        output_root / "dataset_manifest.json",
        {
            "dataset": str(dataset_path),
            "images": len(dataset_rows),
            "gt_hands": sum(len(row.targets) for row in dataset_rows),
            "max_images": args.max_images,
        },
    )

    model_results: list[dict[str, Any]] = []
    onnx_comparisons: list[dict[str, Any]] = []
    for model_set in model_sets:
        spec = MODEL_SETS[model_set]
        model_dir = output_root / model_set
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"== Evaluate OM {model_set} ==", flush=True)
        om_pipeline = make_om_pipeline(args, spec)
        try:
            om_result = evaluate_backend(
                backend_name=f"{model_set}_om",
                pipeline=om_pipeline,
                dataset_rows=dataset_rows,
                targets_map=targets_map,
                args=args,
                output_dir=model_dir / "om",
                save_vis_prefix=f"{model_set}_om",
            )
        finally:
            om_pipeline.close()
        enforce = model_set == "full" or args.fail_on_lite
        add_threshold_verdict(om_result["summary"], args, enforce=enforce)
        write_json(model_dir / "om" / "summary.json", om_result["summary"])
        write_backend_report(model_dir / "om" / "report.md", om_result["summary"])
        model_results.append({"model_set": model_set, **om_result})

        if args.run_onnx:
            print(f"== Evaluate ONNX {model_set} ==", flush=True)
            onnx_pipeline = make_onnx_pipeline(args, spec)
            onnx_result = evaluate_backend(
                backend_name=f"{model_set}_onnx",
                pipeline=onnx_pipeline,
                dataset_rows=dataset_rows,
                targets_map=targets_map,
                args=args,
                output_dir=model_dir / "onnx",
                save_vis_prefix=f"{model_set}_onnx",
            )
            rows, om_unmatched, onnx_unmatched = compare_prediction_sets(
                om_result["predictions"],
                onnx_result["predictions"],
                args.landmark_match_iou,
            )
            comparison = {
                "model_set": model_set,
                "summary": summarize_comparison(rows, om_unmatched, onnx_unmatched),
            }
            onnx_comparisons.append(comparison)
            write_csv(model_dir / "om_vs_onnx_matches.csv", rows)
            write_json(model_dir / "om_vs_onnx_summary.json", comparison["summary"])

    overall_passed = all(
        item["summary"].get("passed") is not False
        for item in model_results
        if item["summary"]["thresholds"]["enforced"]
    )
    report = {
        "task": "eval_hf_hand_dataset_om",
        "dataset": str(dataset_path),
        "output_dir": str(output_root),
        "images": len(dataset_rows),
        "model_sets": model_sets,
        "run_onnx": args.run_onnx,
        "overall_passed": overall_passed,
        "model_results": [{"model_set": item["model_set"], "summary": item["summary"]} for item in model_results],
        "onnx_comparisons": onnx_comparisons,
    }
    write_json(output_root / "summary.json", report)
    write_summary_report(output_root / "summary.md", report)
    write_csv(
        output_root / "summary.csv",
        [
            {
                "model_set": item["model_set"],
                "enforced": item["summary"]["thresholds"]["enforced"],
                "passed": item["summary"]["passed"],
                "recall": item["summary"]["detection"]["recall"],
                "ap50": item["summary"]["detection"]["ap@0.50"],
                "full21_mean_px": item["summary"]["full21_mean_px_mean"],
                "full21_p95_px": item["summary"]["full21_mean_px_p95"],
                "total_mean_ms": item["summary"]["total_mean_ms"],
            }
            for item in model_results
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_default))
    return 0 if overall_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
