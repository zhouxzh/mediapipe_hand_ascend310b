#!/usr/bin/env python3
"""Analyze a palm detector OM model against TFLite references."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.decode import PalmDetection
from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import sigmoid
from hand_pipeline.decode import weighted_nms
from hand_pipeline.eval import PalmPrediction
from hand_pipeline.eval import box_iou
from hand_pipeline.eval import detection_metrics
from hand_pipeline.eval import list_images
from hand_pipeline.eval import load_targets
from hand_pipeline.inference import AclOmModel
from hand_pipeline.inference import AclRuntime
from hand_pipeline.inference import TfliteModel
from hand_pipeline.preprocess import LetterboxInfo
from hand_pipeline.preprocess import image_to_tensor


BOX_DIMS = [
    "box_cx",
    "box_cy",
    "box_w",
    "box_h",
    "kp0_x",
    "kp0_y",
    "kp1_x",
    "kp1_y",
    "kp2_x",
    "kp2_y",
    "kp3_x",
    "kp3_y",
    "kp4_x",
    "kp4_y",
    "kp5_x",
    "kp5_y",
    "kp6_x",
    "kp6_y",
]


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def summarize(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_p99": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
        f"{prefix}_max": float(np.max(arr)),
    }


def abs_metrics(actual: np.ndarray, expected: np.ndarray, prefix: str) -> dict[str, float]:
    diff = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return summarize(diff, prefix)


def safe_ratio(numerator: float, denominator: float) -> float:
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return math.nan
    return float(numerator / denominator)


def letterbox_to_array(info: LetterboxInfo) -> np.ndarray:
    return np.array(
        [
            info.input_size,
            info.orig_width,
            info.orig_height,
            info.resized_width,
            info.resized_height,
            info.pad_left,
            info.pad_top,
            info.pad_right,
            info.pad_bottom,
            *info.normalized_padding,
        ],
        dtype=np.float32,
    )


def letterbox_from_array(values: np.ndarray) -> LetterboxInfo:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return LetterboxInfo(
        input_size=int(arr[0]),
        orig_width=int(arr[1]),
        orig_height=int(arr[2]),
        resized_width=int(arr[3]),
        resized_height=int(arr[4]),
        pad_left=int(arr[5]),
        pad_top=int(arr[6]),
        pad_right=int(arr[7]),
        pad_bottom=int(arr[8]),
        normalized_padding_values=(float(arr[9]), float(arr[10]), float(arr[11]), float(arr[12])),
    )


def decode_all_geometry(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    anchors: np.ndarray,
    letterbox: LetterboxInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(raw_boxes, dtype=np.float32)
    scores = np.asarray(raw_scores, dtype=np.float32)
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 3:
        scores = scores[0]
    scores = scores.reshape(-1)
    x_scale = y_scale = w_scale = h_scale = 192.0
    x_center = boxes[:, 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = boxes[:, 1] / y_scale * anchors[:, 3] + anchors[:, 1]
    w = boxes[:, 2] / w_scale * anchors[:, 2]
    h = boxes[:, 3] / h_scale * anchors[:, 3]
    decoded_boxes = np.stack(
        [x_center - w * 0.5, y_center - h * 0.5, x_center + w * 0.5, y_center + h * 0.5],
        axis=1,
    )
    keypoints = np.zeros((boxes.shape[0], 7, 2), dtype=np.float32)
    for keypoint_id in range(7):
        offset = 4 + keypoint_id * 2
        keypoints[:, keypoint_id, 0] = boxes[:, offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoints[:, keypoint_id, 1] = boxes[:, offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    left, top, right, bottom = letterbox.normalized_padding
    x_scale_pad = max(1.0 - left - right, 1e-9)
    y_scale_pad = max(1.0 - top - bottom, 1e-9)
    decoded_boxes[:, [0, 2]] = (decoded_boxes[:, [0, 2]] - left) / x_scale_pad * letterbox.orig_width
    decoded_boxes[:, [1, 3]] = (decoded_boxes[:, [1, 3]] - top) / y_scale_pad * letterbox.orig_height
    keypoints[:, :, 0] = (keypoints[:, :, 0] - left) / x_scale_pad * letterbox.orig_width
    keypoints[:, :, 1] = (keypoints[:, :, 1] - top) / y_scale_pad * letterbox.orig_height
    probs = sigmoid(np.clip(scores, -100.0, 100.0))
    return decoded_boxes.astype(np.float32), keypoints.astype(np.float32), probs.astype(np.float32)


def detections_to_predictions(
    detections: list[PalmDetection],
    image_id: int,
    image_name: str,
) -> list[PalmPrediction]:
    return [
        PalmPrediction(
            image_id=image_id,
            image_name=image_name,
            score=float(det.score),
            box=det.box.astype(np.float32),
            palm7=det.keypoints.astype(np.float32),
        )
        for det in detections
    ]


def match_detections(
    refs: list[PalmDetection],
    preds: list[PalmDetection],
    iou_threshold: float,
) -> tuple[list[dict[str, float]], int, int]:
    rows: list[dict[str, float]] = []
    used: set[int] = set()
    for ref_index, ref in enumerate(sorted(refs, key=lambda item: item.score, reverse=True)):
        if not preds:
            continue
        boxes = np.stack([pred.box for pred in preds], axis=0).astype(np.float32)
        ious = box_iou(ref.box.astype(np.float32), boxes)
        order = np.argsort(ious)[::-1]
        matched = None
        for pos in order:
            pos_i = int(pos)
            if pos_i not in used and float(ious[pos_i]) >= iou_threshold:
                matched = pos_i
                break
        if matched is None:
            continue
        used.add(matched)
        pred = preds[matched]
        ref_center = np.array([(ref.box[0] + ref.box[2]) * 0.5, (ref.box[1] + ref.box[3]) * 0.5], dtype=np.float32)
        pred_center = np.array([(pred.box[0] + pred.box[2]) * 0.5, (pred.box[1] + pred.box[3]) * 0.5], dtype=np.float32)
        ref_size = np.array([ref.box[2] - ref.box[0], ref.box[3] - ref.box[1]], dtype=np.float32)
        pred_size = np.array([pred.box[2] - pred.box[0], pred.box[3] - pred.box[1]], dtype=np.float32)
        palm7_err = np.linalg.norm(ref.keypoints.astype(np.float32) - pred.keypoints.astype(np.float32), axis=1)
        rows.append(
            {
                "ref_index": float(ref_index),
                "pred_index": float(matched),
                "iou": float(ious[matched]),
                "score_ref": float(ref.score),
                "score_pred": float(pred.score),
                "score_abs": abs(float(ref.score) - float(pred.score)),
                "box_coord_mean_abs": float(np.mean(np.abs(ref.box.astype(np.float32) - pred.box.astype(np.float32)))),
                "center_px": float(np.linalg.norm(ref_center - pred_center)),
                "size_px": float(np.linalg.norm(ref_size - pred_size)),
                "palm7_mean_px": float(np.mean(palm7_err)),
                "palm7_max_px": float(np.max(palm7_err)),
            }
        )
    return rows, len(preds) - len(used), len(refs) - len(rows)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if a.size == 0 or b.size == 0:
        return math.nan
    k = min(k, a.size, b.size)
    a_ids = set(int(i) for i in np.argpartition(a, -k)[-k:])
    b_ids = set(int(i) for i in np.argpartition(b, -k)[-k:])
    return len(a_ids & b_ids) / max(k, 1)


def make_reference(args: argparse.Namespace) -> int:
    data_root = resolve_dataset(args.data)
    output_dir = resolve_path(args.output_dir)
    ref_dir = output_dir / "reference_npz"
    ref_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(data_root / args.split / "images")
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    model = TfliteModel(resolve_path(args.model), num_threads=args.num_threads)
    anchors = generate_palm_anchors()
    manifest_items: list[dict[str, Any]] = []
    predictions: list[PalmPrediction] = []
    infer_ms: list[float] = []
    decode_ms: list[float] = []

    for image_id, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        tensor, letterbox = image_to_tensor(image, input_size=192)
        start = time.perf_counter()
        raw_boxes, raw_scores = model(tensor)
        infer_ms.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        palms = decode_raw_palm(raw_boxes, raw_scores, anchors, letterbox, score_threshold=args.score_threshold)
        palms = weighted_nms(palms, iou_threshold=args.nms_iou, max_detections=args.max_det)
        decode_ms.append((time.perf_counter() - start) * 1000.0)
        predictions.extend(detections_to_predictions(palms, image_id, image_path.name))

        npz_name = f"{image_id:06d}_{image_path.stem}.npz"
        np.savez_compressed(
            ref_dir / npz_name,
            input=tensor.astype(np.float32),
            tflite_boxes=np.asarray(raw_boxes, dtype=np.float32),
            tflite_scores=np.asarray(raw_scores, dtype=np.float32),
            letterbox=letterbox_to_array(letterbox),
        )
        manifest_items.append(
            {
                "image_id": image_id,
                "image": image_path.name,
                "npz": f"reference_npz/{npz_name}",
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "tflite_nms_detections": len(palms),
            }
        )
        if (image_id + 1) % 50 == 0:
            print(f"[reference] {image_id + 1}/{len(image_paths)}", flush=True)

    summary: dict[str, Any] = {
        "task": "make_palm_tflite_reference",
        "data": str(data_root),
        "split": args.split,
        "model": str(resolve_path(args.model)),
        "images": len(image_paths),
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "infer_mean_ms": float(np.mean(infer_ms)) if infer_ms else 0.0,
        "decode_mean_ms": float(np.mean(decode_ms)) if decode_ms else 0.0,
        "tflite_predictions": len(predictions),
    }
    try:
        targets = load_targets(data_root, args.split)
        image_names = {path.name for path in image_paths}
        targets = {name: targets.get(name, []) for name in image_names}
        summary["tflite_gt_metrics"] = detection_metrics(predictions, targets, args.score_threshold, 0.50)
    except Exception as exc:  # keep reference generation usable without labels
        summary["tflite_gt_metrics_error"] = str(exc)

    write_json(output_dir / "manifest.json", {"summary": summary, "items": manifest_items})
    write_json(output_dir / "reference_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def collect_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference-dir", default="runs/palm_om/legacy_full_palm")
    parser.add_argument("--output-dir", default="runs/palm_om/legacy_full_palm/om_compare")
    parser.add_argument(
        "--model",
        default="models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om",
    )
    parser.add_argument("--data", default="")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--keep-model-loaded", action="store_true")


def run_om_outputs(model_path: Path, tensor: np.ndarray, repeat: int, keep_model_loaded: bool) -> tuple[list[list[np.ndarray]], list[float], list[float]]:
    load_ms: list[float] = []
    infer_ms: list[float] = []
    outputs: list[list[np.ndarray]] = []
    if keep_model_loaded:
        start = time.perf_counter()
        model = AclOmModel(model_path)
        load_ms.append((time.perf_counter() - start) * 1000.0)
        try:
            for _ in range(repeat):
                start = time.perf_counter()
                outputs.append(model(tensor))
                infer_ms.append((time.perf_counter() - start) * 1000.0)
        finally:
            model.close()
    else:
        for _ in range(repeat):
            start = time.perf_counter()
            model = AclOmModel(model_path)
            load_ms.append((time.perf_counter() - start) * 1000.0)
            try:
                start = time.perf_counter()
                outputs.append(model(tensor))
                infer_ms.append((time.perf_counter() - start) * 1000.0)
            finally:
                model.close()
    return outputs, load_ms, infer_ms


def add_anchor_group(rows: list[dict[str, Any]], group: str, diff_boxes: np.ndarray, diff_scores: np.ndarray, sl: slice) -> None:
    rows.append(
        {
            "group": group,
            "anchors": int(diff_scores[sl].size),
            "box_mean_abs": float(np.mean(diff_boxes[sl])),
            "box_p95_abs": float(np.percentile(diff_boxes[sl], 95)),
            "box_max_abs": float(np.max(diff_boxes[sl])),
            "score_mean_abs": float(np.mean(diff_scores[sl])),
            "score_p95_abs": float(np.percentile(diff_scores[sl], 95)),
            "score_max_abs": float(np.max(diff_scores[sl])),
        }
    )


def compare_om(args: argparse.Namespace) -> int:
    reference_root = resolve_path(args.reference_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((reference_root / "manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    if args.max_images:
        items = items[: args.max_images]
    model_path = resolve_path(args.model)
    anchors = generate_palm_anchors()

    raw_box_abs: list[np.ndarray] = []
    raw_score_abs: list[np.ndarray] = []
    raw_box_ref_abs: list[np.ndarray] = []
    raw_score_ref_abs: list[np.ndarray] = []
    prob_abs: list[np.ndarray] = []
    selected_box_px: list[float] = []
    selected_palm7_px: list[float] = []
    nms_match_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    channel_sums = np.zeros((18,), dtype=np.float64)
    channel_counts = np.zeros((18,), dtype=np.float64)
    channel_signed_sums = np.zeros((18,), dtype=np.float64)
    anchor_group_accum: list[dict[str, Any]] = []
    load_ms_all: list[float] = []
    infer_ms_all: list[float] = []
    stability_box_abs: list[float] = []
    stability_score_abs: list[float] = []
    tflite_predictions: list[PalmPrediction] = []
    om_predictions: list[PalmPrediction] = []
    threshold_stats = {
        "tflite_positive": 0,
        "om_positive": 0,
        "both_positive": 0,
        "tflite_only_positive": 0,
        "om_only_positive": 0,
    }
    top20_overlaps: list[float] = []
    top100_overlaps: list[float] = []
    nms_totals = {
        "tflite_nms": 0,
        "om_nms": 0,
        "matched": 0,
        "om_unmatched": 0,
        "tflite_unmatched": 0,
    }

    with AclRuntime(args.device_id):
        for index, item in enumerate(items):
            ref = np.load(reference_root / item["npz"])
            tensor = np.asarray(ref["input"], dtype=np.float32)
            ref_boxes = np.asarray(ref["tflite_boxes"], dtype=np.float32)
            ref_scores = np.asarray(ref["tflite_scores"], dtype=np.float32)
            letterbox = letterbox_from_array(ref["letterbox"])
            outputs, load_ms, infer_ms = run_om_outputs(model_path, tensor, max(args.repeat, 1), args.keep_model_loaded)
            load_ms_all.extend(load_ms)
            infer_ms_all.extend(infer_ms)
            om_boxes = np.asarray(outputs[0][0], dtype=np.float32)
            om_scores = np.asarray(outputs[0][1], dtype=np.float32)
            for repeat_outputs in outputs[1:]:
                stability_box_abs.append(float(np.mean(np.abs(np.asarray(repeat_outputs[0], dtype=np.float32) - om_boxes))))
                stability_score_abs.append(float(np.mean(np.abs(np.asarray(repeat_outputs[1], dtype=np.float32) - om_scores))))

            box_diff = np.abs(om_boxes - ref_boxes)
            score_diff = np.abs(om_scores - ref_scores)
            raw_box_abs.append(box_diff.reshape(-1))
            raw_score_abs.append(score_diff.reshape(-1))
            raw_box_ref_abs.append(np.abs(ref_boxes).reshape(-1))
            raw_score_ref_abs.append(np.abs(ref_scores).reshape(-1))
            box_2d = box_diff.reshape(2016, 18)
            signed_2d = (om_boxes - ref_boxes).reshape(2016, 18)
            channel_sums += np.sum(box_2d, axis=0)
            channel_signed_sums += np.sum(signed_2d, axis=0)
            channel_counts += box_2d.shape[0]

            ref_dec_boxes, ref_dec_kp, ref_probs = decode_all_geometry(ref_boxes, ref_scores, anchors, letterbox)
            om_dec_boxes, om_dec_kp, om_probs = decode_all_geometry(om_boxes, om_scores, anchors, letterbox)
            p_diff = np.abs(om_probs - ref_probs)
            prob_abs.append(p_diff.reshape(-1))

            ref_pos = ref_probs >= args.score_threshold
            om_pos = om_probs >= args.score_threshold
            threshold_stats["tflite_positive"] += int(np.sum(ref_pos))
            threshold_stats["om_positive"] += int(np.sum(om_pos))
            threshold_stats["both_positive"] += int(np.sum(ref_pos & om_pos))
            threshold_stats["tflite_only_positive"] += int(np.sum(ref_pos & ~om_pos))
            threshold_stats["om_only_positive"] += int(np.sum(om_pos & ~ref_pos))
            top20_overlaps.append(topk_overlap(ref_probs, om_probs, 20))
            top100_overlaps.append(topk_overlap(ref_probs, om_probs, 100))

            selected = ref_pos | om_pos
            if np.any(selected):
                center_ref = np.stack(
                    [
                        (ref_dec_boxes[selected, 0] + ref_dec_boxes[selected, 2]) * 0.5,
                        (ref_dec_boxes[selected, 1] + ref_dec_boxes[selected, 3]) * 0.5,
                    ],
                    axis=1,
                )
                center_om = np.stack(
                    [
                        (om_dec_boxes[selected, 0] + om_dec_boxes[selected, 2]) * 0.5,
                        (om_dec_boxes[selected, 1] + om_dec_boxes[selected, 3]) * 0.5,
                    ],
                    axis=1,
                )
                selected_box_px.extend(float(v) for v in np.linalg.norm(center_ref - center_om, axis=1))
                selected_palm7_px.extend(float(v) for v in np.linalg.norm(ref_dec_kp[selected] - om_dec_kp[selected], axis=2).reshape(-1))

            add_anchor_group(anchor_group_accum, "stride8_first_1152", box_2d, score_diff.reshape(2016), slice(0, 1152))
            add_anchor_group(anchor_group_accum, "stride16_last_864", box_2d, score_diff.reshape(2016), slice(1152, 2016))

            ref_nms = weighted_nms(
                decode_raw_palm(ref_boxes, ref_scores, anchors, letterbox, score_threshold=args.score_threshold),
                iou_threshold=args.nms_iou,
                max_detections=args.max_det,
            )
            om_nms = weighted_nms(
                decode_raw_palm(om_boxes, om_scores, anchors, letterbox, score_threshold=args.score_threshold),
                iou_threshold=args.nms_iou,
                max_detections=args.max_det,
            )
            tflite_predictions.extend(detections_to_predictions(ref_nms, int(item["image_id"]), item["image"]))
            om_predictions.extend(detections_to_predictions(om_nms, int(item["image_id"]), item["image"]))
            matches, om_unmatched, ref_unmatched = match_detections(ref_nms, om_nms, args.match_iou)
            for row in matches:
                row.update({"image": item["image"], "image_id": int(item["image_id"])})
            nms_match_rows.extend(matches)
            nms_totals["tflite_nms"] += len(ref_nms)
            nms_totals["om_nms"] += len(om_nms)
            nms_totals["matched"] += len(matches)
            nms_totals["om_unmatched"] += om_unmatched
            nms_totals["tflite_unmatched"] += ref_unmatched

            per_image_rows.append(
                {
                    "image": item["image"],
                    "raw_box_mean_abs": float(np.mean(box_diff)),
                    "raw_box_p95_abs": float(np.percentile(box_diff, 95)),
                    "raw_score_mean_abs": float(np.mean(score_diff)),
                    "prob_mean_abs": float(np.mean(p_diff)),
                    "tflite_positive": int(np.sum(ref_pos)),
                    "om_positive": int(np.sum(om_pos)),
                    "tflite_only_positive": int(np.sum(ref_pos & ~om_pos)),
                    "om_only_positive": int(np.sum(om_pos & ~ref_pos)),
                    "top20_overlap": top20_overlaps[-1],
                    "top100_overlap": top100_overlaps[-1],
                    "tflite_nms": len(ref_nms),
                    "om_nms": len(om_nms),
                    "nms_matched": len(matches),
                    "nms_om_unmatched": om_unmatched,
                    "nms_tflite_unmatched": ref_unmatched,
                }
            )
            if (index + 1) % 25 == 0:
                print(f"[compare] {index + 1}/{len(items)}", flush=True)

    raw_box_arr = np.concatenate(raw_box_abs) if raw_box_abs else np.array([], dtype=np.float32)
    raw_score_arr = np.concatenate(raw_score_abs) if raw_score_abs else np.array([], dtype=np.float32)
    raw_box_ref_arr = np.concatenate(raw_box_ref_abs) if raw_box_ref_abs else np.array([], dtype=np.float32)
    raw_score_ref_arr = np.concatenate(raw_score_ref_abs) if raw_score_ref_abs else np.array([], dtype=np.float32)
    prob_arr = np.concatenate(prob_abs) if prob_abs else np.array([], dtype=np.float32)
    channel_rows = []
    for dim, name in enumerate(BOX_DIMS):
        channel_rows.append(
            {
                "dim": dim,
                "name": name,
                "mean_abs": float(channel_sums[dim] / max(channel_counts[dim], 1.0)),
                "signed_mean": float(channel_signed_sums[dim] / max(channel_counts[dim], 1.0)),
            }
        )
    anchor_rows = []
    for group in ("stride8_first_1152", "stride16_last_864"):
        rows = [row for row in anchor_group_accum if row["group"] == group]
        anchor_rows.append(
            {
                "group": group,
                "images": len(rows),
                "box_mean_abs": float(np.mean([row["box_mean_abs"] for row in rows])) if rows else math.nan,
                "box_p95_abs_mean": float(np.mean([row["box_p95_abs"] for row in rows])) if rows else math.nan,
                "score_mean_abs": float(np.mean([row["score_mean_abs"] for row in rows])) if rows else math.nan,
                "score_p95_abs_mean": float(np.mean([row["score_p95_abs"] for row in rows])) if rows else math.nan,
            }
        )

    summary: dict[str, Any] = {
        "task": "compare_palm_om_to_tflite",
        "reference_dir": str(reference_root),
        "model": str(model_path),
        "images": len(items),
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "repeat": args.repeat,
        "keep_model_loaded": args.keep_model_loaded,
        **abs_metrics(raw_box_arr, np.zeros_like(raw_box_arr), "raw_box_abs"),
        **abs_metrics(raw_score_arr, np.zeros_like(raw_score_arr), "raw_score_abs"),
        **summarize(raw_box_ref_arr, "raw_box_ref_abs"),
        **summarize(raw_score_ref_arr, "raw_score_ref_abs"),
        **abs_metrics(prob_arr, np.zeros_like(prob_arr), "prob_abs"),
        **summarize(selected_box_px, "positive_anchor_center_px"),
        **summarize(selected_palm7_px, "positive_anchor_palm7_px"),
        **summarize([row["iou"] for row in nms_match_rows], "nms_match_iou"),
        **summarize([row["center_px"] for row in nms_match_rows], "nms_match_center_px"),
        **summarize([row["palm7_mean_px"] for row in nms_match_rows], "nms_match_palm7_mean_px"),
        **summarize(load_ms_all, "load_ms"),
        **summarize(infer_ms_all, "infer_ms"),
        **summarize(stability_box_abs, "repeat_stability_box_abs"),
        **summarize(stability_score_abs, "repeat_stability_score_abs"),
        "threshold_crossing": threshold_stats,
        "top20_overlap_mean": float(np.mean(top20_overlaps)) if top20_overlaps else math.nan,
        "top100_overlap_mean": float(np.mean(top100_overlaps)) if top100_overlaps else math.nan,
        "nms": nms_totals,
        "box_channels": channel_rows,
        "anchor_groups": anchor_rows,
    }
    summary["raw_box_relative_mean"] = safe_ratio(summary["raw_box_abs_mean"], summary["raw_box_ref_abs_mean"])
    summary["raw_score_relative_mean"] = safe_ratio(summary["raw_score_abs_mean"], summary["raw_score_ref_abs_mean"])
    summary["raw_box_relative_percent"] = summary["raw_box_relative_mean"] * 100.0
    summary["raw_score_relative_percent"] = summary["raw_score_relative_mean"] * 100.0
    try:
        data_root = resolve_dataset(args.data)
        targets = load_targets(data_root, args.split)
        image_names = {item["image"] for item in items}
        targets = {name: targets.get(name, []) for name in image_names}
        summary["tflite_gt_metrics"] = detection_metrics(tflite_predictions, targets, args.score_threshold, 0.50)
        summary["om_gt_metrics"] = detection_metrics(om_predictions, targets, args.score_threshold, 0.50)
    except Exception as exc:
        summary["gt_metrics_error"] = str(exc)

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "per_image.csv", per_image_rows)
    write_csv(output_dir / "nms_matches.csv", nms_match_rows)
    write_csv(output_dir / "box_channel_metrics.csv", channel_rows)
    write_csv(output_dir / "anchor_group_metrics.csv", anchor_rows)
    write_compare_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_compare_report(path: Path, summary: dict[str, Any]) -> None:
    tflite_gt = summary.get("tflite_gt_metrics", {})
    om_gt = summary.get("om_gt_metrics", {})
    threshold = summary["threshold_crossing"]
    nms = summary["nms"]
    lines = [
        "# Palm OM Diagnostic Report",
        "",
        f"- model: `{summary['model']}`",
        f"- reference_dir: `{summary['reference_dir']}`",
        f"- images: `{summary['images']}`",
        f"- repeat: `{summary['repeat']}`",
        f"- keep_model_loaded: `{summary['keep_model_loaded']}`",
        "",
        "## Raw Output Error",
        "",
        "Relative mean is `mean_abs / mean(abs(TFLite raw output))`.",
        "",
        "| Tensor | mean_abs | ref_abs_mean | relative_mean | p95_abs | p99_abs | max_abs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| boxes `[2016,18]` | {fmt(summary['raw_box_abs_mean'])} | "
            f"{fmt(summary['raw_box_ref_abs_mean'])} | {fmt(summary['raw_box_relative_percent'], 6)}% | "
            f"{fmt(summary['raw_box_abs_p95'])} | {fmt(summary['raw_box_abs_p99'])} | {fmt(summary['raw_box_abs_max'])} |"
        ),
        (
            f"| score logits `[2016,1]` | {fmt(summary['raw_score_abs_mean'])} | "
            f"{fmt(summary['raw_score_ref_abs_mean'])} | {fmt(summary['raw_score_relative_percent'], 6)}% | "
            f"{fmt(summary['raw_score_abs_p95'])} | {fmt(summary['raw_score_abs_p99'])} | {fmt(summary['raw_score_abs_max'])} |"
        ),
        (
            f"| sigmoid probabilities | {fmt(summary['prob_abs_mean'], 6)} | "
            f"- | - | {fmt(summary['prob_abs_p95'], 6)} | {fmt(summary['prob_abs_p99'], 6)} | "
            f"{fmt(summary['prob_abs_max'], 6)} |"
        ),
        "",
        "## Score Threshold Crossing",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        f"| TFLite positive anchors | {threshold['tflite_positive']} |",
        f"| OM positive anchors | {threshold['om_positive']} |",
        f"| Both positive | {threshold['both_positive']} |",
        f"| TFLite-only positive | {threshold['tflite_only_positive']} |",
        f"| OM-only positive | {threshold['om_only_positive']} |",
        f"| top20 overlap mean | {fmt(summary['top20_overlap_mean'])} |",
        f"| top100 overlap mean | {fmt(summary['top100_overlap_mean'])} |",
        "",
        "## Decoded Geometry Error on Positive Anchors",
        "",
        "| Item | mean | p95 | max |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| decoded center error px | {fmt(summary['positive_anchor_center_px_mean'])} | "
            f"{fmt(summary['positive_anchor_center_px_p95'])} | {fmt(summary['positive_anchor_center_px_max'])} |"
        ),
        (
            f"| decoded palm7 point error px | {fmt(summary['positive_anchor_palm7_px_mean'])} | "
            f"{fmt(summary['positive_anchor_palm7_px_p95'])} | {fmt(summary['positive_anchor_palm7_px_max'])} |"
        ),
        "",
        "## NMS Output Matching",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| TFLite NMS detections | {nms['tflite_nms']} |",
        f"| OM NMS detections | {nms['om_nms']} |",
        f"| matched detections | {nms['matched']} |",
        f"| OM unmatched | {nms['om_unmatched']} |",
        f"| TFLite unmatched | {nms['tflite_unmatched']} |",
        f"| matched IoU mean | {fmt(summary['nms_match_iou_mean'])} |",
        f"| matched center error mean px | {fmt(summary['nms_match_center_px_mean'])} |",
        f"| matched palm7 mean error px | {fmt(summary['nms_match_palm7_mean_px_mean'])} |",
        "",
        "## GT Metrics on Same Image Subset",
        "",
        "| Backend | predictions | precision | recall | AP50 | mAP |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if tflite_gt and om_gt:
        lines.append(
            f"| TFLite | {tflite_gt.get('predictions', '')} | {fmt(tflite_gt.get('precision', math.nan), 6)} | "
            f"{fmt(tflite_gt.get('recall', math.nan), 6)} | {fmt(tflite_gt.get('ap@0.50', math.nan), 6)} | "
            f"{fmt(tflite_gt.get('map@0.50:0.95', math.nan), 6)} |"
        )
        lines.append(
            f"| OM | {om_gt.get('predictions', '')} | {fmt(om_gt.get('precision', math.nan), 6)} | "
            f"{fmt(om_gt.get('recall', math.nan), 6)} | {fmt(om_gt.get('ap@0.50', math.nan), 6)} | "
            f"{fmt(om_gt.get('map@0.50:0.95', math.nan), 6)} |"
        )
    else:
        lines.append("| unavailable | | | | | |")
    lines.extend(
        [
            "",
            "## Largest Box Channels",
            "",
            "| dim | name | mean_abs | signed_mean |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    top_channels = sorted(summary["box_channels"], key=lambda row: row["mean_abs"], reverse=True)[:8]
    for row in top_channels:
        lines.append(f"| {row['dim']} | `{row['name']}` | {fmt(row['mean_abs'])} | {fmt(row['signed_mean'])} |")
    lines.extend(
        [
            "",
            "## Anchor Groups",
            "",
            "| group | box_mean_abs | box_p95_abs_mean | score_mean_abs | score_p95_abs_mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["anchor_groups"]:
        lines.append(
            f"| `{row['group']}` | {fmt(row['box_mean_abs'])} | {fmt(row['box_p95_abs_mean'])} | "
            f"{fmt(row['score_mean_abs'])} | {fmt(row['score_p95_abs_mean'])} |"
        )
    if summary["repeat"] > 1:
        lines.extend(
            [
                "",
                "## Repeat Stability",
                "",
                "| Tensor | mean_abs vs first repeat | p95_abs | max_abs |",
                "| --- | ---: | ---: | ---: |",
                (
                    f"| boxes | {fmt(summary['repeat_stability_box_abs_mean'])} | "
                    f"{fmt(summary['repeat_stability_box_abs_p95'])} | {fmt(summary['repeat_stability_box_abs_max'])} |"
                ),
                (
                    f"| scores | {fmt(summary['repeat_stability_score_abs_mean'])} | "
                    f"{fmt(summary['repeat_stability_score_abs_p95'])} | {fmt(summary['repeat_stability_score_abs_max'])} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation hints:",
            "",
            "- Large raw box error directly becomes decoded box/keypoint displacement because detector regression outputs are divided by 192 and projected to image pixels.",
            "- Score-logit error changes threshold crossings and top-k order, which can change weighted NMS membership even when boxes are only moderately shifted.",
            "- If repeat stability is non-zero with `--keep-model-loaded`, debug ACL model reuse before optimizing geometry.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ref = sub.add_parser("make-reference", help="Run TFLite on real images and save exact input/output references.")
    ref.add_argument("--data", default="")
    ref.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ref.add_argument("--model", default="models/tflite/mediapipe_legacy_0_10_14_palm_detection_full.tflite")
    ref.add_argument("--output-dir", default="runs/palm_om/legacy_full_palm")
    ref.add_argument("--max-images", type=int, default=200)
    ref.add_argument("--score-threshold", type=float, default=0.5)
    ref.add_argument("--nms-iou", type=float, default=0.3)
    ref.add_argument("--max-det", type=int, default=20)
    ref.add_argument("--num-threads", type=int, default=1)

    cmp_parser = sub.add_parser("compare-om", help="Run OM on saved references and compare every palm stage.")
    collect_compare_args(cmp_parser)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "make-reference":
        return make_reference(args)
    if args.cmd == "compare-om":
        return compare_om(args)
    raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
