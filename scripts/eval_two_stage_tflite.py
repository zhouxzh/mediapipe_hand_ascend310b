#!/usr/bin/env python3
"""Evaluate extracted MediaPipe two-stage TFLite hand pipeline."""

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

from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import weighted_nms
from hand_pipeline.eval import box_iou
from hand_pipeline.inference import TfliteModel
from hand_pipeline.preprocess import image_to_tensor
from hand_pipeline.roi import landmarks_to_original
from hand_pipeline.roi import make_hand_roi
from hand_pipeline.roi import crop_from_normalized_rect
from hand_pipeline.roi import preprocess_landmark_tflite


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
    parser.add_argument(
        "--detector",
        default="models/tflite/mediapipe_legacy_0_10_14_palm_detection_full.tflite",
    )
    parser.add_argument(
        "--landmark",
        default="models/tflite/mediapipe_legacy_0_10_14_hand_landmark_full.tflite",
    )
    parser.add_argument("--official-mediapipe", default="runs/mediapipe_baseline_vs_om/mediapipe_predictions.json")
    parser.add_argument(
        "--legacy-rects",
        default="runs/eval_legacy_graph_test/legacy_hand_predictions.json",
    )
    parser.add_argument("--output-dir", default="runs/eval_two_stage_tflite_vs_mediapipe")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--save-vis", type=int, default=16)
    parser.add_argument("--dst-size-mode", choices=("size", "minus_one"), default="size")
    parser.add_argument("--roi-scale", type=float, default=2.6)
    parser.add_argument("--shift-y", type=float, default=-0.5)
    parser.add_argument("--rotation-offset-degrees", type=float, default=0.0)
    parser.add_argument(
        "--roi-source",
        choices=("palm", "official_box", "legacy_rect"),
        default="palm",
        help="Use decoded palm detections, official hand boxes, or legacy graph exported rects for ROI diagnostics.",
    )
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_official(path: Path, image_names: set[str]) -> dict[str, list[dict[str, Any]]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        image_name = Path(str(item.get("image", "")).replace("\\", "/")).name
        if image_name not in image_names:
            continue
        if not item.get("hand21"):
            continue
        grouped.setdefault(image_name, []).append(item)
    return grouped


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


def match_to_official(
    pred: dict[str, Any],
    refs: list[dict[str, Any]],
    used: set[int],
    match_iou: float,
) -> tuple[int | None, float]:
    available = [i for i in range(len(refs)) if i not in used]
    if not available:
        return None, 0.0
    box = np.array(pred["box"], dtype=np.float32)
    ref_boxes = np.stack([np.array(refs[i]["box"], dtype=np.float32) for i in available], axis=0)
    ious = box_iou(box, ref_boxes)
    best_pos = int(np.argmax(ious))
    best_iou = float(ious[best_pos])
    if best_iou < match_iou:
        return None, best_iou
    ref_idx = available[best_pos]
    used.add(ref_idx)
    return ref_idx, best_iou


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


def draw_compare(image: np.ndarray, preds: list[dict[str, Any]], refs: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for ref in refs:
        points = np.array(ref["hand21"], dtype=np.float32)
        for a, b in HAND_EDGES:
            cv2.line(canvas, tuple(np.round(points[a]).astype(int)), tuple(np.round(points[b]).astype(int)), (0, 180, 255), 2)
        for x, y in points:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (0, 180, 255), -1)
    for pred in preds:
        box = np.array(pred["box"], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(x)) for x in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        points = np.array(pred["hand21"], dtype=np.float32)
        for a, b in HAND_EDGES:
            cv2.line(canvas, tuple(np.round(points[a]).astype(int)), tuple(np.round(points[b]).astype(int)), (255, 80, 0), 1)
        for x, y in points:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (255, 80, 0), -1)
    return canvas


def keypoints_from_box(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    cx = (x1 + x2) * 0.5
    wrist_y = y2
    middle_y = y1
    return np.array(
        [
            [cx, wrist_y],
            [cx, (y1 + y2) * 0.55],
            [cx, middle_y],
            [x1, (y1 + y2) * 0.5],
            [x2, (y1 + y2) * 0.5],
            [x1, y2],
            [x2, y2],
        ],
        dtype=np.float32,
    )


def load_legacy_rects(path: Path, image_names: set[str]) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not item.get("hand_rect_from_palm"):
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


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Two-stage TFLite Hand Pipeline Evaluation",
        "",
        "## Settings",
        "",
        f"- data: `{summary['data']}`",
        f"- split: `{summary['split']}`",
        f"- images: `{summary['images']}`",
        f"- official hands: `{summary['official_hands']}`",
        f"- two-stage hands: `{summary['two_stage_hands']}`",
        f"- matched hands: `{summary['matched_hands']}`",
        f"- official unmatched hands: `{summary['official_unmatched_hands']}`",
        f"- two-stage unmatched hands: `{summary['two_stage_unmatched_hands']}`",
        "",
        "## Landmark Error",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| mean px | {fmt(summary['vs_mediapipe_mean_px'], 4)} |",
        f"| median px | {fmt(summary['vs_mediapipe_median_px'], 4)} |",
        f"| P95 px | {fmt(summary['vs_mediapipe_p95_px'], 4)} |",
        f"| max px | {fmt(summary['vs_mediapipe_max_px'], 4)} |",
        f"| NME | {fmt(summary['vs_mediapipe_nme'], 6)} |",
        f"| PCK@0.01 | {fmt(summary['vs_mediapipe_pck@0.01'], 4)} |",
        f"| PCK@0.02 | {fmt(summary['vs_mediapipe_pck@0.02'], 4)} |",
        f"| PCK@0.05 | {fmt(summary['vs_mediapipe_pck@0.05'], 4)} |",
        f"| PCK@0.10 | {fmt(summary['vs_mediapipe_pck@0.10'], 4)} |",
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
        "",
        "## Notes",
        "",
        "- Use `legacy_rect_landmark` to isolate ROI crop, landmark inference, and projection.",
        "- Remaining error in the full two-stage path usually comes from palm-to-rect geometry.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(Path(args.data) / args.split / "images")
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    image_names = {p.name for p in image_paths}
    official_by_image = load_official(Path(args.official_mediapipe), image_names)
    official_hands = sum(len(v) for v in official_by_image.values())
    legacy_rects_by_image = load_legacy_rects(Path(args.legacy_rects), image_names) if args.roi_source == "legacy_rect" else {}

    detector = TfliteModel(args.detector, num_threads=args.num_threads)
    landmark = TfliteModel(args.landmark, num_threads=args.num_threads)
    anchors = generate_palm_anchors()
    predictions: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    point_errors: list[float] = []
    norm_errors: list[float] = []
    det_pre_ms: list[float] = []
    det_infer_ms: list[float] = []
    det_decode_ms: list[float] = []
    roi_ms: list[float] = []
    landmark_ms: list[float] = []
    landmark_post_ms: list[float] = []
    total_ms: list[float] = []
    matched_hands = 0
    saved_vis = 0
    official_used_total = 0

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
        palms = sorted(palms, key=lambda x: x.score, reverse=True)[: args.max_hands]
        det_decode_ms.append((time.perf_counter() - decode_start) * 1000.0)

        image_predictions: list[dict[str, Any]] = []
        refs = official_by_image.get(image_path.name, [])
        used_refs: set[int] = set()
        if args.roi_source == "official_box":
            roi_inputs = []
            for ref_index, ref in enumerate(refs[: args.max_hands]):
                box = np.array(ref["box"], dtype=np.float32)
                roi_inputs.append(
                    {
                        "hand_index": ref_index,
                        "score": float(ref.get("score", 1.0)),
                        "box": box,
                        "keypoints": keypoints_from_box(box),
                        "forced_ref_index": ref_index,
                        "legacy_rect": None,
                    }
                )
        elif args.roi_source == "legacy_rect":
            roi_inputs = []
            for ref_index, ref in enumerate(legacy_rects_by_image.get(image_path.name, [])[: args.max_hands]):
                box = np.array(ref["box"], dtype=np.float32)
                roi_inputs.append(
                    {
                        "hand_index": int(ref.get("hand_index", ref_index)),
                        "score": float(ref.get("score", 1.0)),
                        "box": box,
                        "keypoints": np.zeros((7, 2), dtype=np.float32),
                        "forced_ref_index": ref_index,
                        "legacy_rect": ref["hand_rect_from_palm"],
                    }
                )
        else:
            roi_inputs = [
                {
                    "hand_index": hand_index,
                    "score": float(palm.score),
                    "box": palm.box,
                    "keypoints": palm.keypoints,
                    "forced_ref_index": None,
                    "legacy_rect": None,
                }
                for hand_index, palm in enumerate(palms)
            ]
        for item in roi_inputs:
            hand_index = int(item["hand_index"])
            box = np.asarray(item["box"], dtype=np.float32)
            keypoints = np.asarray(item["keypoints"], dtype=np.float32)
            roi_start = time.perf_counter()
            if item["legacy_rect"] is not None:
                crop, inverse = crop_from_normalized_rect(image, item["legacy_rect"], input_size=224)
                roi_center = np.array(
                    [item["legacy_rect"]["x_center_px"], item["legacy_rect"]["y_center_px"]],
                    dtype=np.float32,
                )
                roi_size = float(item["legacy_rect"]["width_px"])
                roi_rotation = float(item["legacy_rect"]["rotation"])
            else:
                roi = make_hand_roi(
                    image,
                    box,
                    keypoints,
                    scale=args.roi_scale,
                    shift_y=args.shift_y,
                    rotation_offset_degrees=args.rotation_offset_degrees,
                    dst_size_mode=args.dst_size_mode,
                )
                crop = roi.crop
                inverse = roi.inverse
                roi_center = roi.center
                roi_size = float(roi.size)
                roi_rotation = float(roi.rotation)
            lm_tensor = preprocess_landmark_tflite(crop)
            roi_ms.append((time.perf_counter() - roi_start) * 1000.0)

            lm_start = time.perf_counter()
            lm_outputs = landmark(lm_tensor)
            landmark_ms.append((time.perf_counter() - lm_start) * 1000.0)

            lm_post_start = time.perf_counter()
            lm_crop, hand_score, handedness, _world = pick_landmark_outputs(lm_outputs)
            hand21 = landmarks_to_original(lm_crop, inverse, input_size=224, coord_scale="auto")
            landmark_post_ms.append((time.perf_counter() - lm_post_start) * 1000.0)
            if not math.isnan(hand_score) and hand_score < args.min_hand_score:
                continue

            pred = {
                "image_id": image_id,
                "image": str(image_path),
                "hand_index": hand_index,
                "score": float(item["score"]),
                "hand_score": float(hand_score),
                "handedness": float(handedness),
                "box": box.astype(float).tolist(),
                "palm7": keypoints.astype(float).tolist(),
                "hand21": hand21.astype(float).tolist(),
                "roi_center": roi_center.astype(float).tolist(),
                "roi_size": roi_size,
                "roi_rotation_rad": roi_rotation,
            }
            forced_ref = item["forced_ref_index"]
            if forced_ref is not None:
                ref_idx = int(forced_ref)
                used_refs.add(ref_idx)
                match_iou = 1.0
            else:
                ref_idx, match_iou = match_to_official(pred, refs, used_refs, args.match_iou)
            pred["official_match_index"] = ref_idx
            pred["official_match_iou"] = match_iou
            if ref_idx is not None:
                matched_hands += 1
                ref = refs[ref_idx]
                ref_points = np.array(ref["hand21"], dtype=np.float32)
                err = np.linalg.norm(hand21 - ref_points, axis=1)
                norm = norm_from_box(np.array(ref["box"], dtype=np.float32))
                point_errors.extend(float(x) for x in err)
                norm_errors.extend(float(x / norm) for x in err)
                match_rows.append(
                    {
                        "image": image_path.name,
                        "hand_index": hand_index,
                        "official_index": ref_idx,
                        "match_iou": match_iou,
                        "mean_px": float(np.mean(err)),
                        "max_px": float(np.max(err)),
                        "nme": float(np.mean(err / norm)),
                        "score": float(item["score"]),
                        "hand_score": float(hand_score),
                    }
                )
            predictions.append(pred)
            image_predictions.append(pred)
        official_used_total += len(used_refs)
        total_ms.append((time.perf_counter() - total_start) * 1000.0)
        if args.save_vis and saved_vis < args.save_vis and image_predictions:
            canvas = draw_compare(image, image_predictions, refs)
            cv2.imwrite(str(vis_dir / f"{image_path.stem}.jpg"), canvas)
            saved_vis += 1

    summary = {
        "task": "eval_two_stage_tflite_vs_official_mediapipe",
        "data": args.data,
        "split": args.split,
        "images": len(image_paths),
        "detector": args.detector,
        "landmark": args.landmark,
        "official_mediapipe": args.official_mediapipe,
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "max_hands": args.max_hands,
        "match_iou": args.match_iou,
        "min_hand_score": args.min_hand_score,
        "dst_size_mode": args.dst_size_mode,
        "roi_source": args.roi_source,
        "roi_scale": args.roi_scale,
        "shift_y": args.shift_y,
        "rotation_offset_degrees": args.rotation_offset_degrees,
        "official_hands": official_hands,
        "two_stage_hands": len(predictions),
        "matched_hands": matched_hands,
        "official_unmatched_hands": official_hands - official_used_total,
        "two_stage_unmatched_hands": len(predictions) - matched_hands,
        "visualizations": saved_vis,
        **metric_summary(point_errors, norm_errors, "vs_mediapipe"),
        **summarize_times(det_pre_ms, "det_preprocess"),
        **summarize_times(det_infer_ms, "det_infer"),
        **summarize_times(det_decode_ms, "det_decode"),
        **summarize_times(roi_ms, "roi_preprocess"),
        **summarize_times(landmark_ms, "landmark_infer"),
        **summarize_times(landmark_post_ms, "landmark_post"),
        **summarize_times(total_ms, "total"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "matches.csv", match_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

