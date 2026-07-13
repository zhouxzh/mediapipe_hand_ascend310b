#!/usr/bin/env python3
"""Evaluate Ascend OM video results against MediaPipe graph annotations."""

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

from hand_pipeline.eval import box_iou
from hand_pipeline.two_stage import OmHandPipeline
from hand_pipeline.visualization import HAND_EDGES


MODEL_SETS = {
    "full": {
        "detector": "models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om",
        "landmark": "models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om",
    },
    "lite": {
        "detector": "models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om",
        "landmark": "models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/eval_videos/demo1.mp4")
    parser.add_argument("--annotations", default="")
    parser.add_argument("--model-set", choices=sorted(MODEL_SETS), default="full")
    parser.add_argument("--om-detector", default="")
    parser.add_argument("--om-landmark", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--pipeline-mode", choices=["image", "tracking"], default="tracking")
    parser.add_argument("--reference-stream", choices=["image", "tracking"], default="")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--max-tracking-lost-frames", type=int, default=0)
    parser.add_argument("--max-tracking-rejected-frames", type=int, default=0)
    parser.add_argument("--max-tracking-rotation-delta", type=float, default=math.inf)
    parser.add_argument("--min-tracking-size-ratio", type=float, default=0.0)
    parser.add_argument("--max-tracking-size-ratio", type=float, default=math.inf)
    parser.add_argument("--max-tracking-center-shift", type=float, default=math.inf)
    parser.add_argument("--tracking-rect-smooth-alpha", type=float, default=1.0)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=8)
    parser.add_argument("--reload-detector-each-frame", action="store_true")
    parser.add_argument("--max-mean-hand21-px", type=float, default=5.0)
    parser.add_argument("--max-p95-hand21-px", type=float, default=15.0)
    parser.add_argument("--max-count-mismatch-rate", type=float, default=0.05)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_annotations_path(video_path: Path) -> Path:
    return video_path.parent / "annotations" / video_path.stem / "mediapipe_annotations.json"


def default_output_dir(video_path: Path, model_set: str, pipeline_mode: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / "video_mediapipe_om" / f"{video_path.stem}_{model_set}_{pipeline_mode}_{stamp}"


def hand_bbox_from_points(points: Any, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.asarray([math.nan, math.nan, math.nan, math.nan], dtype=np.float32)
    xy = arr[:, :2]
    return np.asarray(
        [
            float(np.clip(np.nanmin(xy[:, 0]), 0.0, float(width))),
            float(np.clip(np.nanmin(xy[:, 1]), 0.0, float(height))),
            float(np.clip(np.nanmax(xy[:, 0]), 0.0, float(width))),
            float(np.clip(np.nanmax(xy[:, 1]), 0.0, float(height))),
        ],
        dtype=np.float32,
    )


def rect_values(rect: dict[str, Any] | None) -> np.ndarray | None:
    if not rect:
        return None
    pixel = rect.get("pixel", rect)
    keys = ("x_center_px", "y_center_px", "width_px", "height_px", "rotation")
    if not all(key in pixel for key in keys):
        return None
    return np.asarray([float(pixel[key]) for key in keys], dtype=np.float32)


def normalize_ref_hand(hand: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    hand21 = np.asarray(hand.get("hand21_keypoints_px", []), dtype=np.float32)
    palm = hand.get("palm_detection") or {}
    palm_box = palm.get("palm_bbox_xyxy_px") or hand.get("palm_bbox_xyxy_px")
    palm7 = palm.get("palm7_keypoints_px") or hand.get("palm7_keypoints_px")
    return {
        "hand_index": int(hand.get("hand_index", 0)),
        "handedness": hand.get("handedness", ""),
        "handedness_score": float(hand.get("handedness_score", math.nan)),
        "hand21": hand21,
        "hand_bbox": hand_bbox_from_points(hand21, width, height),
        "palm_box": None if palm_box is None else np.asarray(palm_box, dtype=np.float32),
        "palm7": None if palm7 is None else np.asarray(palm7, dtype=np.float32),
        "palm_score": float(palm.get("score", math.nan)),
        "source_roi": str(hand.get("source", "")),
        "next_tracking_roi": rect_values(hand.get("next_tracking_roi")),
    }


def normalize_om_pred(pred: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    hand21 = np.asarray(pred.get("hand21", []), dtype=np.float32)
    return {
        "hand_index": int(pred.get("hand_index", 0)),
        "score": float(pred.get("score", math.nan)),
        "hand_score": float(pred.get("hand_score", math.nan)),
        "handedness": float(pred.get("handedness", math.nan)),
        "hand21": hand21,
        "hand_bbox": hand_bbox_from_points(hand21, width, height),
        "palm_box": np.asarray(pred["box"], dtype=np.float32) if pred.get("box") is not None else None,
        "palm7": np.asarray(pred["palm7"], dtype=np.float32) if pred.get("palm7") is not None else None,
        "source_roi": str(pred.get("source_roi", "")),
        "palm_detector_skipped": bool(pred.get("palm_detector_skipped", False)),
        "next_tracking_roi": rect_values(pred.get("next_tracking_roi")),
    }


def point_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape or a.size == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    err = np.linalg.norm(a[:, :2].astype(np.float32) - b[:, :2].astype(np.float32), axis=1)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
    }


def scalar_summary(values: list[float], prefix: str) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_max": math.nan,
        }
    arr = np.asarray(clean, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
    }


def timing_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return scalar_summary([float(row.get(key, math.nan)) for row in rows], key.replace("_ms", ""))


def match_hands(
    refs: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    match_iou: float,
) -> tuple[list[tuple[int, int, float]], int, int]:
    matches: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    if not refs or not preds:
        return matches, len(refs), len(preds)
    pred_boxes = np.stack([pred["hand_bbox"] for pred in preds], axis=0)
    for ref_index, ref in enumerate(refs):
        ious = box_iou(ref["hand_bbox"], pred_boxes)
        order = np.argsort(ious)[::-1]
        for pos in order:
            pred_index = int(pos)
            if pred_index in used_pred:
                continue
            iou = float(ious[pred_index])
            if iou >= match_iou:
                used_pred.add(pred_index)
                matches.append((ref_index, pred_index, iou))
            break
    return matches, len(refs) - len(matches), len(preds) - len(used_pred)


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
        return "nan" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def draw_overlay(
    frame: np.ndarray,
    refs: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    frame_index: int,
) -> np.ndarray:
    canvas = frame.copy()
    for ref in refs:
        points = ref["hand21"]
        for start, end in HAND_EDGES:
            cv2.line(canvas, tuple(np.round(points[start, :2]).astype(int)), tuple(np.round(points[end, :2]).astype(int)), (255, 120, 40), 2)
        for point in points:
            cv2.circle(canvas, tuple(np.round(point[:2]).astype(int)), 3, (255, 120, 40), -1)
    for pred in preds:
        points = pred["hand21"]
        for start, end in HAND_EDGES:
            cv2.line(canvas, tuple(np.round(points[start, :2]).astype(int)), tuple(np.round(points[end, :2]).astype(int)), (60, 230, 120), 2)
        for point in points:
            cv2.circle(canvas, tuple(np.round(point[:2]).astype(int)), 3, (60, 230, 120), -1)
    cv2.putText(
        canvas,
        f"frame {frame_index} MediaPipe orange, OM green",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_report(path: Path, summary: dict[str, Any]) -> None:
    verdict = "consistent" if summary["consistent"] else "different"
    lines = [
        "# Video MediaPipe vs OM Evaluation",
        "",
        "## Verdict",
        "",
        f"- result: `{verdict}`",
        f"- mode: `{summary['pipeline_mode']}`",
        f"- matched hands: `{summary['matched_hands']}`",
        f"- count mismatch rate: `{fmt(summary['count_mismatch_rate'], 6)}`",
        f"- hand21 mean px: `{fmt(summary['hand21_mean_px_mean'])}`",
        f"- hand21 p95 px: `{fmt(summary['hand21_mean_px_p95'])}`",
        f"- hand bbox IoU mean: `{fmt(summary['hand_bbox_iou_mean'])}`",
        "",
        "## Inputs",
        "",
        f"- video: `{summary['video']}`",
        f"- annotations: `{summary['annotations']}`",
        f"- OM detector: `{summary['om_detector']}`",
        f"- OM landmark: `{summary['om_landmark']}`",
        f"- processed frames: `{summary['processed_frames']}`",
        f"- resolution: `{summary['width']}x{summary['height']}`",
        "",
        "## Error Metrics",
        "",
        "| Metric | Mean | Median | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| hand21 mean px | {fmt(summary['hand21_mean_px_mean'])} | {fmt(summary['hand21_mean_px_median'])} | {fmt(summary['hand21_mean_px_p95'])} | {fmt(summary['hand21_mean_px_max'])} |",
        f"| hand bbox IoU | {fmt(summary['hand_bbox_iou_mean'])} | {fmt(summary['hand_bbox_iou_median'])} | {fmt(summary['hand_bbox_iou_p95'])} | {fmt(summary['hand_bbox_iou_max'])} |",
        f"| palm7 mean px | {fmt(summary['palm7_mean_px_mean'])} | {fmt(summary['palm7_mean_px_median'])} | {fmt(summary['palm7_mean_px_p95'])} | {fmt(summary['palm7_mean_px_max'])} |",
        f"| palm box abs px | {fmt(summary['palm_box_mean_abs_px_mean'])} | {fmt(summary['palm_box_mean_abs_px_median'])} | {fmt(summary['palm_box_mean_abs_px_p95'])} | {fmt(summary['palm_box_mean_abs_px_max'])} |",
        f"| next ROI abs | {fmt(summary['next_roi_mean_abs_mean'])} | {fmt(summary['next_roi_mean_abs_median'])} | {fmt(summary['next_roi_mean_abs_p95'])} | {fmt(summary['next_roi_mean_abs_max'])} |",
        "",
        "## Timing",
        "",
        "| Stage | Mean | Median | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| total ms | {fmt(summary['om_total_mean'])} | {fmt(summary['om_total_median'])} | {fmt(summary['om_total_p95'])} | {fmt(summary['om_total_max'])} |",
        f"| detector ms | {fmt(summary['om_detector_mean'])} | {fmt(summary['om_detector_median'])} | {fmt(summary['om_detector_p95'])} | {fmt(summary['om_detector_max'])} |",
        f"| landmark ms | {fmt(summary['om_landmark_mean'])} | {fmt(summary['om_landmark_median'])} | {fmt(summary['om_landmark_p95'])} | {fmt(summary['om_landmark_max'])} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    video_path = resolve_path(args.video)
    annotations_path = resolve_path(args.annotations) if args.annotations else default_annotations_path(video_path)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(video_path, args.model_set, args.pipeline_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model_defaults = MODEL_SETS[args.model_set]
    om_detector = resolve_path(args.om_detector or model_defaults["detector"])
    om_landmark = resolve_path(args.om_landmark or model_defaults["landmark"])
    for path in (video_path, annotations_path, om_detector, om_landmark):
        if not path.exists():
            raise FileNotFoundError(str(path))

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    frames_by_index = {int(frame["frame_index"]): frame for frame in annotations["frames"]}
    reference_stream = args.reference_stream or args.pipeline_mode

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = OmHandPipeline(
        om_detector,
        om_landmark,
        device_id=args.device_id,
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        max_hands=args.max_hands,
        min_hand_score=args.min_hand_score,
        reload_detector_each_frame=args.reload_detector_each_frame,
        mode=args.pipeline_mode,
        max_tracking_lost_frames=args.max_tracking_lost_frames,
        max_tracking_rejected_frames=args.max_tracking_rejected_frames,
        max_tracking_rotation_delta=args.max_tracking_rotation_delta,
        min_tracking_size_ratio=args.min_tracking_size_ratio,
        max_tracking_size_ratio=args.max_tracking_size_ratio,
        max_tracking_center_shift=args.max_tracking_center_shift,
        tracking_rect_smooth_alpha=args.tracking_rect_smooth_alpha,
    )

    frame_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    saved_vis = 0
    count_mismatch_frames = 0
    unmatched_ref_total = 0
    unmatched_om_total = 0

    try:
        processed_frames = 0
        frame_index = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index < args.start_frame:
                continue
            if (frame_index - args.start_frame) % args.frame_stride != 0:
                continue
            if args.max_frames and processed_frames >= args.max_frames:
                break
            ref_frame = frames_by_index.get(frame_index)
            if ref_frame is None:
                raise KeyError(f"Missing annotation for frame {frame_index}")

            result = pipeline.infer(frame)
            refs = [
                normalize_ref_hand(hand, width, height)
                for hand in ref_frame[reference_stream].get("hands", [])
            ]
            preds = [normalize_om_pred(pred, width, height) for pred in result.predictions]
            matches, unmatched_ref, unmatched_om = match_hands(refs, preds, args.match_iou)
            unmatched_ref_total += unmatched_ref
            unmatched_om_total += unmatched_om
            if len(refs) != len(preds):
                count_mismatch_frames += 1

            frame_row = {
                "frame_index": frame_index,
                "reference_hands": len(refs),
                "om_hands": len(preds),
                "matched_hands": len(matches),
                "unmatched_reference_hands": unmatched_ref,
                "unmatched_om_hands": unmatched_om,
                "reference_palm_detections": len(ref_frame[reference_stream].get("palm_detections", [])),
                "om_palm_detector_skipped": bool(result.timings.get("palm_detector_skipped", False)),
                "om_preprocess_ms": float(result.timings.get("preprocess_ms", math.nan)),
                "om_detector_ms": float(result.timings.get("detector_ms", math.nan)),
                "om_decode_ms": float(result.timings.get("decode_ms", math.nan)),
                "om_roi_ms": float(result.timings.get("roi_ms", math.nan)),
                "om_landmark_ms": float(result.timings.get("landmark_ms", math.nan)),
                "om_post_ms": float(result.timings.get("post_ms", math.nan)),
                "om_total_ms": float(result.timings.get("total_ms", math.nan)),
                "om_tracking_state_kept": int(result.timings.get("tracking_state_kept", 0) or 0),
                "om_tracking_state_rejected": int(result.timings.get("tracking_state_rejected", 0) or 0),
            }
            frame_rows.append(frame_row)

            for ref_index, pred_index, iou in matches:
                ref = refs[ref_index]
                pred = preds[pred_index]
                hand_err = point_error(pred["hand21"], ref["hand21"])
                hand_box_abs = np.abs(pred["hand_bbox"] - ref["hand_bbox"])
                row: dict[str, Any] = {
                    "frame_index": frame_index,
                    "reference_index": ref_index,
                    "om_index": pred_index,
                    "match_iou": iou,
                    "hand21_mean_px": hand_err["mean"],
                    "hand21_median_px": hand_err["median"],
                    "hand21_p95_px": hand_err["p95"],
                    "hand21_max_px": hand_err["max"],
                    "hand_bbox_iou": iou,
                    "hand_bbox_mean_abs_px": float(np.mean(hand_box_abs)),
                    "hand_bbox_max_abs_px": float(np.max(hand_box_abs)),
                    "reference_handedness": ref["handedness"],
                    "reference_handedness_score": ref["handedness_score"],
                    "om_hand_score": pred["hand_score"],
                    "om_source_roi": pred["source_roi"],
                }
                if ref["palm7"] is not None and pred["palm7"] is not None:
                    palm_err = point_error(pred["palm7"], ref["palm7"])
                    row.update(
                        {
                            "palm7_mean_px": palm_err["mean"],
                            "palm7_p95_px": palm_err["p95"],
                            "palm7_max_px": palm_err["max"],
                        }
                    )
                else:
                    row.update({"palm7_mean_px": math.nan, "palm7_p95_px": math.nan, "palm7_max_px": math.nan})
                if ref["palm_box"] is not None and pred["palm_box"] is not None:
                    palm_box_abs = np.abs(pred["palm_box"] - ref["palm_box"])
                    row.update(
                        {
                            "palm_box_mean_abs_px": float(np.mean(palm_box_abs)),
                            "palm_box_max_abs_px": float(np.max(palm_box_abs)),
                            "palm_score_abs": abs(pred["score"] - ref["palm_score"])
                            if math.isfinite(pred["score"]) and math.isfinite(ref["palm_score"])
                            else math.nan,
                        }
                    )
                else:
                    row.update({"palm_box_mean_abs_px": math.nan, "palm_box_max_abs_px": math.nan, "palm_score_abs": math.nan})
                if ref["next_tracking_roi"] is not None and pred["next_tracking_roi"] is not None:
                    roi_abs = np.abs(pred["next_tracking_roi"] - ref["next_tracking_roi"])
                    row.update({"next_roi_mean_abs": float(np.mean(roi_abs)), "next_roi_max_abs": float(np.max(roi_abs))})
                else:
                    row.update({"next_roi_mean_abs": math.nan, "next_roi_max_abs": math.nan})
                match_rows.append(row)

            prediction_rows.append(
                {
                    "frame_index": frame_index,
                    "reference": ref_frame[reference_stream],
                    "om": result.predictions,
                    "om_timings": result.timings,
                    "om_debug": result.debug,
                }
            )

            if args.save_vis and saved_vis < args.save_vis and (refs or preds):
                cv2.imwrite(str(vis_dir / f"frame_{frame_index:06d}.jpg"), draw_overlay(frame, refs, preds, frame_index))
                saved_vis += 1
            processed_frames += 1
    finally:
        cap.release()
        pipeline.close()

    matched_hands = len(match_rows)
    count_mismatch_rate = count_mismatch_frames / max(processed_frames, 1)
    summary: dict[str, Any] = {
        "task": "eval_video_mediapipe_om",
        "video": str(video_path),
        "annotations": str(annotations_path),
        "output_dir": str(output_dir),
        "model_set": args.model_set,
        "om_detector": str(om_detector),
        "om_landmark": str(om_landmark),
        "pipeline_mode": args.pipeline_mode,
        "reference_stream": reference_stream,
        "source_frame_count": source_frame_count,
        "processed_frames": processed_frames,
        "frame_stride": args.frame_stride,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "match_iou": args.match_iou,
        "matched_hands": matched_hands,
        "unmatched_reference_hands": unmatched_ref_total,
        "unmatched_om_hands": unmatched_om_total,
        "count_mismatch_frames": count_mismatch_frames,
        "count_mismatch_rate": count_mismatch_rate,
        "visualizations": saved_vis,
        **scalar_summary([row["hand21_mean_px"] for row in match_rows], "hand21_mean_px"),
        **scalar_summary([row["hand21_p95_px"] for row in match_rows], "hand21_p95_px"),
        **scalar_summary([row["hand21_max_px"] for row in match_rows], "hand21_max_px"),
        **scalar_summary([row["hand_bbox_iou"] for row in match_rows], "hand_bbox_iou"),
        **scalar_summary([row["hand_bbox_mean_abs_px"] for row in match_rows], "hand_bbox_mean_abs_px"),
        **scalar_summary([row["palm7_mean_px"] for row in match_rows], "palm7_mean_px"),
        **scalar_summary([row["palm7_p95_px"] for row in match_rows], "palm7_p95_px"),
        **scalar_summary([row["palm_box_mean_abs_px"] for row in match_rows], "palm_box_mean_abs_px"),
        **scalar_summary([row["palm_score_abs"] for row in match_rows], "palm_score_abs"),
        **scalar_summary([row["next_roi_mean_abs"] for row in match_rows], "next_roi_mean_abs"),
        **scalar_summary([row["next_roi_max_abs"] for row in match_rows], "next_roi_max_abs"),
        **timing_summary(frame_rows, "om_total_ms"),
        **timing_summary(frame_rows, "om_detector_ms"),
        **timing_summary(frame_rows, "om_landmark_ms"),
        "thresholds": {
            "max_mean_hand21_px": args.max_mean_hand21_px,
            "max_p95_hand21_px": args.max_p95_hand21_px,
            "max_count_mismatch_rate": args.max_count_mismatch_rate,
        },
    }
    summary["consistent"] = bool(
        matched_hands > 0
        and summary["hand21_mean_px_mean"] <= args.max_mean_hand21_px
        and summary["hand21_mean_px_p95"] <= args.max_p95_hand21_px
        and count_mismatch_rate <= args.max_count_mismatch_rate
    )

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(prediction_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "frames.csv", frame_rows)
    write_csv(output_dir / "matches.csv", match_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
