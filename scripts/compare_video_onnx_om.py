#!/usr/bin/env python3
"""Compare ONNX and Ascend OM hand pipeline results on a video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from hand_pipeline.eval import box_iou
from hand_pipeline.two_stage import OmHandPipeline
from hand_pipeline.two_stage import OnnxHandPipeline
from hand_pipeline.two_stage import summarize_times
from hand_pipeline.visualization import HAND_EDGES


DEFAULT_ONNX_DETECTOR = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx"
DEFAULT_ONNX_LANDMARK = "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx"
DEFAULT_OM_DETECTOR = "models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om"
DEFAULT_OM_LANDMARK = "models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/eval_videos/test.mp4")
    parser.add_argument("--onnx-detector", default=DEFAULT_ONNX_DETECTOR)
    parser.add_argument("--onnx-landmark", default=DEFAULT_ONNX_LANDMARK)
    parser.add_argument("--om-detector", default=DEFAULT_OM_DETECTOR)
    parser.add_argument("--om-landmark", default=DEFAULT_OM_LANDMARK)
    parser.add_argument("--output-dir", default="runs/video_onnx_om_compare/test")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--roi-scale", type=float, default=2.6)
    parser.add_argument("--shift-y", type=float, default=-0.5)
    parser.add_argument("--rotation-offset-degrees", type=float, default=0.0)
    parser.add_argument("--pipeline-mode", choices=["tracking", "image"], default="tracking")
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=8)
    parser.add_argument("--reload-detector-each-frame", action="store_true")
    parser.add_argument("--max-mean-hand21-px", type=float, default=2.0)
    parser.add_argument("--max-p95-hand21-px", type=float, default=5.0)
    parser.add_argument("--max-count-mismatch-rate", type=float, default=0.01)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mean_point_error(a: Any, b: Any) -> tuple[float, float, float]:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    err = np.linalg.norm(a_arr - b_arr, axis=1)
    return float(np.mean(err)), float(np.max(err)), float(np.percentile(err, 95))


def match_frame_predictions(
    onnx_predictions: list[dict[str, Any]],
    om_predictions: list[dict[str, Any]],
    match_iou: float,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    used_om: set[int] = set()
    for onnx_index, onnx_pred in enumerate(onnx_predictions):
        if not om_predictions:
            continue
        onnx_box = np.asarray(onnx_pred["box"], dtype=np.float32)
        om_boxes = np.stack([np.asarray(pred["box"], dtype=np.float32) for pred in om_predictions], axis=0)
        ious = box_iou(onnx_box, om_boxes)
        order = np.argsort(ious)[::-1]
        matched_om: int | None = None
        matched_iou = 0.0
        for pos in order:
            om_index = int(pos)
            if om_index in used_om:
                continue
            matched_iou = float(ious[om_index])
            if matched_iou >= match_iou:
                matched_om = om_index
                used_om.add(om_index)
            break
        if matched_om is None:
            continue

        om_pred = om_predictions[matched_om]
        box_abs = np.abs(np.asarray(onnx_pred["box"], dtype=np.float32) - np.asarray(om_pred["box"], dtype=np.float32))
        palm_mean, palm_max, palm_p95 = mean_point_error(onnx_pred["palm7"], om_pred["palm7"])
        hand_mean, hand_max, hand_p95 = mean_point_error(onnx_pred["hand21"], om_pred["hand21"])
        rows.append(
            {
                "onnx_index": onnx_index,
                "om_index": matched_om,
                "match_iou": matched_iou,
                "box_mean_abs_px": float(np.mean(box_abs)),
                "box_max_abs_px": float(np.max(box_abs)),
                "palm7_mean_px": palm_mean,
                "palm7_p95_px": palm_p95,
                "palm7_max_px": palm_max,
                "hand21_mean_px": hand_mean,
                "hand21_p95_px": hand_p95,
                "hand21_max_px": hand_max,
                "score_abs": abs(float(onnx_pred["score"]) - float(om_pred["score"])),
                "hand_score_abs": abs(float(onnx_pred["hand_score"]) - float(om_pred["hand_score"])),
            }
        )
    return rows, len(onnx_predictions) - len(rows), len(om_predictions) - len(used_om)


def summarize_values(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_max": math.nan,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
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


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def draw_predictions(canvas: np.ndarray, predictions: list[dict[str, Any]], color: tuple[int, int, int], label: str) -> None:
    for index, pred in enumerate(predictions):
        box = np.asarray(pred["box"], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        points = np.asarray(pred["hand21"], dtype=np.float32)
        for a, b in HAND_EDGES:
            pt_a = tuple(np.round(points[a]).astype(int))
            pt_b = tuple(np.round(points[b]).astype(int))
            cv2.line(canvas, pt_a, pt_b, color, 2)
        for point in points:
            cv2.circle(canvas, tuple(np.round(point[:2]).astype(int)), 3, color, -1)
        cv2.putText(
            canvas,
            f"{label}{index + 1}:{float(pred['score']):.2f}",
            (max(x1, 8), max(y1 - 8, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    verdict = "consistent" if summary["consistent"] else "different"
    lines = [
        "# Video ONNX vs OM Comparison",
        "",
        "## Verdict",
        "",
        f"- result: `{verdict}`",
        f"- count mismatch rate: `{fmt(summary['count_mismatch_rate'], 6)}`",
        f"- matched hands: `{summary['matched_hands']}`",
        f"- hand21 mean px: `{fmt(summary['hand21_mean_px_mean'])}`",
        f"- hand21 p95 px: `{fmt(summary['hand21_mean_px_p95'])}`",
        f"- hand21 max px: `{fmt(summary['hand21_mean_px_max'])}`",
        "",
        "## Settings",
        "",
        f"- video: `{summary['video']}`",
        f"- processed frames: `{summary['processed_frames']}`",
        f"- source frame count: `{summary['source_frame_count']}`",
        f"- resolution: `{summary['width']}x{summary['height']}`",
        f"- fps: `{fmt(summary['fps'])}`",
        f"- ONNX detector: `{summary['onnx_detector']}`",
        f"- ONNX landmark: `{summary['onnx_landmark']}`",
        f"- OM detector: `{summary['om_detector']}`",
        f"- OM landmark: `{summary['om_landmark']}`",
        "",
        "## Difference Metrics",
        "",
        "| Metric | Mean | Median | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| box mean abs px | {fmt(summary['box_mean_abs_px_mean'])} | {fmt(summary['box_mean_abs_px_median'])} | {fmt(summary['box_mean_abs_px_p95'])} | {fmt(summary['box_mean_abs_px_max'])} |",
        f"| palm7 mean px | {fmt(summary['palm7_mean_px_mean'])} | {fmt(summary['palm7_mean_px_median'])} | {fmt(summary['palm7_mean_px_p95'])} | {fmt(summary['palm7_mean_px_max'])} |",
        f"| hand21 mean px | {fmt(summary['hand21_mean_px_mean'])} | {fmt(summary['hand21_mean_px_median'])} | {fmt(summary['hand21_mean_px_p95'])} | {fmt(summary['hand21_mean_px_max'])} |",
        f"| score abs | {fmt(summary['score_abs_mean'], 6)} | {fmt(summary['score_abs_median'], 6)} | {fmt(summary['score_abs_p95'], 6)} | {fmt(summary['score_abs_max'], 6)} |",
        f"| hand score abs | {fmt(summary['hand_score_abs_mean'], 6)} | {fmt(summary['hand_score_abs_median'], 6)} | {fmt(summary['hand_score_abs_p95'], 6)} | {fmt(summary['hand_score_abs_max'], 6)} |",
        "",
        "## Timing",
        "",
        "| Backend | Mean ms | Median ms | P95 ms |",
        "| --- | ---: | ---: | ---: |",
        f"| ONNX total | {fmt(summary['onnx_total_mean_ms'])} | {fmt(summary['onnx_total_median_ms'])} | {fmt(summary['onnx_total_p95_ms'])} |",
        f"| OM total | {fmt(summary['om_total_mean_ms'])} | {fmt(summary['om_total_median_ms'])} | {fmt(summary['om_total_p95_ms'])} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    video_path = resolve_path(args.video)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    onnx_detector = resolve_path(args.onnx_detector)
    onnx_landmark = resolve_path(args.onnx_landmark)
    om_detector = resolve_path(args.om_detector)
    om_landmark = resolve_path(args.om_landmark)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    onnx_pipeline = OnnxHandPipeline(
        onnx_detector,
        onnx_landmark,
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        max_hands=args.max_hands,
        min_hand_score=args.min_hand_score,
        roi_scale=args.roi_scale,
        shift_y=args.shift_y,
        rotation_offset_degrees=args.rotation_offset_degrees,
        mode=args.pipeline_mode,
    )
    om_pipeline = OmHandPipeline(
        om_detector,
        om_landmark,
        device_id=args.device_id,
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        max_hands=args.max_hands,
        min_hand_score=args.min_hand_score,
        roi_scale=args.roi_scale,
        shift_y=args.shift_y,
        rotation_offset_degrees=args.rotation_offset_degrees,
        reload_detector_each_frame=args.reload_detector_each_frame,
        mode=args.pipeline_mode,
    )

    frame_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    predictions_rows: list[dict[str, Any]] = []
    onnx_total_ms: list[float] = []
    om_total_ms: list[float] = []
    saved_vis = 0
    processed_frames = 0
    count_mismatch_frames = 0
    onnx_unmatched_total = 0
    om_unmatched_total = 0

    try:
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

            onnx_result = onnx_pipeline.infer(frame)
            om_result = om_pipeline.infer(frame)
            onnx_total_ms.append(onnx_result.timings["total_ms"])
            om_total_ms.append(om_result.timings["total_ms"])

            frame_matches, onnx_unmatched, om_unmatched = match_frame_predictions(
                onnx_result.predictions,
                om_result.predictions,
                args.match_iou,
            )
            onnx_unmatched_total += onnx_unmatched
            om_unmatched_total += om_unmatched
            if len(onnx_result.predictions) != len(om_result.predictions):
                count_mismatch_frames += 1

            frame_row = {
                "frame_index": frame_index,
                "onnx_hands": len(onnx_result.predictions),
                "om_hands": len(om_result.predictions),
                "matched_hands": len(frame_matches),
                "onnx_unmatched_hands": onnx_unmatched,
                "om_unmatched_hands": om_unmatched,
                "onnx_total_ms": onnx_result.timings["total_ms"],
                "om_total_ms": om_result.timings["total_ms"],
            }
            frame_rows.append(frame_row)
            for row in frame_matches:
                row["frame_index"] = frame_index
                match_rows.append(row)
            predictions_rows.append(
                {
                    "frame_index": frame_index,
                    "onnx": onnx_result.predictions,
                    "om": om_result.predictions,
                }
            )

            if args.save_vis and saved_vis < args.save_vis and (onnx_result.predictions or om_result.predictions):
                canvas = frame.copy()
                draw_predictions(canvas, onnx_result.predictions, (255, 80, 0), "onnx")
                draw_predictions(canvas, om_result.predictions, (40, 220, 120), "om")
                cv2.putText(
                    canvas,
                    "ONNX blue/orange, OM green",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(vis_dir / f"frame_{frame_index:06d}.jpg"), canvas)
                saved_vis += 1
            processed_frames += 1
    finally:
        cap.release()
        om_pipeline.close()

    matched_hands = len(match_rows)
    count_mismatch_rate = count_mismatch_frames / max(processed_frames, 1)
    summary: dict[str, Any] = {
        "task": "compare_video_onnx_om",
        "video": str(video_path),
        "output_dir": str(output_dir),
        "source_frame_count": source_frame_count,
        "processed_frames": processed_frames,
        "frame_stride": args.frame_stride,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "onnx_detector": str(onnx_detector),
        "onnx_landmark": str(onnx_landmark),
        "om_detector": str(om_detector),
        "om_landmark": str(om_landmark),
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "max_hands": args.max_hands,
        "min_hand_score": args.min_hand_score,
        "match_iou": args.match_iou,
        "reload_detector_each_frame": args.reload_detector_each_frame,
        "pipeline_mode": args.pipeline_mode,
        "matched_hands": matched_hands,
        "onnx_unmatched_hands": onnx_unmatched_total,
        "om_unmatched_hands": om_unmatched_total,
        "count_mismatch_frames": count_mismatch_frames,
        "count_mismatch_rate": count_mismatch_rate,
        "visualizations": saved_vis,
        **summarize_values([float(row["box_mean_abs_px"]) for row in match_rows], "box_mean_abs_px"),
        **summarize_values([float(row["palm7_mean_px"]) for row in match_rows], "palm7_mean_px"),
        **summarize_values([float(row["hand21_mean_px"]) for row in match_rows], "hand21_mean_px"),
        **summarize_values([float(row["score_abs"]) for row in match_rows], "score_abs"),
        **summarize_values([float(row["hand_score_abs"]) for row in match_rows], "hand_score_abs"),
        **summarize_times(onnx_total_ms, "onnx_total"),
        **summarize_times(om_total_ms, "om_total"),
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
        and onnx_unmatched_total == 0
        and om_unmatched_total == 0
    )

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "frames.csv", frame_rows)
    write_csv(output_dir / "matches.csv", match_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["consistent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
