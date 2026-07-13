#!/usr/bin/env python3
"""Render side-by-side review videos for MediaPipe-vs-OM video evaluation."""

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

from hand_pipeline.visualization import HAND_EDGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/eval_videos/demo1.mp4")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--draw-roi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-palm", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_csv_by_frame(path: Path) -> dict[int, list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(int(row["frame_index"]), []).append(row)
    return rows


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def draw_text_panel(canvas: np.ndarray, lines: list[str], *, x: int, y: int, width: int) -> None:
    line_height = 24
    height = 12 + line_height * len(lines)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (20, 24, 30), -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (90, 100, 110), 1)
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x + 8, y + 24 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 240, 245),
            2,
            cv2.LINE_AA,
        )


def draw_hand21(canvas: np.ndarray, points: Any, color: tuple[int, int, int], *, thickness: int = 2) -> None:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 21:
        return
    for start, end in HAND_EDGES:
        cv2.line(
            canvas,
            tuple(np.round(arr[start, :2]).astype(int)),
            tuple(np.round(arr[end, :2]).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    for point in arr:
        xy = tuple(np.round(point[:2]).astype(int))
        cv2.circle(canvas, xy, 3, (245, 245, 245), -1)
        cv2.circle(canvas, xy, 3, color, 1)


def draw_box(canvas: np.ndarray, box: Any, color: tuple[int, int, int], label: str = "") -> None:
    if box is None:
        return
    arr = np.asarray(box, dtype=np.float32)
    if arr.shape[0] < 4 or not np.all(np.isfinite(arr[:4])):
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in arr[:4]]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(canvas, label, (max(x1, 8), max(y1 - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def draw_points(canvas: np.ndarray, points: Any, color: tuple[int, int, int]) -> None:
    if points is None:
        return
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2:
        return
    for point in arr:
        cv2.circle(canvas, tuple(np.round(point[:2]).astype(int)), 3, color, -1)


def draw_roi(canvas: np.ndarray, roi: dict[str, Any] | None, color: tuple[int, int, int]) -> None:
    if not roi:
        return
    pixel = roi.get("pixel", roi)
    required = ("x_center_px", "y_center_px", "width_px", "height_px", "rotation")
    if not all(key in pixel for key in required):
        return
    center = (float(pixel["x_center_px"]), float(pixel["y_center_px"]))
    size = (float(pixel["width_px"]), float(pixel["height_px"]))
    angle = float(pixel["rotation"]) * 180.0 / math.pi
    pts = cv2.boxPoints((center, size, angle)).astype(np.int32)
    cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)


def draw_reference_panel(
    canvas: np.ndarray,
    reference: dict[str, Any],
    *,
    draw_palm: bool,
    draw_roi_enabled: bool,
) -> None:
    if draw_palm:
        for palm in reference.get("palm_detections", []):
            draw_box(canvas, palm.get("palm_bbox_xyxy_px"), (30, 180, 255), f"palm {float_or_nan(palm.get('score')):.2f}")
            draw_points(canvas, palm.get("palm7_keypoints_px"), (30, 180, 255))
    for hand in reference.get("hands", []):
        draw_hand21(canvas, hand.get("hand21_keypoints_px"), (255, 120, 35), thickness=2)
        draw_box(canvas, hand.get("hand_bbox_xyxy_px"), (255, 120, 35), f"mp {hand.get('hand_index', 0)}")
        if draw_roi_enabled:
            draw_roi(canvas, hand.get("roi_from_palm_detection"), (0, 150, 255))
            draw_roi(canvas, hand.get("next_tracking_roi"), (255, 220, 80))


def draw_om_panel(
    canvas: np.ndarray,
    predictions: list[dict[str, Any]],
    *,
    draw_palm: bool,
    draw_roi_enabled: bool,
) -> None:
    for pred in predictions:
        if draw_palm:
            draw_box(canvas, pred.get("box"), (30, 180, 255), f"palm {float_or_nan(pred.get('score')):.2f}")
            draw_points(canvas, pred.get("palm7"), (30, 180, 255))
        draw_hand21(canvas, pred.get("hand21"), (60, 230, 120), thickness=2)
        draw_box(canvas, _hand_box_from_points(pred.get("hand21")), (60, 230, 120), f"om {pred.get('hand_index', 0)}")
        if draw_roi_enabled:
            draw_roi(canvas, pred.get("palm_roi"), (0, 150, 255))
            draw_roi(canvas, pred.get("hand_roi"), (255, 220, 80))
            draw_roi(canvas, pred.get("next_tracking_roi"), (80, 180, 255))


def _hand_box_from_points(points: Any) -> list[float] | None:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    xy = arr[:, :2]
    return [float(np.min(xy[:, 0])), float(np.min(xy[:, 1])), float(np.max(xy[:, 0])), float(np.max(xy[:, 1]))]


def frame_metric_lines(
    frame_index: int,
    frame_rows: dict[int, list[dict[str, str]]],
    match_rows: dict[int, list[dict[str, str]]],
    predictions: dict[str, Any],
) -> list[str]:
    frame = frame_rows.get(frame_index, [{}])[0]
    matches = match_rows.get(frame_index, [])
    h21 = [float_or_nan(row.get("hand21_mean_px")) for row in matches]
    h21 = [value for value in h21 if math.isfinite(value)]
    p95 = [float_or_nan(row.get("hand21_p95_px")) for row in matches]
    p95 = [value for value in p95 if math.isfinite(value)]
    source_counts: dict[str, int] = {}
    for pred in predictions.get("om", []):
        source = str(pred.get("source_roi", ""))
        source_counts[source] = source_counts.get(source, 0) + 1
    return [
        f"frame {frame_index}",
        f"MP hands {frame.get('reference_hands', '?')}  OM hands {frame.get('om_hands', '?')}  matched {frame.get('matched_hands', '?')}",
        f"unmatched MP {frame.get('unmatched_reference_hands', '?')}  unmatched OM {frame.get('unmatched_om_hands', '?')}",
        f"hand21 mean {np.mean(h21):.2f}px  p95 {np.mean(p95):.2f}px" if h21 else "hand21 mean nan",
        f"OM total {float_or_nan(frame.get('om_total_ms')):.2f} ms  source {source_counts}",
    ]


def main() -> int:
    args = parse_args()
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")
    video_path = resolve_path(args.video)
    eval_dir = resolve_path(args.eval_dir)
    output_path = resolve_path(args.output) if args.output else eval_dir / "review_side_by_side.mp4"

    predictions = json.loads((eval_dir / "predictions.json").read_text(encoding="utf-8"))
    predictions_by_frame = {int(item["frame_index"]): item for item in predictions}
    frame_rows = load_csv_by_frame(eval_dir / "frames.csv")
    match_rows = load_csv_by_frame(eval_dir / "matches.csv")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaled_size = (max(1, int(round(source_width * args.scale))), max(1, int(round(source_height * args.scale))))
    out_size = (scaled_size[0] * 2, scaled_size[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    try:
        frame_index = -1
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            item = predictions_by_frame.get(frame_index)
            if item is None:
                continue
            left = frame.copy()
            right = frame.copy()
            draw_reference_panel(left, item["reference"], draw_palm=args.draw_palm, draw_roi_enabled=args.draw_roi)
            draw_om_panel(right, item["om"], draw_palm=args.draw_palm, draw_roi_enabled=args.draw_roi)
            draw_text_panel(left, ["MediaPipe reference"] + frame_metric_lines(frame_index, frame_rows, match_rows, item), x=12, y=12, width=650)
            draw_text_panel(right, ["Ascend OM pipeline"] + frame_metric_lines(frame_index, frame_rows, match_rows, item), x=12, y=12, width=650)
            if args.scale != 1.0:
                left = cv2.resize(left, scaled_size, interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, scaled_size, interpolation=cv2.INTER_AREA)
            writer.write(np.concatenate([left, right], axis=1))
            written += 1
            if args.max_frames and written >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()

    print(json.dumps({"output": str(output_path), "frames": written, "fps": source_fps, "size": out_size}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
