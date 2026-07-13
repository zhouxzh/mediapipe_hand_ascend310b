#!/usr/bin/env python3
"""Annotate a video with MediaPipe legacy graph outputs.

This script directly runs MediaPipe's legacy hand tracking graph instead of the
high-level ``mp.solutions.hands.Hands`` wrapper. It subscribes to the graph
streams that are needed as references for this repository:

- ``palm_detections``: raw MediaPipe palm detector detections.
- ``hand_rects_from_palm_detections``: landmark ROIs produced from palms.
- ``multi_hand_landmarks``: 21 image-space hand landmarks.
- ``hand_rects_from_landmarks``: next-frame tracking ROIs.

By default, two graph instances are run on the same video:

- ``image``: ``use_prev_landmarks=False`` so palm detection runs every frame.
- ``tracking``: ``use_prev_landmarks=True`` so MediaPipe's video tracking graph
  can skip palm detection after a hand is tracked.

Use ``--running-mode image`` or ``--running-mode tracking`` to export only one
stream for full-dataset runs where output size and runtime matter.
"""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.visualization import HAND_EDGES  # noqa: E402


BINARY_GRAPH_PATH = "mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb"
GRAPH_OUTPUTS = [
    "multi_hand_landmarks",
    "multi_hand_world_landmarks",
    "multi_handedness",
    "palm_detections",
    "hand_rects_from_palm_detections",
    "hand_rects_from_landmarks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/eval_videos/test.mp4")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max-num-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--running-mode", choices=["both", "image", "tracking"], default="both")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-vis", type=int, default=16)
    parser.add_argument("--draw-palm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-landmarks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-roi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing annotation output directory.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_output_dir(video_path: Path) -> Path:
    return video_path.parent / "annotations" / video_path.stem


def format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean_ms": math.nan,
            f"{prefix}_median_ms": math.nan,
            f"{prefix}_p95_ms": math.nan,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean_ms": float(np.mean(arr)),
        f"{prefix}_median_ms": float(np.median(arr)),
        f"{prefix}_p95_ms": float(np.percentile(arr, 95)),
    }


def selected_modes(running_mode: str) -> list[str]:
    if running_mode == "both":
        return ["image", "tracking"]
    return [running_mode]


def empty_stream() -> dict[str, Any]:
    return {
        "palm_detections": [],
        "hand_rects_from_palm_detections": [],
        "hands": [],
        "hand_rects_from_landmarks": [],
    }


class MediaPipeHandGraph:
    def __init__(
        self,
        *,
        mediapipe_module: Any,
        use_prev_landmarks: bool,
        model_complexity: int,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        from mediapipe.python.solution_base import SolutionBase

        self.graph = SolutionBase(
            binary_graph_path=BINARY_GRAPH_PATH,
            side_inputs={
                "model_complexity": int(model_complexity),
                "num_hands": int(max_num_hands),
                "use_prev_landmarks": bool(use_prev_landmarks),
            },
            calculator_params={
                "palmdetectioncpu__TensorsToDetectionsCalculator.min_score_thresh": float(
                    min_detection_confidence
                ),
                "handlandmarkcpu__ThresholdingCalculator.threshold": float(min_tracking_confidence),
            },
            outputs=GRAPH_OUTPUTS,
        )
        self.mediapipe_version = str(mediapipe_module.__version__)

    def process(self, image_rgb: np.ndarray) -> Any:
        return self.graph.process({"image": image_rgb})

    def close(self) -> None:
        self.graph.close()


def normalized_landmarks_to_pixel_array(landmarks: Any, width: int, height: int) -> np.ndarray:
    points = []
    for landmark in landmarks.landmark:
        points.append(
            [
                float(landmark.x) * float(width),
                float(landmark.y) * float(height),
                float(landmark.z) * float(width),
            ]
        )
    return np.asarray(points, dtype=np.float32)


def normalized_landmarks_to_list(landmarks: Any) -> list[list[float]]:
    return [[float(point.x), float(point.y), float(point.z)] for point in landmarks.landmark]


def hand_bbox_from_landmarks(hand21_px: np.ndarray, width: int, height: int) -> list[float]:
    xy = np.asarray(hand21_px, dtype=np.float32)[:, :2]
    x1 = float(np.clip(np.nanmin(xy[:, 0]), 0.0, float(width)))
    y1 = float(np.clip(np.nanmin(xy[:, 1]), 0.0, float(height)))
    x2 = float(np.clip(np.nanmax(xy[:, 0]), 0.0, float(width)))
    y2 = float(np.clip(np.nanmax(xy[:, 1]), 0.0, float(height)))
    return [x1, y1, x2, y2]


def handedness_to_dict(item: Any | None) -> dict[str, Any]:
    if item is None or not getattr(item, "classification", None):
        return {"label": "", "score": math.nan, "index": -1}
    cls = item.classification[0]
    return {
        "label": str(cls.label),
        "score": float(cls.score),
        "index": int(cls.index),
    }


def detection_to_dict(detection: Any, detection_index: int, width: int, height: int) -> dict[str, Any]:
    location = detection.location_data
    relative_box = location.relative_bounding_box
    x1_norm = float(relative_box.xmin)
    y1_norm = float(relative_box.ymin)
    x2_norm = float(relative_box.xmin + relative_box.width)
    y2_norm = float(relative_box.ymin + relative_box.height)
    keypoints_norm = [[float(point.x), float(point.y)] for point in location.relative_keypoints]
    keypoints_px = [[point[0] * float(width), point[1] * float(height)] for point in keypoints_norm]
    box_px = [
        x1_norm * float(width),
        y1_norm * float(height),
        x2_norm * float(width),
        y2_norm * float(height),
    ]
    return {
        "detection_index": int(detection_index),
        "score": float(detection.score[0]) if detection.score else math.nan,
        "label": str(detection.label[0]) if detection.label else "",
        "label_id": int(detection.label_id[0]) if detection.label_id else -1,
        "location_format": int(location.format),
        "palm_bbox_xyxy_px": box_px,
        "palm_bbox_xyxy_norm": [x1_norm, y1_norm, x2_norm, y2_norm],
        "palm7_keypoints_px": keypoints_px,
        "palm7_keypoints_norm": keypoints_norm,
        "palm_source": "mediapipe_palm_detections",
    }


def rect_to_dict(rect: Any, rect_index: int, width: int, height: int) -> dict[str, Any]:
    normalized = {
        "x_center": float(rect.x_center),
        "y_center": float(rect.y_center),
        "width": float(rect.width),
        "height": float(rect.height),
        "rotation": float(rect.rotation),
    }
    pixel = {
        "x_center_px": float(rect.x_center) * float(width),
        "y_center_px": float(rect.y_center) * float(height),
        "width_px": float(rect.width) * float(width),
        "height_px": float(rect.height) * float(height),
        "rotation": float(rect.rotation),
    }
    return {"rect_index": int(rect_index), "normalized": normalized, "pixel": pixel}


def graph_outputs_to_streams(results: Any, width: int, height: int, mode: str) -> dict[str, Any]:
    palms = [
        detection_to_dict(detection, index, width, height)
        for index, detection in enumerate(list(results.palm_detections or []))
    ]
    palm_rois = [
        rect_to_dict(rect, index, width, height)
        for index, rect in enumerate(list(results.hand_rects_from_palm_detections or []))
    ]
    tracking_rois = [
        rect_to_dict(rect, index, width, height)
        for index, rect in enumerate(list(results.hand_rects_from_landmarks or []))
    ]
    hands = hand_records_from_graph_outputs(
        results,
        palms=palms,
        palm_rois=palm_rois,
        tracking_rois=tracking_rois,
        width=width,
        height=height,
        mode=mode,
    )
    return {
        "palm_detections": palms,
        "hand_rects_from_palm_detections": palm_rois,
        "hands": hands,
        "hand_rects_from_landmarks": tracking_rois,
    }


def hand_records_from_graph_outputs(
    results: Any,
    *,
    palms: list[dict[str, Any]],
    palm_rois: list[dict[str, Any]],
    tracking_rois: list[dict[str, Any]],
    width: int,
    height: int,
    mode: str,
) -> list[dict[str, Any]]:
    landmarks_list = list(results.multi_hand_landmarks or [])
    world_landmarks_list = list(results.multi_hand_world_landmarks or [])
    handedness_list = list(results.multi_handedness or [])
    records: list[dict[str, Any]] = []
    for hand_index, landmarks in enumerate(landmarks_list):
        hand21_px = normalized_landmarks_to_pixel_array(landmarks, width, height)
        hand21_norm = normalized_landmarks_to_list(landmarks)
        world = (
            normalized_landmarks_to_list(world_landmarks_list[hand_index])
            if hand_index < len(world_landmarks_list)
            else []
        )
        handness = handedness_to_dict(handedness_list[hand_index] if hand_index < len(handedness_list) else None)
        palm = palms[hand_index] if hand_index < len(palms) else None
        palm_roi = palm_rois[hand_index] if hand_index < len(palm_rois) else None
        next_tracking_roi = tracking_rois[hand_index] if hand_index < len(tracking_rois) else None
        records.append(
            {
                "hand_index": int(hand_index),
                "source": mode,
                "handedness": handness["label"],
                "handedness_score": handness["score"],
                "handedness_index": handness["index"],
                "palm_detection_index": int(palm["detection_index"]) if palm is not None else -1,
                "palm_detection": palm,
                "palm_source": palm["palm_source"] if palm is not None else "none_detector_skipped_or_unmatched",
                "palm_bbox_xyxy_px": palm["palm_bbox_xyxy_px"] if palm is not None else None,
                "palm_bbox_xyxy_norm": palm["palm_bbox_xyxy_norm"] if palm is not None else None,
                "palm7_keypoints_px": palm["palm7_keypoints_px"] if palm is not None else None,
                "palm7_keypoints_norm": palm["palm7_keypoints_norm"] if palm is not None else None,
                "hand_bbox_xyxy_px": hand_bbox_from_landmarks(hand21_px, width, height),
                "hand21_keypoints_px": hand21_px.astype(float).tolist(),
                "hand21_keypoints_norm": hand21_norm,
                "world21_keypoints_m": world,
                "roi_from_palm_detection": palm_roi,
                "next_tracking_roi": next_tracking_roi,
            }
        )
    return records


def draw_records(
    frame_bgr: np.ndarray,
    image_streams: dict[str, Any],
    tracking_streams: dict[str, Any],
    *,
    draw_palm: bool,
    draw_landmarks: bool,
    draw_roi: bool,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if draw_palm:
        _draw_palms(canvas, image_streams["palm_detections"], (35, 180, 255), "img palm")
        _draw_palms(canvas, tracking_streams["palm_detections"], (80, 230, 120), "trk palm")
    if draw_roi:
        _draw_rois(canvas, image_streams["hand_rects_from_palm_detections"], (0, 150, 255))
        _draw_rois(canvas, tracking_streams["hand_rects_from_landmarks"], (90, 220, 110))
    if draw_landmarks:
        _draw_hands(canvas, image_streams["hands"], (255, 100, 40), "img")
        _draw_hands(canvas, tracking_streams["hands"], (50, 230, 110), "trk")
    cv2.putText(
        canvas,
        "image-mode orange, tracking green",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_palms(canvas: np.ndarray, palms: list[dict[str, Any]], color: tuple[int, int, int], prefix: str) -> None:
    for palm in palms:
        box = np.asarray(palm["palm_bbox_xyxy_px"], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        for point in np.asarray(palm["palm7_keypoints_px"], dtype=np.float32):
            cv2.circle(canvas, tuple(np.round(point[:2]).astype(int)), 3, color, -1)
        score = float(palm["score"])
        label = f"{prefix}{int(palm['detection_index']) + 1}:{score:.2f}"
        cv2.putText(
            canvas,
            label,
            (max(x1, 8), max(y1 - 8, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_hands(canvas: np.ndarray, hands: list[dict[str, Any]], color: tuple[int, int, int], prefix: str) -> None:
    for hand in hands:
        points = np.asarray(hand["hand21_keypoints_px"], dtype=np.float32)
        for start, end in HAND_EDGES:
            a = tuple(np.round(points[start, :2]).astype(int))
            b = tuple(np.round(points[end, :2]).astype(int))
            cv2.line(canvas, a, b, color, 2)
        for point in points:
            xy = tuple(np.round(point[:2]).astype(int))
            cv2.circle(canvas, xy, 3, (245, 245, 245), -1)
            cv2.circle(canvas, xy, 3, color, 1)
        box = np.asarray(hand["hand_bbox_xyxy_px"], dtype=np.float32)
        x1, y1 = [int(round(float(v))) for v in box[:2]]
        label = f"{prefix}{int(hand['hand_index']) + 1}"
        if hand.get("handedness"):
            label += f" {hand['handedness']}"
        cv2.putText(
            canvas,
            label,
            (max(x1, 8), max(y1 - 8, 46)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_rois(canvas: np.ndarray, rois: list[dict[str, Any]], color: tuple[int, int, int]) -> None:
    for roi in rois:
        pixel = roi["pixel"]
        center = (float(pixel["x_center_px"]), float(pixel["y_center_px"]))
        size = (float(pixel["width_px"]), float(pixel["height_px"]))
        angle_degrees = float(pixel["rotation"]) * 180.0 / math.pi
        points = cv2.boxPoints((center, size, angle_degrees)).astype(np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


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


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MediaPipe Video Annotation",
        "",
        "## Inputs",
        "",
        f"- video: `{summary['video']}`",
        f"- graph: `{summary['binary_graph_path']}`",
        f"- mediapipe version: `{summary['mediapipe_version']}`",
        f"- running mode: `{summary['running_mode']}`",
        f"- frame count: `{summary['source_frame_count']}`",
        f"- processed frames: `{summary['processed_frames']}`",
        f"- resolution: `{summary['width']}x{summary['height']}`",
        f"- fps: `{format_float(summary['fps'])}`",
        "",
        "## Outputs",
        "",
        f"- annotations: `{summary['annotations_json']}`",
        f"- frames csv: `{summary['frames_csv']}`",
        f"- hands csv: `{summary['hands_csv']}`",
        f"- palm detections csv: `{summary['palm_detections_csv']}`",
        f"- palm ROI csv: `{summary['palm_rois_csv']}`",
        f"- tracking ROI csv: `{summary['tracking_rois_csv']}`",
        f"- visualization frames: `{summary['visualizations']}`",
        f"- annotated video: `{summary.get('annotated_video', '')}`",
        "",
        "## Reference Streams",
        "",
        "- `image` runs the same graph with `use_prev_landmarks=False`; the palm detector is evaluated every frame.",
        "- `tracking` runs the same graph with `use_prev_landmarks=True`; palm detections are only present on frames where the graph invokes the detector.",
        "- `palm_detections` are raw MediaPipe graph detections, including bbox and seven palm keypoints.",
        "- `hand_rects_from_landmarks` is the MediaPipe tracking ROI to compare against this repository's tracking logic.",
        "",
        "## Timing",
        "",
        "| Mode | Mean ms | Median ms | P95 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in summary.get("streams", ["image", "tracking"]):
        lines.append(
            f"| {mode} | "
            f"{format_float(summary[f'{mode}_mean_ms'])} | "
            f"{format_float(summary[f'{mode}_median_ms'])} | "
            f"{format_float(summary[f'{mode}_p95_ms'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_palm_rows(rows: list[dict[str, Any]], frame_index: int, mode: str, palms: list[dict[str, Any]]) -> None:
    for palm in palms:
        box = palm["palm_bbox_xyxy_px"]
        rows.append(
            {
                "frame_index": frame_index,
                "mode": mode,
                "detection_index": palm["detection_index"],
                "score": palm["score"],
                "label": palm["label"],
                "label_id": palm["label_id"],
                "palm_x1": box[0],
                "palm_y1": box[1],
                "palm_x2": box[2],
                "palm_y2": box[3],
                "palm_keypoints_json": json.dumps(palm["palm7_keypoints_px"], ensure_ascii=False),
            }
        )


def append_roi_rows(
    rows: list[dict[str, Any]],
    frame_index: int,
    mode: str,
    roi_source: str,
    rois: list[dict[str, Any]],
) -> None:
    for roi in rois:
        normalized = roi.get("normalized", {})
        pixel = roi.get("pixel", {})
        rows.append(
            {
                "frame_index": frame_index,
                "mode": mode,
                "roi_source": roi_source,
                "rect_index": roi.get("rect_index", -1),
                "x_center": normalized.get("x_center", math.nan),
                "y_center": normalized.get("y_center", math.nan),
                "width": normalized.get("width", math.nan),
                "height": normalized.get("height", math.nan),
                "rotation": normalized.get("rotation", math.nan),
                "x_center_px": pixel.get("x_center_px", math.nan),
                "y_center_px": pixel.get("y_center_px", math.nan),
                "width_px": pixel.get("width_px", math.nan),
                "height_px": pixel.get("height_px", math.nan),
                "rotation_px": pixel.get("rotation", math.nan),
            }
        )


def append_hand_rows(rows: list[dict[str, Any]], frame_index: int, mode: str, hands: list[dict[str, Any]]) -> None:
    for hand in hands:
        hand_box = hand["hand_bbox_xyxy_px"]
        palm_box = hand["palm_bbox_xyxy_px"] or [math.nan, math.nan, math.nan, math.nan]
        palm_roi = hand["roi_from_palm_detection"] or {}
        tracking_roi = hand["next_tracking_roi"] or {}
        palm_roi_px = palm_roi.get("pixel", {})
        tracking_roi_px = tracking_roi.get("pixel", {})
        rows.append(
            {
                "frame_index": frame_index,
                "mode": mode,
                "hand_index": hand["hand_index"],
                "handedness": hand["handedness"],
                "handedness_score": hand["handedness_score"],
                "palm_detection_index": hand["palm_detection_index"],
                "palm_source": hand["palm_source"],
                "palm_x1": palm_box[0],
                "palm_y1": palm_box[1],
                "palm_x2": palm_box[2],
                "palm_y2": palm_box[3],
                "hand_x1": hand_box[0],
                "hand_y1": hand_box[1],
                "hand_x2": hand_box[2],
                "hand_y2": hand_box[3],
                "palm_roi_x_center_px": palm_roi_px.get("x_center_px", math.nan),
                "palm_roi_y_center_px": palm_roi_px.get("y_center_px", math.nan),
                "palm_roi_width_px": palm_roi_px.get("width_px", math.nan),
                "palm_roi_height_px": palm_roi_px.get("height_px", math.nan),
                "palm_roi_rotation": palm_roi_px.get("rotation", math.nan),
                "tracking_roi_x_center_px": tracking_roi_px.get("x_center_px", math.nan),
                "tracking_roi_y_center_px": tracking_roi_px.get("y_center_px", math.nan),
                "tracking_roi_width_px": tracking_roi_px.get("width_px", math.nan),
                "tracking_roi_height_px": tracking_roi_px.get("height_px", math.nan),
                "tracking_roi_rotation": tracking_roi_px.get("rotation", math.nan),
            }
        )


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "mediapipe is required. Run this in the local conda env, for example: "
            "conda activate mediapipe_legacy"
        ) from exc

    video_path = resolve_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(video_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. Use --force to overwrite."
            )
        for child in output_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    annotated_video_path = ""
    if args.save_video:
        annotated_video_path = str(output_dir / "annotated_mediapipe.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps or 25.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create annotated video: {annotated_video_path}")

    graph_kwargs = {
        "mediapipe_module": mp,
        "model_complexity": args.model_complexity,
        "max_num_hands": args.max_num_hands,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
    }
    modes = selected_modes(args.running_mode)
    graphs: dict[str, MediaPipeHandGraph] = {}
    if "image" in modes:
        graphs["image"] = MediaPipeHandGraph(use_prev_landmarks=False, **graph_kwargs)
    if "tracking" in modes:
        graphs["tracking"] = MediaPipeHandGraph(use_prev_landmarks=True, **graph_kwargs)

    frame_rows: list[dict[str, Any]] = []
    hand_rows: list[dict[str, Any]] = []
    palm_rows: list[dict[str, Any]] = []
    palm_roi_rows: list[dict[str, Any]] = []
    tracking_roi_rows: list[dict[str, Any]] = []
    annotation_frames: list[dict[str, Any]] = []
    stream_ms: dict[str, list[float]] = {mode: [] for mode in modes}
    saved_vis = 0
    processed_frames = 0

    try:
        frame_index = -1
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index < args.start_frame:
                continue
            if (frame_index - args.start_frame) % args.frame_stride != 0:
                continue
            if args.max_frames and processed_frames >= args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            streams: dict[str, dict[str, Any]] = {}
            elapsed_by_mode: dict[str, float] = {}
            for mode, graph in graphs.items():
                t0 = time.perf_counter()
                graph_results = graph.process(frame_rgb)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                elapsed_by_mode[mode] = elapsed_ms
                stream_ms[mode].append(elapsed_ms)
                streams[mode] = graph_outputs_to_streams(graph_results, width, height, mode)

            image_streams = streams.get("image", empty_stream())
            tracking_streams = streams.get("tracking", empty_stream())
            frame_row = {
                "frame_index": frame_index,
                "timestamp_sec": frame_index / fps if fps > 0 else math.nan,
            }
            for mode in ("image", "tracking"):
                stream = streams.get(mode, empty_stream())
                frame_row[f"{mode}_palm_detections"] = len(stream["palm_detections"])
                frame_row[f"{mode}_palm_rois"] = len(stream["hand_rects_from_palm_detections"])
                frame_row[f"{mode}_hands"] = len(stream["hands"])
                frame_row[f"{mode}_tracking_rois"] = len(stream["hand_rects_from_landmarks"])
                frame_row[f"{mode}_ms"] = elapsed_by_mode.get(mode, math.nan)
            frame_rows.append(frame_row)

            annotation_frame = {
                "frame_index": frame_index,
                "timestamp_sec": frame_index / fps if fps > 0 else math.nan,
                "width": width,
                "height": height,
            }
            for mode in modes:
                stream = streams[mode]
                append_palm_rows(palm_rows, frame_index, mode, stream["palm_detections"])
                append_roi_rows(
                    palm_roi_rows,
                    frame_index,
                    mode,
                    "hand_rects_from_palm_detections",
                    stream["hand_rects_from_palm_detections"],
                )
                append_roi_rows(
                    tracking_roi_rows,
                    frame_index,
                    mode,
                    "hand_rects_from_landmarks",
                    stream["hand_rects_from_landmarks"],
                )
                append_hand_rows(hand_rows, frame_index, mode, stream["hands"])
                annotation_frame[mode] = stream
            annotation_frames.append(annotation_frame)

            if args.save_vis and saved_vis < args.save_vis and (
                image_streams["hands"]
                or image_streams["palm_detections"]
                or tracking_streams["hands"]
                or tracking_streams["palm_detections"]
            ):
                canvas = draw_records(
                    frame_bgr,
                    image_streams,
                    tracking_streams,
                    draw_palm=args.draw_palm,
                    draw_landmarks=args.draw_landmarks,
                    draw_roi=args.draw_roi,
                )
                cv2.imwrite(str(vis_dir / f"frame_{frame_index:06d}.jpg"), canvas)
                saved_vis += 1
            if writer is not None:
                canvas = draw_records(
                    frame_bgr,
                    image_streams,
                    tracking_streams,
                    draw_palm=args.draw_palm,
                    draw_landmarks=args.draw_landmarks,
                    draw_roi=args.draw_roi,
                )
                writer.write(canvas)

            processed_frames += 1
            if processed_frames % 100 == 0:
                print(f"[mediapipe-annotate] processed {processed_frames} frames", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        for graph in graphs.values():
            graph.close()

    annotations_path = output_dir / "mediapipe_annotations.json"
    frames_csv = output_dir / "frames.csv"
    hands_csv = output_dir / "hands.csv"
    palm_csv = output_dir / "palm_detections.csv"
    palm_rois_csv = output_dir / "palm_rois.csv"
    tracking_rois_csv = output_dir / "tracking_rois.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    timing_summary: dict[str, float] = {}
    for mode in ("image", "tracking"):
        timing_summary.update(summarize(stream_ms.get(mode, []), mode))

    summary: dict[str, Any] = {
        "task": "annotate_mediapipe_video",
        "video": str(video_path),
        "output_dir": str(output_dir),
        "binary_graph_path": BINARY_GRAPH_PATH,
        "mediapipe_version": str(mp.__version__),
        "running_mode": args.running_mode,
        "streams": modes,
        "source_frame_count": source_frame_count,
        "processed_frames": processed_frames,
        "frame_stride": args.frame_stride,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "model_complexity": args.model_complexity,
        "max_num_hands": args.max_num_hands,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "annotations_json": str(annotations_path),
        "frames_csv": str(frames_csv),
        "hands_csv": str(hands_csv),
        "palm_detections_csv": str(palm_csv),
        "palm_rois_csv": str(palm_rois_csv),
        "tracking_rois_csv": str(tracking_rois_csv),
        "report_md": str(report_path),
        "visualizations": saved_vis,
        "annotated_video": annotated_video_path,
        **timing_summary,
    }

    annotations = {
        "summary": summary,
        "schema": {
            "image": "MediaPipe hand graph with use_prev_landmarks=False.",
            "tracking": "MediaPipe hand graph with use_prev_landmarks=True.",
            "palm_detections": "Raw graph stream PALM_DETECTIONS:palm_detections.",
            "hand_rects_from_palm_detections": "Raw graph stream HAND_ROIS_FROM_PALM_DETECTIONS.",
            "hands": "Graph multi_hand_landmarks matched by graph output order.",
            "hand_rects_from_landmarks": "Raw graph stream HAND_ROIS_FROM_LANDMARKS used for tracking.",
        },
        "frames": annotation_frames,
    }
    annotations_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(frames_csv, frame_rows)
    write_csv(hands_csv, hand_rows)
    write_csv(palm_csv, palm_rows)
    write_csv(palm_rois_csv, palm_roi_rows)
    write_csv(tracking_rois_csv, tracking_roi_rows)
    write_report(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
