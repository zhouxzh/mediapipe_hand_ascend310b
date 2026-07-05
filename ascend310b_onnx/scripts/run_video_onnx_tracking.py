#!/usr/bin/env python3
"""Run video tracking through the portable pipeline with ONNX models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from hand_pipeline.runtimes.onnx import OnnxModel
from hand_pipeline.tracking import HandTracker
from hand_pipeline.tracking import TrackingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--detector",
        type=Path,
        default=PROJECT_ROOT / "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx",
    )
    parser.add_argument(
        "--landmark",
        type=Path,
        default=PROJECT_ROOT / "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/onnx_video_predictions.jsonl")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    video_path = resolve(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    detector_path = resolve(args.detector)
    landmark_path = resolve(args.landmark)
    detector = OnnxModel(detector_path, num_threads=args.num_threads)
    landmark = OnnxModel(landmark_path, num_threads=args.num_threads)
    tracker = HandTracker(
        detector,
        landmark,
        config=TrackingConfig(
            score_threshold=args.score_threshold,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
            max_hands=args.max_hands,
            min_hand_score=args.min_hand_score,
        ),
    )

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    rows = 0
    with output.open("w", encoding="utf-8") as file:
        while True:
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            timing, hands, debug = tracker.process(frame)
            record: dict[str, Any] = {
                "frame_index": frame_index,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "runtime": "onnx",
                "timing": timing,
                "hands": hands,
                "tracking_debug": debug,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1
            frame_index += 1

    cap.release()
    tracker.close()
    print(f"[done] {output} frames={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

