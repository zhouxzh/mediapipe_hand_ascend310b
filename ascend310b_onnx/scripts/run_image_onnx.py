#!/usr/bin/env python3
"""Run one image through the portable two-stage pipeline with ONNX models."""

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

from hand_pipeline.pipeline import PipelineConfig
from hand_pipeline.pipeline import hand_prediction_to_dict
from hand_pipeline.pipeline import run_two_stage
from hand_pipeline.runtimes.onnx import OnnxModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
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
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/onnx_image_result.json")
    parser.add_argument("--num-threads", type=int, default=1)
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
    image_path = resolve(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detector_path = resolve(args.detector)
    landmark_path = resolve(args.landmark)
    detector = OnnxModel(detector_path, num_threads=args.num_threads)
    landmark = OnnxModel(landmark_path, num_threads=args.num_threads)
    config = PipelineConfig(
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        max_hands=args.max_hands,
        min_hand_score=args.min_hand_score,
    )
    palms, hands = run_two_stage(image, detector, landmark, config=config)
    result: dict[str, Any] = {
        "image": str(image_path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "runtime": "onnx",
        "detector": str(detector_path),
        "landmark": str(landmark_path),
        "palms": [
            {
                "score": float(palm.score),
                "box": palm.box.astype(float).tolist(),
                "palm7": palm.keypoints.astype(float).tolist(),
            }
            for palm in palms
        ],
        "hands": [hand_prediction_to_dict(hand) for hand in hands],
    }

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

