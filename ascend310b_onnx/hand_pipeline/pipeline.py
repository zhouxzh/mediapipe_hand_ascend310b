"""Reusable two-stage hand pipeline primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from hand_pipeline.decode import PalmDetection
from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import weighted_nms
from hand_pipeline.outputs import pick_landmark_outputs
from hand_pipeline.preprocess import image_to_tensor
from hand_pipeline.runtime import ModelRunner
from hand_pipeline.roi import HandRoi
from hand_pipeline.roi import NormalizedRect
from hand_pipeline.roi import landmarks_to_original
from hand_pipeline.roi import make_hand_roi
from hand_pipeline.roi import normalized_rect_from_palm_detection
from hand_pipeline.roi import preprocess_landmark_tflite
from hand_pipeline.roi import project_landmarks_with_normalized_rect
from hand_pipeline.roi import hand_roi_from_normalized_rect


@dataclass(frozen=True)
class PipelineConfig:
    score_threshold: float = 0.5
    nms_iou: float = 0.3
    max_det: int = 20
    max_hands: int = 2
    min_hand_score: float = 0.5
    landmark_input_size: int = 224
    roi_mode: str = "normalized_rect"
    roi_scale: float = 2.6
    shift_y: float = -0.5
    rotation_offset_degrees: float = 0.0
    crop_border_mode: str = "replicate"
    crop_dst_size_mode: str = "size"


@dataclass(frozen=True)
class HandPrediction:
    hand_index: int
    score: float
    hand_score: float
    handedness: float
    box: np.ndarray
    palm7: np.ndarray
    hand21: np.ndarray
    roi_center: np.ndarray
    roi_size: float
    roi_rotation: float
    normalized_rect: NormalizedRect | None
    crop_landmarks: np.ndarray
    world_landmarks: np.ndarray | None = None


def detect_palms(
    image_bgr: np.ndarray,
    detector: ModelRunner,
    anchors: np.ndarray | None = None,
    config: PipelineConfig | None = None,
) -> list[PalmDetection]:
    config = config or PipelineConfig()
    anchors = generate_palm_anchors() if anchors is None else anchors
    tensor, letterbox = image_to_tensor(image_bgr, input_size=192)
    raw_boxes, raw_scores = detector(tensor)
    palms = decode_raw_palm(raw_boxes, raw_scores, anchors, letterbox, score_threshold=config.score_threshold)
    palms = weighted_nms(palms, iou_threshold=config.nms_iou, max_detections=config.max_det)
    return sorted(palms, key=lambda item: item.score, reverse=True)[: config.max_hands]


def run_landmark_on_palm(
    image_bgr: np.ndarray,
    landmark: ModelRunner,
    palm: PalmDetection,
    hand_index: int,
    config: PipelineConfig | None = None,
) -> HandPrediction | None:
    config = config or PipelineConfig()
    roi, rect = _roi_from_palm(image_bgr, palm, config)
    outputs = pick_landmark_outputs(landmark(preprocess_landmark_tflite(roi.crop)))
    if not math.isnan(outputs.hand_score) and outputs.hand_score < config.min_hand_score:
        return None
    if rect is None:
        hand21 = landmarks_to_original(
            outputs.landmarks,
            roi.inverse,
            input_size=config.landmark_input_size,
            coord_scale="auto",
        )
    else:
        height, width = image_bgr.shape[:2]
        hand21 = project_landmarks_with_normalized_rect(
            outputs.landmarks,
            rect,
            width,
            height,
            input_size=config.landmark_input_size,
            coord_scale="auto",
        )
    return HandPrediction(
        hand_index=hand_index,
        score=float(palm.score),
        hand_score=float(outputs.hand_score),
        handedness=float(outputs.handedness),
        box=palm.box.astype(np.float32),
        palm7=palm.keypoints.astype(np.float32),
        hand21=hand21.astype(np.float32),
        roi_center=roi.center.astype(np.float32),
        roi_size=float(roi.size),
        roi_rotation=float(roi.rotation),
        normalized_rect=rect,
        crop_landmarks=outputs.landmarks,
        world_landmarks=outputs.world_landmarks,
    )


def run_two_stage(
    image_bgr: np.ndarray,
    detector: ModelRunner,
    landmark: ModelRunner,
    anchors: np.ndarray | None = None,
    config: PipelineConfig | None = None,
) -> tuple[list[PalmDetection], list[HandPrediction]]:
    config = config or PipelineConfig()
    palms = detect_palms(image_bgr, detector, anchors=anchors, config=config)
    predictions: list[HandPrediction] = []
    for hand_index, palm in enumerate(palms):
        prediction = run_landmark_on_palm(image_bgr, landmark, palm, hand_index, config=config)
        if prediction is not None:
            predictions.append(prediction)
    return palms, predictions


def hand_prediction_to_dict(prediction: HandPrediction) -> dict[str, object]:
    result: dict[str, object] = {
        "hand_index": prediction.hand_index,
        "score": prediction.score,
        "hand_score": prediction.hand_score,
        "handedness": prediction.handedness,
        "box": prediction.box.astype(float).tolist(),
        "palm7": prediction.palm7.astype(float).tolist(),
        "hand21": prediction.hand21.astype(float).tolist(),
        "roi_center": prediction.roi_center.astype(float).tolist(),
        "roi_size": prediction.roi_size,
        "roi_rotation_rad": prediction.roi_rotation,
    }
    if prediction.normalized_rect is not None:
        result["hand_rect_from_palm"] = {
            "x_center": float(prediction.normalized_rect.x_center),
            "y_center": float(prediction.normalized_rect.y_center),
            "width": float(prediction.normalized_rect.width),
            "height": float(prediction.normalized_rect.height),
            "rotation": float(prediction.normalized_rect.rotation),
        }
    return result


def _roi_from_palm(
    image_bgr: np.ndarray,
    palm: PalmDetection,
    config: PipelineConfig,
) -> tuple[HandRoi, NormalizedRect | None]:
    if config.roi_mode == "normalized_rect":
        height, width = image_bgr.shape[:2]
        rect = normalized_rect_from_palm_detection(palm.box, palm.keypoints, width, height)
        roi = hand_roi_from_normalized_rect(
            image_bgr,
            rect,
            input_size=config.landmark_input_size,
            border_mode=config.crop_border_mode,
            dst_size_mode=config.crop_dst_size_mode,
        )
        return roi, rect
    if config.roi_mode == "legacy_affine":
        roi = make_hand_roi(
            image_bgr,
            palm.box,
            palm.keypoints,
            input_size=config.landmark_input_size,
            scale=config.roi_scale,
            shift_y=config.shift_y,
            rotation_offset_degrees=config.rotation_offset_degrees,
            dst_size_mode=config.crop_dst_size_mode,
        )
        return roi, None
    raise ValueError(f"Unknown ROI mode: {config.roi_mode}")
