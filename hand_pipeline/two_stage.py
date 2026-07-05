"""Reusable two-stage MediaPipe hand inference helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import weighted_nms
from hand_pipeline.inference import OnnxModel
from hand_pipeline.om_runtime import PersistentAclModel
from hand_pipeline.om_runtime import PersistentAclRuntime
from hand_pipeline.preprocess import image_to_tensor
from hand_pipeline.roi import landmarks_to_original
from hand_pipeline.roi import make_hand_roi
from hand_pipeline.roi import preprocess_landmark_tflite


@dataclass
class FrameResult:
    predictions: list[dict[str, Any]]
    timings: dict[str, float]


def pick_landmark_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    landmarks = None
    world = None
    scalar_outputs: list[np.ndarray] = []
    for value in outputs:
        arr = np.asarray(value)
        if arr.size == 63 and landmarks is None:
            landmarks = arr.reshape(21, 3)
        elif arr.size == 63:
            world = arr.reshape(21, 3)
        elif arr.size == 1:
            scalar_outputs.append(arr.reshape(-1))
    if landmarks is None:
        raise ValueError(f"Could not find 63-value landmark output: {[x.shape for x in outputs]}")
    hand_score = float(scalar_outputs[0][0]) if len(scalar_outputs) >= 1 else math.nan
    handedness = float(scalar_outputs[1][0]) if len(scalar_outputs) >= 2 else math.nan
    return landmarks.astype(np.float32), hand_score, handedness, None if world is None else world.astype(np.float32)


def summarize_times(values: list[float], prefix: str) -> dict[str, float]:
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


def infer_two_stage(
    image_bgr: np.ndarray,
    *,
    detector: Any,
    landmark: Any,
    anchors: np.ndarray,
    score_threshold: float,
    nms_iou: float,
    max_det: int,
    max_hands: int,
    min_hand_score: float,
    roi_scale: float,
    shift_y: float,
    rotation_offset_degrees: float,
) -> FrameResult:
    total_start = time.perf_counter()
    pre_start = time.perf_counter()
    tensor, letterbox = image_to_tensor(image_bgr, input_size=192)
    preprocess_ms = (time.perf_counter() - pre_start) * 1000.0

    det_start = time.perf_counter()
    raw_boxes, raw_scores = detector(tensor)
    detector_ms = (time.perf_counter() - det_start) * 1000.0

    decode_start = time.perf_counter()
    palms = decode_raw_palm(raw_boxes, raw_scores, anchors, letterbox, score_threshold=score_threshold)
    palms = weighted_nms(palms, iou_threshold=nms_iou, max_detections=max(max_det, max_hands))
    palms = sorted(palms, key=lambda item: item.score, reverse=True)[:max_hands]
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    roi_ms = 0.0
    landmark_ms = 0.0
    post_ms = 0.0
    predictions: list[dict[str, Any]] = []
    for hand_index, palm in enumerate(palms):
        roi_start = time.perf_counter()
        roi = make_hand_roi(
            image_bgr,
            palm.box,
            palm.keypoints,
            scale=roi_scale,
            shift_y=shift_y,
            rotation_offset_degrees=rotation_offset_degrees,
        )
        lm_tensor = preprocess_landmark_tflite(roi.crop)
        roi_ms += (time.perf_counter() - roi_start) * 1000.0

        lm_start = time.perf_counter()
        lm_outputs = landmark(lm_tensor)
        landmark_ms += (time.perf_counter() - lm_start) * 1000.0

        post_start = time.perf_counter()
        lm_crop, hand_score, handedness, _world = pick_landmark_outputs(lm_outputs)
        if not math.isnan(hand_score) and hand_score < min_hand_score:
            post_ms += (time.perf_counter() - post_start) * 1000.0
            continue
        hand21 = landmarks_to_original(lm_crop, roi.inverse, input_size=224, coord_scale="auto")
        post_ms += (time.perf_counter() - post_start) * 1000.0
        predictions.append(
            {
                "hand_index": hand_index,
                "score": float(palm.score),
                "hand_score": float(hand_score),
                "handedness": float(handedness),
                "box": palm.box.astype(float).tolist(),
                "palm7": palm.keypoints.astype(float).tolist(),
                "hand21": hand21.astype(float).tolist(),
                "roi_center": roi.center.astype(float).tolist(),
                "roi_size": float(roi.size),
                "roi_rotation_rad": float(roi.rotation),
            }
        )

    return FrameResult(
        predictions=predictions,
        timings={
            "preprocess_ms": preprocess_ms,
            "detector_ms": detector_ms,
            "decode_ms": decode_ms,
            "roi_ms": roi_ms,
            "landmark_ms": landmark_ms,
            "post_ms": post_ms,
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
        },
    )


class OnnxHandPipeline:
    def __init__(
        self,
        detector_path: Path,
        landmark_path: Path,
        *,
        score_threshold: float,
        nms_iou: float,
        max_hands: int,
        min_hand_score: float,
        max_det: int,
        roi_scale: float,
        shift_y: float,
        rotation_offset_degrees: float,
    ) -> None:
        self.detector = OnnxModel(detector_path)
        self.landmark = OnnxModel(landmark_path)
        self.anchors = generate_palm_anchors()
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_hands = max_hands
        self.min_hand_score = min_hand_score
        self.max_det = max_det
        self.roi_scale = roi_scale
        self.shift_y = shift_y
        self.rotation_offset_degrees = rotation_offset_degrees

    def infer(self, image_bgr: np.ndarray) -> FrameResult:
        return infer_two_stage(
            image_bgr,
            detector=self.detector,
            landmark=self.landmark,
            anchors=self.anchors,
            score_threshold=self.score_threshold,
            nms_iou=self.nms_iou,
            max_det=self.max_det,
            max_hands=self.max_hands,
            min_hand_score=self.min_hand_score,
            roi_scale=self.roi_scale,
            shift_y=self.shift_y,
            rotation_offset_degrees=self.rotation_offset_degrees,
        )

    def close(self) -> None:
        return None


class OmHandPipeline:
    def __init__(
        self,
        detector_path: Path,
        landmark_path: Path,
        *,
        device_id: int,
        score_threshold: float,
        nms_iou: float,
        max_hands: int,
        min_hand_score: float,
        max_det: int,
        roi_scale: float,
        shift_y: float,
        rotation_offset_degrees: float,
        reload_detector_each_frame: bool = False,
        finalize_on_release: bool = True,
    ) -> None:
        self.runtime = PersistentAclRuntime(device_id=device_id, finalize_on_release=finalize_on_release)
        self.detector_path = detector_path
        self.reload_detector_each_frame = reload_detector_each_frame
        self.detector: PersistentAclModel | None = None
        if not reload_detector_each_frame:
            self.detector = PersistentAclModel(detector_path, runtime=self.runtime)
        self.landmark = PersistentAclModel(landmark_path, runtime=self.runtime)
        self.anchors = generate_palm_anchors()
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_hands = max_hands
        self.min_hand_score = min_hand_score
        self.max_det = max_det
        self.roi_scale = roi_scale
        self.shift_y = shift_y
        self.rotation_offset_degrees = rotation_offset_degrees

    def close(self) -> None:
        if self.detector is not None:
            self.detector.release()
            self.detector = None
        self.landmark.release()
        self.runtime.release()

    def _detector_call(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self.reload_detector_each_frame:
            detector = PersistentAclModel(self.detector_path, runtime=self.runtime)
            try:
                return detector.infer(tensor)
            finally:
                detector.release()
        if self.detector is None:
            self.detector = PersistentAclModel(self.detector_path, runtime=self.runtime)
        return self.detector.infer(tensor)

    def infer(self, image_bgr: np.ndarray) -> FrameResult:
        return infer_two_stage(
            image_bgr,
            detector=self._detector_call,
            landmark=self.landmark.infer,
            anchors=self.anchors,
            score_threshold=self.score_threshold,
            nms_iou=self.nms_iou,
            max_det=self.max_det,
            max_hands=self.max_hands,
            min_hand_score=self.min_hand_score,
            roi_scale=self.roi_scale,
            shift_y=self.shift_y,
            rotation_offset_degrees=self.rotation_offset_degrees,
        )
