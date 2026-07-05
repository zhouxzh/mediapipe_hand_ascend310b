"""Portable MediaPipe-style hand tracking loop."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from hand_pipeline.decode import PalmDetection
from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import weighted_nms
from hand_pipeline.outputs import pick_landmark_outputs
from hand_pipeline.preprocess import image_to_tensor
from hand_pipeline.runtime import ModelRunner
from hand_pipeline.roi import NormalizedRect
from hand_pipeline.roi import associate_normalized_rects
from hand_pipeline.roi import hand_roi_from_normalized_rect
from hand_pipeline.roi import landmarks_to_tracking_roi
from hand_pipeline.roi import landmarks_to_tracking_roi_float32
from hand_pipeline.roi import normalized_rect_from_palm_detection
from hand_pipeline.roi import normalized_rect_from_palm_detection_float32
from hand_pipeline.roi import normalized_rect_to_dict
from hand_pipeline.roi import normalized_rect_to_pixel_dict
from hand_pipeline.roi import preprocess_landmark_tflite
from hand_pipeline.roi import project_landmarks_normalized_with_rect_float32
from hand_pipeline.roi import project_landmarks_with_normalized_rect


@dataclass(frozen=True)
class TrackingConfig:
    score_threshold: float = 0.5
    nms_iou: float = 0.3
    max_det: int = 20
    max_hands: int = 2
    min_hand_score: float = 0.5
    roi_precision: str = "python"
    projection_precision: str = "python"
    landmark_crop_border_mode: str = "replicate"
    landmark_crop_dst_size_mode: str = "size"


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def normalize_xy(points_xy: np.ndarray, width: int, height: int) -> list[list[float]]:
    points = np.asarray(points_xy, dtype=np.float32).copy()
    points[:, 0] /= max(float(width), 1.0)
    points[:, 1] /= max(float(height), 1.0)
    return points.astype(float).tolist()


def normalize_box_xyxy(box: np.ndarray, width: int, height: int) -> list[float]:
    values = np.asarray(box, dtype=np.float32).copy()
    values[[0, 2]] /= max(float(width), 1.0)
    values[[1, 3]] /= max(float(height), 1.0)
    return values.astype(float).tolist()


def rect_record(rect: NormalizedRect, width: int, height: int, source: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        **normalized_rect_to_dict(rect),
        **normalized_rect_to_pixel_dict(rect, width, height),
    }
    if source is not None:
        record["source"] = source
    return record


def run_palm_detector(
    frame: np.ndarray,
    detector: ModelRunner,
    anchors: np.ndarray,
    config: TrackingConfig,
) -> tuple[dict[str, float], list[PalmDetection]]:
    pre_start = time.perf_counter()
    tensor, letterbox = image_to_tensor(frame, input_size=192)
    det_pre_ms = elapsed_ms(pre_start)

    det_start = time.perf_counter()
    raw_boxes, raw_scores = detector(tensor)
    det_infer_ms = elapsed_ms(det_start)

    decode_start = time.perf_counter()
    palms = decode_raw_palm(raw_boxes, raw_scores, anchors, letterbox, score_threshold=config.score_threshold)
    palms = weighted_nms(palms, iou_threshold=config.nms_iou, max_detections=config.max_det)
    palms = sorted(palms, key=lambda item: item.score, reverse=True)[: config.max_hands]
    det_decode_ms = elapsed_ms(decode_start)
    return {
        "det_pre_ms": det_pre_ms,
        "det_infer_ms": det_infer_ms,
        "det_decode_ms": det_decode_ms,
    }, palms


def run_two_stage_image_mode(
    frame: np.ndarray,
    detector: ModelRunner,
    landmark: ModelRunner,
    anchors: np.ndarray | None = None,
    config: TrackingConfig | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    config = config or TrackingConfig()
    anchors = generate_palm_anchors() if anchors is None else anchors
    height, width = frame.shape[:2]
    det_timing, palms = run_palm_detector(frame, detector, anchors, config)

    roi_ms = 0.0
    landmark_infer_ms = 0.0
    landmark_post_ms = 0.0
    hand_records: list[dict[str, Any]] = []
    for hand_index, palm in enumerate(palms):
        roi_start = time.perf_counter()
        palm_roi = normalized_rect_from_palm_detection(palm.box, palm.keypoints, width, height)
        roi = hand_roi_from_normalized_rect(frame, palm_roi)
        lm_tensor = preprocess_landmark_tflite(roi.crop)
        roi_ms += elapsed_ms(roi_start)

        lm_start = time.perf_counter()
        lm_outputs = landmark(lm_tensor)
        landmark_infer_ms += elapsed_ms(lm_start)

        post_start = time.perf_counter()
        outputs = pick_landmark_outputs(lm_outputs)
        hand21 = project_landmarks_with_normalized_rect(
            outputs.landmarks,
            palm_roi,
            width,
            height,
            input_size=224,
            coord_scale="auto",
        )
        landmark_post_ms += elapsed_ms(post_start)
        if math.isnan(outputs.hand_score) or outputs.hand_score > config.min_hand_score:
            hand_records.append(
                {
                    "hand_index": hand_index,
                    "score": float(palm.score),
                    "hand_score": float(outputs.hand_score),
                    "palm_bbox_xyxy_px": palm.box.astype(float).tolist(),
                    "palm_bbox_xyxy_norm": normalize_box_xyxy(palm.box, width, height),
                    "palm7_keypoints_px": palm.keypoints.astype(float).tolist(),
                    "palm7_keypoints_norm": normalize_xy(palm.keypoints, width, height),
                    "hand21_keypoints_px": hand21.astype(float).tolist(),
                    "hand21_keypoints_norm": normalize_xy(hand21, width, height),
                    "hand21_keypoints_crop_xyz": outputs.landmarks.astype(float).tolist(),
                    "source_roi": "palm_detection",
                    "palm_roi": rect_record(palm_roi, width, height),
                    "roi_center_px": roi.center.astype(float).tolist(),
                    "roi_size_px": float(roi.size),
                    "roi_rotation_rad": float(roi.rotation),
                }
            )

    total_ms = (
        det_timing["det_pre_ms"]
        + det_timing["det_infer_ms"]
        + det_timing["det_decode_ms"]
        + roi_ms
        + landmark_infer_ms
        + landmark_post_ms
    )
    timing = {
        "palms": len(palms),
        "hands": len(hand_records),
        **det_timing,
        "roi_ms": roi_ms,
        "landmark_infer_ms": landmark_infer_ms,
        "landmark_post_ms": landmark_post_ms,
        "total_ms": total_ms,
    }
    return timing, hand_records


class HandTracker:
    """MediaPipe-style hand tracking loop with a portable Python state machine."""

    def __init__(
        self,
        detector: ModelRunner,
        landmark: ModelRunner,
        anchors: np.ndarray | None = None,
        config: TrackingConfig | None = None,
    ) -> None:
        self.detector = detector
        self.landmark = landmark
        self.anchors = generate_palm_anchors() if anchors is None else anchors
        self.config = config or TrackingConfig()
        self.prev_hand_rects_from_landmarks: list[NormalizedRect] = []

    def reset(self) -> None:
        self.prev_hand_rects_from_landmarks = []

    def close(self) -> None:
        self.reset()

    def process(self, frame: np.ndarray) -> tuple[dict[str, float | int | bool], list[dict[str, Any]], dict[str, Any]]:
        height, width = frame.shape[:2]
        prev_rects = list(self.prev_hand_rects_from_landmarks)
        used_prev_landmarks = len(prev_rects) > 0
        prev_has_enough_hands = len(prev_rects) >= self.config.max_hands

        det_timing = {"det_pre_ms": 0.0, "det_infer_ms": 0.0, "det_decode_ms": 0.0}
        palms: list[PalmDetection] = []
        if not prev_has_enough_hands:
            det_timing, palms = run_palm_detector(frame, self.detector, self.anchors, self.config)

        roi_start = time.perf_counter()
        palm_rects = self._palm_rects(palms, width, height)
        palm_detection_records = [
            {
                "score": float(palm.score),
                "palm_bbox_xyxy_px": palm.box.astype(float).tolist(),
                "palm_bbox_xyxy_norm": normalize_box_xyxy(palm.box, width, height),
                "palm7_keypoints_px": palm.keypoints.astype(float).tolist(),
                "palm7_keypoints_norm": normalize_xy(palm.keypoints, width, height),
            }
            for palm in palms
        ]
        palm_sources = {id(rect): idx for idx, rect in enumerate(palm_rects)}
        prev_sources = {id(rect): idx for idx, rect in enumerate(prev_rects)}
        hand_rects = associate_normalized_rects([palm_rects, prev_rects], min_similarity_threshold=0.5)
        roi_ms = elapsed_ms(roi_start)

        crop_ms = 0.0
        landmark_infer_ms = 0.0
        landmark_post_ms = 0.0
        hand_records: list[dict[str, Any]] = []
        landmark_roi_results: list[dict[str, Any]] = []
        next_rects: list[NormalizedRect] = []
        for hand_index, hand_rect in enumerate(hand_rects):
            source_roi = "previous_landmarks" if id(hand_rect) in prev_sources else "palm_detection"
            source_index = prev_sources[id(hand_rect)] if id(hand_rect) in prev_sources else palm_sources.get(id(hand_rect))

            crop_start = time.perf_counter()
            roi = hand_roi_from_normalized_rect(
                frame,
                hand_rect,
                border_mode=self.config.landmark_crop_border_mode,
                dst_size_mode=self.config.landmark_crop_dst_size_mode,
            )
            lm_tensor = preprocess_landmark_tflite(roi.crop)
            crop_ms += elapsed_ms(crop_start)

            lm_start = time.perf_counter()
            lm_outputs = self.landmark(lm_tensor)
            landmark_infer_ms += elapsed_ms(lm_start)

            post_start = time.perf_counter()
            outputs = pick_landmark_outputs(lm_outputs)
            if self.config.projection_precision == "float32":
                hand21_norm = project_landmarks_normalized_with_rect_float32(
                    outputs.landmarks,
                    hand_rect,
                    input_size=224,
                    coord_scale="auto",
                )
                hand21 = hand21_norm.copy()
                hand21[:, 0] *= float(width)
                hand21[:, 1] *= float(height)
                next_roi_landmarks = hand21_norm
            else:
                hand21 = project_landmarks_with_normalized_rect(
                    outputs.landmarks,
                    hand_rect,
                    width,
                    height,
                    input_size=224,
                    coord_scale="auto",
                )
                next_roi_landmarks = hand21

            keep_hand = math.isnan(outputs.hand_score) or outputs.hand_score > self.config.min_hand_score
            next_rect = self._next_rect(next_roi_landmarks, width, height) if keep_hand else None
            landmark_post_ms += elapsed_ms(post_start)

            landmark_roi_results.append(
                {
                    "hand_index": int(hand_index),
                    "landmark_backend": "tflite",
                    "source_roi": source_roi,
                    "source_index": None if source_index is None else int(source_index),
                    "hand_score": float(outputs.hand_score),
                    "kept": bool(keep_hand),
                    "landmark_backend_has_hand": bool(keep_hand),
                    "handedness": None,
                    "hand_roi": rect_record(hand_rect, width, height),
                    "subgraph_next_tracking_roi": None,
                    "next_tracking_roi": None if next_rect is None else rect_record(next_rect, width, height),
                }
            )
            if not keep_hand:
                continue

            next_rects.append(next_rect)
            hand_records.append(
                {
                    "hand_index": hand_index,
                    "landmark_backend": "tflite",
                    "hand_score": float(outputs.hand_score),
                    "handedness": None,
                    "source_roi": source_roi,
                    "source_index": None if source_index is None else int(source_index),
                    "hand_roi": rect_record(hand_rect, width, height),
                    "next_tracking_roi": rect_record(next_rect, width, height),
                    "hand21_keypoints_px": hand21.astype(float).tolist(),
                    "hand21_keypoints_norm": normalize_xy(hand21, width, height),
                    "hand21_keypoints_crop_xyz": outputs.landmarks.astype(float).tolist(),
                    "roi_center_px": roi.center.astype(float).tolist(),
                    "roi_size_px": float(roi.size),
                    "roi_rotation_rad": float(roi.rotation),
                }
            )

        self.prev_hand_rects_from_landmarks = next_rects
        total_ms = (
            det_timing["det_pre_ms"]
            + det_timing["det_infer_ms"]
            + det_timing["det_decode_ms"]
            + roi_ms
            + crop_ms
            + landmark_infer_ms
            + landmark_post_ms
        )
        timing: dict[str, float | int | bool] = {
            "palms": len(palm_detection_records),
            "hands": len(hand_records),
            "prev_rects": len(prev_rects),
            "palm_rects": len(palm_rects),
            "hand_rects": len(hand_rects),
            "next_rects": len(next_rects),
            "used_prev_landmarks": used_prev_landmarks,
            "prev_has_enough_hands": prev_has_enough_hands,
            "palm_detector_skipped": prev_has_enough_hands,
            **det_timing,
            "roi_ms": roi_ms,
            "crop_ms": crop_ms,
            "landmark_infer_ms": landmark_infer_ms,
            "landmark_post_ms": landmark_post_ms,
            "total_ms": total_ms,
        }
        debug = {
            "palm_backend": "tflite",
            "landmark_backend": "tflite",
            "previous_tracking_rois": [rect_record(rect, width, height, source="previous_landmarks") for rect in prev_rects],
            "palm_detections": palm_detection_records,
            "palm_rois": [rect_record(rect, width, height, source="palm_detection") for rect in palm_rects],
            "associated_hand_rois": [
                rect_record(
                    rect,
                    width,
                    height,
                    source="previous_landmarks" if id(rect) in prev_sources else "palm_detection",
                )
                for rect in hand_rects
            ],
            "next_tracking_rois": [rect_record(rect, width, height, source="landmarks") for rect in next_rects],
            "landmark_roi_results": landmark_roi_results,
        }
        return timing, hand_records, debug

    def _palm_rects(self, palms: list[PalmDetection], width: int, height: int) -> list[NormalizedRect]:
        if self.config.roi_precision == "float32":
            return [normalized_rect_from_palm_detection_float32(palm.box, palm.keypoints, width, height) for palm in palms]
        return [normalized_rect_from_palm_detection(palm.box, palm.keypoints, width, height) for palm in palms]

    def _next_rect(self, landmarks: np.ndarray, width: int, height: int) -> NormalizedRect:
        if self.config.roi_precision == "float32":
            return landmarks_to_tracking_roi_float32(landmarks, width, height)
        return landmarks_to_tracking_roi(landmarks, width, height)
