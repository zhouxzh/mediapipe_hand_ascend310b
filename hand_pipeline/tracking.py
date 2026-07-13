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
from hand_pipeline.preprocess import nv12_image_to_tensor
from hand_pipeline.runtime import ModelRunner
from hand_pipeline.roi import NormalizedRect
from hand_pipeline.roi import associate_normalized_rects
from hand_pipeline.roi import hand_roi_tensor_from_nv12
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
    max_tracking_lost_frames: int = 0
    max_tracking_rejected_frames: int = 0
    max_tracking_rotation_delta: float = math.inf
    min_tracking_size_ratio: float = 0.0
    max_tracking_size_ratio: float = math.inf
    max_tracking_center_shift: float = math.inf
    tracking_rect_smooth_alpha: float = 1.0
    roi_precision: str = "float32"
    projection_precision: str = "float32"
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


def _limit_associated_rects(
    rects: list[NormalizedRect],
    previous_rect_ids: set[int],
    max_hands: int,
) -> list[NormalizedRect]:
    """Limit associated ROIs while retaining previous-landmark priority."""
    limit = max(0, int(max_hands))
    if len(rects) <= limit:
        return rects
    prioritized = [rect for rect in rects if id(rect) in previous_rect_ids]
    prioritized.extend(rect for rect in rects if id(rect) not in previous_rect_ids)
    selected_ids = {id(rect) for rect in prioritized[:limit]}
    return [rect for rect in rects if id(rect) in selected_ids]


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


def run_palm_detector_nv12(
    nv12: np.ndarray,
    image_width: int,
    image_height: int,
    detector: ModelRunner,
    anchors: np.ndarray,
    config: TrackingConfig,
) -> tuple[dict[str, float], list[PalmDetection]]:
    pre_start = time.perf_counter()
    tensor, letterbox = nv12_image_to_tensor(nv12, image_width, image_height, input_size=192)
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
        if config.roi_precision == "float32":
            palm_roi = normalized_rect_from_palm_detection_float32(palm.box, palm.keypoints, width, height)
        else:
            palm_roi = normalized_rect_from_palm_detection(palm.box, palm.keypoints, width, height)
        roi = hand_roi_from_normalized_rect(
            frame,
            palm_roi,
            border_mode=config.landmark_crop_border_mode,
            dst_size_mode=config.landmark_crop_dst_size_mode,
        )
        lm_tensor = preprocess_landmark_tflite(roi.crop)
        roi_ms += elapsed_ms(roi_start)

        lm_start = time.perf_counter()
        lm_outputs = landmark(lm_tensor)
        landmark_infer_ms += elapsed_ms(lm_start)

        post_start = time.perf_counter()
        outputs = pick_landmark_outputs(lm_outputs)
        if config.projection_precision == "float32":
            hand21_norm = project_landmarks_normalized_with_rect_float32(
                outputs.landmarks,
                palm_roi,
                input_size=224,
                coord_scale="auto",
            )
            hand21 = hand21_norm.copy()
            hand21[:, 0] *= float(width)
            hand21[:, 1] *= float(height)
        else:
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
                    "handedness": float(outputs.handedness),
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
        self.prev_hand_rect_lost_counts: list[int] = []
        self.prev_hand_rect_rejected_counts: list[int] = []

    def reset(self) -> None:
        self.prev_hand_rects_from_landmarks = []
        self.prev_hand_rect_lost_counts = []
        self.prev_hand_rect_rejected_counts = []

    def close(self) -> None:
        self.reset()

    def process(self, frame: np.ndarray) -> tuple[dict[str, float | int | bool], list[dict[str, Any]], dict[str, Any]]:
        height, width = frame.shape[:2]
        return self._process_common(frame, None, width, height)

    def process_nv12(
        self,
        nv12: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple[dict[str, float | int | bool], list[dict[str, Any]], dict[str, Any]]:
        return self._process_common(None, nv12, int(image_width), int(image_height))

    def _process_common(
        self,
        frame: np.ndarray | None,
        nv12: np.ndarray | None,
        width: int,
        height: int,
    ) -> tuple[dict[str, float | int | bool], list[dict[str, Any]], dict[str, Any]]:
        prev_rects = list(self.prev_hand_rects_from_landmarks)
        prev_lost_counts = list(self.prev_hand_rect_lost_counts)
        if len(prev_lost_counts) != len(prev_rects):
            prev_lost_counts = [0] * len(prev_rects)
        prev_rejected_counts = list(self.prev_hand_rect_rejected_counts)
        if len(prev_rejected_counts) != len(prev_rects):
            prev_rejected_counts = [0] * len(prev_rects)
        used_prev_landmarks = len(prev_rects) > 0
        prev_has_enough_hands = len(prev_rects) >= self.config.max_hands

        det_timing = {"det_pre_ms": 0.0, "det_infer_ms": 0.0, "det_decode_ms": 0.0}
        palms: list[PalmDetection] = []
        if not prev_has_enough_hands:
            if nv12 is not None:
                det_timing, palms = run_palm_detector_nv12(nv12, width, height, self.detector, self.anchors, self.config)
            else:
                if frame is None:
                    raise ValueError("frame is required when nv12 is not provided")
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
        hand_rects = _limit_associated_rects(hand_rects, set(prev_sources), self.config.max_hands)
        roi_ms = elapsed_ms(roi_start)

        crop_ms = 0.0
        landmark_infer_ms = 0.0
        landmark_post_ms = 0.0
        hand_records: list[dict[str, Any]] = []
        landmark_roi_results: list[dict[str, Any]] = []
        next_rects: list[NormalizedRect] = []
        next_lost_counts: list[int] = []
        next_rejected_counts: list[int] = []
        for hand_index, hand_rect in enumerate(hand_rects):
            source_roi = "previous_landmarks" if id(hand_rect) in prev_sources else "palm_detection"
            source_index = prev_sources[id(hand_rect)] if id(hand_rect) in prev_sources else palm_sources.get(id(hand_rect))
            source_lost_count = (
                prev_lost_counts[int(source_index)]
                if source_roi == "previous_landmarks"
                and source_index is not None
                and int(source_index) < len(prev_lost_counts)
                else 0
            )
            source_rejected_count = (
                prev_rejected_counts[int(source_index)]
                if source_roi == "previous_landmarks"
                and source_index is not None
                and int(source_index) < len(prev_rejected_counts)
                else 0
            )

            crop_start = time.perf_counter()
            if nv12 is not None:
                lm_tensor, roi = hand_roi_tensor_from_nv12(
                    nv12,
                    width,
                    height,
                    hand_rect,
                    border_mode=self.config.landmark_crop_border_mode,
                    dst_size_mode=self.config.landmark_crop_dst_size_mode,
                )
            else:
                if frame is None:
                    raise ValueError("frame is required when nv12 is not provided")
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

            landmark_has_hand = math.isnan(outputs.hand_score) or outputs.hand_score > self.config.min_hand_score
            emit_hand = landmark_has_hand
            candidate_next_rect = self._next_rect(next_roi_landmarks, width, height) if landmark_has_hand else None
            next_rect = candidate_next_rect
            tracking_state_kept = False
            tracking_state_rejected = False
            reject_reason = ""
            next_lost_count = 0
            next_rejected_count = 0
            if landmark_has_hand and source_roi == "previous_landmarks" and candidate_next_rect is not None:
                reject_reason = self._tracking_reject_reason(hand_rect, candidate_next_rect)
                if reject_reason:
                    tracking_state_rejected = True
                    if reject_reason != "size_shrink":
                        emit_hand = False
                    next_rejected_count = source_rejected_count + 1
                    if next_rejected_count <= self.config.max_tracking_rejected_frames:
                        next_rect = hand_rect
                        tracking_state_kept = True
                    else:
                        next_rect = None
                else:
                    next_rect = self._smooth_tracking_rect(hand_rect, candidate_next_rect)
            if not landmark_has_hand and source_roi == "previous_landmarks":
                next_lost_count = source_lost_count + 1
                if next_lost_count <= self.config.max_tracking_lost_frames:
                    next_rect = hand_rect
                    tracking_state_kept = True
            landmark_post_ms += elapsed_ms(post_start)

            landmark_roi_results.append(
                {
                    "hand_index": int(hand_index),
                    "landmark_backend": "tflite",
                    "source_roi": source_roi,
                    "source_index": None if source_index is None else int(source_index),
                    "source_lost_count": int(source_lost_count),
                    "source_rejected_count": int(source_rejected_count),
                    "hand_score": float(outputs.hand_score),
                    "kept": bool(emit_hand),
                    "tracking_state_kept": bool(tracking_state_kept),
                    "tracking_state_rejected": bool(tracking_state_rejected),
                    "tracking_reject_reason": reject_reason,
                    "next_lost_count": int(next_lost_count),
                    "next_rejected_count": int(next_rejected_count),
                    "landmark_backend_has_hand": bool(landmark_has_hand),
                    "handedness": float(outputs.handedness),
                    "hand_roi": rect_record(hand_rect, width, height),
                    "subgraph_next_tracking_roi": None
                    if candidate_next_rect is None
                    else rect_record(candidate_next_rect, width, height),
                    "next_tracking_roi": None if next_rect is None else rect_record(next_rect, width, height),
                }
            )
            if next_rect is not None:
                next_rects.append(next_rect)
                next_lost_counts.append(next_lost_count)
                next_rejected_counts.append(next_rejected_count)
            if not emit_hand:
                continue

            hand_records.append(
                {
                    "hand_index": hand_index,
                    "landmark_backend": "tflite",
                    "hand_score": float(outputs.hand_score),
                    "handedness": float(outputs.handedness),
                    "source_roi": source_roi,
                    "source_index": None if source_index is None else int(source_index),
                    "hand_roi": rect_record(hand_rect, width, height),
                    "next_tracking_roi": None if next_rect is None else rect_record(next_rect, width, height),
                    "subgraph_next_tracking_roi": None
                    if candidate_next_rect is None
                    else rect_record(candidate_next_rect, width, height),
                    "tracking_state_rejected": bool(tracking_state_rejected),
                    "tracking_reject_reason": reject_reason,
                    "hand21_keypoints_px": hand21.astype(float).tolist(),
                    "hand21_keypoints_norm": normalize_xy(hand21, width, height),
                    "hand21_keypoints_crop_xyz": outputs.landmarks.astype(float).tolist(),
                    "roi_center_px": roi.center.astype(float).tolist(),
                    "roi_size_px": float(roi.size),
                    "roi_rotation_rad": float(roi.rotation),
                }
            )
            if source_roi == "palm_detection" and source_index is not None and source_index < len(palms):
                palm = palms[int(source_index)]
                hand_records[-1].update(
                    {
                        "score": float(palm.score),
                        "palm_bbox_xyxy_px": palm.box.astype(float).tolist(),
                        "palm_bbox_xyxy_norm": normalize_box_xyxy(palm.box, width, height),
                        "palm7_keypoints_px": palm.keypoints.astype(float).tolist(),
                        "palm7_keypoints_norm": normalize_xy(palm.keypoints, width, height),
                    }
                )

        self.prev_hand_rects_from_landmarks = next_rects
        self.prev_hand_rect_lost_counts = next_lost_counts
        self.prev_hand_rect_rejected_counts = next_rejected_counts
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
            "tracking_state_kept": sum(1 for item in landmark_roi_results if item.get("tracking_state_kept")),
            "tracking_state_rejected": sum(
                1 for item in landmark_roi_results if item.get("tracking_state_rejected")
            ),
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
            "previous_tracking_lost_counts": [int(value) for value in prev_lost_counts],
            "previous_tracking_rejected_counts": [int(value) for value in prev_rejected_counts],
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
            "next_tracking_lost_counts": [int(value) for value in next_lost_counts],
            "next_tracking_rejected_counts": [int(value) for value in next_rejected_counts],
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

    def _tracking_reject_reason(self, current: NormalizedRect, next_rect: NormalizedRect) -> str:
        if current.width <= 0.0 or current.height <= 0.0 or next_rect.width <= 0.0 or next_rect.height <= 0.0:
            return "invalid_size"
        rotation_delta = abs(normalize_angle_delta(next_rect.rotation - current.rotation))
        if rotation_delta > self.config.max_tracking_rotation_delta:
            return "rotation_delta"
        current_size = max(float(current.width), float(current.height))
        next_size = max(float(next_rect.width), float(next_rect.height))
        size_ratio = next_size / max(current_size, 1e-6)
        if size_ratio < self.config.min_tracking_size_ratio:
            return "size_shrink"
        if size_ratio > self.config.max_tracking_size_ratio:
            return "size_expand"
        center_shift = math.hypot(
            float(next_rect.x_center) - float(current.x_center),
            float(next_rect.y_center) - float(current.y_center),
        )
        if center_shift > self.config.max_tracking_center_shift * current_size:
            return "center_shift"
        return ""

    def _smooth_tracking_rect(self, current: NormalizedRect, next_rect: NormalizedRect) -> NormalizedRect:
        alpha = float(self.config.tracking_rect_smooth_alpha)
        if alpha >= 1.0:
            return next_rect
        if alpha <= 0.0:
            return current
        rotation = float(current.rotation) + alpha * normalize_angle_delta(float(next_rect.rotation) - float(current.rotation))
        return NormalizedRect(
            x_center=float(current.x_center) + alpha * (float(next_rect.x_center) - float(current.x_center)),
            y_center=float(current.y_center) + alpha * (float(next_rect.y_center) - float(current.y_center)),
            width=float(current.width) + alpha * (float(next_rect.width) - float(current.width)),
            height=float(current.height) + alpha * (float(next_rect.height) - float(current.height)),
            rotation=normalize_angle_delta(rotation),
        )


def normalize_angle_delta(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
