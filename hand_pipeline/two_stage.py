"""Reusable two-stage MediaPipe hand inference wrappers.

This module keeps the repository's stable OM/ONNX pipeline API while delegating
MediaPipe-style ROI projection and tracking behavior to ``pipeline`` and
``tracking``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.inference import OnnxModel
from hand_pipeline.om_runtime import PersistentAclModel
from hand_pipeline.om_runtime import PersistentAclRuntime
from hand_pipeline.pipeline import PipelineConfig
from hand_pipeline.pipeline import hand_prediction_to_dict
from hand_pipeline.pipeline import run_two_stage
from hand_pipeline.tracking import HandTracker
from hand_pipeline.tracking import TrackingConfig
from hand_pipeline.tracking import run_two_stage_image_mode


@dataclass
class FrameResult:
    predictions: list[dict[str, Any]]
    timings: dict[str, float]
    debug: dict[str, Any] | None = None


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


def _normalize_mode(mode: str) -> str:
    if mode not in {"image", "tracking"}:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    return mode


def _legacy_timings(timing: dict[str, Any]) -> dict[str, float]:
    """Map MediaPipe-style timing names to the repository's historical names."""
    preprocess = float(timing.get("det_pre_ms", timing.get("preprocess_ms", 0.0)) or 0.0)
    detector = float(timing.get("det_infer_ms", timing.get("detector_ms", 0.0)) or 0.0)
    decode = float(timing.get("det_decode_ms", timing.get("decode_ms", 0.0)) or 0.0)
    roi_only = float(timing.get("roi_ms", 0.0) or 0.0)
    crop = float(timing.get("crop_ms", 0.0) or 0.0)
    roi = roi_only + crop
    landmark = float(timing.get("landmark_infer_ms", timing.get("landmark_ms", 0.0)) or 0.0)
    post = float(timing.get("landmark_post_ms", timing.get("post_ms", 0.0)) or 0.0)
    total = float(timing.get("total_ms", preprocess + detector + decode + roi + landmark + post) or 0.0)
    result = {
        "preprocess_ms": preprocess,
        "detector_ms": detector,
        "decode_ms": decode,
        "roi_ms": roi,
        "landmark_ms": landmark,
        "post_ms": post,
        "total_ms": total,
        "det_pre_ms": preprocess,
        "det_npu_ms": detector,
        "det_post_ms": decode,
        "roi_only_ms": roi_only,
        "crop_ms": crop,
        "landmark_npu_ms": landmark,
        "landmark_post_ms": post,
    }
    for key in (
        "palm_detector_skipped",
        "palms",
        "hands",
        "prev_rects",
        "hand_rects",
        "next_rects",
        "tracking_state_kept",
        "tracking_state_rejected",
    ):
        if key in timing:
            result[key] = float(timing[key]) if isinstance(timing[key], (int, float, bool)) else timing[key]
    return result


def _box_from_hand21(hand21: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    xy = np.asarray(hand21, dtype=np.float32)[:, :2]
    x1 = float(np.clip(np.nanmin(xy[:, 0]), 0.0, float(image_width)))
    y1 = float(np.clip(np.nanmin(xy[:, 1]), 0.0, float(image_height)))
    x2 = float(np.clip(np.nanmax(xy[:, 0]), 0.0, float(image_width)))
    y2 = float(np.clip(np.nanmax(xy[:, 1]), 0.0, float(image_height)))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _palm7_from_hand21(hand21: np.ndarray) -> np.ndarray:
    points = np.asarray(hand21, dtype=np.float32)
    # MediaPipe palm detector keypoint order is wrist plus selected finger joints.
    return points[[0, 5, 9, 13, 17, 2, 3], :2].astype(np.float32)


def _rect_box(record: dict[str, Any], image_width: int, image_height: int) -> np.ndarray | None:
    required = ("x_center_px", "y_center_px", "width_px", "height_px")
    if not all(key in record for key in required):
        return None
    x_center = float(record["x_center_px"])
    y_center = float(record["y_center_px"])
    width = float(record["width_px"])
    height = float(record["height_px"])
    return np.asarray(
        [
            max(0.0, x_center - width * 0.5),
            max(0.0, y_center - height * 0.5),
            min(float(image_width), x_center + width * 0.5),
            min(float(image_height), y_center + height * 0.5),
        ],
        dtype=np.float32,
    )


def _compat_prediction(record: dict[str, Any], frame_shape: tuple[int, ...]) -> dict[str, Any]:
    image_height, image_width = int(frame_shape[0]), int(frame_shape[1])
    hand21 = np.asarray(
        record.get("hand21_keypoints_px", record.get("hand21")),
        dtype=np.float32,
    )
    if hand21.ndim != 2 or hand21.shape[0] != 21:
        raise ValueError(f"Expected 21 hand landmarks, got {hand21.shape}")

    palm_box = record.get("palm_bbox_xyxy_px", record.get("box"))
    if palm_box is None:
        palm_box = _rect_box(record.get("hand_roi", {}), image_width, image_height)
    if palm_box is None:
        palm_box = _box_from_hand21(hand21, image_width, image_height)
    palm7 = record.get("palm7_keypoints_px", record.get("palm7"))
    if palm7 is None:
        palm7 = _palm7_from_hand21(hand21)

    source_roi = str(record.get("source_roi", "palm_detection"))
    score = record.get("score")
    if score is None:
        score = math.nan
    result: dict[str, Any] = {
        "hand_index": int(record.get("hand_index", 0)),
        "score": float(score),
        "hand_score": float(record.get("hand_score", math.nan)),
        "handedness": float(record.get("handedness", math.nan))
        if record.get("handedness") is not None
        else math.nan,
        "box": np.asarray(palm_box, dtype=np.float32).astype(float).tolist(),
        "palm7": np.asarray(palm7, dtype=np.float32).astype(float).tolist(),
        "hand21": hand21.astype(float).tolist(),
        "roi_center": np.asarray(record.get("roi_center_px", [math.nan, math.nan]), dtype=np.float32)
        .astype(float)
        .tolist(),
        "roi_size": float(record.get("roi_size_px", math.nan)),
        "roi_rotation_rad": float(record.get("roi_rotation_rad", math.nan)),
        "source_roi": source_roi,
        "palm_detector_skipped": source_roi == "previous_landmarks",
    }
    for key in ("hand_roi", "palm_roi", "next_tracking_roi", "source_index"):
        if key in record:
            result[key] = record[key]
    return result


def _image_predictions_from_primitives(
    image_bgr: np.ndarray,
    *,
    detector: Any,
    landmark: Any,
    anchors: np.ndarray,
    config: PipelineConfig,
) -> FrameResult:
    # This path is useful for callers that need the typed primitive API. The
    # timing-focused image path below is used by default by the wrapper classes.
    _palms, predictions = run_two_stage(image_bgr, detector, landmark, anchors=anchors, config=config)
    return FrameResult(
        predictions=[hand_prediction_to_dict(item) for item in predictions],
        timings={},
        debug=None,
    )


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
    roi_scale: float = 2.6,
    shift_y: float = -0.5,
    rotation_offset_degrees: float = 0.0,
    mode: str = "image",
    tracker: HandTracker | None = None,
    roi_precision: str = "float32",
    projection_precision: str = "float32",
) -> FrameResult:
    mode = _normalize_mode(mode)
    if mode == "tracking":
        if tracker is None:
            raise ValueError("tracking mode requires a HandTracker")
        timing, records, debug = tracker.process(image_bgr)
        return FrameResult(
            predictions=[_compat_prediction(item, image_bgr.shape) for item in records],
            timings=_legacy_timings(timing),
            debug=debug,
        )

    config = TrackingConfig(
        score_threshold=score_threshold,
        nms_iou=nms_iou,
        max_det=max_det,
        max_hands=max_hands,
        min_hand_score=min_hand_score,
        roi_precision=roi_precision,
        projection_precision=projection_precision,
    )
    timing, records = run_two_stage_image_mode(image_bgr, detector, landmark, anchors=anchors, config=config)
    return FrameResult(
        predictions=[_compat_prediction(item, image_bgr.shape) for item in records],
        timings=_legacy_timings(timing),
        debug=None,
    )


class _OmDetectorRunner:
    def __init__(
        self,
        detector_path: Path,
        runtime: PersistentAclRuntime,
        *,
        reload_each_call: bool,
    ) -> None:
        self.model_path = Path(detector_path)
        self.runtime = runtime
        self.reload_each_call = bool(reload_each_call)
        self.model: PersistentAclModel | None = None
        if not self.reload_each_call:
            self.model = PersistentAclModel(self.model_path, runtime=self.runtime)

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self.reload_each_call:
            detector = PersistentAclModel(self.model_path, runtime=self.runtime)
            try:
                return detector.infer(tensor)
            finally:
                detector.release()
        if self.model is None:
            self.model = PersistentAclModel(self.model_path, runtime=self.runtime)
        return self.model.infer(tensor)

    def close(self) -> None:
        if self.model is not None:
            self.model.release()
            self.model = None


class _OmLandmarkRunner:
    def __init__(self, landmark_path: Path, runtime: PersistentAclRuntime) -> None:
        self.model_path = Path(landmark_path)
        self.model = PersistentAclModel(self.model_path, runtime=runtime)

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.model.infer(tensor)

    def close(self) -> None:
        self.model.release()


class _BaseHandPipeline:
    def __init__(
        self,
        *,
        score_threshold: float,
        nms_iou: float,
        max_hands: int,
        min_hand_score: float,
        max_det: int,
        roi_scale: float = 2.6,
        shift_y: float = -0.5,
        rotation_offset_degrees: float = 0.0,
        mode: str = "image",
        roi_precision: str = "float32",
        projection_precision: str = "float32",
        max_tracking_lost_frames: int = 0,
        max_tracking_rejected_frames: int = 0,
        max_tracking_rotation_delta: float = math.inf,
        min_tracking_size_ratio: float = 0.0,
        max_tracking_size_ratio: float = math.inf,
        max_tracking_center_shift: float = math.inf,
        tracking_rect_smooth_alpha: float = 1.0,
    ) -> None:
        self.anchors = generate_palm_anchors()
        self.score_threshold = float(score_threshold)
        self.nms_iou = float(nms_iou)
        self.max_hands = int(max_hands)
        self.min_hand_score = float(min_hand_score)
        self.max_det = int(max_det)
        self.roi_scale = float(roi_scale)
        self.shift_y = float(shift_y)
        self.rotation_offset_degrees = float(rotation_offset_degrees)
        self.mode = _normalize_mode(mode)
        self.roi_precision = roi_precision
        self.projection_precision = projection_precision
        self.max_tracking_lost_frames = int(max_tracking_lost_frames)
        self.max_tracking_rejected_frames = int(max_tracking_rejected_frames)
        self.max_tracking_rotation_delta = float(max_tracking_rotation_delta)
        self.min_tracking_size_ratio = float(min_tracking_size_ratio)
        self.max_tracking_size_ratio = float(max_tracking_size_ratio)
        self.max_tracking_center_shift = float(max_tracking_center_shift)
        self.tracking_rect_smooth_alpha = float(tracking_rect_smooth_alpha)
        self.tracker: HandTracker | None = None

    def _tracking_config(self) -> TrackingConfig:
        return TrackingConfig(
            score_threshold=self.score_threshold,
            nms_iou=self.nms_iou,
            max_det=self.max_det,
            max_hands=self.max_hands,
            min_hand_score=self.min_hand_score,
            max_tracking_lost_frames=self.max_tracking_lost_frames,
            max_tracking_rejected_frames=self.max_tracking_rejected_frames,
            max_tracking_rotation_delta=self.max_tracking_rotation_delta,
            min_tracking_size_ratio=self.min_tracking_size_ratio,
            max_tracking_size_ratio=self.max_tracking_size_ratio,
            max_tracking_center_shift=self.max_tracking_center_shift,
            tracking_rect_smooth_alpha=self.tracking_rect_smooth_alpha,
            roi_precision=self.roi_precision,
            projection_precision=self.projection_precision,
        )

    def _ensure_tracker(self) -> HandTracker:
        if self.tracker is None:
            self.tracker = HandTracker(
                self.detector,
                self.landmark,
                anchors=self.anchors,
                config=self._tracking_config(),
            )
        return self.tracker

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

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
            mode=self.mode,
            tracker=self._ensure_tracker() if self.mode == "tracking" else None,
            roi_precision=self.roi_precision,
            projection_precision=self.projection_precision,
        )

    def infer_nv12(self, image_nv12: np.ndarray, image_width: int, image_height: int) -> FrameResult:
        if self.mode != "tracking":
            raise ValueError("NV12 inference currently supports tracking mode only")
        tracker = self._ensure_tracker()
        timing, records, debug = tracker.process_nv12(image_nv12, image_width, image_height)
        frame_shape = (int(image_height), int(image_width), 3)
        return FrameResult(
            predictions=[_compat_prediction(item, frame_shape) for item in records],
            timings=_legacy_timings(timing),
            debug=debug,
        )


class OnnxHandPipeline(_BaseHandPipeline):
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
        roi_scale: float = 2.6,
        shift_y: float = -0.5,
        rotation_offset_degrees: float = 0.0,
        mode: str = "image",
        roi_precision: str = "float32",
        projection_precision: str = "float32",
        max_tracking_lost_frames: int = 0,
        max_tracking_rejected_frames: int = 0,
        max_tracking_rotation_delta: float = math.inf,
        min_tracking_size_ratio: float = 0.0,
        max_tracking_size_ratio: float = math.inf,
        max_tracking_center_shift: float = math.inf,
        tracking_rect_smooth_alpha: float = 1.0,
    ) -> None:
        self.detector = OnnxModel(detector_path)
        self.landmark = OnnxModel(landmark_path)
        super().__init__(
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            max_hands=max_hands,
            min_hand_score=min_hand_score,
            max_det=max_det,
            roi_scale=roi_scale,
            shift_y=shift_y,
            rotation_offset_degrees=rotation_offset_degrees,
            mode=mode,
            roi_precision=roi_precision,
            projection_precision=projection_precision,
            max_tracking_lost_frames=max_tracking_lost_frames,
            max_tracking_rejected_frames=max_tracking_rejected_frames,
            max_tracking_rotation_delta=max_tracking_rotation_delta,
            min_tracking_size_ratio=min_tracking_size_ratio,
            max_tracking_size_ratio=max_tracking_size_ratio,
            max_tracking_center_shift=max_tracking_center_shift,
            tracking_rect_smooth_alpha=tracking_rect_smooth_alpha,
        )

    def close(self) -> None:
        self.reset()


class OmHandPipeline(_BaseHandPipeline):
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
        roi_scale: float = 2.6,
        shift_y: float = -0.5,
        rotation_offset_degrees: float = 0.0,
        reload_detector_each_frame: bool = False,
        finalize_on_release: bool = True,
        mode: str = "image",
        roi_precision: str = "float32",
        projection_precision: str = "float32",
        max_tracking_lost_frames: int = 0,
        max_tracking_rejected_frames: int = 0,
        max_tracking_rotation_delta: float = math.inf,
        min_tracking_size_ratio: float = 0.0,
        max_tracking_size_ratio: float = math.inf,
        max_tracking_center_shift: float = math.inf,
        tracking_rect_smooth_alpha: float = 1.0,
    ) -> None:
        self.runtime = PersistentAclRuntime(device_id=device_id, finalize_on_release=finalize_on_release)
        try:
            self.detector = _OmDetectorRunner(
                Path(detector_path),
                self.runtime,
                reload_each_call=reload_detector_each_frame,
            )
            self.landmark = _OmLandmarkRunner(Path(landmark_path), self.runtime)
            self.reload_detector_each_frame = bool(reload_detector_each_frame)
            super().__init__(
                score_threshold=score_threshold,
                nms_iou=nms_iou,
                max_hands=max_hands,
                min_hand_score=min_hand_score,
                max_det=max_det,
                roi_scale=roi_scale,
                shift_y=shift_y,
                rotation_offset_degrees=rotation_offset_degrees,
                mode=mode,
                roi_precision=roi_precision,
                projection_precision=projection_precision,
                max_tracking_lost_frames=max_tracking_lost_frames,
                max_tracking_rejected_frames=max_tracking_rejected_frames,
                max_tracking_rotation_delta=max_tracking_rotation_delta,
                min_tracking_size_ratio=min_tracking_size_ratio,
                max_tracking_size_ratio=max_tracking_size_ratio,
                max_tracking_center_shift=max_tracking_center_shift,
                tracking_rect_smooth_alpha=tracking_rect_smooth_alpha,
            )
        except Exception:
            detector = getattr(self, "detector", None)
            if detector is not None:
                detector.close()
            landmark = getattr(self, "landmark", None)
            if landmark is not None:
                landmark.close()
            self.runtime.release()
            raise

    def close(self) -> None:
        self.reset()
        self.detector.close()
        self.landmark.close()
        self.runtime.release()
