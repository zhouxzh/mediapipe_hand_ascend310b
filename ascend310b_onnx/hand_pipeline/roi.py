"""MediaPipe-style hand ROI geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HandRoi:
    matrix: np.ndarray
    inverse: np.ndarray
    center: np.ndarray
    size: float
    rotation: float
    crop: np.ndarray


@dataclass(frozen=True)
class NormalizedRect:
    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float


def normalize_radians(angle: float) -> float:
    return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))


def _f32(value: float) -> float:
    return float(np.float32(value))


def normalize_radians_float32(angle: float) -> float:
    angle = _f32(angle)
    return _f32(angle - _f32(2.0 * math.pi) * math.floor((angle + _f32(math.pi)) / _f32(2.0 * math.pi)))


def local_axes(rotation: float) -> tuple[np.ndarray, np.ndarray]:
    x_axis = np.array([math.cos(rotation), math.sin(rotation)], dtype=np.float32)
    y_axis = np.array([-math.sin(rotation), math.cos(rotation)], dtype=np.float32)
    return x_axis, y_axis


def rotation_from_keypoints(
    keypoints: np.ndarray,
    start_idx: int = 0,
    end_idx: int = 2,
    target_angle_degrees: float = 90.0,
) -> float:
    start = keypoints[start_idx].astype(np.float32)
    end = keypoints[end_idx].astype(np.float32)
    vector = end - start
    if float(np.linalg.norm(vector)) < 1e-6:
        return 0.0
    target = math.radians(target_angle_degrees)
    return normalize_radians(target - math.atan2(float(-(vector[1])), float(vector[0])))


def rotation_from_keypoints_float32(
    keypoints: np.ndarray,
    start_idx: int = 0,
    end_idx: int = 2,
    target_angle_degrees: float = 90.0,
) -> float:
    start = keypoints[start_idx].astype(np.float32)
    end = keypoints[end_idx].astype(np.float32)
    vector_x = _f32(_f32(end[0]) - _f32(start[0]))
    vector_y = _f32(_f32(end[1]) - _f32(start[1]))
    if _f32(math.hypot(vector_x, vector_y)) < _f32(1e-6):
        return 0.0
    target = _f32(_f32(math.pi) * _f32(target_angle_degrees) / _f32(180.0))
    return normalize_radians_float32(_f32(target - _f32(math.atan2(_f32(-vector_y), vector_x))))


def make_hand_roi(
    image: np.ndarray,
    box_xyxy: np.ndarray,
    keypoints: np.ndarray,
    input_size: int = 224,
    scale: float = 2.6,
    shift_y: float = -0.5,
    rotation_start: int = 0,
    rotation_end: int = 2,
    rotation_offset_degrees: float = 0.0,
    dst_size_mode: str = "size",
) -> HandRoi:
    """Create the hand landmark crop from a palm detection.

    The geometry mirrors MediaPipe's PalmDetectionDetectionToRoi subgraph:
    DetectionsToRectsCalculator followed by RectTransformationCalculator with
    scale_x=scale_y=2.6, shift_y=-0.5, and square_long=true.
    """
    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    rotation = normalize_radians(
        rotation_from_keypoints(keypoints, rotation_start, rotation_end) + math.radians(rotation_offset_degrees)
    )

    if rotation == 0.0:
        center[1] += height * shift_y
    else:
        center[0] += -height * shift_y * math.sin(rotation)
        center[1] += height * shift_y * math.cos(rotation)

    long_side = max(width, height)
    roi_size = long_side * scale
    x_axis, y_axis = local_axes(rotation)
    src = np.array(
        [
            center - 0.5 * roi_size * x_axis - 0.5 * roi_size * y_axis,
            center + 0.5 * roi_size * x_axis - 0.5 * roi_size * y_axis,
            center - 0.5 * roi_size * x_axis + 0.5 * roi_size * y_axis,
        ],
        dtype=np.float32,
    )
    if dst_size_mode == "size":
        dst_extent = float(input_size)
    elif dst_size_mode == "minus_one":
        dst_extent = float(input_size - 1)
    else:
        raise ValueError(f"Unknown dst_size_mode: {dst_size_mode}")
    dst = np.array([[0.0, 0.0], [dst_extent, 0.0], [0.0, dst_extent]], dtype=np.float32)
    matrix = cv2.getAffineTransform(src, dst)
    inverse = cv2.invertAffineTransform(matrix)
    crop = cv2.warpAffine(
        image,
        matrix,
        (input_size, input_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return HandRoi(matrix=matrix, inverse=inverse, center=center, size=roi_size, rotation=rotation, crop=crop)


def transform_normalized_rect(
    rect: NormalizedRect,
    image_width: int,
    image_height: int,
    scale_x: float,
    scale_y: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    square_long: bool = True,
) -> NormalizedRect:
    """Mirror MediaPipe RectTransformationCalculator for NormalizedRect."""
    width = float(rect.width)
    height = float(rect.height)
    rotation = float(rect.rotation)
    image_width_f = max(float(image_width), 1.0)
    image_height_f = max(float(image_height), 1.0)

    if rotation == 0.0:
        x_center = float(rect.x_center) + width * shift_x
        y_center = float(rect.y_center) + height * shift_y
    else:
        x_shift = (
            image_width_f * width * shift_x * math.cos(rotation)
            - image_height_f * height * shift_y * math.sin(rotation)
        ) / image_width_f
        y_shift = (
            image_width_f * width * shift_x * math.sin(rotation)
            + image_height_f * height * shift_y * math.cos(rotation)
        ) / image_height_f
        x_center = float(rect.x_center) + x_shift
        y_center = float(rect.y_center) + y_shift

    if square_long:
        long_side = max(width * image_width_f, height * image_height_f)
        width = long_side / image_width_f
        height = long_side / image_height_f

    return NormalizedRect(
        x_center=float(x_center),
        y_center=float(y_center),
        width=float(width * scale_x),
        height=float(height * scale_y),
        rotation=rotation,
    )


def transform_normalized_rect_float32(
    rect: NormalizedRect,
    image_width: int,
    image_height: int,
    scale_x: float,
    scale_y: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    square_long: bool = True,
) -> NormalizedRect:
    """Mirror RectTransformationCalculator with float32-style intermediates."""
    width = _f32(rect.width)
    height = _f32(rect.height)
    rotation = _f32(rect.rotation)
    image_width_f = _f32(max(float(image_width), 1.0))
    image_height_f = _f32(max(float(image_height), 1.0))
    scale_x_f = _f32(scale_x)
    scale_y_f = _f32(scale_y)
    shift_x_f = _f32(shift_x)
    shift_y_f = _f32(shift_y)

    if rotation == 0.0:
        x_center = _f32(_f32(rect.x_center) + _f32(width * shift_x_f))
        y_center = _f32(_f32(rect.y_center) + _f32(height * shift_y_f))
    else:
        cos_r = _f32(math.cos(rotation))
        sin_r = _f32(math.sin(rotation))
        x_shift = _f32(
            _f32(_f32(image_width_f * width) * shift_x_f) * cos_r
            - _f32(_f32(image_height_f * height) * shift_y_f) * sin_r
        )
        x_shift = _f32(x_shift / image_width_f)
        y_shift = _f32(
            _f32(_f32(image_width_f * width) * shift_x_f) * sin_r
            + _f32(_f32(image_height_f * height) * shift_y_f) * cos_r
        )
        y_shift = _f32(y_shift / image_height_f)
        x_center = _f32(_f32(rect.x_center) + x_shift)
        y_center = _f32(_f32(rect.y_center) + y_shift)

    if square_long:
        long_side = _f32(max(_f32(width * image_width_f), _f32(height * image_height_f)))
        width = _f32(long_side / image_width_f)
        height = _f32(long_side / image_height_f)

    return NormalizedRect(
        x_center=x_center,
        y_center=y_center,
        width=_f32(width * scale_x_f),
        height=_f32(height * scale_y_f),
        rotation=rotation,
    )


def normalized_rect_from_palm_detection(
    box_xyxy: np.ndarray,
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
) -> NormalizedRect:
    """Create the MediaPipe PalmDetectionDetectionToRoi NormalizedRect."""
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32)
    image_width_f = max(float(image_width), 1.0)
    image_height_f = max(float(image_height), 1.0)
    raw_rect = NormalizedRect(
        x_center=float((x1 + x2) * 0.5 / image_width_f),
        y_center=float((y1 + y2) * 0.5 / image_height_f),
        width=float(max(float(x2 - x1), 1.0) / image_width_f),
        height=float(max(float(y2 - y1), 1.0) / image_height_f),
        rotation=rotation_from_keypoints(np.asarray(keypoints, dtype=np.float32), 0, 2, 90.0),
    )
    return transform_normalized_rect(
        raw_rect,
        image_width=image_width,
        image_height=image_height,
        scale_x=2.6,
        scale_y=2.6,
        shift_y=-0.5,
        square_long=True,
    )


def normalized_rect_from_palm_detection_float32(
    box_xyxy: np.ndarray,
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
) -> NormalizedRect:
    """Create PalmDetectionDetectionToRoi NormalizedRect with float32 intermediates."""
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32)
    image_width_f = _f32(max(float(image_width), 1.0))
    image_height_f = _f32(max(float(image_height), 1.0))
    raw_width_px = _f32(max(_f32(_f32(x2) - _f32(x1)), _f32(1.0)))
    raw_height_px = _f32(max(_f32(_f32(y2) - _f32(y1)), _f32(1.0)))
    raw_rect = NormalizedRect(
        x_center=_f32(_f32(_f32(_f32(x1) + _f32(x2)) * _f32(0.5)) / image_width_f),
        y_center=_f32(_f32(_f32(_f32(y1) + _f32(y2)) * _f32(0.5)) / image_height_f),
        width=_f32(raw_width_px / image_width_f),
        height=_f32(raw_height_px / image_height_f),
        rotation=rotation_from_keypoints_float32(np.asarray(keypoints, dtype=np.float32), 0, 2, 90.0),
    )
    return transform_normalized_rect_float32(
        raw_rect,
        image_width=image_width,
        image_height=image_height,
        scale_x=2.6,
        scale_y=2.6,
        shift_y=-0.5,
        square_long=True,
    )


def normalized_rect_to_pixel_dict(rect: NormalizedRect, image_width: int, image_height: int) -> dict[str, float]:
    return {
        "x_center_px": float(rect.x_center) * float(image_width),
        "y_center_px": float(rect.y_center) * float(image_height),
        "width_px": float(rect.width) * float(image_width),
        "height_px": float(rect.height) * float(image_height),
        "rotation": float(rect.rotation),
    }


def normalized_rect_to_dict(rect: NormalizedRect) -> dict[str, float]:
    return {
        "x_center": float(rect.x_center),
        "y_center": float(rect.y_center),
        "width": float(rect.width),
        "height": float(rect.height),
        "rotation": float(rect.rotation),
    }


def normalized_rect_from_dict(rect: dict[str, float], image_width: int, image_height: int) -> NormalizedRect:
    if "x_center" in rect:
        return NormalizedRect(
            x_center=float(rect["x_center"]),
            y_center=float(rect["y_center"]),
            width=float(rect["width"]),
            height=float(rect["height"]),
            rotation=float(rect.get("rotation", 0.0)),
        )
    return NormalizedRect(
        x_center=float(rect["x_center_px"]) / max(float(image_width), 1.0),
        y_center=float(rect["y_center_px"]) / max(float(image_height), 1.0),
        width=float(rect["width_px"]) / max(float(image_width), 1.0),
        height=float(rect["height_px"]) / max(float(image_height), 1.0),
        rotation=float(rect.get("rotation", 0.0)),
    )


def hand_roi_from_normalized_rect(
    image: np.ndarray,
    rect: NormalizedRect,
    input_size: int = 224,
    border_mode: str = "replicate",
    dst_size_mode: str = "size",
) -> HandRoi:
    height, width = image.shape[:2]
    pixel_rect = normalized_rect_to_pixel_dict(rect, width, height)
    center = np.array([pixel_rect["x_center_px"], pixel_rect["y_center_px"]], dtype=np.float32)
    rect_width = float(pixel_rect["width_px"])
    rect_height = float(pixel_rect["height_px"])
    rotation = float(pixel_rect["rotation"])
    src = cv2.boxPoints(
        (
            (float(center[0]), float(center[1])),
            (float(rect_width), float(rect_height)),
            float(rotation * 180.0 / math.pi),
        )
    ).astype(np.float32)
    if dst_size_mode == "size":
        dst_extent = float(input_size)
    elif dst_size_mode == "minus_one":
        dst_extent = float(input_size - 1)
    else:
        raise ValueError(f"Unknown dst_size_mode: {dst_size_mode}")
    if border_mode == "replicate":
        cv_border_mode = cv2.BORDER_REPLICATE
    elif border_mode in {"zero", "constant"}:
        cv_border_mode = cv2.BORDER_CONSTANT
    else:
        raise ValueError(f"Unknown border_mode: {border_mode}")
    dst = np.array(
        [
            [0.0, dst_extent],
            [0.0, 0.0],
            [dst_extent, 0.0],
            [dst_extent, dst_extent],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getAffineTransform(src[:3], dst[:3])
    inverse = cv2.invertAffineTransform(matrix)
    projection_matrix = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(
        image,
        projection_matrix,
        (input_size, input_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv_border_mode,
    )
    return HandRoi(
        matrix=matrix,
        inverse=inverse,
        center=center,
        size=max(rect_width, rect_height),
        rotation=rotation,
        crop=crop,
    )


def crop_from_normalized_rect(
    image: np.ndarray,
    rect: dict[str, float] | NormalizedRect,
    input_size: int = 224,
    border_mode: str = "replicate",
    dst_size_mode: str = "size",
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(rect, NormalizedRect):
        rect = normalized_rect_from_dict(rect, image.shape[1], image.shape[0])
    roi = hand_roi_from_normalized_rect(
        image,
        rect,
        input_size=input_size,
        border_mode=border_mode,
        dst_size_mode=dst_size_mode,
    )
    return roi.crop, roi.inverse


def landmarks_to_tracking_roi(
    landmarks: np.ndarray,
    image_width: int,
    image_height: int,
) -> NormalizedRect:
    """Mirror MediaPipe HandLandmarksToRectCalculator plus tracking rect transform."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected landmarks with shape (N, >=2), got {points.shape}")
    xy = points[:, :2].copy()
    image_width_f = max(float(image_width), 1.0)
    image_height_f = max(float(image_height), 1.0)
    if float(np.nanmax(np.abs(xy))) > 2.0:
        xy[:, 0] /= image_width_f
        xy[:, 1] /= image_height_f

    if xy.shape[0] == 21:
        xy = xy[[0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]]

    wrist_joint = 0
    middle_finger_pip_joint = 6
    index_finger_pip_joint = 4
    ring_finger_pip_joint = 8
    x0 = float(xy[wrist_joint, 0]) * image_width_f
    y0 = float(xy[wrist_joint, 1]) * image_height_f
    x1 = (float(xy[index_finger_pip_joint, 0]) + float(xy[ring_finger_pip_joint, 0])) / 2.0
    y1 = (float(xy[index_finger_pip_joint, 1]) + float(xy[ring_finger_pip_joint, 1])) / 2.0
    x1 = (x1 + float(xy[middle_finger_pip_joint, 0])) / 2.0 * image_width_f
    y1 = (y1 + float(xy[middle_finger_pip_joint, 1])) / 2.0 * image_height_f
    rotation = normalize_radians((math.pi * 0.5) - math.atan2(-(y1 - y0), x1 - x0))
    reverse_angle = normalize_radians(-rotation)

    axis_aligned_center_x = float((np.max(xy[:, 0]) + np.min(xy[:, 0])) / 2.0)
    axis_aligned_center_y = float((np.max(xy[:, 1]) + np.min(xy[:, 1])) / 2.0)
    original_x = (xy[:, 0] - axis_aligned_center_x) * image_width_f
    original_y = (xy[:, 1] - axis_aligned_center_y) * image_height_f
    projected_x = original_x * math.cos(reverse_angle) - original_y * math.sin(reverse_angle)
    projected_y = original_x * math.sin(reverse_angle) + original_y * math.cos(reverse_angle)

    min_x = float(np.min(projected_x))
    max_x = float(np.max(projected_x))
    min_y = float(np.min(projected_y))
    max_y = float(np.max(projected_y))
    projected_center_x = (max_x + min_x) / 2.0
    projected_center_y = (max_y + min_y) / 2.0
    center_x = (
        projected_center_x * math.cos(rotation)
        - projected_center_y * math.sin(rotation)
        + image_width_f * axis_aligned_center_x
    )
    center_y = (
        projected_center_x * math.sin(rotation)
        + projected_center_y * math.cos(rotation)
        + image_height_f * axis_aligned_center_y
    )
    raw_rect = NormalizedRect(
        x_center=float(center_x / image_width_f),
        y_center=float(center_y / image_height_f),
        width=float((max_x - min_x) / image_width_f),
        height=float((max_y - min_y) / image_height_f),
        rotation=float(rotation),
    )
    return transform_normalized_rect(
        raw_rect,
        image_width=image_width,
        image_height=image_height,
        scale_x=2.0,
        scale_y=2.0,
        shift_y=-0.1,
        square_long=True,
    )


def landmarks_to_tracking_roi_float32(
    landmarks: np.ndarray,
    image_width: int,
    image_height: int,
) -> NormalizedRect:
    """Mirror HandLandmarksToRectCalculator with float32-style intermediates."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected landmarks with shape (N, >=2), got {points.shape}")
    xy = points[:, :2].astype(np.float32, copy=True)
    image_width_f = _f32(max(float(image_width), 1.0))
    image_height_f = _f32(max(float(image_height), 1.0))
    if float(np.nanmax(np.abs(xy))) > 2.0:
        xy[:, 0] = (xy[:, 0] / np.float32(image_width_f)).astype(np.float32)
        xy[:, 1] = (xy[:, 1] / np.float32(image_height_f)).astype(np.float32)

    if xy.shape[0] == 21:
        xy = xy[[0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]]

    wrist_joint = 0
    middle_finger_pip_joint = 6
    index_finger_pip_joint = 4
    ring_finger_pip_joint = 8
    x0 = _f32(_f32(xy[wrist_joint, 0]) * image_width_f)
    y0 = _f32(_f32(xy[wrist_joint, 1]) * image_height_f)
    x1 = _f32((_f32(xy[index_finger_pip_joint, 0]) + _f32(xy[ring_finger_pip_joint, 0])) / _f32(2.0))
    y1 = _f32((_f32(xy[index_finger_pip_joint, 1]) + _f32(xy[ring_finger_pip_joint, 1])) / _f32(2.0))
    x1 = _f32(_f32(_f32(x1 + _f32(xy[middle_finger_pip_joint, 0])) / _f32(2.0)) * image_width_f)
    y1 = _f32(_f32(_f32(y1 + _f32(xy[middle_finger_pip_joint, 1])) / _f32(2.0)) * image_height_f)
    rotation = normalize_radians_float32(_f32(_f32(math.pi * 0.5) - _f32(math.atan2(_f32(-(y1 - y0)), _f32(x1 - x0)))))
    reverse_angle = normalize_radians_float32(-rotation)
    cos_reverse = _f32(math.cos(reverse_angle))
    sin_reverse = _f32(math.sin(reverse_angle))
    cos_rotation = _f32(math.cos(rotation))
    sin_rotation = _f32(math.sin(rotation))

    min_x_norm = _f32(np.min(xy[:, 0]))
    max_x_norm = _f32(np.max(xy[:, 0]))
    min_y_norm = _f32(np.min(xy[:, 1]))
    max_y_norm = _f32(np.max(xy[:, 1]))
    axis_aligned_center_x = _f32(_f32(max_x_norm + min_x_norm) / _f32(2.0))
    axis_aligned_center_y = _f32(_f32(max_y_norm + min_y_norm) / _f32(2.0))

    projected_x_values: list[float] = []
    projected_y_values: list[float] = []
    for point in xy:
        original_x = _f32(_f32(_f32(point[0]) - axis_aligned_center_x) * image_width_f)
        original_y = _f32(_f32(_f32(point[1]) - axis_aligned_center_y) * image_height_f)
        projected_x_values.append(_f32(_f32(original_x * cos_reverse) - _f32(original_y * sin_reverse)))
        projected_y_values.append(_f32(_f32(original_x * sin_reverse) + _f32(original_y * cos_reverse)))

    min_x = _f32(min(projected_x_values))
    max_x = _f32(max(projected_x_values))
    min_y = _f32(min(projected_y_values))
    max_y = _f32(max(projected_y_values))
    projected_center_x = _f32(_f32(max_x + min_x) / _f32(2.0))
    projected_center_y = _f32(_f32(max_y + min_y) / _f32(2.0))
    center_x = _f32(
        _f32(projected_center_x * cos_rotation)
        - _f32(projected_center_y * sin_rotation)
        + _f32(image_width_f * axis_aligned_center_x)
    )
    center_y = _f32(
        _f32(projected_center_x * sin_rotation)
        + _f32(projected_center_y * cos_rotation)
        + _f32(image_height_f * axis_aligned_center_y)
    )
    raw_rect = NormalizedRect(
        x_center=_f32(center_x / image_width_f),
        y_center=_f32(center_y / image_height_f),
        width=_f32(_f32(max_x - min_x) / image_width_f),
        height=_f32(_f32(max_y - min_y) / image_height_f),
        rotation=rotation,
    )
    return transform_normalized_rect_float32(
        raw_rect,
        image_width=image_width,
        image_height=image_height,
        scale_x=2.0,
        scale_y=2.0,
        shift_y=-0.1,
        square_long=True,
    )


def normalized_rect_iou_axis_aligned(rect: NormalizedRect, other: NormalizedRect) -> float:
    x1 = max(float(rect.x_center - rect.width * 0.5), float(other.x_center - other.width * 0.5))
    y1 = max(float(rect.y_center - rect.height * 0.5), float(other.y_center - other.height * 0.5))
    x2 = min(float(rect.x_center + rect.width * 0.5), float(other.x_center + other.width * 0.5))
    y2 = min(float(rect.y_center + rect.height * 0.5), float(other.y_center + other.height * 0.5))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, float(rect.width)) * max(0.0, float(rect.height))
    other_area = max(0.0, float(other.width)) * max(0.0, float(other.height))
    denom = area + other_area - intersection
    return intersection / denom if denom > 0.0 else 0.0


def associate_normalized_rects(
    rect_vectors: list[list[NormalizedRect]],
    min_similarity_threshold: float = 0.5,
) -> list[NormalizedRect]:
    """Mirror AssociationNormRectCalculator without PREV-tag ID propagation."""
    result: list[NormalizedRect] = []
    non_empty_id = 0
    for index, rects in enumerate(rect_vectors):
        if rects:
            non_empty_id = index
            for rect in rects:
                _add_association_rect(rect, result, min_similarity_threshold)
            break
    for rects in rect_vectors[non_empty_id + 1 :]:
        for rect in rects:
            _add_association_rect(rect, result, min_similarity_threshold)
    return result


def _add_association_rect(
    rect: NormalizedRect,
    current: list[NormalizedRect],
    min_similarity_threshold: float,
) -> None:
    kept = [
        item
        for item in current
        if normalized_rect_iou_axis_aligned(rect, item) <= min_similarity_threshold
    ]
    kept.append(rect)
    current[:] = kept


def preprocess_landmark_tflite(crop: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb[None])


def landmarks_to_original(
    landmarks_crop: np.ndarray,
    inverse_affine: np.ndarray,
    input_size: int = 224,
    coord_scale: str = "auto",
) -> np.ndarray:
    points = landmarks_crop[:, :2].astype(np.float32).copy()
    if coord_scale == "normalized" or (coord_scale == "auto" and float(np.nanmax(np.abs(points))) <= 2.0):
        points *= input_size
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return homogeneous @ inverse_affine.T


def project_landmarks_with_normalized_rect(
    landmarks_crop: np.ndarray,
    rect: NormalizedRect,
    image_width: int,
    image_height: int,
    input_size: int = 224,
    coord_scale: str = "auto",
) -> np.ndarray:
    """Mirror MediaPipe LandmarkProjectionCalculator with NORM_RECT input."""
    landmarks = np.asarray(landmarks_crop, dtype=np.float32).copy()
    if landmarks.ndim != 2 or landmarks.shape[1] < 2:
        raise ValueError(f"Expected landmarks with shape (N, >=2), got {landmarks.shape}")
    if coord_scale == "pixel" or (coord_scale == "auto" and float(np.nanmax(np.abs(landmarks[:, :2]))) > 2.0):
        landmarks[:, 0] /= float(input_size)
        landmarks[:, 1] /= float(input_size)
        if landmarks.shape[1] > 2:
            landmarks[:, 2] /= float(input_size)

    x = landmarks[:, 0] - 0.5
    y = landmarks[:, 1] - 0.5
    angle = float(rect.rotation)
    new_x = math.cos(angle) * x - math.sin(angle) * y
    new_y = math.sin(angle) * x + math.cos(angle) * y
    output = landmarks.copy()
    output[:, 0] = (new_x * float(rect.width) + float(rect.x_center)) * float(image_width)
    output[:, 1] = (new_y * float(rect.height) + float(rect.y_center)) * float(image_height)
    if output.shape[1] > 2:
        output[:, 2] = landmarks[:, 2] * float(rect.width)
    return output


def project_landmarks_normalized_with_rect_float32(
    landmarks_crop: np.ndarray,
    rect: NormalizedRect,
    input_size: int = 224,
    coord_scale: str = "auto",
) -> np.ndarray:
    """Mirror TensorsToLandmarks + LandmarkProjectionCalculator with float outputs.

    The result stays in full-image normalized coordinates, matching the
    NormalizedLandmarkList consumed by HandLandmarksToRectCalculator.
    """
    landmarks = np.asarray(landmarks_crop, dtype=np.float32).copy()
    if landmarks.ndim != 2 or landmarks.shape[1] < 2:
        raise ValueError(f"Expected landmarks with shape (N, >=2), got {landmarks.shape}")
    input_size_f = _f32(input_size)
    if coord_scale == "pixel" or (coord_scale == "auto" and float(np.nanmax(np.abs(landmarks[:, :2]))) > 2.0):
        landmarks[:, 0] = (landmarks[:, 0] / np.float32(input_size_f)).astype(np.float32)
        landmarks[:, 1] = (landmarks[:, 1] / np.float32(input_size_f)).astype(np.float32)
        if landmarks.shape[1] > 2:
            # TensorsToLandmarksCalculator normalizes z by input width and normalize_z=0.4.
            landmarks[:, 2] = (landmarks[:, 2] / np.float32(input_size_f) / np.float32(0.4)).astype(np.float32)

    angle = _f32(rect.rotation)
    rect_width = _f32(rect.width)
    rect_height = _f32(rect.height)
    rect_x_center = _f32(rect.x_center)
    rect_y_center = _f32(rect.y_center)
    output = landmarks.copy()
    for index in range(len(output)):
        x = _f32(_f32(landmarks[index, 0]) - _f32(0.5))
        y = _f32(_f32(landmarks[index, 1]) - _f32(0.5))
        new_x = _f32(float(math.cos(angle)) * x - float(math.sin(angle)) * y)
        new_y = _f32(float(math.sin(angle)) * x + float(math.cos(angle)) * y)
        # The second assignment in LandmarkProjectionCalculator is all float.
        new_x = _f32(_f32(new_x * rect_width) + rect_x_center)
        new_y = _f32(_f32(new_y * rect_height) + rect_y_center)
        output[index, 0] = new_x
        output[index, 1] = new_y
        if output.shape[1] > 2:
            output[index, 2] = _f32(_f32(landmarks[index, 2]) * rect_width)
    return output
