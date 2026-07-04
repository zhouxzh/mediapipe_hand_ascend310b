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


def normalize_radians(angle: float) -> float:
    return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))


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


def crop_from_normalized_rect(
    image: np.ndarray,
    rect: dict[str, float],
    input_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([float(rect["x_center_px"]), float(rect["y_center_px"])], dtype=np.float32)
    width = float(rect["width_px"])
    height = float(rect["height_px"])
    rotation = float(rect["rotation"])
    x_axis, y_axis = local_axes(rotation)
    src = np.array(
        [
            center - 0.5 * width * x_axis - 0.5 * height * y_axis,
            center + 0.5 * width * x_axis - 0.5 * height * y_axis,
            center - 0.5 * width * x_axis + 0.5 * height * y_axis,
        ],
        dtype=np.float32,
    )
    dst_extent = float(input_size)
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
    return crop, inverse


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
