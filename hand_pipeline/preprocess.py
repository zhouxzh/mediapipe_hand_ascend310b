"""Preprocessing for MediaPipe hand models."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    input_size: int
    orig_width: int
    orig_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    normalized_padding_values: tuple[float, float, float, float] | None = None

    @property
    def normalized_padding(self) -> tuple[float, float, float, float]:
        if self.normalized_padding_values is not None:
            return self.normalized_padding_values
        size = float(self.input_size)
        return (
            self.pad_left / size,
            self.pad_top / size,
            self.pad_right / size,
            self.pad_bottom / size,
        )


def _padded_full_image_roi(
    orig_width: int,
    orig_height: int,
    input_width: int,
    input_height: int,
    keep_aspect_ratio: bool,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return MediaPipe ImageToTensor full-image ROI and letterbox padding."""
    center_x = 0.5 * float(orig_width)
    center_y = 0.5 * float(orig_height)
    roi_width = float(orig_width)
    roi_height = float(orig_height)
    if not keep_aspect_ratio:
        return (center_x, center_y, roi_width, roi_height), (0.0, 0.0, 0.0, 0.0)

    tensor_aspect_ratio = float(input_height) / float(input_width)
    roi_aspect_ratio = roi_height / roi_width
    horizontal_padding = 0.0
    vertical_padding = 0.0
    if tensor_aspect_ratio > roi_aspect_ratio:
        roi_height = roi_width * tensor_aspect_ratio
        vertical_padding = (1.0 - roi_aspect_ratio / tensor_aspect_ratio) * 0.5
    else:
        roi_width = roi_height / tensor_aspect_ratio
        horizontal_padding = (1.0 - tensor_aspect_ratio / roi_aspect_ratio) * 0.5
    return (
        center_x,
        center_y,
        roi_width,
        roi_height,
    ), (
        horizontal_padding,
        vertical_padding,
        horizontal_padding,
        vertical_padding,
    )


def image_to_tensor(
    image_bgr: np.ndarray,
    input_size: int = 192,
    keep_aspect_ratio: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Convert BGR image to MediaPipe ImageToTensor-style NHWC float32 tensor.

    PalmDetectionCpu uses ImageToTensorCalculator. On CPU that path pads the
    full-image ROI in continuous coordinates and samples it with
    cv::warpPerspective, not with a discrete resize-then-pad operation.
    """
    orig_height, orig_width = image_bgr.shape[:2]
    if keep_aspect_ratio:
        scale = min(input_size / orig_width, input_size / orig_height)
        resized_width = int(np.ceil(orig_width * scale))
        resized_height = int(np.ceil(orig_height * scale))
    else:
        resized_width = input_size
        resized_height = input_size

    pad_left = (input_size - resized_width) // 2
    pad_top = (input_size - resized_height) // 2
    pad_right = input_size - resized_width - pad_left
    pad_bottom = input_size - resized_height - pad_top
    (center_x, center_y, roi_width, roi_height), normalized_padding = _padded_full_image_roi(
        orig_width,
        orig_height,
        input_size,
        input_size,
        keep_aspect_ratio,
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    src_points = cv2.boxPoints(((center_x, center_y), (roi_width, roi_height), 0.0)).astype(np.float32)
    dst_points = np.array(
        [
            [0.0, float(input_size)],
            [0.0, 0.0],
            [float(input_size), 0.0],
            [float(input_size), float(input_size)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    tensor_image = cv2.warpPerspective(
        rgb,
        matrix,
        (input_size, input_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    tensor = tensor_image.astype(np.float32) / 255.0
    info = LetterboxInfo(
        input_size=input_size,
        orig_width=orig_width,
        orig_height=orig_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        normalized_padding_values=normalized_padding,
    )
    return np.ascontiguousarray(tensor[None]), info


def _nv12_planes(nv12: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if nv12.ndim != 2:
        raise ValueError(f"NV12 frame must be 2D, got shape={nv12.shape}")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"NV12 width/height must be positive even values, got {width}x{height}")
    if nv12.shape[0] < height + height // 2 or nv12.shape[1] < width:
        raise ValueError(f"NV12 frame shape {nv12.shape} is too small for {width}x{height}")
    y = nv12[:height, :width]
    uv = nv12[height : height + height // 2, :width].reshape(height // 2, width // 2, 2)
    return y, uv


def nv12_to_rgb_crop(
    nv12: np.ndarray,
    width: int,
    height: int,
    projection_matrix: np.ndarray,
    output_size: int,
    *,
    border_mode: int = cv2.BORDER_CONSTANT,
) -> np.ndarray:
    """Warp an NV12 frame directly into a small RGB crop."""
    output_size = int(output_size)
    if output_size <= 0 or output_size % 2:
        raise ValueError(f"NV12 crop output_size must be a positive even value, got {output_size}")

    y_plane, uv_plane = _nv12_planes(nv12, width, height)
    matrix = np.asarray(projection_matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"projection_matrix must have shape (3, 3), got {matrix.shape}")

    y_crop = cv2.warpPerspective(
        y_plane,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0,
    )

    src_uv_to_src_y = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    dst_y_to_dst_uv = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    uv_matrix = dst_y_to_dst_uv @ matrix @ src_uv_to_src_y
    uv_crop = cv2.warpPerspective(
        uv_plane,
        uv_matrix,
        (output_size // 2, output_size // 2),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=(128, 128),
    )

    crop_nv12 = np.empty((output_size * 3 // 2, output_size), dtype=np.uint8)
    crop_nv12[:output_size, :] = y_crop
    crop_nv12[output_size:, :] = uv_crop.reshape(output_size // 2, output_size)
    return cv2.cvtColor(crop_nv12, cv2.COLOR_YUV2RGB_NV12)


def nv12_image_to_tensor(
    nv12: np.ndarray,
    width: int,
    height: int,
    input_size: int = 192,
    keep_aspect_ratio: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Convert NV12 frame to MediaPipe ImageToTensor-style NHWC float32 tensor."""
    orig_width = int(width)
    orig_height = int(height)
    if keep_aspect_ratio:
        scale = min(input_size / orig_width, input_size / orig_height)
        resized_width = int(np.ceil(orig_width * scale))
        resized_height = int(np.ceil(orig_height * scale))
    else:
        resized_width = input_size
        resized_height = input_size

    pad_left = (input_size - resized_width) // 2
    pad_top = (input_size - resized_height) // 2
    pad_right = input_size - resized_width - pad_left
    pad_bottom = input_size - resized_height - pad_top
    (center_x, center_y, roi_width, roi_height), normalized_padding = _padded_full_image_roi(
        orig_width,
        orig_height,
        input_size,
        input_size,
        keep_aspect_ratio,
    )
    src_points = cv2.boxPoints(((center_x, center_y), (roi_width, roi_height), 0.0)).astype(np.float32)
    dst_points = np.array(
        [
            [0.0, float(input_size)],
            [0.0, 0.0],
            [float(input_size), 0.0],
            [float(input_size), float(input_size)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    tensor_image = nv12_to_rgb_crop(
        nv12,
        orig_width,
        orig_height,
        matrix,
        input_size,
        border_mode=cv2.BORDER_CONSTANT,
    )
    tensor = tensor_image.astype(np.float32) / 255.0
    info = LetterboxInfo(
        input_size=input_size,
        orig_width=orig_width,
        orig_height=orig_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        normalized_padding_values=normalized_padding,
    )
    return np.ascontiguousarray(tensor[None]), info
