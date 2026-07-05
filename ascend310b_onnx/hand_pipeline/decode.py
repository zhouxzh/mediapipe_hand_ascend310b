"""MediaPipe palm detector decoding and NMS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_pipeline.preprocess import LetterboxInfo


@dataclass(frozen=True)
class Anchor:
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class PalmDetection:
    box: np.ndarray
    score: float
    keypoints: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def calculate_scale(min_scale: float, max_scale: float, stride_index: int, num_strides: int) -> float:
    if num_strides == 1:
        return (min_scale + max_scale) * 0.5
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def generate_palm_anchors() -> np.ndarray:
    """Generate the 2016 anchors used by MediaPipe PalmDetectionCpu."""
    num_layers = 4
    min_scale = 0.1484375
    max_scale = 0.75
    input_size = 192
    anchor_offset_x = 0.5
    anchor_offset_y = 0.5
    strides = [8, 16, 16, 16]
    aspect_ratios = [1.0]
    interpolated_scale_aspect_ratio = 1.0
    fixed_anchor_size = True

    anchors: list[Anchor] = []
    layer_id = 0
    while layer_id < num_layers:
        anchor_height: list[float] = []
        anchor_width: list[float] = []
        last_same_stride_layer = layer_id
        while last_same_stride_layer < num_layers and strides[last_same_stride_layer] == strides[layer_id]:
            scale = calculate_scale(min_scale, max_scale, last_same_stride_layer, num_layers)
            for aspect_ratio in aspect_ratios:
                ratio_sqrt = float(np.sqrt(aspect_ratio))
                anchor_height.append(scale / ratio_sqrt)
                anchor_width.append(scale * ratio_sqrt)
            if interpolated_scale_aspect_ratio > 0.0:
                scale_next = (
                    1.0
                    if last_same_stride_layer == num_layers - 1
                    else calculate_scale(min_scale, max_scale, last_same_stride_layer + 1, num_layers)
                )
                scale_interp = float(np.sqrt(scale * scale_next))
                ratio_sqrt = float(np.sqrt(interpolated_scale_aspect_ratio))
                anchor_height.append(scale_interp / ratio_sqrt)
                anchor_width.append(scale_interp * ratio_sqrt)
            last_same_stride_layer += 1

        stride = strides[layer_id]
        feature_map_height = int(np.ceil(input_size / stride))
        feature_map_width = int(np.ceil(input_size / stride))
        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + anchor_offset_x) / feature_map_width
                    y_center = (y + anchor_offset_y) / feature_map_height
                    if fixed_anchor_size:
                        width = 1.0
                        height = 1.0
                    else:
                        width = anchor_width[anchor_id]
                        height = anchor_height[anchor_id]
                    anchors.append(Anchor(x_center, y_center, width, height))
        layer_id = last_same_stride_layer

    arr = np.array([[a.x_center, a.y_center, a.width, a.height] for a in anchors], dtype=np.float32)
    if arr.shape != (2016, 4):
        raise ValueError(f"Expected 2016 palm anchors, got {arr.shape}")
    return arr


def remove_letterbox_from_points(points: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    left, top, right, bottom = info.normalized_padding
    x_scale = max(1.0 - left - right, 1e-9)
    y_scale = max(1.0 - top - bottom, 1e-9)
    out = points.copy()
    out[..., 0] = (out[..., 0] - left) / x_scale
    out[..., 1] = (out[..., 1] - top) / y_scale
    return out


def decode_raw_palm(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    anchors: np.ndarray,
    letterbox: LetterboxInfo,
    score_threshold: float = 0.5,
) -> list[PalmDetection]:
    boxes = np.asarray(raw_boxes, dtype=np.float32)
    scores = np.asarray(raw_scores, dtype=np.float32)
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 3:
        scores = scores[0]
    scores = scores.reshape(-1)
    if boxes.shape != (2016, 18):
        raise ValueError(f"Expected raw boxes shape (2016, 18), got {boxes.shape}")

    clipped_scores = np.clip(scores, -100.0, 100.0)
    probs = sigmoid(clipped_scores)
    keep = probs >= score_threshold
    if not np.any(keep):
        return []

    boxes = boxes[keep]
    probs = probs[keep]
    anchors = anchors[keep]
    x_scale = y_scale = w_scale = h_scale = 192.0

    # MediaPipe maps reverse_output_order=true to XYWH, including keypoints.
    x_center = boxes[:, 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = boxes[:, 1] / y_scale * anchors[:, 3] + anchors[:, 1]
    w = boxes[:, 2] / w_scale * anchors[:, 2]
    h = boxes[:, 3] / h_scale * anchors[:, 3]
    decoded_boxes = np.stack(
        [x_center - w * 0.5, y_center - h * 0.5, x_center + w * 0.5, y_center + h * 0.5],
        axis=1,
    )

    keypoints = np.zeros((boxes.shape[0], 7, 2), dtype=np.float32)
    for keypoint_id in range(7):
        offset = 4 + keypoint_id * 2
        keypoints[:, keypoint_id, 0] = boxes[:, offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoints[:, keypoint_id, 1] = boxes[:, offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    corners = decoded_boxes.reshape(-1, 2, 2)
    corners = remove_letterbox_from_points(corners, letterbox).reshape(-1, 4)
    keypoints = remove_letterbox_from_points(keypoints, letterbox)

    corners[:, [0, 2]] *= letterbox.orig_width
    corners[:, [1, 3]] *= letterbox.orig_height
    keypoints[:, :, 0] *= letterbox.orig_width
    keypoints[:, :, 1] *= letterbox.orig_height

    corners[:, [0, 2]] = np.clip(corners[:, [0, 2]], 0, letterbox.orig_width)
    corners[:, [1, 3]] = np.clip(corners[:, [1, 3]], 0, letterbox.orig_height)
    keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, letterbox.orig_width)
    keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, letterbox.orig_height)

    return [
        PalmDetection(box=corners[i], score=float(probs[i]), keypoints=keypoints[i])
        for i in range(corners.shape[0])
    ]


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def weighted_nms(detections: list[PalmDetection], iou_threshold: float = 0.3, max_detections: int = 20) -> list[PalmDetection]:
    if not detections:
        return []
    boxes = np.stack([d.box for d in detections], axis=0).astype(np.float32)
    scores = np.array([d.score for d in detections], dtype=np.float32)
    keypoints = np.stack([d.keypoints for d in detections], axis=0).astype(np.float32)
    remaining = scores.argsort()[::-1]
    results: list[PalmDetection] = []

    while remaining.size > 0 and len(results) < max_detections:
        current = int(remaining[0])
        if remaining.size == 1:
            overlap_ids = np.array([current], dtype=np.int64)
            remaining = remaining[1:]
        else:
            ious = box_iou(boxes[current], boxes[remaining])
            overlap_mask = ious > iou_threshold
            overlap_ids = remaining[overlap_mask]
            if overlap_ids.size == 0:
                overlap_ids = np.array([current], dtype=np.int64)
                overlap_mask = remaining == current
            remaining = remaining[~overlap_mask]

        weights = scores[overlap_ids]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            merged_box = boxes[current]
            merged_keypoints = keypoints[current]
        else:
            merged_box = np.sum(boxes[overlap_ids] * weights[:, None], axis=0) / weight_sum
            merged_keypoints = np.sum(keypoints[overlap_ids] * weights[:, None, None], axis=0) / weight_sum
        results.append(
            PalmDetection(
                box=merged_box.astype(np.float32),
                score=float(np.max(scores[overlap_ids])),
                keypoints=merged_keypoints.astype(np.float32),
            )
        )
    return results

