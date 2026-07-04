"""Realtime drawing helpers for MediaPipe hand predictions."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def _point_xy(point: Any) -> tuple[int, int]:
    arr = np.asarray(point, dtype=np.float32)
    return int(round(float(arr[0]))), int(round(float(arr[1])))


def draw_hand_predictions(
    image_bgr: np.ndarray,
    predictions: list[dict[str, Any]],
    *,
    show_palm: bool = True,
    show_landmarks: bool = True,
) -> np.ndarray:
    canvas = image_bgr.copy()
    for index, pred in enumerate(predictions):
        color = (32, 210, 120) if index == 0 else (255, 170, 40)
        palm_color = (60, 180, 255)
        if show_palm and pred.get("box") is not None:
            box = np.asarray(pred["box"], dtype=np.float32)
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), palm_color, 2)
            if pred.get("palm7") is not None:
                for point in np.asarray(pred["palm7"], dtype=np.float32):
                    cv2.circle(canvas, _point_xy(point), 3, palm_color, -1)

        if show_landmarks and pred.get("hand21") is not None:
            points = np.asarray(pred["hand21"], dtype=np.float32)
            for a, b in HAND_EDGES:
                cv2.line(canvas, _point_xy(points[a]), _point_xy(points[b]), color, 2)
            for point in points:
                cv2.circle(canvas, _point_xy(point), 3, (245, 245, 245), -1)
                cv2.circle(canvas, _point_xy(point), 3, color, 1)

        label = f"hand {index + 1}"
        score = pred.get("score")
        hand_score = pred.get("hand_score")
        if isinstance(score, (float, int)):
            label += f" palm={float(score):.2f}"
        if isinstance(hand_score, (float, int)) and np.isfinite(hand_score):
            label += f" lm={float(hand_score):.2f}"
        origin = _point_xy(np.asarray(pred.get("box", [12, 30, 0, 0]), dtype=np.float32)[:2])
        cv2.putText(
            canvas,
            label,
            (max(origin[0], 8), max(origin[1] - 8, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_status_overlay(
    image_bgr: np.ndarray,
    *,
    capture_fps: float = 0.0,
    infer_fps: float = 0.0,
    infer_ms: float = 0.0,
    hands: int = 0,
    backend: str = "",
) -> np.ndarray:
    canvas = image_bgr
    lines = [
        f"hands {hands}",
        f"infer {infer_ms:.1f} ms / {infer_fps:.1f} fps",
        f"capture {capture_fps:.1f} fps",
    ]
    if backend:
        lines.append(backend)
    x, y = 12, 24
    for line in lines:
        (width, height), _baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        cv2.rectangle(canvas, (x - 5, y - height - 6), (x + width + 5, y + 6), (18, 24, 32), -1)
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 245, 240), 2, cv2.LINE_AA)
        y += 25
    return canvas
