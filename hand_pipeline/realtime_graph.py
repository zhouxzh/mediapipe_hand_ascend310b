"""Realtime MediaPipe Hand graph adapter for WebRTC.

The portable ``HandTracker`` already implements the fixed MediaPipe hand
tracking graph order.  This module gives the WebRTC path an explicit packet and
stream boundary without introducing a generic MediaPipe scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass
class RealtimeFramePacket:
    frame_index: int
    timestamp: float
    captured_at: float
    image_bgr: np.ndarray | None = None
    image_nv12: np.ndarray | None = None
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class RealtimeGraphResult:
    frame_index: int
    timestamp: float
    captured_at: float
    completed_at: float
    predictions: list[dict[str, Any]]
    timings: dict[str, Any]
    debug: dict[str, Any] | None
    streams: dict[str, Any]


class MediaPipeHandRealtimeGraph:
    """Fixed hand tracking graph executor used by the realtime WebRTC path."""

    STREAM_NAMES = (
        "palm_detections",
        "hand_rects_from_palm_detections",
        "gated_prev_hand_rects_from_landmarks",
        "hand_rects",
        "multi_hand_landmarks",
        "hand_rects_from_landmarks",
    )

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.last_streams: dict[str, Any] = {name: [] for name in self.STREAM_NAMES}

    def process(self, packet: RealtimeFramePacket) -> RealtimeGraphResult:
        if (
            packet.image_nv12 is not None
            and hasattr(self.pipeline, "infer_nv12")
            and getattr(self.pipeline, "mode", None) == "tracking"
        ):
            width = int(packet.image_width)
            height = int(packet.image_height)
            result = self.pipeline.infer_nv12(packet.image_nv12, width, height)
        else:
            if packet.image_bgr is None:
                raise ValueError("BGR image is required when NV12 tracking inference is unavailable")
            result = self.pipeline.infer(packet.image_bgr)
        streams = self._streams_from_result(result.predictions, result.debug)
        self.last_streams = streams
        return RealtimeGraphResult(
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
            captured_at=packet.captured_at,
            completed_at=time.perf_counter(),
            predictions=result.predictions,
            timings=result.timings,
            debug=result.debug,
            streams=streams,
        )

    def _streams_from_result(
        self,
        predictions: list[dict[str, Any]],
        debug: dict[str, Any] | None,
    ) -> dict[str, Any]:
        debug = debug or {}
        return {
            "palm_detections": debug.get("palm_detections", []),
            "hand_rects_from_palm_detections": debug.get("palm_rois", []),
            "gated_prev_hand_rects_from_landmarks": debug.get("previous_tracking_rois", []),
            "hand_rects": debug.get("associated_hand_rois", []),
            "multi_hand_landmarks": [item.get("hand21") for item in predictions],
            "hand_rects_from_landmarks": debug.get("next_tracking_rois", []),
        }
