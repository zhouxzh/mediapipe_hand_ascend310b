#!/usr/bin/env python3
"""WebRTC H.264 realtime MediaPipe hand OM demo for Ascend 310B."""

from __future__ import annotations

import argparse
import asyncio
import fractions
import logging
import math
import os
import queue
import socket
import sys
import time
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack
from aiortc import RTCPeerConnection
from aiortc import RTCRtpSender
from aiortc import RTCSessionDescription
from aiortc.mediastreams import MediaStreamError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import weighted_nms
from hand_pipeline.om_runtime import PersistentAclModel
from hand_pipeline.om_runtime import PersistentAclRuntime
from hand_pipeline.preprocess import image_to_tensor
from hand_pipeline.roi import landmarks_to_original
from hand_pipeline.roi import make_hand_roi
from hand_pipeline.roi import preprocess_landmark_tflite
from hand_pipeline.visualization import draw_hand_predictions
from hand_pipeline.visualization import draw_status_overlay

try:
    from webrtc_app.cann_encoder import (
        CannH264Encoder,
        _try_import_cann,
        collect_venc_diagnostics,
        probe_cann_venc,
        set_encoder_status_callback,
        set_session_encoder_mode,
        set_session_bitrate_override_kbps,
    )
    from webrtc_app.dvpp_jpegd import DvppJpegDecoder
    from webrtc_app.v4l2_capture import V4l2MjpegCapture
    from webrtc_app.v4l2_raw import V4l2RawCapture
except ImportError:
    CannH264Encoder = None  # type: ignore[assignment]
    probe_cann_venc = None  # type: ignore[assignment]
    DvppJpegDecoder = None  # type: ignore[assignment]
    V4l2MjpegCapture = None  # type: ignore[assignment]
    V4l2RawCapture = None  # type: ignore[assignment]
    _try_import_cann = None  # type: ignore[assignment]
    collect_venc_diagnostics = None  # type: ignore[assignment]
    set_encoder_status_callback = None  # type: ignore[assignment]
    set_session_encoder_mode = None  # type: ignore[assignment]
    set_session_bitrate_override_kbps = None  # type: ignore[assignment]


WEB_DIR = ROOT / "web"
MODELS_DIR = ROOT / "models" / "om"
LOG_DIR = ROOT / "logs"
DEFAULT_DETECTOR = "mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om"
DEFAULT_LANDMARK = "mediapipe_legacy_0_10_14_hand_landmark_full.om"
DEFAULT_H264_BITRATE_KBPS = 4000
DEFAULT_INFER_EVERY_N = 1
DEFAULT_CANN_VENC_RETRY_SECONDS = 300
CAMERA_BACKEND_OPENCV = "opencv"
CAMERA_BACKEND_DVPP = "dvpp"
VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)

pcs: set[RTCPeerConnection] = set()
pc_tracks: dict[RTCPeerConnection, "HandOmVideoTrack"] = {}
latest_track_stats: dict[str, object] = {}
latest_stats_lock = threading.Lock()
app_logger = logging.getLogger("webrtc_hand_om")
encoder_state = {
    "mode": "cpu",
    "hardware_requested": False,
    "hardware_active": False,
    "name": "cpu-libx264-stable",
    "last_error": "",
    "cann_blocked_until": 0.0,
    "cann_blocked_reason": "",
}


def no_store_file_response(path: Path) -> web.FileResponse:
    response = web.FileResponse(path)
    response.headers["Cache-Control"] = "no-store"
    return response


def patch_h264_encoder() -> bool:
    encoder_state["hardware_active"] = False
    encoder_state["name"] = "cpu-libx264-stable"
    encoder_state["last_error"] = ""

    if CannH264Encoder is None:
        app_logger.warning("WebRTC H.264 patch encoder is unavailable; using aiortc default H.264 encoder")
        return False
    import aiortc.codecs as codecs_module
    import aiortc.codecs.h264 as h264_module
    import aiortc.rtcrtpsender as rtcrtpsender_module

    original_get_encoder = codecs_module.get_encoder

    def get_encoder(codec):
        if codec.mimeType.lower() == "video/h264":
            return CannH264Encoder()
        return original_get_encoder(codec)

    def update_encoder_status(name: str, hardware_active: bool, reason: str = "") -> None:
        encoder_state["name"] = name
        encoder_state["hardware_active"] = bool(hardware_active)
        encoder_state["last_error"] = reason

    if set_encoder_status_callback is not None:
        set_encoder_status_callback(update_encoder_status)

    h264_module.H264Encoder = CannH264Encoder
    codecs_module.H264Encoder = CannH264Encoder
    codecs_module.get_encoder = get_encoder
    rtcrtpsender_module.get_encoder = get_encoder

    app_logger.info("WebRTC H.264 encoder patched; runtime mode is selected per offer")
    return True


def set_active_encoder_mode(mode: str) -> None:
    mode = str(mode).lower()
    if mode not in {"cpu", "cann"}:
        raise ValueError(f"Unsupported encoder mode: {mode}")
    encoder_state["mode"] = mode
    encoder_state["hardware_requested"] = mode == "cann"
    encoder_state["hardware_active"] = mode == "cann"
    encoder_state["name"] = "cann-venc-h264" if mode == "cann" else "cpu-libx264-stable"
    encoder_state["last_error"] = ""
    if set_session_encoder_mode is not None:
        set_session_encoder_mode(mode)


def cann_venc_blocked_seconds() -> int:
    blocked_until = float(encoder_state.get("cann_blocked_until", 0.0) or 0.0)
    return max(0, int(math.ceil(blocked_until - time.monotonic())))


def clear_cann_venc_block() -> None:
    encoder_state["cann_blocked_until"] = 0.0
    encoder_state["cann_blocked_reason"] = ""


def raise_if_cann_venc_blocked() -> None:
    remaining_seconds = cann_venc_blocked_seconds()
    if remaining_seconds <= 0:
        clear_cann_venc_block()
        return
    reason = str(encoder_state.get("cann_blocked_reason") or "previous CANN VENC create failure")
    raise RuntimeError(
        "CANN VENC is temporarily disabled after a previous create failure "
        f"to avoid repeated driver-side memory pressure. Retry in {remaining_seconds}s "
        f"or restart the service after cleaning/rebooting the board. Previous failure: {reason}"
    )


def block_cann_venc(reason: str, retry_seconds: int) -> None:
    retry_seconds = max(0, int(retry_seconds))
    if retry_seconds <= 0:
        clear_cann_venc_block()
        return
    encoder_state["cann_blocked_until"] = time.monotonic() + retry_seconds
    encoder_state["cann_blocked_reason"] = reason


def set_offer_bitrate_override(bitrate_kbps: int | None) -> None:
    if set_session_bitrate_override_kbps is not None:
        set_session_bitrate_override_kbps(bitrate_kbps)


def source_to_device_path(source: str) -> str:
    source_text = str(source)
    if source_text.isdigit():
        return f"/dev/video{int(source_text)}"
    return source_text


def nv12_to_bgr(nv12: np.ndarray, width: int, height: int) -> np.ndarray:
    if nv12.ndim != 2:
        raise ValueError(f"NV12 frame must be 2D, got shape={nv12.shape}")
    tight_nv12 = np.empty((height * 3 // 2, width), dtype=np.uint8)
    tight_nv12[:height, :] = nv12[:height, :width]
    tight_nv12[height:, :] = nv12[height : height + height // 2, :width]
    return cv2.cvtColor(tight_nv12, cv2.COLOR_YUV2BGR_NV12)


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    if height % 2 or width % 2:
        bgr = bgr[: height - (height % 2), : width - (width % 2)]
        height, width = bgr.shape[:2]
    yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    yuv_flat = yuv_i420.reshape(-1)
    y_size = height * width
    uv_size = y_size // 4
    y = yuv_flat[:y_size].reshape(height, width)
    u = yuv_flat[y_size : y_size + uv_size].reshape(height // 2, width // 2)
    v = yuv_flat[y_size + uv_size : y_size + uv_size * 2].reshape(height // 2, width // 2)
    nv12 = np.empty((height * 3 // 2, width), dtype=np.uint8)
    nv12[:height, :] = y
    uv = nv12[height:, :].reshape(height // 2, width // 2, 2)
    uv[:, :, 0] = u
    uv[:, :, 1] = v
    return nv12


def decode_fourcc(value: float | int) -> str:
    code = int(value or 0)
    if code <= 0:
        return ""
    chars: list[str] = []
    for shift in (0, 8, 16, 24):
        char = chr((code >> shift) & 0xFF)
        if char.isprintable():
            chars.append(char)
    return "".join(chars)


def pick_landmark_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    landmarks = None
    world = None
    one_value: list[np.ndarray] = []
    for value in outputs:
        arr = np.asarray(value)
        if arr.size == 63 and landmarks is None:
            landmarks = arr.reshape(21, 3)
        elif arr.size == 63:
            world = arr.reshape(21, 3)
        elif arr.size == 1:
            one_value.append(arr.reshape(-1))
    if landmarks is None:
        raise ValueError(f"Could not find 63-value landmark output: {[x.shape for x in outputs]}")
    hand_score = float(one_value[0][0]) if len(one_value) >= 1 else math.nan
    handedness = float(one_value[1][0]) if len(one_value) >= 2 else math.nan
    return landmarks.astype(np.float32), hand_score, handedness, None if world is None else world.astype(np.float32)


async def index(_: web.Request) -> web.FileResponse:
    return no_store_file_response(WEB_DIR / "webrtc_index.html")


async def client_js(_: web.Request) -> web.FileResponse:
    return no_store_file_response(WEB_DIR / "webrtc_client.js")


async def styles_css(_: web.Request) -> web.FileResponse:
    return no_store_file_response(WEB_DIR / "webrtc_styles.css")


def _model_role(path: Path) -> str:
    name = path.name.lower()
    if "landmark" in name:
        return "landmark"
    if "palm" in name or "detector" in name:
        return "detector"
    return "other"


def list_om_models() -> dict[str, list[dict[str, object]]]:
    items = {"detectors": [], "landmarks": [], "other": []}
    for path in sorted(MODELS_DIR.glob("*.om")):
        role = _model_role(path)
        if role == "landmark":
            input_text = "224x224"
        elif role == "detector":
            input_text = "192x192"
        else:
            input_text = "unknown"
        item = {"name": path.name, "role": role, "input": input_text, "size_bytes": path.stat().st_size}
        if role == "detector":
            items["detectors"].append(item)
        elif role == "landmark":
            items["landmarks"].append(item)
        else:
            items["other"].append(item)
    return items


def resolve_model_path(model_value: str | os.PathLike[str]) -> Path:
    model_path = Path(str(model_value))
    if model_path.is_absolute():
        return model_path
    if model_path.parent == Path("."):
        return MODELS_DIR / model_path
    if model_path.exists():
        return model_path
    return ROOT / model_path


def default_model_name(configured: str | os.PathLike[str] | None, default_name: str, role: str) -> str:
    if configured:
        configured_path = resolve_model_path(configured)
        if configured_path.exists():
            return configured_path.name
    default_path = MODELS_DIR / default_name
    if default_path.exists():
        return default_path.name
    models = list_om_models()
    key = "detectors" if role == "detector" else "landmarks"
    if models[key]:
        return str(models[key][0]["name"])
    return default_name


async def health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "runtime_target": "ascend-310b",
            "transport": "webrtc",
            "video_codec": "h264",
            "encoder": encoder_state["name"],
            "encoder_mode": encoder_state["mode"],
            "hardware_encode": bool(encoder_state["hardware_active"]),
            "hardware_encode_requested": bool(encoder_state["hardware_requested"]),
            "encoder_last_error": encoder_state["last_error"],
            "cann_venc_blocked_seconds": cann_venc_blocked_seconds(),
            "cann_venc_blocked_reason": encoder_state["cann_blocked_reason"],
            "default_detector": request.config_dict.get("default_detector", DEFAULT_DETECTOR),
            "default_landmark": request.config_dict.get("default_landmark", DEFAULT_LANDMARK),
            "default_source": request.config_dict.get("default_source", "/dev/video0"),
            "default_device_id": request.config_dict.get("device_id", 0),
            "defaults": {
                "width": request.config_dict.get("camera_width", 1280),
                "height": request.config_dict.get("camera_height", 720),
                "fps": request.config_dict.get("camera_fps", 30),
                "infer_every_n": request.config_dict.get("infer_every_n", DEFAULT_INFER_EVERY_N),
                "score_threshold": request.config_dict.get("score_threshold", 0.5),
                "nms_iou": request.config_dict.get("nms_iou", 0.3),
                "max_hands": request.config_dict.get("max_hands", 2),
                "min_hand_score": request.config_dict.get("min_hand_score", 0.5),
                "bitrate_kbps": request.config_dict.get("bitrate_kbps", DEFAULT_H264_BITRATE_KBPS),
                "encoder_mode": request.config_dict.get("encoder_mode", "cpu"),
                "cann_venc_retry_seconds": request.config_dict.get("cann_venc_retry_seconds", DEFAULT_CANN_VENC_RETRY_SECONDS),
                "camera_backend": request.config_dict.get("camera_backend", CAMERA_BACKEND_OPENCV),
                "camera_fourcc": request.config_dict.get("camera_fourcc", "MJPG"),
            },
        }
    )


async def models(request: web.Request) -> web.Response:
    items = list_om_models()
    return web.json_response(
        {
            **items,
            "default_detector": request.config_dict.get("default_detector", default_model_name(None, DEFAULT_DETECTOR, "detector")),
            "default_landmark": request.config_dict.get("default_landmark", default_model_name(None, DEFAULT_LANDMARK, "landmark")),
        }
    )


async def stats(_: web.Request) -> web.Response:
    with latest_stats_lock:
        return web.json_response(dict(latest_track_stats))


def parse_positive_int(value: object, name: str, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text=f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise web.HTTPBadRequest(text=f"{name} must be positive.")
    return parsed


def parse_non_negative_int(value: object, name: str, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text=f"{name} must be an integer.") from exc
    if parsed < 0:
        raise web.HTTPBadRequest(text=f"{name} must be non-negative.")
    return parsed


def parse_float_range(value: object, name: str, default: float, lower: float, upper: float) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text=f"{name} must be a number.") from exc
    if parsed < lower or parsed > upper:
        raise web.HTTPBadRequest(text=f"{name} must be in [{lower}, {upper}].")
    return parsed


def parse_offer_payload(params: dict[str, object], default_device_id: int = 0) -> dict[str, object]:
    try:
        offer = RTCSessionDescription(sdp=str(params["sdp"]), type=str(params["type"]))
    except KeyError as exc:
        raise web.HTTPBadRequest(text=f"Missing WebRTC offer field: {exc}") from exc

    width = parse_positive_int(params.get("width"), "width", 1280)
    height = parse_positive_int(params.get("height"), "height", 720)
    fps = parse_positive_int(params.get("fps"), "fps", 30)
    infer_every_n = parse_positive_int(params.get("infer_every_n"), "infer_every_n", DEFAULT_INFER_EVERY_N)
    device_id = parse_non_negative_int(params.get("device_id"), "device_id", default_device_id)
    score_threshold = parse_float_range(params.get("score_threshold"), "score_threshold", 0.5, 0.01, 0.99)
    nms_iou = parse_float_range(params.get("nms_iou"), "nms_iou", 0.3, 0.01, 0.99)
    min_hand_score = parse_float_range(params.get("min_hand_score"), "min_hand_score", 0.5, 0.0, 1.0)
    max_hands = parse_positive_int(params.get("max_hands"), "max_hands", 2)

    bitrate_kbps = params.get("bitrate_kbps")
    if bitrate_kbps in (None, "", 0, "0"):
        bitrate_kbps = None
    else:
        bitrate_kbps = parse_positive_int(bitrate_kbps, "bitrate_kbps", DEFAULT_H264_BITRATE_KBPS)

    camera_backend = str(params.get("camera_backend") or CAMERA_BACKEND_OPENCV).lower()
    if camera_backend not in {CAMERA_BACKEND_OPENCV, CAMERA_BACKEND_DVPP}:
        raise web.HTTPBadRequest(text="camera_backend must be opencv or dvpp.")
    camera_fourcc = str(params.get("camera_fourcc") or "MJPG").upper()
    if camera_fourcc not in {"MJPG", "YUYV", "DEFAULT"}:
        raise web.HTTPBadRequest(text="camera_fourcc must be MJPG, YUYV, or DEFAULT.")
    encoder_mode = str(params.get("encoder_mode") or "cpu").lower()
    if encoder_mode not in {"cpu", "cann"}:
        raise web.HTTPBadRequest(text="encoder_mode must be cpu or cann.")

    return {
        "offer": offer,
        "detector_name": str(params.get("detector") or DEFAULT_DETECTOR),
        "landmark_name": str(params.get("landmark") or DEFAULT_LANDMARK),
        "source": str(params.get("source") or "/dev/video0"),
        "width": width,
        "height": height,
        "fps": fps,
        "bitrate_kbps": bitrate_kbps,
        "encoder_mode": encoder_mode,
        "infer_every_n": infer_every_n,
        "score_threshold": score_threshold,
        "nms_iou": nms_iou,
        "max_hands": max_hands,
        "min_hand_score": min_hand_score,
        "device_id": device_id,
        "camera_backend": camera_backend,
        "camera_fourcc": camera_fourcc,
    }


def _offer_has_h264(sdp: str) -> bool:
    return any(
        line.startswith("a=rtpmap:") and line.strip().split(None, 1)[-1].split("/", 1)[0].lower() == "h264"
        for line in sdp.splitlines()
    )


def _local_h264_codecs():
    return [codec for codec in RTCRtpSender.getCapabilities("video").codecs if codec.mimeType.lower() == "video/h264"]


def _prefer_h264_for_sender(pc: RTCPeerConnection, sender: RTCRtpSender) -> None:
    codecs = _local_h264_codecs()
    if not codecs:
        raise web.HTTPBadRequest(text="No local video/H264 codec capability found.")
    for transceiver in pc.getTransceivers():
        if transceiver.sender == sender:
            transceiver.setCodecPreferences(codecs)
            app_logger.info("Video transceiver codec preference set to video/H264")
            return
    raise web.HTTPInternalServerError(text="Could not find sender transceiver.")


def estimate_h264_bitrate_kbps(width: int, height: int, fps: int) -> int:
    bitrate = round(width * height * fps * 0.04 / 1000)
    return max(500, min(bitrate, 10000))


def set_video_bitrate_in_sdp(sdp: str, bitrate_kbps: int) -> str:
    lines = sdp.splitlines()
    output: list[str] = []
    in_video = False
    inserted = False
    for line in lines:
        if line.startswith("m="):
            if in_video and not inserted:
                output.append(f"b=AS:{bitrate_kbps}")
            in_video = line.startswith("m=video")
            inserted = False
            output.append(line)
            continue
        if in_video and line.startswith("b=AS:"):
            if not inserted:
                output.append(f"b=AS:{bitrate_kbps}")
                inserted = True
            continue
        output.append(line)
        if in_video and not inserted and line.startswith("c="):
            output.append(f"b=AS:{bitrate_kbps}")
            inserted = True
    if in_video and not inserted:
        output.append(f"b=AS:{bitrate_kbps}")
    return "\r\n".join(output) + "\r\n"


class HandOmPipeline:
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
        reload_detector_each_call: bool = False,
    ) -> None:
        self.detector_path = detector_path
        self.landmark_path = landmark_path
        self.device_id = device_id
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_hands = max_hands
        self.min_hand_score = min_hand_score
        self.reload_detector_each_call = reload_detector_each_call
        self.runtime = PersistentAclRuntime(device_id=device_id, finalize_on_release=False)
        self.detector: PersistentAclModel | None = None
        self.landmark: PersistentAclModel | None = None
        self.anchors = generate_palm_anchors()
        if not self.reload_detector_each_call:
            self.detector = PersistentAclModel(detector_path, runtime=self.runtime)
        self.landmark = PersistentAclModel(landmark_path, runtime=self.runtime)

    def close(self) -> None:
        if self.detector is not None:
            self.detector.release()
            self.detector = None
        if self.landmark is not None:
            self.landmark.release()
            self.landmark = None
        if self.runtime is not None:
            self.runtime.release()

    def _detector_infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self.reload_detector_each_call:
            detector = PersistentAclModel(self.detector_path, runtime=self.runtime)
            try:
                return detector.infer(tensor)
            finally:
                detector.release()
        if self.detector is None:
            self.detector = PersistentAclModel(self.detector_path, runtime=self.runtime)
        return self.detector.infer(tensor)

    def infer(self, image_bgr: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
        total_start = time.perf_counter()
        pre_start = time.perf_counter()
        tensor, letterbox = image_to_tensor(image_bgr, input_size=192)
        preprocess_ms = (time.perf_counter() - pre_start) * 1000.0

        det_start = time.perf_counter()
        raw_boxes, raw_scores = self._detector_infer(tensor)
        detector_ms = (time.perf_counter() - det_start) * 1000.0

        decode_start = time.perf_counter()
        palms = decode_raw_palm(raw_boxes, raw_scores, self.anchors, letterbox, score_threshold=self.score_threshold)
        palms = weighted_nms(palms, iou_threshold=self.nms_iou, max_detections=max(20, self.max_hands))
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        predictions: list[dict[str, Any]] = []
        landmark_ms = 0.0
        post_ms = 0.0
        landmark = self.landmark
        if landmark is None:
            raise RuntimeError("Landmark OM model is not loaded")

        for hand_index, palm in enumerate(sorted(palms, key=lambda item: item.score, reverse=True)[: self.max_hands]):
            roi_start = time.perf_counter()
            roi = make_hand_roi(image_bgr, palm.box, palm.keypoints)
            lm_tensor = preprocess_landmark_tflite(roi.crop)
            roi_ms = (time.perf_counter() - roi_start) * 1000.0

            lm_start = time.perf_counter()
            lm_outputs = landmark.infer(lm_tensor)
            landmark_ms += (time.perf_counter() - lm_start) * 1000.0

            post_start = time.perf_counter()
            lm_crop, hand_score, handedness, _world = pick_landmark_outputs(lm_outputs)
            if not math.isnan(hand_score) and hand_score < self.min_hand_score:
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
                    "roi_ms": roi_ms,
                }
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return predictions, {
            "preprocess_ms": preprocess_ms,
            "detector_ms": detector_ms,
            "decode_ms": decode_ms,
            "landmark_ms": landmark_ms,
            "post_ms": post_ms,
            "total_ms": total_ms,
        }


class HandOmVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        detector_path: Path,
        landmark_path: Path,
        source: str,
        width: int,
        height: int,
        fps: int,
        score_threshold: float,
        nms_iou: float,
        max_hands: int,
        min_hand_score: float,
        infer_every_n: int,
        device_id: int,
        camera_backend: str = CAMERA_BACKEND_OPENCV,
        camera_fourcc: str = "MJPG",
        reload_detector_each_call: bool = False,
    ) -> None:
        super().__init__()
        self.detector_path = detector_path
        self.landmark_path = landmark_path
        self.source = source
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.width = width
        self.height = height
        self.fps = fps
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_hands = max_hands
        self.min_hand_score = min_hand_score
        self.infer_every_n = max(1, infer_every_n)
        self.device_id = device_id
        self.camera_backend = camera_backend
        self.camera_fourcc = camera_fourcc
        self.reload_detector_each_call = reload_detector_each_call
        self.pipeline: HandOmPipeline | None = None
        self.cap: cv2.VideoCapture | None = None
        self.capture_impl = None
        self.jpegd = None
        self._actual_fourcc = ""
        self._start: float | None = None
        self._timestamp = 0
        self._frame_time = 1 / max(self.fps, 1)
        self._frame_index = 0
        self._closed = False
        self._state_lock = threading.Lock()
        self._last_predictions: list[dict[str, Any]] = []
        self._last_infer_ms = 0.0
        self._last_infer_total_ms = 0.0
        self._last_capture_ms = 0.0
        self._last_nv12_ms = 0.0
        self._last_pipeline_ms = 0.0
        self._last_frame_t: float | None = None
        self._capture_fps_start = time.perf_counter()
        self._capture_fps_frames = 0
        self._capture_fps = 0.0
        self._infer_fps_start = time.perf_counter()
        self._infer_fps_frames = 0
        self._infer_fps = 0.0
        self._track_fps_start = time.perf_counter()
        self._track_fps_frames = 0
        self._track_fps = 0.0
        self._capture_error = ""
        self._infer_error = ""
        self._render_error = ""
        self._open()

    def _open(self) -> None:
        if not self.detector_path.exists():
            raise FileNotFoundError(f"Detector OM model not found: {self.detector_path}")
        if not self.landmark_path.exists():
            raise FileNotFoundError(f"Landmark OM model not found: {self.landmark_path}")
        self.pipeline = HandOmPipeline(
            self.detector_path,
            self.landmark_path,
            device_id=self.device_id,
            score_threshold=self.score_threshold,
            nms_iou=self.nms_iou,
            max_hands=self.max_hands,
            min_hand_score=self.min_hand_score,
            reload_detector_each_call=self.reload_detector_each_call,
        )
        if self.camera_backend == CAMERA_BACKEND_DVPP:
            self._open_dvpp_camera()
        else:
            self._open_opencv_camera()
        app_logger.info(
            "Opened WebRTC hand track detector=%s landmark=%s source=%s capture=%sx%s@%s backend=%s fourcc=%s",
            self.detector_path.name,
            self.landmark_path.name,
            self.source,
            self.width,
            self.height,
            self.fps,
            self.camera_backend,
            self._actual_fourcc or self.camera_fourcc,
        )
        self._publish_stats()

    def _open_opencv_camera(self) -> None:
        source: int | str = int(self.source) if str(self.source).isdigit() else self.source
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")
        if self.camera_fourcc != "DEFAULT":
            fourcc = cv2.VideoWriter_fourcc(*self.camera_fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.requested_width)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.requested_height)
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS) or self.requested_fps))
        self._frame_time = 1 / max(self.fps, 1)
        self._actual_fourcc = decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC))

    def _open_dvpp_camera(self) -> None:
        if DvppJpegDecoder is None:
            raise RuntimeError("DVPP JPEGD modules are unavailable.")
        device_path = source_to_device_path(self.source)
        if V4l2RawCapture is not None:
            self.capture_impl = V4l2RawCapture(device=device_path, width=self.requested_width, height=self.requested_height, fps=self.requested_fps)
        elif V4l2MjpegCapture is not None:
            self.capture_impl = V4l2MjpegCapture(device=device_path, width=self.requested_width, height=self.requested_height, fps=self.requested_fps)
        else:
            raise RuntimeError("V4L2 MJPEG capture modules are unavailable.")
        self.capture_impl.start()
        self.jpegd = DvppJpegDecoder()
        self.width = int(self.capture_impl.width)
        self.height = int(self.capture_impl.height)
        self.fps = self.requested_fps
        self._frame_time = 1 / max(self.fps, 1)
        self._actual_fourcc = "MJPG"

    def describe_settings(self, bitrate_kbps: int | None = None) -> dict[str, object]:
        return {
            "model": "mediapipe_hand_two_stage_om",
            "detector": self.detector_path.name,
            "landmark": self.landmark_path.name,
            "source": self.source,
            "model_input": "palm 192x192 + landmark 224x224",
            "encoder": encoder_state["name"],
            "encoder_mode": encoder_state["mode"],
            "applied": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "bitrate_kbps": bitrate_kbps,
                "encoder_mode": encoder_state["mode"],
                "camera_backend": self.camera_backend,
                "camera_fourcc": self.camera_fourcc,
                "actual_fourcc": self._actual_fourcc,
                "infer_every_n": self.infer_every_n,
                "score_threshold": self.score_threshold,
                "nms_iou": self.nms_iou,
                "max_hands": self.max_hands,
            },
        }

    def _publish_stats(self) -> None:
        with latest_stats_lock:
            latest_track_stats.clear()
            latest_track_stats.update(
                {
                    "closed": self._closed,
                    "detector": self.detector_path.name,
                    "landmark": self.landmark_path.name,
                    "source": self.source,
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps,
                    "camera_backend": self.camera_backend,
                    "camera_fourcc": self.camera_fourcc,
                    "actual_fourcc": self._actual_fourcc,
                    "encoder": encoder_state["name"],
                    "encoder_mode": encoder_state["mode"],
                    "hardware_encode": encoder_state["hardware_active"],
                    "infer_every_n": self.infer_every_n,
                    "hands": len(self._last_predictions),
                    "npu_latency_ms": self._last_infer_ms,
                    "infer_total_ms": self._last_infer_total_ms,
                    "capture_ms": self._last_capture_ms,
                    "nv12_ms": self._last_nv12_ms,
                    "pipeline_ms": self._last_pipeline_ms,
                    "capture_fps": self._capture_fps,
                    "infer_fps": self._infer_fps,
                    "track_fps": self._track_fps,
                    "capture_error": self._capture_error,
                    "infer_error": self._infer_error,
                    "render_error": self._render_error,
                }
            )

    def _read_bgr_frame(self) -> np.ndarray:
        capture_start = time.perf_counter()
        try:
            if self.camera_backend == CAMERA_BACKEND_DVPP:
                if self.capture_impl is None or self.jpegd is None:
                    raise RuntimeError("DVPP camera is not open")
                jpeg_bytes = self.capture_impl.read(timeout=2.0)
                nv12_flat = self.jpegd.decode(jpeg_bytes)
                nv12 = nv12_flat.reshape(self.jpegd.nv12_shape)
                frame = nv12_to_bgr(nv12, self.width, self.height)
            else:
                if self.cap is None:
                    raise RuntimeError("OpenCV camera is not open")
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera read returned no frame")
            if frame.shape[0] % 2 or frame.shape[1] % 2:
                frame = frame[: frame.shape[0] - (frame.shape[0] % 2), : frame.shape[1] - (frame.shape[1] % 2)]
            self._capture_error = ""
            return frame
        except queue.Empty as exc:
            self._capture_error = "camera read timeout"
            raise RuntimeError(self._capture_error) from exc
        finally:
            now = time.perf_counter()
            self._last_capture_ms = (now - capture_start) * 1000.0
            self._capture_fps_frames += 1
            elapsed = now - self._capture_fps_start
            if elapsed >= 1.0:
                self._capture_fps = self._capture_fps_frames / elapsed
                self._capture_fps_frames = 0
                self._capture_fps_start = now

    def _read_output_frame(self):
        frame_start = time.perf_counter()
        frame = self._read_bgr_frame()
        predictions = self._last_predictions
        if self._frame_index % self.infer_every_n == 0:
            try:
                if self.pipeline is None:
                    raise RuntimeError("Hand OM pipeline is not open")
                predictions, timing = self.pipeline.infer(frame)
                self._last_predictions = predictions
                self._last_infer_ms = float(timing["detector_ms"] + timing["landmark_ms"])
                self._last_infer_total_ms = float(timing["total_ms"])
                self._infer_error = ""
                now = time.perf_counter()
                self._infer_fps_frames += 1
                elapsed = now - self._infer_fps_start
                if elapsed >= 1.0:
                    self._infer_fps = self._infer_fps_frames / elapsed
                    self._infer_fps_frames = 0
                    self._infer_fps_start = now
            except Exception as exc:
                self._infer_error = str(exc)
                app_logger.exception("Hand OM inference failed")
        try:
            rendered = draw_hand_predictions(frame, predictions)
            rendered = draw_status_overlay(
                rendered,
                capture_fps=self._capture_fps,
                infer_fps=self._infer_fps,
                infer_ms=self._last_infer_total_ms,
                hands=len(predictions),
                backend=f"{self.camera_backend}/{self._actual_fourcc or self.camera_fourcc}",
            )
            nv12_start = time.perf_counter()
            nv12 = bgr_to_nv12(rendered)
            self._last_nv12_ms = (time.perf_counter() - nv12_start) * 1000.0
            self._render_error = ""
        except Exception as exc:
            self._render_error = str(exc)
            raise

        self._frame_index += 1
        now = time.perf_counter()
        self._track_fps_frames += 1
        elapsed = now - self._track_fps_start
        if elapsed >= 1.0:
            self._track_fps = self._track_fps_frames / elapsed
            self._track_fps_frames = 0
            self._track_fps_start = now
        self._last_pipeline_ms = (time.perf_counter() - frame_start) * 1000.0
        self._publish_stats()
        return nv12

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError
        if self._start is None:
            self._start = time.time()
            self._timestamp = 0
        else:
            self._timestamp += int(self._frame_time * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        loop = asyncio.get_running_loop()
        try:
            frame = await loop.run_in_executor(None, self._read_output_frame)
        except RuntimeError as exc:
            raise MediaStreamError(str(exc)) from exc
        video_frame = av.VideoFrame.from_ndarray(frame, format="nv12")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def _cleanup(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._publish_stats()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.capture_impl is not None:
            self.capture_impl.stop()
            self.capture_impl = None
        if self.jpegd is not None:
            self.jpegd.destroy()
            self.jpegd = None
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None

    def stop(self) -> None:
        app_logger.info("Stopping WebRTC hand track detector=%s landmark=%s", self.detector_path.name, self.landmark_path.name)
        self._cleanup()
        try:
            super().stop()
        except Exception:
            return


async def offer(request: web.Request) -> web.Response:
    params = parse_offer_payload(await request.json(), default_device_id=int(request.config_dict.get("device_id", 0)))
    offer_sdp: RTCSessionDescription = params["offer"]  # type: ignore[assignment]
    if not _offer_has_h264(offer_sdp.sdp):
        raise web.HTTPBadRequest(text="Browser offer does not contain video/H264.")

    detector_path = resolve_model_path(str(params["detector_name"]))
    landmark_path = resolve_model_path(str(params["landmark_name"]))
    bitrate_kbps = params["bitrate_kbps"]
    if bitrate_kbps is None:
        bitrate_kbps = estimate_h264_bitrate_kbps(int(params["width"]), int(params["height"]), int(params["fps"]))
    encoder_mode = str(params["encoder_mode"])
    try:
        set_active_encoder_mode(encoder_mode)
        if encoder_mode == "cann":
            raise_if_cann_venc_blocked()
            if _try_import_cann is None or probe_cann_venc is None:
                raise RuntimeError("CANN VENC module is unavailable.")
            if not _try_import_cann():
                raise RuntimeError("CANN ACL import failed; cannot use CANN VENC.")
            probe_cann_venc(
                width=int(params["width"]),
                height=int(params["height"]),
                fps=int(params["fps"]),
                bitrate=int(bitrate_kbps),
            )
            clear_cann_venc_block()
            set_active_encoder_mode("cann")
    except Exception as exc:
        set_active_encoder_mode("cpu")
        error_message = str(exc)
        if collect_venc_diagnostics is not None:
            diagnostics = collect_venc_diagnostics()
            if diagnostics:
                error_message = f"{error_message} Diagnostics: {diagnostics}"
        encoder_state["last_error"] = error_message
        if encoder_mode == "cann":
            block_cann_venc(error_message, int(request.config_dict.get("cann_venc_retry_seconds", DEFAULT_CANN_VENC_RETRY_SECONDS)))
        app_logger.exception("Requested encoder mode %s is unavailable", encoder_mode)
        raise web.HTTPBadRequest(text=f"Requested encoder '{encoder_mode}' is unavailable: {error_message}") from exc

    if pcs:
        app_logger.info("Closing %s stale peer connection(s) before new offer", len(pcs))
        await asyncio.gather(*[close_peer_connection(pc) for pc in list(pcs)], return_exceptions=True)
        pcs.clear()
        await asyncio.sleep(0.3)

    pc = RTCPeerConnection()
    pcs.add(pc)
    track: HandOmVideoTrack | None = None
    connect_timeout_task: asyncio.Task | None = None

    try:
        track = HandOmVideoTrack(
            detector_path=detector_path,
            landmark_path=landmark_path,
            source=str(params["source"]),
            width=int(params["width"]),
            height=int(params["height"]),
            fps=int(params["fps"]),
            score_threshold=float(params["score_threshold"]),
            nms_iou=float(params["nms_iou"]),
            max_hands=int(params["max_hands"]),
            min_hand_score=float(params["min_hand_score"]),
            infer_every_n=int(params["infer_every_n"]),
            device_id=int(params["device_id"]),
            camera_backend=str(params["camera_backend"]),
            camera_fourcc=str(params["camera_fourcc"]),
            reload_detector_each_call=bool(request.config_dict.get("reload_detector_each_call", False)),
        )
        pc_tracks[pc] = track
        sender = pc.addTrack(track)
        _prefer_h264_for_sender(pc, sender)
        set_offer_bitrate_override(int(bitrate_kbps))

        async def close_if_not_connected(timeout: float = 30.0) -> None:
            await asyncio.sleep(timeout)
            if pc in pcs and pc.connectionState not in {"connected", "closed"} and pc.iceConnectionState not in {"completed", "closed"}:
                app_logger.warning("PeerConnection %s did not connect within %.1fs; closing stale track", id(pc), timeout)
                await close_peer_connection(pc)

        connect_timeout_task = asyncio.create_task(close_if_not_connected())

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            app_logger.info("PeerConnection %s state -> %s", id(pc), pc.connectionState)
            if pc.connectionState == "connected" and connect_timeout_task is not None:
                connect_timeout_task.cancel()
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await close_peer_connection(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange() -> None:
            app_logger.info("PeerConnection %s ICE -> %s", id(pc), pc.iceConnectionState)
            if pc.iceConnectionState == "completed" and connect_timeout_task is not None:
                connect_timeout_task.cancel()
            if pc.iceConnectionState in {"failed", "closed", "disconnected"}:
                await close_peer_connection(pc)

        await pc.setRemoteDescription(offer_sdp)
        answer = await pc.createAnswer()
        answer = RTCSessionDescription(sdp=set_video_bitrate_in_sdp(answer.sdp, int(bitrate_kbps)), type=answer.type)
        await pc.setLocalDescription(answer)
        return web.json_response(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "source_settings": track.describe_settings(bitrate_kbps=int(bitrate_kbps)),
            }
        )
    except web.HTTPException:
        if connect_timeout_task is not None:
            connect_timeout_task.cancel()
        raise
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if connect_timeout_task is not None:
            connect_timeout_task.cancel()
        app_logger.exception("Offer handling failed: %s", exc)
        await close_peer_connection(pc)
        raise web.HTTPBadRequest(text=f"Failed to create WebRTC answer: {exc}") from exc
    except Exception as exc:
        if connect_timeout_task is not None:
            connect_timeout_task.cancel()
        app_logger.exception("Offer handling failed")
        await close_peer_connection(pc)
        raise web.HTTPInternalServerError(text=f"Failed to create WebRTC answer: {exc}. Check logs/webrtc_hand_om_app.log.") from exc


async def close_peer_connection(pc: RTCPeerConnection) -> None:
    track = pc_tracks.pop(pc, None)
    if pc in pcs:
        pcs.discard(pc)
        try:
            await pc.close()
        except Exception:
            app_logger.exception("Failed to close PeerConnection %s cleanly", id(pc))
    if track is not None:
        try:
            track.stop()
        except Exception:
            app_logger.exception("Failed to stop source track for PeerConnection %s", id(pc))
    if not pc_tracks:
        with latest_stats_lock:
            latest_track_stats.clear()
    set_offer_bitrate_override(None)


@web.middleware
async def error_logging_middleware(request: web.Request, handler) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception:
        app_logger.exception("Unhandled server error while processing %s %s", request.method, request.path)
        raise web.HTTPInternalServerError(text="Unhandled server error. Check logs/webrtc_hand_om_app.log.")


async def on_shutdown(_: web.Application) -> None:
    app_logger.info("Shutting down server, closing %s peer connections", len(pcs))
    await asyncio.gather(*[close_peer_connection(pc) for pc in list(pcs)], return_exceptions=True)
    pcs.clear()


def build_app(args: argparse.Namespace) -> web.Application:
    app = web.Application(middlewares=[error_logging_middleware])
    app["default_detector"] = default_model_name(args.detector, DEFAULT_DETECTOR, "detector")
    app["default_landmark"] = default_model_name(args.landmark, DEFAULT_LANDMARK, "landmark")
    app["default_source"] = args.source
    app["device_id"] = args.device_id
    app["camera_width"] = args.camera_width
    app["camera_height"] = args.camera_height
    app["camera_fps"] = args.camera_fps
    app["infer_every_n"] = args.infer_every_n
    app["score_threshold"] = args.score_threshold
    app["nms_iou"] = args.nms_iou
    app["max_hands"] = args.max_hands
    app["min_hand_score"] = args.min_hand_score
    app["bitrate_kbps"] = args.bitrate_kbps
    app["encoder_mode"] = args.encoder_mode
    app["cann_venc_retry_seconds"] = args.cann_venc_retry_seconds
    app["camera_backend"] = args.camera_backend
    app["camera_fourcc"] = args.camera_fourcc
    app["reload_detector_each_call"] = args.reload_detector_each_call
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", client_js)
    app.router.add_get("/styles.css", styles_css)
    app.router.add_get("/health", health)
    app.router.add_get("/models", models)
    app.router.add_get("/stats", stats)
    app.router.add_post("/offer", offer)
    return app


def port_is_free(host: str, port: int) -> bool:
    bind_host = "0.0.0.0" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, int(port)))
        except OSError:
            return False
    return True


def choose_port(host: str, start_port: int, port_range: int, strict_port: bool) -> int:
    if strict_port:
        return start_port
    attempts = max(1, int(port_range))
    for offset in range(attempts):
        port = int(start_port) + offset
        if port_is_free(host, port):
            if port != start_port:
                print(f"Port {start_port} is busy. Using {port}.", flush=True)
            return port
    raise OSError(f"Cannot find an empty port in range: {start_port}-{start_port + attempts - 1}")


def local_ipv4_addresses() -> list[str]:
    addresses: list[str] = []
    try:
        hostname_ips = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        hostname_ips = []
    for ip in hostname_ips:
        if not ip.startswith("127.") and ip not in addresses:
            addresses.append(ip)
    for target in ("8.8.8.8", "1.1.1.1"):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                sock.connect((target, 80))
                ip = sock.getsockname()[0]
            except OSError:
                continue
        if ip and not ip.startswith("127.") and ip not in addresses:
            addresses.insert(0, ip)
    preferred = [ip for ip in addresses if not ip.startswith("172.")]
    secondary = [ip for ip in addresses if ip.startswith("172.")]
    return preferred + secondary


def print_access_urls(host: str, port: int) -> None:
    hostname = socket.gethostname()
    print("", flush=True)
    print("MediaPipe Hand WebRTC H.264 app is starting. Open one of these URLs:", flush=True)
    if host in ("0.0.0.0", "::"):
        print(f"  http://{hostname}:{port}", flush=True)
        for ip in local_ipv4_addresses():
            print(f"  http://{ip}:{port}", flush=True)
        print(f"  http://127.0.0.1:{port}  (only from this board or SSH port forwarding)", flush=True)
    else:
        print(f"  http://{host}:{port}", flush=True)
    print("", flush=True)


def setup_logging(log_level: str, log_file: str) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, log_level))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    file_handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", default=MODELS_DIR / DEFAULT_DETECTOR, type=Path)
    parser.add_argument("--landmark", default=MODELS_DIR / DEFAULT_LANDMARK, type=Path)
    parser.add_argument("-s", "--source", default="/dev/video0", type=str)
    parser.add_argument("--host", default=os.environ.get("WEBRTC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEBRTC_PORT", "8080")))
    parser.add_argument("--port-range", type=int, default=20)
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--camera-width", default=1280, type=int)
    parser.add_argument("--camera-height", default=720, type=int)
    parser.add_argument("--camera-fps", default=30, type=int)
    parser.add_argument("--infer-every-n", default=DEFAULT_INFER_EVERY_N, type=int)
    parser.add_argument("--score-threshold", default=0.5, type=float)
    parser.add_argument("--nms-iou", default=0.3, type=float)
    parser.add_argument("--max-hands", default=2, type=int)
    parser.add_argument("--min-hand-score", default=0.5, type=float)
    parser.add_argument("--bitrate-kbps", default=DEFAULT_H264_BITRATE_KBPS, type=int)
    parser.add_argument("--camera-backend", default=CAMERA_BACKEND_OPENCV, choices=[CAMERA_BACKEND_OPENCV, CAMERA_BACKEND_DVPP])
    parser.add_argument("--camera-fourcc", default="MJPG", choices=["MJPG", "YUYV", "DEFAULT"])
    parser.add_argument("--reload-detector-each-call", action="store_true")
    parser.add_argument("--encoder-mode", default="cpu", choices=["cpu", "cann"])
    parser.add_argument(
        "--cann-venc-retry-seconds",
        default=int(os.environ.get("CANN_VENC_RETRY_SECONDS", str(DEFAULT_CANN_VENC_RETRY_SECONDS))),
        type=int,
        help="Cooldown after a CANN VENC create failure. Use 0 to disable the guard.",
    )
    parser.add_argument("--opencv-threads", default=1, type=int)
    parser.add_argument("--log-level", default=os.environ.get("WEBRTC_LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=os.environ.get("WEBRTC_LOG_FILE", str(LOG_DIR / "webrtc_hand_om_app.log")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(max(1, args.opencv_threads))
    setup_logging(args.log_level, args.log_file)
    patch_h264_encoder()
    set_active_encoder_mode(args.encoder_mode)
    args.port = choose_port(args.host, args.port, args.port_range, args.strict_port)
    app_logger.info("Starting MediaPipe Hand WebRTC OM app on %s:%s", args.host, args.port)
    print_access_urls(args.host, args.port)
    web.run_app(build_app(args), host=args.host, port=args.port, access_log=app_logger)


if __name__ == "__main__":
    main()
