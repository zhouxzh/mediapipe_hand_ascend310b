#!/usr/bin/env python3
"""WebRTC H.264 realtime MediaPipe hand OM demo for Ascend 310B."""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
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
from textwrap import dedent
from typing import Any


def _prepend_user_ssl_fix_path() -> None:
    """Use the local OpenSSL repair copy before importing Python's ssl module."""
    fix_dir = _user_ssl_fix_path()
    if fix_dir is None:
        return
    fix_text = str(fix_dir)
    if fix_text not in sys.path:
        sys.path.insert(0, fix_text)
    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [part for part in pythonpath.split(os.pathsep) if part]
    if fix_text not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([fix_text, *parts])


def _remove_user_ssl_fix_path() -> None:
    """Prefer the active conda env's ssl module when it works."""
    fix_dir = _user_ssl_fix_path()
    if fix_dir is None:
        return
    fix_text = str(fix_dir)
    sys.path[:] = [part for part in sys.path if part != fix_text]
    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [part for part in pythonpath.split(os.pathsep) if part and part != fix_text]
    if parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    else:
        os.environ.pop("PYTHONPATH", None)


def _user_ssl_fix_path() -> Path | None:
    """Return the optional user-side _ssl repair path when present."""
    fix_dir = (
        Path.home()
        / ".local"
        / "mediapipe_hand_ssl_fix"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "lib-dynload"
    )
    if not fix_dir.exists():
        return None
    return fix_dir


def _ensure_user_pycache_prefix() -> None:
    """Avoid stale or incompatible root-owned pyc files in the board base env."""
    if getattr(sys, "pycache_prefix", None):
        return
    pycache_prefix = Path.home() / ".cache" / "mediapipe_hand_pycache"
    try:
        pycache_prefix.mkdir(parents=True, exist_ok=True)
        sys.pycache_prefix = str(pycache_prefix)
        os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache_prefix))
    except OSError:
        pass


def _verify_python_ssl_runtime() -> None:
    """Fail early when conda's OpenSSL runtime is broken."""
    _remove_user_ssl_fix_path()
    try:
        import ssl  # noqa: F401
        return
    except ImportError as first_exc:
        sys.modules.pop("ssl", None)
        sys.modules.pop("_ssl", None)
        _prepend_user_ssl_fix_path()
    try:
        import ssl  # noqa: F401
        return
    except ImportError as exc:
        prefix = Path(sys.prefix)
        cached_openssl = prefix / "pkgs" / "openssl-1.1.1w-h2f4d8fa_0" / "lib"
        repair_hint = ""
        if cached_openssl.exists():
            repair_hint = dedent(f"""

This board has a cached OpenSSL package that can be used to repair the base env:

sudo cp -a {prefix}/lib/libcrypto.so.1.1 {prefix}/lib/libcrypto.so.1.1.bak.$(date +%Y%m%d_%H%M%S)
sudo cp -a {prefix}/lib/libssl.so.1.1 {prefix}/lib/libssl.so.1.1.bak.$(date +%Y%m%d_%H%M%S)
sudo cp -a {cached_openssl}/libcrypto.so.1.1 {prefix}/lib/libcrypto.so.1.1
sudo cp -a {cached_openssl}/libssl.so.1.1 {prefix}/lib/libssl.so.1.1
""")
        message = dedent(f"""
Python ssl runtime is broken before aiortc is imported.

Python: {sys.executable}
Prefix: {prefix}
Original error: {first_exc}
Repair-path retry error: {exc}

WebRTC requires Python's ssl module through aioice/aiortc. Fix the conda
OpenSSL runtime first, then verify with:

python - <<'PY'
import ssl
print(ssl.OPENSSL_VERSION)
import aiortc
print(aiortc.__version__)
PY
{repair_hint}""").strip()
        raise SystemExit(message) from exc


faulthandler.enable(all_threads=True)
_ensure_user_pycache_prefix()
_verify_python_ssl_runtime()

import av
import cv2
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack
from aiortc import RTCConfiguration
from aiortc import RTCIceServer
from aiortc import RTCPeerConnection
from aiortc import RTCRtpSender
from aiortc import RTCSessionDescription
from aiortc.mediastreams import MediaStreamError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_pipeline.two_stage import OmHandPipeline
from hand_pipeline.realtime_graph import MediaPipeHandRealtimeGraph
from hand_pipeline.realtime_graph import RealtimeFramePacket
from hand_pipeline.visualization import HAND_EDGES
from hand_pipeline.visualization import _point_xy
from hand_pipeline.visualization import draw_hand_predictions
from hand_pipeline.visualization import draw_status_overlay
from hand_pipeline.visualization import mirror_predictions_horizontal

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
    _cann_import_error: Exception | None = None
except Exception as exc:
    CannH264Encoder = None  # type: ignore[assignment]
    probe_cann_venc = None  # type: ignore[assignment]
    _try_import_cann = None  # type: ignore[assignment]
    collect_venc_diagnostics = None  # type: ignore[assignment]
    set_encoder_status_callback = None  # type: ignore[assignment]
    set_session_encoder_mode = None  # type: ignore[assignment]
    set_session_bitrate_override_kbps = None  # type: ignore[assignment]
    _cann_import_error = exc

DvppJpegDecoder = None  # type: ignore[assignment]
V4l2RawCapture = None  # type: ignore[assignment]
_dvpp_import_error: Exception | None = None


def load_dvpp_camera_modules() -> None:
    """Import optional DVPP camera helpers only when that backend is requested."""
    global DvppJpegDecoder, V4l2RawCapture, _dvpp_import_error
    if DvppJpegDecoder is not None and V4l2RawCapture is not None:
        return
    try:
        from webrtc_app.dvpp_jpegd import DvppJpegDecoder as _DvppJpegDecoder
        from webrtc_app.v4l2_raw import V4l2RawCapture as _V4l2RawCapture
    except Exception as exc:
        _dvpp_import_error = exc
        raise RuntimeError(f"DVPP camera modules are unavailable: {exc}") from exc
    DvppJpegDecoder = _DvppJpegDecoder
    V4l2RawCapture = _V4l2RawCapture
    _dvpp_import_error = None


WEB_DIR = ROOT / "web"
MODELS_DIR = ROOT / "models" / "om"
PIANOVAM_VIDEO_DIR = ROOT / "data" / "PianoVAM_v1" / "Video"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
LOG_DIR = ROOT / "logs"
DEFAULT_DETECTOR = "mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om"
DEFAULT_LANDMARK = "mediapipe_legacy_0_10_14_hand_landmark_full.om"
DEFAULT_H264_BITRATE_KBPS = 4000
DEFAULT_INFER_EVERY_N = 1
DEFAULT_CANN_VENC_RETRY_SECONDS = 300
DEFAULT_ICE_SERVERS = ""
CAMERA_BACKEND_OPENCV = "opencv"
CAMERA_BACKEND_DVPP = "dvpp"
DEFAULT_CAMERA_BACKEND = CAMERA_BACKEND_DVPP
DEFAULT_ENCODER_MODE = "cann"
THREADING_MODE_SERIAL = "serial"
THREADING_MODE_PIPELINE = "pipeline"
DEFAULT_THREADING_MODE = THREADING_MODE_PIPELINE
DEFAULT_PIPELINE_QUEUE_SIZE = 1
PIPELINE_QUEUE_SIZES = (1, 2, 4, 8)
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
_NV12_UV_DIRECT_DRAW: bool | None = None


def no_store_file_response(path: Path) -> web.FileResponse:
    response = web.FileResponse(path)
    response.headers["Cache-Control"] = "no-store"
    return response


def patch_h264_encoder() -> bool:
    encoder_state["hardware_active"] = False
    encoder_state["name"] = "cpu-libx264-stable"
    encoder_state["last_error"] = ""

    if CannH264Encoder is None:
        detail = f": {_cann_import_error}" if _cann_import_error else ""
        app_logger.warning("WebRTC H.264 patch encoder is unavailable%s; using aiortc default H.264 encoder", detail)
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


def flip_nv12_horizontal(nv12: np.ndarray, width: int, height: int) -> np.ndarray:
    if nv12.ndim != 2:
        raise ValueError(f"NV12 frame must be 2D, got shape={nv12.shape}")
    y = nv12[:height, :width][:, ::-1]
    uv = nv12[height : height + height // 2, :width].reshape(height // 2, width // 2, 2)[:, ::-1, :]
    flipped = np.empty((height * 3 // 2, width), dtype=np.uint8)
    flipped[:height, :] = y
    flipped[height:, :] = uv.reshape(height // 2, width)
    return flipped


def bgr_color_to_yuv(color_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = (float(color_bgr[0]), float(color_bgr[1]), float(color_bgr[2]))
    y = 16.0 + 0.098 * b + 0.504 * g + 0.257 * r
    u = 128.0 + 0.439 * b - 0.291 * g - 0.148 * r
    v = 128.0 - 0.071 * b - 0.368 * g + 0.439 * r
    return tuple(int(np.clip(round(value), 0, 255)) for value in (y, u, v))  # type: ignore[return-value]


def _nv12_yuv_planes(nv12: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    y = nv12[:height, :width]
    uv = nv12[height : height + height // 2, :width].reshape(height // 2, width // 2, 2)
    return y, uv


def _uv_point(point: tuple[int, int]) -> tuple[int, int]:
    return int(point[0]) // 2, int(point[1]) // 2


def _paint_uv_mask(uv_plane: np.ndarray, mask: np.ndarray, u_value: int, v_value: int) -> None:
    selected = mask > 0
    uv_plane[:, :, 0][selected] = u_value
    uv_plane[:, :, 1][selected] = v_value


def _draw_uv_direct(
    uv_plane: np.ndarray,
    draw_func: str,
    args: tuple[Any, ...],
    color: tuple[int, int],
    thickness: int,
    line_type: int | None = None,
) -> bool:
    global _NV12_UV_DIRECT_DRAW
    if _NV12_UV_DIRECT_DRAW is False:
        return False
    try:
        func = getattr(cv2, draw_func)
        params: list[Any] = [uv_plane, *args, color, thickness]
        if line_type is None:
            func(*params)
        else:
            func(*params, line_type)
        _NV12_UV_DIRECT_DRAW = True
        return True
    except Exception:
        _NV12_UV_DIRECT_DRAW = False
        return False


def draw_nv12_line(
    nv12: np.ndarray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    y_plane, uv_plane = _nv12_yuv_planes(nv12, width, height)
    y_value, u_value, v_value = bgr_color_to_yuv(color_bgr)
    uv_thickness = max(1, int(math.ceil(thickness / 2.0)))
    cv2.line(y_plane, start, end, y_value, thickness, cv2.LINE_AA)
    if _draw_uv_direct(
        uv_plane,
        "line",
        (_uv_point(start), _uv_point(end)),
        (u_value, v_value),
        uv_thickness,
        cv2.LINE_AA,
    ):
        return
    mask = np.zeros(uv_plane.shape[:2], dtype=np.uint8)
    cv2.line(mask, _uv_point(start), _uv_point(end), 255, uv_thickness, cv2.LINE_AA)
    _paint_uv_mask(uv_plane, mask, u_value, v_value)


def draw_nv12_rectangle(
    nv12: np.ndarray,
    width: int,
    height: int,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    y_plane, uv_plane = _nv12_yuv_planes(nv12, width, height)
    y_value, u_value, v_value = bgr_color_to_yuv(color_bgr)
    uv_thickness = thickness if thickness < 0 else max(1, int(math.ceil(thickness / 2.0)))
    cv2.rectangle(y_plane, p1, p2, y_value, thickness)
    if _draw_uv_direct(
        uv_plane,
        "rectangle",
        (_uv_point(p1), _uv_point(p2)),
        (u_value, v_value),
        uv_thickness,
    ):
        return
    mask = np.zeros(uv_plane.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, _uv_point(p1), _uv_point(p2), 255, uv_thickness)
    _paint_uv_mask(uv_plane, mask, u_value, v_value)


def draw_nv12_circle(
    nv12: np.ndarray,
    width: int,
    height: int,
    center: tuple[int, int],
    radius: int,
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    y_plane, uv_plane = _nv12_yuv_planes(nv12, width, height)
    y_value, u_value, v_value = bgr_color_to_yuv(color_bgr)
    uv_radius = max(1, int(math.ceil(radius / 2.0)))
    uv_thickness = thickness if thickness < 0 else max(1, int(math.ceil(thickness / 2.0)))
    cv2.circle(y_plane, center, radius, y_value, thickness, cv2.LINE_AA)
    if _draw_uv_direct(
        uv_plane,
        "circle",
        (_uv_point(center), uv_radius),
        (u_value, v_value),
        uv_thickness,
        cv2.LINE_AA,
    ):
        return
    mask = np.zeros(uv_plane.shape[:2], dtype=np.uint8)
    cv2.circle(mask, _uv_point(center), uv_radius, 255, uv_thickness, cv2.LINE_AA)
    _paint_uv_mask(uv_plane, mask, u_value, v_value)


def draw_nv12_text(
    nv12: np.ndarray,
    width: int,
    height: int,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    y_plane, uv_plane = _nv12_yuv_planes(nv12, width, height)
    y_value, u_value, v_value = bgr_color_to_yuv(color_bgr)
    uv_origin = _uv_point(origin)
    uv_scale = max(0.1, font_scale * 0.5)
    uv_thickness = max(1, int(math.ceil(thickness / 2.0)))
    cv2.putText(y_plane, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, y_value, thickness, cv2.LINE_AA)
    global _NV12_UV_DIRECT_DRAW
    if _NV12_UV_DIRECT_DRAW is not False:
        try:
            cv2.putText(
                uv_plane,
                text,
                uv_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                uv_scale,
                (u_value, v_value),
                uv_thickness,
                cv2.LINE_AA,
            )
            _NV12_UV_DIRECT_DRAW = True
            return
        except Exception:
            _NV12_UV_DIRECT_DRAW = False
    mask = np.zeros(uv_plane.shape[:2], dtype=np.uint8)
    cv2.putText(mask, text, uv_origin, cv2.FONT_HERSHEY_SIMPLEX, uv_scale, 255, uv_thickness, cv2.LINE_AA)
    _paint_uv_mask(uv_plane, mask, u_value, v_value)


def draw_nv12_hand_predictions(
    nv12: np.ndarray,
    width: int,
    height: int,
    predictions: list[dict[str, Any]],
) -> None:
    for index, pred in enumerate(predictions):
        color = (32, 210, 120) if index == 0 else (255, 170, 40)
        palm_color = (60, 180, 255)
        if pred.get("box") is not None:
            box = np.asarray(pred["box"], dtype=np.float32)
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            draw_nv12_rectangle(nv12, width, height, (x1, y1), (x2, y2), palm_color, 2)
            if pred.get("palm7") is not None:
                for point in np.asarray(pred["palm7"], dtype=np.float32):
                    draw_nv12_circle(nv12, width, height, _point_xy(point), 3, palm_color, -1)

        if pred.get("hand21") is not None:
            points = np.asarray(pred["hand21"], dtype=np.float32)
            for a, b in HAND_EDGES:
                draw_nv12_line(nv12, width, height, _point_xy(points[a]), _point_xy(points[b]), color, 2)
            for point in points:
                xy = _point_xy(point)
                draw_nv12_circle(nv12, width, height, xy, 3, (245, 245, 245), -1)
                draw_nv12_circle(nv12, width, height, xy, 3, color, 1)

        label = f"hand {index + 1}"
        score = pred.get("score")
        hand_score = pred.get("hand_score")
        if isinstance(score, (float, int)):
            label += f" palm={float(score):.2f}"
        if isinstance(hand_score, (float, int)) and np.isfinite(hand_score):
            label += f" lm={float(hand_score):.2f}"
        origin = _point_xy(np.asarray(pred.get("box", [12, 30, 0, 0]), dtype=np.float32)[:2])
        draw_nv12_text(nv12, width, height, label, (max(origin[0], 8), max(origin[1] - 8, 22)), 0.55, color, 2)


def draw_nv12_status_overlay(
    nv12: np.ndarray,
    width: int,
    height: int,
    *,
    capture_fps: float = 0.0,
    infer_fps: float = 0.0,
    infer_ms: float = 0.0,
    hands: int = 0,
    backend: str = "",
) -> None:
    lines = [
        f"hands {hands}",
        f"infer {infer_ms:.1f} ms / {infer_fps:.1f} fps",
        f"capture {capture_fps:.1f} fps",
    ]
    if backend:
        lines.append(backend)
    x, y = 12, 24
    for line in lines:
        (text_width, text_height), _baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        draw_nv12_rectangle(
            nv12,
            width,
            height,
            (x - 5, y - text_height - 6),
            (x + text_width + 5, y + 6),
            (18, 24, 32),
            -1,
        )
        draw_nv12_text(nv12, width, height, line, (x, y), 0.56, (220, 245, 240), 2)
        y += 25


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


def list_dataset_videos(
    video_dir: Path | None = None,
    root: Path | None = None,
) -> list[dict[str, object]]:
    directory = PIANOVAM_VIDEO_DIR if video_dir is None else Path(video_dir)
    path_root = ROOT if root is None else Path(root)
    if not directory.is_dir():
        return []
    items: list[dict[str, object]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            source_path = path.relative_to(path_root).as_posix()
        except ValueError:
            source_path = str(path.resolve())
        items.append(
            {
                "name": path.name,
                "path": source_path,
                "size_bytes": path.stat().st_size,
            }
        )
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
                "pipeline_mode": request.config_dict.get("pipeline_mode", "tracking"),
                "threading_mode": request.config_dict.get("threading_mode", DEFAULT_THREADING_MODE),
                "pipeline_queue_size": request.config_dict.get("pipeline_queue_size", DEFAULT_PIPELINE_QUEUE_SIZE),
                "bitrate_kbps": request.config_dict.get("bitrate_kbps", DEFAULT_H264_BITRATE_KBPS),
                "encoder_mode": request.config_dict.get("encoder_mode", DEFAULT_ENCODER_MODE),
                "cann_venc_retry_seconds": request.config_dict.get("cann_venc_retry_seconds", DEFAULT_CANN_VENC_RETRY_SECONDS),
                "camera_backend": request.config_dict.get("camera_backend", DEFAULT_CAMERA_BACKEND),
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


async def videos(_: web.Request) -> web.Response:
    return web.json_response(
        {
            "videos": list_dataset_videos(),
            "directory": PIANOVAM_VIDEO_DIR.relative_to(ROOT).as_posix(),
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


def parse_choice(value: object, name: str, default: str, choices: set[str]) -> str:
    parsed = str(value or default).lower()
    if parsed not in choices:
        allowed = ", ".join(sorted(choices))
        raise web.HTTPBadRequest(text=f"{name} must be one of: {allowed}.")
    return parsed


def parse_offer_payload(
    params: dict[str, object],
    default_device_id: int = 0,
    default_pipeline_mode: str = "tracking",
    default_threading_mode: str = DEFAULT_THREADING_MODE,
    default_pipeline_queue_size: int = DEFAULT_PIPELINE_QUEUE_SIZE,
) -> dict[str, object]:
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

    camera_backend = str(params.get("camera_backend") or DEFAULT_CAMERA_BACKEND).lower()
    if camera_backend not in {CAMERA_BACKEND_OPENCV, CAMERA_BACKEND_DVPP}:
        raise web.HTTPBadRequest(text="camera_backend must be opencv or dvpp.")
    camera_fourcc = str(params.get("camera_fourcc") or "MJPG").upper()
    if camera_fourcc not in {"MJPG", "YUYV", "DEFAULT"}:
        raise web.HTTPBadRequest(text="camera_fourcc must be MJPG, YUYV, or DEFAULT.")
    pipeline_mode = parse_choice(params.get("pipeline_mode"), "pipeline_mode", default_pipeline_mode, {"tracking", "image"})
    threading_mode = parse_choice(
        params.get("threading_mode"),
        "threading_mode",
        default_threading_mode,
        {THREADING_MODE_SERIAL, THREADING_MODE_PIPELINE},
    )
    pipeline_queue_size = parse_positive_int(
        params.get("pipeline_queue_size"),
        "pipeline_queue_size",
        default_pipeline_queue_size,
    )
    if pipeline_queue_size not in PIPELINE_QUEUE_SIZES:
        raise web.HTTPBadRequest(text=f"pipeline_queue_size must be one of: {', '.join(map(str, PIPELINE_QUEUE_SIZES))}.")
    encoder_mode = parse_choice(params.get("encoder_mode"), "encoder_mode", DEFAULT_ENCODER_MODE, {"cpu", "cann"})

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
        "pipeline_mode": pipeline_mode,
        "threading_mode": threading_mode,
        "pipeline_queue_size": pipeline_queue_size,
    }


def _offer_has_h264(sdp: str) -> bool:
    return any(
        line.startswith("a=rtpmap:") and line.strip().split(None, 1)[-1].split("/", 1)[0].lower() == "h264"
        for line in sdp.splitlines()
    )


def _local_h264_codecs():
    return [codec for codec in RTCRtpSender.getCapabilities("video").codecs if codec.mimeType.lower() == "video/h264"]


def parse_ice_servers(value: str | None) -> list[RTCIceServer]:
    if not value:
        return []
    servers: list[RTCIceServer] = []
    for item in str(value).split(","):
        url = item.strip()
        if url:
            servers.append(RTCIceServer(urls=url))
    return servers


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
        camera_backend: str = DEFAULT_CAMERA_BACKEND,
        camera_fourcc: str = "MJPG",
        reload_detector_each_call: bool = False,
        pipeline_mode: str = "tracking",
        threading_mode: str = DEFAULT_THREADING_MODE,
        pipeline_queue_size: int = DEFAULT_PIPELINE_QUEUE_SIZE,
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
        self.pipeline_mode = pipeline_mode
        self.threading_mode = threading_mode
        self.pipeline_queue_size = max(1, int(pipeline_queue_size))
        self.pipeline: OmHandPipeline | None = None
        self.realtime_graph: MediaPipeHandRealtimeGraph | None = None
        self.cap: cv2.VideoCapture | None = None
        self.capture_impl = None
        self.jpegd = None
        self._actual_fourcc = ""
        self._source_type = "camera"
        self._source_is_file = False
        self._source_loop_count = 0
        self._start: float | None = None
        self._timestamp = 0
        self._frame_time = 1 / max(self.fps, 1)
        self._frame_index = 0
        self._closed = False
        self._state_lock = threading.Lock()
        self._prediction_lock = threading.Lock()
        self._frame_condition = threading.Condition()
        self._pipeline_stop = threading.Event()
        self._pipeline_reset_requested = threading.Event()
        self._infer_queue: queue.Queue[RealtimeFramePacket] | None = None
        self._inference_thread: threading.Thread | None = None
        self._latest_frame_packet: RealtimeFramePacket | None = None
        self._last_rendered_frame_index = -1
        self._infer_candidate_index = 0
        self._last_predictions: list[dict[str, Any]] = []
        self._last_graph_streams: dict[str, Any] = {}
        self._last_debug: dict[str, Any] | None = None
        self._last_infer_ms = 0.0
        self._last_infer_total_ms = 0.0
        self._last_det_pre_ms = 0.0
        self._last_det_npu_ms = 0.0
        self._last_det_post_ms = 0.0
        self._last_roi_ms = 0.0
        self._last_crop_ms = 0.0
        self._last_landmark_npu_ms = 0.0
        self._last_landmark_post_ms = 0.0
        self._last_palm_detector_skipped = False
        self._last_prediction_at: float | None = None
        self._last_prediction_frame_index = -1
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
        self._dropped_pipeline_frames = 0
        self._open()

    def _normalize_nv12_for_frame(self, nv12: np.ndarray) -> np.ndarray:
        if nv12.ndim != 2:
            raise ValueError(f"NV12 frame must be 2D, got shape={nv12.shape}")
        rows = self.height + self.height // 2
        if nv12.shape[0] < rows or nv12.shape[1] < self.width:
            raise ValueError(f"NV12 frame shape {nv12.shape} is too small for {self.width}x{self.height}")
        return np.ascontiguousarray(nv12[:rows, : self.width])

    def _open(self) -> None:
        if not self.detector_path.exists():
            raise FileNotFoundError(f"Detector OM model not found: {self.detector_path}")
        if not self.landmark_path.exists():
            raise FileNotFoundError(f"Landmark OM model not found: {self.landmark_path}")
        try:
            self.pipeline = OmHandPipeline(
                self.detector_path,
                self.landmark_path,
                device_id=self.device_id,
                score_threshold=self.score_threshold,
                nms_iou=self.nms_iou,
                max_hands=self.max_hands,
                min_hand_score=self.min_hand_score,
                max_det=20,
                mode=self.pipeline_mode,
                reload_detector_each_frame=self.reload_detector_each_call,
                finalize_on_release=False,
            )
            self.realtime_graph = MediaPipeHandRealtimeGraph(self.pipeline)
            if self.camera_backend == CAMERA_BACKEND_DVPP:
                self._open_dvpp_camera()
            else:
                self._open_opencv_camera()
            if self.threading_mode == THREADING_MODE_PIPELINE:
                self._start_pipeline_threads()
        except Exception:
            try:
                self._cleanup()
            except Exception:
                app_logger.exception("Failed to roll back partially opened WebRTC hand track")
            raise
        app_logger.info(
            "Opened WebRTC hand track detector=%s landmark=%s source=%s capture=%sx%s@%s backend=%s fourcc=%s pipeline=%s threading=%s queue=%s",
            self.detector_path.name,
            self.landmark_path.name,
            self.source,
            self.width,
            self.height,
            self.fps,
            self.camera_backend,
            self._actual_fourcc or self.camera_fourcc,
            self.pipeline_mode,
            self.threading_mode,
            self.pipeline_queue_size,
        )
        self._publish_stats()

    def _open_opencv_camera(self) -> None:
        source_text = str(self.source).strip()
        candidate = Path(source_text).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        self._source_is_file = candidate.is_file()
        if self._source_is_file:
            source: int | str = str(candidate.resolve())
            self._source_type = "video"
            self.cap = cv2.VideoCapture(source)
        else:
            source = int(source_text) if source_text.isdigit() else source_text
            self._source_type = "camera"
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")
        if not self._source_is_file:
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
        load_dvpp_camera_modules()
        device_path = source_to_device_path(self.source)
        if V4l2RawCapture is not None:
            self.capture_impl = V4l2RawCapture(device=device_path, width=self.requested_width, height=self.requested_height, fps=self.requested_fps)
        else:
            detail = f": {_dvpp_import_error}" if _dvpp_import_error else ""
            raise RuntimeError(f"V4L2 raw capture module is unavailable{detail}.")
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
            "source_type": self._source_type,
            "loop_video": self._source_is_file,
            "model_input": "palm 192x192 + landmark 224x224",
            "pipeline_mode": self.pipeline_mode,
            "threading_mode": self.threading_mode,
            "pipeline_queue_size": self.pipeline_queue_size,
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
                "pipeline_mode": self.pipeline_mode,
                "threading_mode": self.threading_mode,
                "pipeline_queue_size": self.pipeline_queue_size,
                "score_threshold": self.score_threshold,
                "nms_iou": self.nms_iou,
                "max_hands": self.max_hands,
            },
        }

    def _publish_stats(self) -> None:
        now = time.perf_counter()
        latest_frame_age_ms = 0.0
        latest_frame_index = -1
        with self._frame_condition:
            if self._latest_frame_packet is not None:
                latest_frame_index = self._latest_frame_packet.frame_index
                latest_frame_age_ms = max(0.0, (now - self._latest_frame_packet.captured_at) * 1000.0)
        prediction_age_ms = 0.0
        if self._last_prediction_at is not None:
            prediction_age_ms = max(0.0, (now - self._last_prediction_at) * 1000.0)
        with latest_stats_lock:
            latest_track_stats.clear()
            latest_track_stats.update(
                {
                    "closed": self._closed,
                    "detector": self.detector_path.name,
                    "landmark": self.landmark_path.name,
                    "source": self.source,
                    "source_type": self._source_type,
                    "source_loop_count": self._source_loop_count,
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
                    "pipeline_mode": self.pipeline_mode,
                    "threading_mode": self.threading_mode,
                    "pipeline_queue_size": self.pipeline_queue_size,
                    "dropped_frames": self._dropped_pipeline_frames,
                    "dropped_capture_frames": 0,
                    "dropped_pipeline_frames": self._dropped_pipeline_frames,
                    "latest_frame_age_ms": latest_frame_age_ms,
                    "prediction_age_ms": prediction_age_ms,
                    "latest_frame_index": latest_frame_index,
                    "prediction_frame_index": self._last_prediction_frame_index,
                    "rendered_frame_index": self._last_rendered_frame_index,
                    "hands": len(self._last_predictions),
                    "npu_latency_ms": self._last_infer_ms,
                    "infer_total_ms": self._last_infer_total_ms,
                    "det_pre_ms": self._last_det_pre_ms,
                    "det_npu_ms": self._last_det_npu_ms,
                    "det_post_ms": self._last_det_post_ms,
                    "roi_ms": self._last_roi_ms,
                    "crop_ms": self._last_crop_ms,
                    "landmark_npu_ms": self._last_landmark_npu_ms,
                    "landmark_post_ms": self._last_landmark_post_ms,
                    "palm_detector_skipped": self._last_palm_detector_skipped,
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
        frame, _nv12 = self._read_capture_frame()
        if frame is None:
            raise RuntimeError("BGR frame is unavailable for the selected camera backend")
        return frame

    def _frame_bgr_for_fallback(self, frame: np.ndarray | None, frame_nv12: np.ndarray | None) -> np.ndarray | None:
        if frame is not None:
            return frame
        if frame_nv12 is not None and self.pipeline_mode != "tracking":
            return nv12_to_bgr(frame_nv12, self.width, self.height)
        return None

    def _mark_video_looped(self) -> None:
        self._source_loop_count += 1
        self._pipeline_reset_requested.set()
        with self._prediction_lock:
            self._last_predictions = []
            self._last_graph_streams = {}
            self._last_debug = None
            self._last_prediction_at = None
            self._last_prediction_frame_index = -1

    def _reset_pipeline_if_requested(self) -> bool:
        if not self._pipeline_reset_requested.is_set():
            return False
        self._pipeline_reset_requested.clear()
        if self.pipeline is not None:
            self.pipeline.reset()
        if self.realtime_graph is not None:
            self.realtime_graph.last_streams = {
                name: [] for name in self.realtime_graph.STREAM_NAMES
            }
        return True

    def _read_capture_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        capture_start = time.perf_counter()
        try:
            if self.camera_backend == CAMERA_BACKEND_DVPP:
                if self.capture_impl is None or self.jpegd is None:
                    raise RuntimeError("DVPP camera is not open")
                jpeg_bytes = self.capture_impl.read(timeout=2.0)
                nv12_flat = self.jpegd.decode(jpeg_bytes)
                nv12 = nv12_flat.reshape(self.jpegd.nv12_shape)
                tight_nv12 = self._normalize_nv12_for_frame(nv12)
                frame = None
            else:
                if self.cap is None:
                    raise RuntimeError("OpenCV camera is not open")
                ok, frame = self.cap.read()
                if (not ok or frame is None) and self._source_is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                    if ok and frame is not None:
                        self._mark_video_looped()
                if not ok or frame is None:
                    raise RuntimeError("Camera read returned no frame")
                tight_nv12 = None
            if frame is not None and (frame.shape[0] % 2 or frame.shape[1] % 2):
                frame = frame[: frame.shape[0] - (frame.shape[0] % 2), : frame.shape[1] - (frame.shape[1] % 2)]
            self._capture_error = ""
            return frame, tight_nv12
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

    def _start_pipeline_threads(self) -> None:
        self._infer_queue = queue.Queue(maxsize=self.pipeline_queue_size)
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="webrtc-hand-infer",
            daemon=True,
        )
        self._inference_thread.start()

    def _join_pipeline_threads(self) -> bool:
        thread = self._inference_thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        if thread.is_alive():
            thread.join(timeout=2.0)
        if thread.is_alive():
            app_logger.warning("Inference thread did not stop within 2 seconds; deferring pipeline release")
            return False
        self._inference_thread = None
        return True

    def _release_pipeline(self) -> None:
        with self._state_lock:
            pipeline = self.pipeline
            self.pipeline = None
            self.realtime_graph = None
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                app_logger.exception("Failed to release OM hand pipeline")

    def _put_latest_infer_packet(self, packet: RealtimeFramePacket) -> None:
        if self._infer_queue is None:
            return
        dropped = 0
        while not self._pipeline_stop.is_set():
            try:
                self._infer_queue.put_nowait(packet)
                break
            except queue.Full:
                try:
                    self._infer_queue.get_nowait()
                    dropped += 1
                except queue.Empty:
                    continue
        if dropped:
            self._dropped_pipeline_frames += dropped

    def _inference_loop(self) -> None:
        try:
            self._run_inference_loop()
        finally:
            with self._state_lock:
                if self._inference_thread is threading.current_thread():
                    self._inference_thread = None
                self._infer_queue = None
                release_pipeline = self._closed
            if release_pipeline:
                self._release_pipeline()

    def _run_inference_loop(self) -> None:
        while not self._pipeline_stop.is_set():
            infer_queue = self._infer_queue
            if infer_queue is None:
                return
            try:
                packet = infer_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            drained = 0
            while True:
                try:
                    packet = infer_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            if drained:
                self._dropped_pipeline_frames += drained

            reset_for_loop = self._reset_pipeline_if_requested()
            candidate_index = self._infer_candidate_index
            self._infer_candidate_index += 1
            if not reset_for_loop and candidate_index % self.infer_every_n != 0:
                continue
            try:
                if self.realtime_graph is None:
                    raise RuntimeError("Realtime hand graph is not open")
                result = self.realtime_graph.process(packet)
                self._apply_graph_result(result)
            except Exception as exc:
                self._infer_error = str(exc)
                app_logger.exception("Hand OM realtime inference failed")

    def _apply_graph_result(self, result: Any) -> None:
        timing = result.timings
        det_npu_ms = float(timing.get("det_npu_ms", timing.get("detector_ms", 0.0)) or 0.0)
        landmark_npu_ms = float(timing.get("landmark_npu_ms", timing.get("landmark_ms", 0.0)) or 0.0)
        with self._prediction_lock:
            self._last_predictions = result.predictions
            self._last_debug = result.debug
            self._last_graph_streams = result.streams
            self._last_det_pre_ms = float(timing.get("det_pre_ms", timing.get("preprocess_ms", 0.0)) or 0.0)
            self._last_det_npu_ms = det_npu_ms
            self._last_det_post_ms = float(timing.get("det_post_ms", timing.get("decode_ms", 0.0)) or 0.0)
            self._last_roi_ms = float(timing.get("roi_only_ms", timing.get("roi_ms", 0.0)) or 0.0)
            self._last_crop_ms = float(timing.get("crop_ms", 0.0) or 0.0)
            self._last_landmark_npu_ms = landmark_npu_ms
            self._last_landmark_post_ms = float(timing.get("landmark_post_ms", timing.get("post_ms", 0.0)) or 0.0)
            self._last_palm_detector_skipped = bool(timing.get("palm_detector_skipped", False))
            self._last_infer_ms = det_npu_ms + landmark_npu_ms
            self._last_infer_total_ms = float(timing.get("total_ms", 0.0) or 0.0)
            self._last_prediction_at = float(result.completed_at)
            self._last_prediction_frame_index = int(result.frame_index)
            self._infer_error = ""
        self._update_infer_fps()

    def _snapshot_predictions(self) -> list[dict[str, Any]]:
        with self._prediction_lock:
            return list(self._last_predictions)

    def _update_infer_fps(self) -> None:
        now = time.perf_counter()
        self._infer_fps_frames += 1
        elapsed = now - self._infer_fps_start
        if elapsed >= 1.0:
            self._infer_fps = self._infer_fps_frames / elapsed
            self._infer_fps_frames = 0
            self._infer_fps_start = now

    def _update_track_fps(self) -> None:
        now = time.perf_counter()
        self._track_fps_frames += 1
        elapsed = now - self._track_fps_start
        if elapsed >= 1.0:
            self._track_fps = self._track_fps_frames / elapsed
            self._track_fps_frames = 0
            self._track_fps_start = now

    def _render_frame_to_nv12(
        self,
        frame: np.ndarray | None,
        frame_nv12: np.ndarray | None,
        predictions: list[dict[str, Any]],
    ) -> np.ndarray:
        try:
            nv12_start = time.perf_counter()
            if frame_nv12 is not None:
                nv12 = flip_nv12_horizontal(frame_nv12, self.width, self.height)
                rendered_predictions = mirror_predictions_horizontal(predictions, self.width)
                draw_nv12_hand_predictions(nv12, self.width, self.height, rendered_predictions)
                draw_nv12_status_overlay(
                    nv12,
                    self.width,
                    self.height,
                    capture_fps=self._capture_fps,
                    infer_fps=self._infer_fps,
                    infer_ms=self._last_infer_total_ms,
                    hands=len(predictions),
                    backend=f"{self.camera_backend}/{self._actual_fourcc or self.camera_fourcc}",
                )
            else:
                if frame is None:
                    raise RuntimeError("Cannot render without BGR or NV12 frame")
                rendered_frame = cv2.flip(frame, 1)
                rendered_predictions = mirror_predictions_horizontal(predictions, frame.shape[1])
                rendered = draw_hand_predictions(rendered_frame, rendered_predictions, copy_image=False)
                rendered = draw_status_overlay(
                    rendered,
                    capture_fps=self._capture_fps,
                    infer_fps=self._infer_fps,
                    infer_ms=self._last_infer_total_ms,
                    hands=len(predictions),
                    backend=f"{self.camera_backend}/{self._actual_fourcc or self.camera_fourcc}",
                )
                nv12 = bgr_to_nv12(rendered)
            self._last_nv12_ms = (time.perf_counter() - nv12_start) * 1000.0
            self._render_error = ""
            return nv12
        except Exception as exc:
            self._render_error = str(exc)
            raise

    def _read_output_frame(self):
        frame_start = time.perf_counter()
        frame, frame_nv12 = self._read_capture_frame()
        reset_for_loop = self._reset_pipeline_if_requested()
        predictions = self._snapshot_predictions()
        if reset_for_loop or self._frame_index % self.infer_every_n == 0:
            try:
                if self.realtime_graph is None:
                    raise RuntimeError("Realtime hand graph is not open")
                now = time.perf_counter()
                packet = RealtimeFramePacket(
                    frame_index=self._frame_index,
                    timestamp=now,
                    captured_at=now,
                    image_bgr=self._frame_bgr_for_fallback(frame, frame_nv12),
                    image_nv12=frame_nv12,
                    image_width=self.width,
                    image_height=self.height,
                )
                result = self.realtime_graph.process(packet)
                predictions = result.predictions
                self._apply_graph_result(result)
            except Exception as exc:
                self._infer_error = str(exc)
                app_logger.exception("Hand OM inference failed")
        nv12 = self._render_frame_to_nv12(frame, frame_nv12, predictions)

        self._frame_index += 1
        self._last_rendered_frame_index = self._frame_index - 1
        self._update_track_fps()
        self._last_pipeline_ms = (time.perf_counter() - frame_start) * 1000.0
        self._publish_stats()
        return nv12

    def _read_pipeline_output_frame(self):
        frame_start = time.perf_counter()
        frame, frame_nv12 = self._read_capture_frame()
        now = time.perf_counter()
        packet = RealtimeFramePacket(
            frame_index=self._frame_index,
            timestamp=now,
            captured_at=now,
            image_bgr=self._frame_bgr_for_fallback(frame, frame_nv12),
            image_nv12=frame_nv12,
            image_width=self.width,
            image_height=self.height,
        )
        with self._frame_condition:
            self._latest_frame_packet = packet
        self._put_latest_infer_packet(packet)

        predictions = self._snapshot_predictions()
        nv12 = self._render_frame_to_nv12(frame, frame_nv12, predictions)
        self._frame_index += 1
        self._last_rendered_frame_index = packet.frame_index
        self._update_track_fps()
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
            if self.threading_mode == THREADING_MODE_PIPELINE:
                frame = await loop.run_in_executor(None, self._read_pipeline_output_frame)
            else:
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
            self._pipeline_stop.set()
            with self._frame_condition:
                self._frame_condition.notify_all()
            self._publish_stats()
        inference_stopped = self._join_pipeline_threads()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                app_logger.exception("Failed to release OpenCV camera")
            self.cap = None
        if self.capture_impl is not None:
            try:
                self.capture_impl.stop()
            except Exception:
                app_logger.exception("Failed to stop V4L2 capture")
            self.capture_impl = None
        if self.jpegd is not None:
            try:
                self.jpegd.destroy()
            except Exception:
                app_logger.exception("Failed to destroy JPEG decoder")
            self.jpegd = None
        if inference_stopped:
            self._release_pipeline()

    def stop(self) -> None:
        app_logger.info("Stopping WebRTC hand track detector=%s landmark=%s", self.detector_path.name, self.landmark_path.name)
        self._cleanup()
        try:
            super().stop()
        except Exception:
            return


async def offer(request: web.Request) -> web.Response:
    params = parse_offer_payload(
        await request.json(),
        default_device_id=int(request.config_dict.get("device_id", 0)),
        default_pipeline_mode=str(request.config_dict.get("pipeline_mode", "tracking")),
        default_threading_mode=str(request.config_dict.get("threading_mode", DEFAULT_THREADING_MODE)),
        default_pipeline_queue_size=int(request.config_dict.get("pipeline_queue_size", DEFAULT_PIPELINE_QUEUE_SIZE)),
    )
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

    ice_servers = request.config_dict.get("ice_servers", [])
    pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
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
            pipeline_mode=str(params["pipeline_mode"]),
            threading_mode=str(params["threading_mode"]),
            pipeline_queue_size=int(params["pipeline_queue_size"]),
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
    app["pipeline_mode"] = args.pipeline_mode
    app["threading_mode"] = args.threading_mode
    app["pipeline_queue_size"] = args.pipeline_queue_size
    app["bitrate_kbps"] = args.bitrate_kbps
    app["encoder_mode"] = args.encoder_mode
    app["cann_venc_retry_seconds"] = args.cann_venc_retry_seconds
    app["ice_servers"] = parse_ice_servers(args.ice_servers)
    app["camera_backend"] = args.camera_backend
    app["camera_fourcc"] = args.camera_fourcc
    app["reload_detector_each_call"] = args.reload_detector_each_call
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", client_js)
    app.router.add_get("/styles.css", styles_css)
    app.router.add_get("/health", health)
    app.router.add_get("/models", models)
    app.router.add_get("/videos", videos)
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
    parser.add_argument("--pipeline-mode", default="tracking", choices=["tracking", "image"])
    parser.add_argument("--threading-mode", default=DEFAULT_THREADING_MODE, choices=[THREADING_MODE_PIPELINE, THREADING_MODE_SERIAL])
    parser.add_argument("--pipeline-queue-size", default=DEFAULT_PIPELINE_QUEUE_SIZE, type=int, choices=PIPELINE_QUEUE_SIZES)
    parser.add_argument("--bitrate-kbps", default=DEFAULT_H264_BITRATE_KBPS, type=int)
    parser.add_argument("--camera-backend", default=DEFAULT_CAMERA_BACKEND, choices=[CAMERA_BACKEND_OPENCV, CAMERA_BACKEND_DVPP])
    parser.add_argument("--camera-fourcc", default="MJPG", choices=["MJPG", "YUYV", "DEFAULT"])
    parser.add_argument("--reload-detector-each-call", action="store_true")
    parser.add_argument("--encoder-mode", default=DEFAULT_ENCODER_MODE, choices=["cpu", "cann"])
    parser.add_argument(
        "--ice-servers",
        default=os.environ.get("WEBRTC_ICE_SERVERS", DEFAULT_ICE_SERVERS),
        help="Comma-separated ICE server URLs. Empty default uses LAN-only host candidates and avoids public STUN on the board.",
    )
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
    faulthandler.enable(all_threads=True)
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
