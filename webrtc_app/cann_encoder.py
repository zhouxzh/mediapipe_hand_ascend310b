import logging
import os
import queue
import subprocess
import threading
import time
from typing import Iterator, Optional

import av
import numpy as np
from aiortc.codecs.h264 import H264Encoder

logger = logging.getLogger("cann_encoder")

VENC_AUTO_BITS_PER_PIXEL = 0.04
VENC_MIN_BITRATE_KBPS = 500
VENC_MAX_BITRATE_KBPS = 10_000
_SESSION_BITRATE_OVERRIDE_KBPS: Optional[int] = None
_ENCODER_STATUS_CALLBACK = None
_SESSION_ENCODER_MODE = "cpu"


def _clamp_venc_bitrate_kbps(bitrate_kbps: int) -> int:
    return max(VENC_MIN_BITRATE_KBPS, min(bitrate_kbps, VENC_MAX_BITRATE_KBPS))


def _is_success(ret) -> bool:
    return ret is None or ret == ACL_SUCCESS


def _normalize_video_codec(codec: str) -> str:
    return "h264"


def estimate_venc_bitrate_kbps(
    width: int,
    height: int,
    fps: int,
    codec: str = "h264",
) -> int:
    """Estimate a practical VENC bitrate in kbps for the source format."""
    _normalize_video_codec(codec)
    bitrate = round(width * height * fps * VENC_AUTO_BITS_PER_PIXEL / 1000)
    return _clamp_venc_bitrate_kbps(bitrate)


def resolve_venc_bitrate_kbps(
    width: int,
    height: int,
    fps: int,
    codec: str = "h264",
) -> int:
    """Resolve VENC bitrate from source dimensions and codec."""
    return estimate_venc_bitrate_kbps(width, height, fps, codec=codec)


def _target_bitrate_kbps(target_bitrate_bps: int) -> int:
    if target_bitrate_bps <= 0:
        return 0
    return _clamp_venc_bitrate_kbps(target_bitrate_bps // 1000)


def _resolve_venc_bitrate_kbps(
    width: int,
    height: int,
    fps: int,
    codec: str,
    target_bitrate_bps: int,
) -> int:
    bitrate = resolve_venc_bitrate_kbps(width, height, fps, codec=codec)
    target_kbps = _target_bitrate_kbps(target_bitrate_bps)
    if target_kbps:
        bitrate = min(bitrate, target_kbps)
    return bitrate


def set_session_bitrate_override_kbps(bitrate_kbps: Optional[int]) -> None:
    global _SESSION_BITRATE_OVERRIDE_KBPS
    if bitrate_kbps is None:
        _SESSION_BITRATE_OVERRIDE_KBPS = None
        return
    _SESSION_BITRATE_OVERRIDE_KBPS = _clamp_venc_bitrate_kbps(int(bitrate_kbps))


def get_session_bitrate_override_kbps() -> Optional[int]:
    return _SESSION_BITRATE_OVERRIDE_KBPS


def set_encoder_status_callback(callback) -> None:
    global _ENCODER_STATUS_CALLBACK
    _ENCODER_STATUS_CALLBACK = callback


def set_session_encoder_mode(mode: str) -> None:
    """Select the encoder mode for newly-created aiortc H.264 encoders."""
    global _SESSION_ENCODER_MODE
    mode = str(mode).lower()
    if mode not in {"cpu", "cann"}:
        raise ValueError(f"Unsupported encoder mode: {mode}")
    _SESSION_ENCODER_MODE = mode


def get_session_encoder_mode() -> str:
    return _SESSION_ENCODER_MODE


def _notify_encoder_status(name: str, hardware_active: bool, reason: str = "") -> None:
    if _ENCODER_STATUS_CALLBACK is None:
        return
    try:
        _ENCODER_STATUS_CALLBACK(name, hardware_active, reason)
    except Exception:
        logger.exception("Encoder status callback failed")


# ---------------------------------------------------------------------------
#  CANN constants
# ---------------------------------------------------------------------------
ENTYPE_H264_BASE = 1
ENTYPE_H264_MAIN = 2
ENTYPE_H264_HIGH = 3
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1  # NV12
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_SUCCESS = 0
ACL_ALREADY_INITIALIZED = 100002

# ---------------------------------------------------------------------------
#  Optional CANN import
# ---------------------------------------------------------------------------
_acl = None
_acl_media = None
_acl_rt = None
_acl_util = None
_CANN_READY = False

_ACL_INIT_LOCK = threading.Lock()
_ACL_INITIALIZED = False
_ACL_CONTEXT = None
_JOINED_USERMEMORY = False


def _try_import_cann():
    """Import CANN modules with proper library paths set."""
    global _acl, _acl_media, _acl_rt, _acl_util, _CANN_READY

    if _CANN_READY:
        return True
    if _acl is not None:
        return _CANN_READY

    cann_paths = [
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit/8.3.RC1",
    ]
    for base in cann_paths:
        # Python site-packages is directly under the toolkit, NOT under aarch64-linux
        py_path = os.path.join(base, "python", "site-packages")
        lib_path = os.path.join(base, "aarch64-linux", "lib64")
        if os.path.isdir(py_path) and os.path.isdir(lib_path):
            os.environ.setdefault("LD_LIBRARY_PATH", "")
            if lib_path not in os.environ["LD_LIBRARY_PATH"]:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{lib_path}:{os.environ['LD_LIBRARY_PATH']}"
                )
            if py_path not in os.environ.get("PYTHONPATH", ""):
                os.environ["PYTHONPATH"] = (
                    f"{py_path}:{os.environ.get('PYTHONPATH', '')}"
                )
            import sys
            if py_path not in sys.path:
                sys.path.insert(0, py_path)
            break

    try:
        import acl as _acl_mod
        _acl = _acl_mod
        _acl_media = _acl.media
        _acl_rt = _acl.rt
        _acl_util = _acl.util
        _CANN_READY = True
        logger.info("CANN ACL imported successfully")
        return True
    except ImportError as exc:
        _acl = False
        logger.warning("CANN ACL not available: %s", exc)
        return False


def _join_usermemory_cgroup() -> None:
    """Best-effort join of Ascend's usermemory cgroup on Orange Pi AI Pro.

    Some CANN/DVPP services expect user processes to be in this cgroup before
    allocating media buffers. Failure is not fatal because board images differ.
    """
    global _JOINED_USERMEMORY
    if _JOINED_USERMEMORY or os.name != "posix":
        return

    tasks_path = "/sys/fs/cgroup/memory/usermemory/tasks"
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as file:
            if any(":memory:/usermemory" in line for line in file):
                _JOINED_USERMEMORY = True
                return
    except OSError:
        pass

    try:
        with open(tasks_path, "a", encoding="utf-8") as file:
            file.write(f"{os.getpid()}\n")
        _JOINED_USERMEMORY = True
        logger.info("Joined Ascend usermemory cgroup for DVPP/VENC allocations")
    except OSError as exc:
        logger.warning(
            "Could not join %s: %s. CANN VENC may fail if the board image "
            "requires usermemory cgroup membership.",
            tasks_path,
            exc,
        )


def _run_text_command(command: list[str], timeout: float = 2.0) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return f"{command[0]} unavailable: {exc}"
    return completed.stdout.strip()


def _recent_venc_driver_error() -> str:
    if os.name != "posix":
        return ""
    output = _run_text_command(
        ["bash", "-lc", "dmesg | grep -iE 'venc|h264e|h265e|encoder node|rc_' | tail -12"],
        timeout=2.0,
    )
    for line in reversed(output.splitlines()):
        lowered = line.lower()
        if "failed" in lowered or "error" in lowered or "err" in lowered:
            return line.strip()
    return output.splitlines()[-1].strip() if output.splitlines() else ""


def collect_venc_diagnostics() -> str:
    """Return a compact board-side diagnostic snapshot for VENC failures."""
    lines: list[str] = []
    if os.name == "posix":
        try:
            with open("/proc/self/cgroup", "r", encoding="utf-8") as file:
                memory_lines = [line.strip() for line in file if ":memory:" in line]
            if memory_lines:
                lines.append(f"cgroup={';'.join(memory_lines)}")
        except OSError:
            pass
        for path, label in [
            ("/proc/meminfo", "hugepages"),
            ("/proc/umap/venc", "venc"),
        ]:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as file:
                    text = file.read()
            except OSError:
                continue
            if label == "hugepages":
                huge = [
                    line.strip()
                    for line in text.splitlines()
                    if line.startswith(("HugePages_Total", "HugePages_Free", "HugePages_Rsvd"))
                ]
                if huge:
                    lines.append(" ".join(huge))
            else:
                for line in text.splitlines():
                    if "Detail Venc Chn Id Info" in line:
                        break
                active = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip().startswith("0 ") or "UserChnId" in line
                ]
                if active:
                    lines.append("venc_proc=" + " | ".join(active[-3:]))
        npu_smi = _run_text_command(
            ["bash", "-lc", "source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true; npu-smi info 2>/dev/null | grep -E '310B|Memory-Usage|Hugepages-Usage|Alarm|Health'"],
            timeout=3.0,
        )
        if npu_smi:
            lines.append("npu_smi=" + " | ".join(npu_smi.splitlines()[:6]))
    driver_error = _recent_venc_driver_error()
    if driver_error:
        lines.append(f"driver={driver_error}")
    return "; ".join(lines)


def _check_acl_call(name: str, ret) -> None:
    if not _is_success(ret):
        raise RuntimeError(f"{name} failed: {ret}")


def _resolve_h264_entype() -> int:
    raw_value = os.environ.get("CANN_VENC_H264_PROFILE", "baseline").strip().lower()
    aliases = {
        "base": ENTYPE_H264_BASE,
        "baseline": ENTYPE_H264_BASE,
        "1": ENTYPE_H264_BASE,
        "main": ENTYPE_H264_MAIN,
        "2": ENTYPE_H264_MAIN,
        "high": ENTYPE_H264_HIGH,
        "3": ENTYPE_H264_HIGH,
    }
    if raw_value not in aliases:
        logger.warning("Unsupported CANN_VENC_H264_PROFILE=%s; using baseline", raw_value)
        return ENTYPE_H264_BASE
    return aliases[raw_value]


def _should_set_src_rate() -> bool:
    return os.environ.get("CANN_VENC_SET_SRC_RATE", "0").strip().lower() in {"1", "true", "yes"}


def _format_venc_create_error(ret: int, width: int, height: int, fps: int, bitrate: int, entype: int) -> str:
    detail = _recent_venc_driver_error()
    message = (
        f"venc_create_channel failed: {ret} (0x{ret:x}) "
        f"for {width}x{height}@{fps}, bitrate={bitrate}kbps, entype={entype}."
    )
    if detail:
        message += f" Driver says: {detail}"
    else:
        message += " Check dmesg for HiDvpp/VENC driver details."
    return message


def _init_acl(device_id: int = 0) -> bool:
    """One-time ACL runtime initialization (thread-safe)."""
    global _ACL_CONTEXT, _ACL_INITIALIZED
    if _ACL_INITIALIZED:
        if _ACL_CONTEXT is not None:
            ret = _acl_rt.set_context(_ACL_CONTEXT)
            if ret != ACL_SUCCESS:
                logger.error("acl.rt.set_context() failed: %s", ret)
                return False
        return True
    if not _try_import_cann():
        return False
    with _ACL_INIT_LOCK:
        if _ACL_INITIALIZED:
            if _ACL_CONTEXT is not None:
                ret = _acl_rt.set_context(_ACL_CONTEXT)
                if ret != ACL_SUCCESS:
                    logger.error("acl.rt.set_context() failed: %s", ret)
                    return False
            return True
        _join_usermemory_cgroup()
        ret = _acl.init()
        if ret not in (ACL_SUCCESS, ACL_ALREADY_INITIALIZED):
            logger.error("acl.init() failed: %s", ret)
            return False
        ret = _acl_rt.set_device(device_id)
        if ret != ACL_SUCCESS:
            logger.error("acl.rt.set_device(%s) failed: %s", device_id, ret)
            return False
        ctx, ret = _acl_rt.create_context(device_id)
        if ret != ACL_SUCCESS:
            logger.error("acl.rt.create_context(%s) failed: %s", device_id, ret)
            return False
        ret = _acl_rt.set_context(ctx)
        if ret != ACL_SUCCESS:
            logger.error("acl.rt.set_context() failed: %s", ret)
            return False
        _ACL_CONTEXT = ctx
        _ACL_INITIALIZED = True
        logger.info("ACL initialized  device=%s  soc=%s", device_id, _acl.get_soc_name())
        return True


# ---------------------------------------------------------------------------
#  NV12 conversion helpers
# ---------------------------------------------------------------------------
def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR (H,W,3) uint8 numpy array to NV12 (H*3/2, W) uint8."""
    h, w = bgr.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"NV12 requires even width/height, got {w}x{h}")

    b = bgr[..., 0].astype(np.int32)
    g = bgr[..., 1].astype(np.int32)
    r = bgr[..., 2].astype(np.int32)

    y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
    y = np.clip(y, 0, 255).astype(np.uint8)

    b2 = b.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    g2 = g.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    r2 = r.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))

    u_sub = (((-38 * r2 - 74 * g2 + 112 * b2 + 512) >> 10) + 128)
    v_sub = (((112 * r2 - 94 * g2 - 18 * b2 + 512) >> 10) + 128)

    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = np.clip(u_sub, 0, 255).astype(np.uint8)
    uv[:, 1::2] = np.clip(v_sub, 0, 255).astype(np.uint8)
    return np.vstack([y, uv])


# ---------------------------------------------------------------------------
#  Synchronous CANN VENC wrapper
# ---------------------------------------------------------------------------
class CannVenc:
    """Synchronous wrapper around the async CANN VENC callback API."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 30,
        bitrate: Optional[int] = None,  # kbps; VENC unit is kbps
        entype: Optional[int] = None,
        channel_id: int = 10,
    ):
        if not _init_acl():
            raise RuntimeError("CANN ACL initialization failed")

        self.width = width
        self.height = height
        self.fps = fps
        if bitrate is not None and bitrate <= 0:
            raise ValueError(f"bitrate must be positive, got {bitrate}")
        self.bitrate = bitrate or resolve_venc_bitrate_kbps(
            width,
            height,
            fps,
            codec="h264",
        )
        self.entype = _resolve_h264_entype() if entype is None else entype
        self._channel_id = channel_id
        self._channel_desc = None
        self._frame_config = None
        self._callback_tid = None
        self._channel_created = False
        self._ctx = _ACL_CONTEXT
        if self._ctx is None:
            raise RuntimeError("ACL context is not initialized")
        ret = _acl_rt.set_context(self._ctx)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.rt.set_context() failed: {ret}")
        self._running = True

        # Callback synchronization
        self._cb_event = threading.Event()
        self._cb_lock = threading.Lock()
        self._encoded_data: Optional[bytes] = None
        self._encoded_size: int = 0
        self._cb_error: int = 0
        self._cb_queue: queue.Queue = queue.Queue(maxsize=64)

        # Input alignment (CANN VENC requires 16-aligned width for NV12)
        self._align = 16
        self._stride = ((width + self._align - 1) // self._align) * self._align
        self._nv12_size = self._stride * height * 3 // 2
        # Python ACL VENC binding requires a caller-created output stream desc.
        self._out_buf_size = width * height * 3 // 2

        try:
            self._create_channel()
        except Exception:
            self.destroy()
            raise

    def _callback_thread(self, _args):
        _acl_rt.set_context(self._ctx)
        while self._running:
            _acl_rt.process_report(300)

    def _venc_callback(self, input_pic_desc, output_stream_desc, _user_data):
        """Called by CANN when a frame is encoded."""
        try:
            size = _acl_media.dvpp_get_stream_desc_size(output_stream_desc)
            if size > 0:
                data_ptr = _acl_media.dvpp_get_stream_desc_data(output_stream_desc)
                # Copy encoded data from DVPP memory to host
                host_buf, ret = _acl_rt.malloc_host(size)
                if ret == ACL_SUCCESS:
                    _acl_rt.memcpy(host_buf, size, data_ptr, size,
                                   ACL_MEMCPY_DEVICE_TO_HOST)
                    encoded = ctypes_copy_bytes(host_buf, size)
                    _acl_rt.free_host(host_buf)
                    self._cb_queue.put(encoded)
                else:
                    self._cb_queue.put(None)
            else:
                self._cb_queue.put(None)
        except Exception as exc:
            logger.error("VENC callback error: %s", exc)
            try:
                self._cb_queue.put(None)
            except Exception:
                pass
        finally:
            # CANN owns the output stream desc. The caller frees the input DVPP
            # buffer after the callback has been consumed.
            if input_pic_desc is not None:
                _acl_media.dvpp_destroy_pic_desc(input_pic_desc)

    def _create_channel(self):
        self._channel_desc = _acl_media.venc_create_channel_desc()
        if self._channel_desc is None:
            raise RuntimeError("venc_create_channel_desc failed")

        tid, ret = _acl_util.start_thread(self._callback_thread, [])
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.util.start_thread failed: {ret}")
        self._callback_tid = tid

        _check_acl_call("venc_set_channel_desc_thread_id",
                        _acl_media.venc_set_channel_desc_thread_id(self._channel_desc, tid))
        _check_acl_call("venc_set_channel_desc_callback",
                        _acl_media.venc_set_channel_desc_callback(self._channel_desc, self._venc_callback))
        _check_acl_call("venc_set_channel_desc_entype",
                        _acl_media.venc_set_channel_desc_entype(self._channel_desc, self.entype))
        _check_acl_call(
            "venc_set_channel_desc_pic_format",
            _acl_media.venc_set_channel_desc_pic_format(self._channel_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420),
        )
        _check_acl_call("venc_set_channel_desc_pic_width",
                        _acl_media.venc_set_channel_desc_pic_width(self._channel_desc, self.width))
        _check_acl_call("venc_set_channel_desc_pic_height",
                        _acl_media.venc_set_channel_desc_pic_height(self._channel_desc, self.height))
        _check_acl_call(
            "venc_set_channel_desc_key_frame_interval",
            _acl_media.venc_set_channel_desc_key_frame_interval(self._channel_desc, max(self.fps, 1)),
        )
        # ACLLite's CANN 8.0 VENC sample does not set src_rate. Keep it off by
        # default and expose an env switch for boards that need it.
        if _should_set_src_rate():
            _check_acl_call(
                "venc_set_channel_desc_src_rate",
                _acl_media.venc_set_channel_desc_src_rate(self._channel_desc, max(self.fps, 1)),
            )
        _check_acl_call("venc_set_channel_desc_max_bit_rate",
                        _acl_media.venc_set_channel_desc_max_bit_rate(self._channel_desc, self.bitrate))
        _check_acl_call("venc_set_channel_desc_rc_mode",
                        _acl_media.venc_set_channel_desc_rc_mode(self._channel_desc, 2))  # CBR

        ret = _acl_media.venc_create_channel(self._channel_desc)
        if ret != ACL_SUCCESS:
            raise RuntimeError(_format_venc_create_error(
                ret, self.width, self.height, self.fps, self.bitrate, self.entype,
            ))
        self._channel_created = True

        self._frame_config = _acl_media.venc_create_frame_config()
        if self._frame_config is None:
            raise RuntimeError("venc_create_frame_config failed")

        logger.info(
            "CANN VENC channel created  %dx%d@%d  bitrate=%d  entype=%d",
            self.width, self.height, self.fps, self.bitrate, self.entype,
        )

    def encode(self, nv12_data: np.ndarray, force_keyframe: bool = False,
               pre_padded: bool = False) -> bytes:
        """Encode one NV12 frame. Returns Annex-B bitstream bytes.

        Args:
            nv12_data: NV12 numpy array (H*3/2, W) tightly packed, or
                       (stride*H*3/2) pre-padded when pre_padded=True.
            force_keyframe: Force this frame to be an I-frame.
            pre_padded: If True, nv12_data is already stride-aligned (from JPEGD).
        """
        if not self._running:
            raise RuntimeError("VENC channel is closed")

        _acl_rt.set_context(self._ctx)

        h = self.height
        w = self.width
        stride = self._stride

        if pre_padded:
            nv12_padded = nv12_data.ravel()
            # Derive height stride from padded buffer: rows = size / stride_w
            padded_rows = nv12_padded.nbytes // stride
            height_stride = padded_rows * 2 // 3
        else:
            height_stride = h
            # Build padded NV12 for VENC (Y plane padded to stride, UV plane padded to stride)
            nv12_padded = np.zeros(stride * h * 3 // 2, dtype=np.uint8).reshape(-1, stride)
            nv12_src = nv12_data.reshape(-1, w)
            # Y plane
            for row in range(h):
                nv12_padded[row, :w] = nv12_src[row, :w]
            # UV plane
            for row in range(h // 2):
                nv12_padded[h + row, :w] = nv12_src[h + row, :w]
            nv12_padded = nv12_padded.ravel()

        input_size = nv12_padded.nbytes
        input_buffer, ret = _acl_media.dvpp_malloc(input_size)
        if ret != ACL_SUCCESS or input_buffer is None:
            raise RuntimeError(f"dvpp_malloc input failed: {ret}")

        _acl_rt.memcpy(input_buffer, input_size,
                       nv12_padded.ctypes.data, input_size,
                       ACL_MEMCPY_HOST_TO_DEVICE)

        pic_desc = _acl_media.dvpp_create_pic_desc()
        _acl_media.dvpp_set_pic_desc_data(pic_desc, input_buffer)
        _acl_media.dvpp_set_pic_desc_size(pic_desc, input_size)
        _acl_media.dvpp_set_pic_desc_format(pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        _acl_media.dvpp_set_pic_desc_width(pic_desc, self.width)
        _acl_media.dvpp_set_pic_desc_height(pic_desc, self.height)
        _acl_media.dvpp_set_pic_desc_width_stride(pic_desc, self._stride)
        _acl_media.dvpp_set_pic_desc_height_stride(pic_desc, height_stride)

        out_buffer, ret = _acl_media.dvpp_malloc(self._out_buf_size)
        if ret != ACL_SUCCESS or out_buffer is None:
            _acl_media.dvpp_free(input_buffer)
            _acl_media.dvpp_destroy_pic_desc(pic_desc)
            raise RuntimeError(f"dvpp_malloc output failed: {ret}")

        stream_desc = _acl_media.dvpp_create_stream_desc()
        if stream_desc is None:
            _acl_media.dvpp_free(input_buffer)
            _acl_media.dvpp_free(out_buffer)
            _acl_media.dvpp_destroy_pic_desc(pic_desc)
            raise RuntimeError("dvpp_create_stream_desc failed")
        _acl_media.dvpp_set_stream_desc_data(stream_desc, out_buffer)
        _acl_media.dvpp_set_stream_desc_size(stream_desc, self._out_buf_size)

        if force_keyframe:
            _acl_media.venc_set_frame_config_force_i_frame(self._frame_config, True)

        # Drain callback queue before sending; leftover data indicates a
        # previous consume failure; log it so silent frame loss is visible.
        drained = 0
        while not self._cb_queue.empty():
            try:
                self._cb_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.warning("VENC callback queue drained %d leftover frame(s)", drained)

        ret = _acl_media.venc_send_frame(self._channel_desc, pic_desc,
                                          stream_desc, self._frame_config, None)
        if ret != ACL_SUCCESS:
            _acl_media.dvpp_free(input_buffer)
            _acl_media.dvpp_free(out_buffer)
            _acl_media.dvpp_destroy_pic_desc(pic_desc)
            _acl_media.dvpp_destroy_stream_desc(stream_desc)
            raise RuntimeError(f"venc_send_frame failed: {ret}")

        try:
            encoded = self._cb_queue.get(timeout=5.0)
        except queue.Empty:
            encoded = None

        if encoded is None:
            logger.debug("VENC produced no output for this frame (encoder buffering)")
            encoded = b""

        # Cleanup
        _acl_media.dvpp_free(input_buffer)
        _acl_media.dvpp_free(out_buffer)
        _acl_media.dvpp_destroy_stream_desc(stream_desc)
        # pic_desc is destroyed in callback

        if force_keyframe:
            _acl_media.venc_set_frame_config_force_i_frame(self._frame_config, False)

        return encoded or b""

    def destroy(self):
        self._running = False
        if self._channel_desc is not None:
            channel_desc = self._channel_desc
            self._channel_desc = None
            if self._channel_created:
                ret = _acl_media.venc_destroy_channel(channel_desc)
                if not _is_success(ret):
                    logger.warning("venc_destroy_channel failed: %s", ret)
                self._channel_created = False
            if hasattr(_acl_media, "venc_destroy_channel_desc"):
                ret = _acl_media.venc_destroy_channel_desc(channel_desc)
                if not _is_success(ret):
                    logger.warning("venc_destroy_channel_desc failed: %s", ret)
        if self._callback_tid is not None:
            stop_thread = getattr(_acl_util, "stop_thread", None)
            if stop_thread is not None:
                ret = stop_thread(self._callback_tid)
                if not _is_success(ret):
                    logger.warning("acl.util.stop_thread failed: %s", ret)
            self._callback_tid = None
        if self._frame_config is not None:
            ret = _acl_media.venc_destroy_frame_config(self._frame_config)
            if not _is_success(ret):
                logger.warning("venc_destroy_frame_config failed: %s", ret)
            self._frame_config = None
        logger.info("CANN VENC channel destroyed")


def probe_cann_venc(width: int, height: int, fps: int = 30, bitrate: Optional[int] = None) -> None:
    """Create and destroy one VENC channel to fail early before WebRTC answer."""
    venc = CannVenc(width=width, height=height, fps=fps, bitrate=bitrate)
    venc.destroy()


def ctypes_copy_bytes(ptr, size):
    """Copy bytes from a ctypes pointer to a Python bytes object."""
    import ctypes
    return ctypes.string_at(ptr, size)


# ---------------------------------------------------------------------------
#  aiortc-compatible H264 encoder using CANN VENC
# ---------------------------------------------------------------------------
class CannH264Encoder(H264Encoder):
    """aiortc H264 encoder backed by CANN VENC hardware.

    Replaces libx264 encoding with Ascend 310B VENC.
    Inherits _packetize / pack / _split_bitstream from H264Encoder.
    """

    def __init__(self):
        self._target_bitrate_bps: int = 1_000_000
        super().__init__()
        self._venc: Optional[CannVenc] = None
        self._hardware_disabled_reason: str = ""
        self._last_width: int = 0
        self._last_height: int = 0
        self._last_fps: int = 0
        self._last_bitrate: int = 0
        self._last_timestamp_sec: Optional[float] = None
        self._perf_log_count: int = 0

    def close(self) -> None:
        if self._venc is not None:
            self._venc.destroy()
            self._venc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def target_bitrate(self) -> int:
        return self._target_bitrate_bps or 1_000_000

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        self._target_bitrate_bps = max(0, int(bitrate))

    def _estimate_fps(self, frame: av.VideoFrame) -> int:
        if frame.pts is None or frame.time_base is None:
            return self._last_fps or 30

        timestamp_sec = float(frame.pts * frame.time_base)
        if self._last_timestamp_sec is None:
            self._last_timestamp_sec = timestamp_sec
            return self._last_fps or 30

        delta = timestamp_sec - self._last_timestamp_sec
        self._last_timestamp_sec = timestamp_sec
        if delta <= 0:
            return self._last_fps or 30

        fps = round(1.0 / delta)
        return max(1, min(fps, 120))

    def _ensure_venc(self, width: int, height: int, fps: int):
        session_bitrate_kbps = get_session_bitrate_override_kbps()
        if session_bitrate_kbps is not None:
            bitrate = session_bitrate_kbps
        else:
            bitrate = _resolve_venc_bitrate_kbps(
                width=width,
                height=height,
                fps=fps,
                codec="h264",
                target_bitrate_bps=self.target_bitrate,
            )
        if (self._venc is not None
                and self._last_width == width
                and self._last_height == height
                and self._last_fps == fps
                and self._last_bitrate == bitrate):
            return
        if self._venc is not None:
            self._venc.destroy()
        self._venc = CannVenc(width=width, height=height, fps=fps, bitrate=bitrate)
        _notify_encoder_status("cann-venc-h264", True)
        self._last_width = width
        self._last_height = height
        self._last_fps = fps
        self._last_bitrate = bitrate
        self.buffer_data = b""
        self.buffer_pts = None

    def _encode_frame_cpu(
        self,
        frame: av.VideoFrame,
        force_keyframe: bool,
    ) -> Iterator[bytes]:
        """Use aiortc's CPU H.264 encoder after CANN VENC is unavailable.

        With aiortc 1.5 + PyAV 10 on the 310B image, a libx264 codec context
        can report bit_rate=None on the next frame. aiortc's bitrate-change
        check divides by that field, so reset the codec before delegating.
        """
        codec = getattr(self, "codec", None)
        if codec is not None and getattr(codec, "bit_rate", None) in (None, 0):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None
        yield from super()._encode_frame(frame, force_keyframe)

    def _encode_frame(
        self, frame: av.VideoFrame, force_keyframe: bool
    ) -> Iterator[bytes]:
        encoder_mode = get_session_encoder_mode()
        if encoder_mode == "cpu":
            _notify_encoder_status("cpu-libx264-stable", False)
            yield from self._encode_frame_cpu(frame, force_keyframe)
            return
        if not _CANN_READY:
            reason = "CANN ACL is not available for VENC"
            _notify_encoder_status("cann-venc-failed", False, reason)
            raise RuntimeError(reason)
        if self._hardware_disabled_reason:
            _notify_encoder_status("cann-venc-failed", False, self._hardware_disabled_reason)
            raise RuntimeError(self._hardware_disabled_reason)

        fps = self._estimate_fps(frame)
        try:
            self._ensure_venc(frame.width, frame.height, fps=fps)
        except RuntimeError as exc:
            logger.error("CANN VENC initialization failed: %s", exc)
            self.close()
            self._hardware_disabled_reason = str(exc)
            _notify_encoder_status("cann-venc-failed", False, self._hardware_disabled_reason)
            raise

        # NV12 passthrough: if the track already prepared NV12, skip PyAV's
        # expensive RGB/BGR -> NV12 reformat step. VENC still pads rows when
        # the source width is not 16-aligned.
        if getattr(frame.format, "name", None) == "nv12":
            t0 = time.perf_counter()
            nv12 = frame.to_ndarray(format="nv12")
            pre_padded = frame.width % 16 == 0
            convert_ms = (time.perf_counter() - t0) * 1000
            if self._perf_log_count < 5:
                logger.info(
                    "VENC input ndarray frame=%d format=nv12 pre_padded=%s ndarray_ms=%.1f",
                    self._perf_log_count + 1,
                    pre_padded,
                    convert_ms,
                )
        else:
            t0 = time.perf_counter()
            nv12_frame = frame.reformat(format="nv12")
            t1 = time.perf_counter()
            nv12 = nv12_frame.to_ndarray(format="nv12")
            pre_padded = False
            convert_ms = (time.perf_counter() - t1) * 1000
            if self._perf_log_count < 5:
                logger.info(
                    "VENC input convert frame=%d reformat_ms=%.1f ndarray_ms=%.1f",
                    self._perf_log_count + 1,
                    (t1 - t0) * 1000,
                    convert_ms,
                )

        try:
            t0 = time.perf_counter()
            encoded = self._venc.encode(nv12, force_keyframe=force_keyframe, pre_padded=pre_padded)
            encode_ms = (time.perf_counter() - t0) * 1000
        except RuntimeError as exc:
            logger.error("CANN VENC encode failed: %s", exc)
            self.close()
            self._hardware_disabled_reason = str(exc)
            _notify_encoder_status("cann-venc-failed", False, self._hardware_disabled_reason)
            raise

        if self._perf_log_count < 5:
            logger.info(
                "VENC encode frame=%d size=%dx%d fps=%d pre_padded=%s encode_ms=%.1f bytes=%d",
                self._perf_log_count + 1,
                frame.width,
                frame.height,
                fps,
                pre_padded,
                encode_ms,
                len(encoded),
            )
            self._perf_log_count += 1

        if encoded:
            yield from self._split_bitstream(encoded)
