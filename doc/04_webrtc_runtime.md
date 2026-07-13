# WebRTC Runtime Design

This document records the WebRTC runtime path for the Ascend 310B hand
pipeline. It separates camera capture, DVPP/VENC, WebRTC transport, and the
two-stage hand OM inference path.

## 1. Reused Components

The runtime reuses generic transport, capture, and hardware-encoding modules
from the earlier `case8` sample. It does not reuse the YOLO detector or YOLO
post-processing code.

| Target path | Origin | Role |
| --- | --- | --- |
| `webrtc_app/cann_encoder.py` | `case8/webrtc_app/cann_encoder.py` plus board-side ACLLite VENC sample | Replaces aiortc H.264 encoding with CANN VENC. |
| `webrtc_app/dvpp_jpegd.py` | `case8/webrtc_app/dvpp_jpegd.py` | Decodes MJPEG camera frames to NV12 with DVPP JPEGD. |
| `webrtc_app/v4l2_capture.py` | `case8/webrtc_app/v4l2_capture.py` | Reads V4L2 MJPEG through PyAV. |
| `webrtc_app/v4l2_raw.py` | `case8/webrtc_app/v4l2_raw.py` | Reads V4L2 MJPEG directly through ioctl and mmap. |

The hand inference entry point is this repository's own pipeline:

```text
camera frame
  -> OpenCV BGR or V4L2 MJPEG + DVPP JPEGD
  -> MediaPipe-style palm ImageToTensor sampling
  -> palm detector OM
  -> palm decode and weighted NMS
  -> rotated hand ROI
  -> hand landmark OM
  -> 21-point projection back to the source image
  -> overlay
  -> CPU libx264 or CANN VENC H.264
  -> browser playback
```

## 2. Runtime Entry Point

The main WebRTC entry point is:

```bash
python scripts/webrtc_hand_om_app.py
```

It supports:

- selecting detector and landmark OM files;
- `tracking` and `image` pipeline modes;
- CPU libx264 or CANN VENC H.264 encoding;
- OpenCV, V4L2, and DVPP JPEGD capture paths;
- runtime controls through the browser UI.

## 3. Default Models

The default full deployment path uses:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

The first model is the optimized full palm detector OM. The second model is the
legacy full hand landmark OM.

## 4. Board Dependencies

The board-side Python environment must be able to import:

```text
acl
numpy
cv2
aiortc
av
```

The CANN environment must be loaded before OM inference or CANN VENC/DVPP use.
Typical board setup:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
```

WebRTC does not call ATC, but ACL OM inference, DVPP, and VENC all depend on
the CANN runtime.

## 5. OpenSSL Note

Some board images have a broken conda `base` OpenSSL runtime. If importing
Python `ssl` fails before the WebRTC application starts, fix the conda OpenSSL
installation first. Do not use `LD_PRELOAD` as the production startup path; it
can mask the issue and destabilize native WebRTC dependencies.

## 6. Startup Example

On an Ascend 310B board:

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base

python scripts/webrtc_hand_om_app.py \
  --host 0.0.0.0 \
  --port 8080 \
  --pipeline-mode tracking \
  --encoder-mode cpu
```

Open the printed URL in a browser. For LAN debugging the default ICE
configuration uses no public STUN server. If cross-NAT access is required,
configure STUN/TURN explicitly.

## 7. Common Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--detector` | optimized full palm OM | Palm detector OM path. |
| `--landmark` | legacy full landmark OM | Hand landmark OM path. |
| `--score-threshold` | `0.5` | Palm score threshold after sigmoid. |
| `--nms-iou` | `0.3` | Weighted NMS IoU threshold. |
| `--max-hands` | `2` | Maximum hands per frame. |
| `--min-hand-score` | `0.5` | Landmark hand-presence threshold. |
| `--pipeline-mode` | `tracking` | `tracking` reuses previous ROIs; `image` runs palm detection every frame. |
| `--infer-every-n` | `1` | Run inference every N frames and reuse the last result otherwise. |
| `--encoder-mode` | `cpu` | `cpu` uses libx264; `cann` uses CANN VENC H.264. |
| `--ice-servers` | empty | LAN-only ICE by default. |
| `--reload-detector-each-call` | disabled | Debug option for detector runtime reuse issues. |

## 8. CANN VENC Boundary

If `CANN VENC H.264` fails, the runtime reports the error and does not silently
fall back to CPU encoding. This avoids misreading CPU-encoded streams as
hardware-encoded performance.

Past board testing narrowed CANN VENC failures to the CANN 8.0 VENC driver or
protected DVPP memory mapping path, not to the browser, H.264 negotiation,
camera capture, CPU streaming, or OM inference. Repeated VENC creation failures
can temporarily increase NPU memory usage, so avoid looped probing after a
driver-side failure.

## 9. Current Boundary

- The WebRTC app is a real-time demo and board integration tool. It does not
  replace offline dataset validation.
- The Python ACL runner uses persistent buffers and is suitable for interactive
  testing.
- Hardware H.264 failures require explicit diagnosis; CPU fallback must be
  selected explicitly.
- For production, the OM runner and post-processing path should eventually move
  to C++/ACL while keeping this browser UI as a debugging entry point.
