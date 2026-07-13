# Scripts

`scripts/` contains Python entry points needed on the Ascend 310B deployment board or needed to reproduce the deployed OM models.

Before running board-side scripts:

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Realtime WebRTC App

Before starting WebRTC, verify that conda's Python `ssl` and `aiortc` imports
work. If this fails with `_ssl ... undefined symbol: X509_get_version`, repair
the conda OpenSSL files as documented in `doc/04_webrtc_runtime.md`.

```bash
python - <<'PY'
import ssl
print(ssl.OPENSSL_VERSION)
import aiortc
print(aiortc.__version__)
PY
```

```bash
python scripts/webrtc_hand_om_app.py \
  --source /dev/video0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --camera-backend opencv \
  --camera-fourcc MJPG \
  --encoder-mode cpu \
  --pipeline-mode tracking \
  --port 8080
```

The server uses LAN-only ICE by default (`--ice-servers ""`). Keep this default
for board-side testing on the same LAN. The 20T base environment has reproduced
native `Illegal instruction` crashes in aiortc's public STUN path; configure
STUN/TURN explicitly only when NAT traversal is required.

The default model pair is:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

## Dataset Download

Download PianoVAM v1 into the repository `data/` directory with the Hugging Face
CLI:

```bash
hf download PianoVAM/PianoVAM_v1 \
  --repo-type dataset \
  --local-dir data/PianoVAM_v1
```

Use the HF mirror when needed:

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download PianoVAM/PianoVAM_v1 \
  --repo-type dataset \
  --local-dir data/PianoVAM_v1
```

On Windows PowerShell:

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
hf download PianoVAM/PianoVAM_v1 `
  --repo-type dataset `
  --local-dir data/PianoVAM_v1
```

If the download fails with `cas-server.xethub.hf.co` or `401 Unauthorized`,
disable the Xet/CAS backend and rerun the same command:

```powershell
$env:HF_HUB_DISABLE_XET="1"
hf download PianoVAM/PianoVAM_v1 `
  --repo-type dataset `
  --local-dir data/PianoVAM_v1
```

Use include patterns for a partial download:

```bash
hf download PianoVAM/PianoVAM_v1 \
  --repo-type dataset \
  --include "*.json" "*.txt" "*.md" \
  --local-dir data/PianoVAM_v1
```

## ONNX/OM Validation

Raw tensor-level comparison:

```bash
python scripts/compare_onnx_om_raw.py \
  --onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --om models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om \
  --shape 1,192,192,3 \
  --samples 100 \
  --warmup 2 \
  --output-dir runs/onnx_om_raw_compare/lite_palm_fullstyle_origin_dtype_persistent_100samples
```

Use `--reload-om-each-sample` only when checking a suspect palm detector OM for model-handle reuse drift.

Video-level ONNX/OM comparison:

```bash
python scripts/run_video_onnx_om_checks.py --video data/eval_videos/demo1.mp4
```

Video validation defaults to MediaPipe-style tracking. Add `--pipeline-mode image`
to force palm detection on every processed frame.

Quick smoke test:

```bash
python scripts/run_video_onnx_om_checks.py --video data/eval_videos/demo1.mp4 --max-frames 5 --save-vis 2
```

Generate MediaPipe legacy graph annotations for a video evaluation asset on the
PC:

```bash
python scripts/annotate_mediapipe_video.py \
  --video data/eval_videos/demo1.mp4 \
  --save-vis 16 \
  --save-video
```

The annotation script directly subscribes to MediaPipe graph streams including
`palm_detections`, `hand_rects_from_palm_detections`, `multi_hand_landmarks`,
and `hand_rects_from_landmarks`. It writes results under
`data/eval_videos/annotations/<video_stem>/` by default. For
`data/eval_videos/demo1.mp4`, the output folder is
`data/eval_videos/annotations/demo1/`. Use `--force` only when intentionally
regenerating an existing annotation folder.

Evaluate the Ascend OM video pipeline against the MediaPipe reference answer:

```bash
python scripts/eval_video_mediapipe_om.py \
  --video data/eval_videos/demo1.mp4 \
  --pipeline-mode tracking \
  --model-set full \
  --save-vis 8
```

Use `--pipeline-mode image` to evaluate frame-by-frame palm detection against
the MediaPipe image reference stream.

Tracking mode defaults to the pure MediaPipe-style loop with float32 ROI and
projection math. The optional `--tracking-rect-smooth-alpha` and
`--max-tracking-*` arguments are debugging/robustness controls; they are not
enabled by default and should not be treated as strict MediaPipe graph behavior.

For a fully explicit pair:

```bash
python scripts/compare_video_onnx_om.py \
  --video data/eval_videos/demo1.mp4 \
  --onnx-detector models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx \
  --onnx-landmark models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx \
  --om-detector models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om \
  --om-landmark models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om \
  --output-dir runs/video_onnx_om_compare/original_onnx_vs_optimized_om \
  --pipeline-mode tracking \
  --reload-detector-each-frame
```

Dataset-level OM evaluation on the portable HaGRIDv2 MediaPipe test split:

```bash
python scripts/eval_hf_hand_dataset_om.py \
  --model-set full \
  --max-images 20 \
  --save-vis 2
```

Dataset evaluation always uses independent image mode because the test split is
not a continuous video stream.

Full `1663` image validation for the deployed full OM and accepted lite OM:

```bash
python scripts/eval_hf_hand_dataset_om.py --model-set full,lite
```

The dataset evaluator now writes cascade and single-model results in the same
run by default:

```bash
python scripts/eval_hf_hand_dataset_om.py --model-set full,lite --components cascade,palm,landmark
```

Use targeted component runs when isolating a model:

```bash
python scripts/eval_hf_hand_dataset_om.py --model-set full --components palm
python scripts/eval_hf_hand_dataset_om.py --model-set full --components landmark
```

Use `--run-onnx` to also compute ONNX reference metrics and OM-vs-ONNX
differences. Reports are written under `runs/hf_hand_dataset_om/`.

## OM Inference Latency

```bash
python scripts/benchmark_om_inference.py \
  --warmup 20 \
  --iterations 200
```

Benchmark a specific OM:

```bash
python scripts/benchmark_om_inference.py \
  --model models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om \
  --warmup 20 \
  --iterations 200
```

## Rebuild Optimized Palm ONNX/OM

Regenerate the optimized full palm ONNX from the original full ONNX:

```bash
python scripts/build_optimized_palm_om.py \
  --input-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
```

Rebuild the production full palm OM on the board:

```bash
python scripts/build_optimized_palm_om.py \
  --skip-rewrite \
  --compile-om \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx \
  --om-output models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype \
  --atc-log runs/palm_om/atc_logs/downsample_resize_maxpool_slices_origin_dtype.log \
  --precision-mode must_keep_origin_dtype
```

Generate the optimized lite palm ONNX:

```bash
python scripts/build_optimized_palm_om.py \
  --input-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --downsample-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample.onnx \
  --resize-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize.onnx \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

Verify the optimized lite ONNX still matches the original lite ONNX:

```bash
python scripts/compare_onnx_raw.py \
  --left models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --right models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx \
  --shape 1,192,192,3 \
  --samples 10 \
  --output-dir runs/onnx_raw_compare/lite_palm_original_vs_full_style_optimized
```

Build the 8T optimized lite palm OM. `run_clean_atc.py` does not pass graph-parallel ATC flags by default:

```bash
python scripts/run_clean_atc.py \
  --model models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx \
  --output models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype \
  --log runs/atc_8t/logs/lite_palm_fullstyle_origin_dtype.log \
  --report runs/atc_8t/lite_palm_fullstyle_origin_dtype.json \
  --soc-version Ascend310B4 \
  --precision-mode must_keep_origin_dtype \
  --env-mode python_runtime \
  --cache-mode force
```

## 20T/8T Benchmark Rebuilds

Compile the deployed full model set for the runtime SoC:

```bash
python scripts/build_20t_om_models.py --soc-version auto --model-set deployed_full
```

If a board-specific rebuild produces raw outputs identical to the existing OM,
do not keep a hardware-suffixed duplicate model in `models/om/`.

## CANN VENC Status

Read-only status check:

```bash
python scripts/check_venc_runtime.py
```

Creating a VENC channel is guarded because failed CANN 8.0 VENC attempts can leave driver-side memory pressure:

```bash
python scripts/check_venc_runtime.py \
  --join-usermemory \
  --probe \
  --i-understand-venc-probe-risk \
  --width 640 \
  --height 480 \
  --fps 30 \
  --bitrate-kbps 4000
```
