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

```bash
python scripts/webrtc_hand_om_app.py \
  --source /dev/video0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --camera-backend opencv \
  --camera-fourcc MJPG \
  --encoder-mode cpu \
  --port 8080
```

The default model pair is:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
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
python scripts/run_video_onnx_om_checks.py --video video/test.mp4
```

Quick smoke test:

```bash
python scripts/run_video_onnx_om_checks.py --video video/test.mp4 --max-frames 5 --save-vis 2
```

For a fully explicit pair:

```bash
python scripts/compare_video_onnx_om.py \
  --video video/test.mp4 \
  --onnx-detector models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx \
  --onnx-landmark models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx \
  --om-detector models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om \
  --om-landmark models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om \
  --output-dir runs/video_onnx_om_compare/original_onnx_vs_optimized_om \
  --reload-detector-each-frame
```

Dataset-level OM evaluation on the portable HaGRIDv2 MediaPipe test split:

```bash
python scripts/eval_hf_hand_dataset_om.py \
  --model-set full \
  --max-images 20 \
  --save-vis 2
```

Full `1663` image validation for the deployed full OM and accepted lite OM:

```bash
python scripts/eval_hf_hand_dataset_om.py --model-set full,lite
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
