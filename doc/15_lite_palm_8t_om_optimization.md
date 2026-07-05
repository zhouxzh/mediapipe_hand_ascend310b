# Lite Palm 8T OM Optimization Record

Date: 2026-07-05

This document is the final record for the lite palm OM optimization work. It keeps the useful findings and reproduction commands after temporary debug scripts and intermediate `runs/` outputs are removed from the deployment repository.

## Objective

Create an Ascend 310B OM model for `mediapipe_legacy_0_10_14_palm_detection_lite.onnx` whose raw outputs stay close to the original ONNX model.

The accepted result is the 8T `must_keep_origin_dtype` OM. Its average relative error is below `1%`; the boxes p95 relative metric is slightly above `1%`, but the user accepted this as close enough for the objective.

## Board And Toolchain

| Item | Value |
| --- | --- |
| board | `ascend8t`, Orange Pi AI Pro 8T |
| SoC | `Ascend310B4` |
| CANN | `/usr/local/Ascend/ascend-toolkit/latest`, ATC 8.3 RC1 environment |
| Python | `/usr/local/miniconda3/bin/python` |
| CANN note | do not use `--enable_graph_parallel` / `--ac_parallel_enable`; this ATC build rejects `--enable_graph_parallel` |

## Model Path

Original ONNX:

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx
```

Optimized ATC input ONNX:

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

Accepted OM:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype_ascend310b4_singlethread.om
```

| Field | Value |
| --- | --- |
| size | `10930367` bytes |
| SHA256 | `ec2a1fb352d8c782cf71d7bc020fbe41f4f9b0d85474ecc51a49f74224ef88a1` |
| precision | `must_keep_origin_dtype` |
| ATC log | historical path before cleanup: `runs/atc_8t/logs/lite_palm_fullstyle_origin_dtype_ascend310b4_singlethread.log` |
| raw compare report | historical path before cleanup: `runs/onnx_om_raw_compare/lite_palm_fullstyle_ascend310b4_singlethread_persistent_100samples/summary.json` |
| benchmark report | historical path before cleanup: `runs/om_inference_benchmark/om_benchmark_20260705_170921.json` |

## ONNX Rewrites

The lite palm model uses the same sensitive structures as full palm: downsample residual tail padding, half-pixel bilinear resize, and fixed MaxPool. The accepted lite ATC input is produced by the same full-style rewrite chain:

```text
original lite ONNX
  -> downsample residual Pad/Add rewrite
  -> bilinear half-pixel Resize rewrite
  -> MaxPool to Slice+Max rewrite
  -> optimized lite ONNX
```

Reproduction command:

```bash
python scripts/build_optimized_palm_om.py \
  --input-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --downsample-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample.onnx \
  --resize-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize.onnx \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

The optimized lite ONNX is equivalent to the original lite ONNX at raw-output level:

| output | max_abs | mean_abs | p95_abs |
| ---: | ---: | ---: | ---: |
| `0` boxes/keypoints | `5.340576e-05` | `3.211731e-06` | `1.001358e-05` |
| `1` scores | `8.583069e-06` | `1.155112e-06` | `2.861023e-06` |

## ATC Command

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/Documents/mediapipe_hand_ascend310b

python scripts/run_clean_atc.py \
  --model models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx \
  --output models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype_ascend310b4_singlethread \
  --log runs/atc_8t/logs/lite_palm_fullstyle_origin_dtype_ascend310b4_singlethread.log \
  --report runs/atc_8t/lite_palm_fullstyle_origin_dtype_ascend310b4_singlethread.json \
  --soc-version Ascend310B4 \
  --precision-mode must_keep_origin_dtype \
  --env-mode python_runtime \
  --cache-mode force
```

`run_clean_atc.py` sets low-pressure ATC environment variables and does not pass graph-parallel ATC flags by default.

## Raw Output Accuracy

Command:

```bash
python scripts/compare_onnx_om_raw.py \
  --onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --om models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype_ascend310b4_singlethread.om \
  --shape 1,192,192,3 \
  --samples 100 \
  --warmup 2 \
  --output-dir runs/onnx_om_raw_compare/lite_palm_fullstyle_ascend310b4_singlethread_persistent_100samples
```

100 random inputs:

| output | shape | max_abs max | mean_abs mean | p95_abs max | mean_rel mean | p95_rel mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| boxes/keypoints | `[1,2016,18]` | `0.0826664` | `0.00392648` | `0.0155512` | `0.829947%` | `1.046142%` |
| scores | `[1,2016,1]` | `0.0229282` | `0.00116983` | `0.00409853` | `0.014741%` | `0.040398%` |

20 random inputs showed identical results for persistent model reuse and fresh-load per sample, so this optimized OM does not have the direct-lite OM handle reuse drift.

## Additional ATC Precision Attempts

| ATC mode | Result |
| --- | --- |
| direct lite palm OM | Fast but raw output is wrong; not deployable |
| optimized lite ONNX + `must_keep_origin_dtype` | accepted best candidate |
| optimized lite ONNX + `force_fp32` | compiled, but worse: boxes mean_rel `1.601705%`, boxes p95_rel `2.108870%` |
| optimized lite ONNX + `precision_mode_v2=origin` | compiled, but raw-output error was identical to `must_keep_origin_dtype` |
| split stage1/stage2 ONNX | ONNX-equivalent locally, but ATC failed on 20T; diagnostic route only |
| output affine calibration | reduced direct lite error but did not meet target; diagnostic route only |

## Latency

8T, `warmup=20`, `iterations=200`:

| model | execute mean | execute p95 | h2d+execute mean | full mean | full p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| optimized lite palm OM | `24.799 ms` | `24.882 ms` | `24.907 ms` | `25.600 ms` | `25.709 ms` |
| optimized full palm OM | `26.931 ms` | `27.010 ms` | `27.018 ms` | `27.738 ms` | `27.847 ms` |
| direct lite palm OM, not accurate | `3.358 ms` | `3.386 ms` | `3.465 ms` | `4.053 ms` | `4.093 ms` |

The optimized lite OM is only about `7.9%` faster than optimized full palm on 8T. The direct lite OM is much faster but not numerically valid.

## Final Decision

- The accepted lite candidate is `mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype_ascend310b4_singlethread.om`.
- Use optimized full palm as the default production detector unless lite is specifically requested.
- Keep only the full-style lite optimized ONNX and accepted OM for deployment reproduction.
- Remove split/debug/calibration scripts and `runs/` outputs from the deployment package; this document preserves their conclusions.

