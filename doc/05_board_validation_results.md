# Board Validation Results

This document summarizes the current Ascend 310B validation status for the
MediaPipe hand pipeline. It records the deployment recommendation and the
important known results rather than every one-off debug run.

## 1. Boards

| Host | Board | Reported SoC | Notes |
| --- | --- | --- | --- |
| `ascend8t` | Orange Pi AI Pro 8T | Ascend310B4 | Used for full/lite OM validation and WebRTC checks. |
| `ascend20t` | Orange Pi AI Pro 20T | Ascend310B1 | Used for full/lite OM validation, OM rebuild checks, and runtime comparison. |

## 2. Default Deployment Models

| Component | Default OM |
| --- | --- |
| palm detector | `models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om` |
| hand landmark | `models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om` |

The default full palm OM is built from the optimized full palm ONNX with downsample/resize/maxpool-slices preprocessing. The default landmark OM is the legacy full hand landmark model.

## 3. Validation Policy

Formal acceptance considers:

- palm detector AP/recall on the palm validation set;
- 21-point landmark error on the landmark validation set;
- end-to-end full pipeline behavior;
- board-side runtime speed;
- video tracking regression behavior when the task is video-specific.

Lite models are report-only candidates. They may be benchmarked, but they are not the default acceptance path.

## 4. Dataset Validation Results

The formal image-dataset check uses
`data/portable-hagridv2-mediapipe-hand/test-00000.parquet` with `1663` images
and `1664` hands.

| Board | Model set | Palm OM class | Enforced | Passed | Recall | AP50 | Full21 mean px | Full21 P95 px | Total mean ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ascend20t` | full | current default `origin_dtype` | yes | yes | 0.997596 | 0.994933 | 0.1278 | 0.2477 | 22.9441 |
| `ascend20t` | lite | current lite `origin_dtype` | no | report-only | 0.974760 | 0.983212 | 1.3184 | 2.4846 | 21.4419 |
| `ascend8t` | full | current default `origin_dtype` | yes | yes | 0.997596 | 0.994933 | 0.1278 | 0.2477 | 39.7796 |
| `ascend8t` | lite | current lite `origin_dtype` | no | report-only | 0.974760 | 0.983212 | 1.3184 | 2.4857 | 36.2475 |
| `ascend20t` | full | mix-precision candidate | yes | yes | 0.997596 | 0.994933 | 0.1286 | 0.2535 | 17.9832 |
| `ascend8t` | full | mix-precision candidate | yes | yes | 0.997596 | 0.994933 | 0.1286 | 0.2535 | 29.0740 |

The 20T current-default rows come from
`runs/fp16_om/hf_eval_20t_baseline/20260707_192632`. The 8T current-default
rows come from `runs/hf_hand_dataset_om/20260705_220608`. The 20T
mix-precision candidate row comes from
`runs/fp16_om/hf_eval_20t_recommended/20260707_192948`. The 8T mix-precision
candidate row comes from
`runs/fp16_om/hf_eval_8t_mix_full_clean/20260707_221804`.

The mix-precision full palm OM is a speed comparison candidate. It has not been
promoted to the repository-wide default because the code defaults still point to
the `origin_dtype` OM. Promotion should update code defaults, this document,
[03_models.md](03_models.md), and [04_webrtc_runtime.md](04_webrtc_runtime.md)
together.

## 5. Palm And Landmark Component Results

The 20T current-default full pipeline has the following component-level results:

| Component | Model | Key result | Runtime |
| --- | --- | --- | ---: |
| palm detector | full `origin_dtype` | recall `1.000000`, AP50 `0.994206`, palm7 mean `0.2152 px` | detector mean `12.3195 ms` |
| hand landmark | full | passed full21 mean `0.0638 px`, passed full21 P95 `0.1006 px` | landmark mean `2.6616 ms` |
| palm detector | lite `origin_dtype` | recall `0.975361`, AP50 `0.982114`, palm7 mean `5.7885 px` | detector mean `11.4120 ms` |
| hand landmark | lite | passed full21 mean `1.2215 px`, passed full21 P95 `2.2706 px` | landmark mean `2.1594 ms` |

The lite path is faster only in parts of the pipeline and loses substantial
landmark accuracy, so it remains report-only.

## 6. Current Conclusions

- The `origin_dtype` full optimized palm OM plus full landmark OM is the current
  default.
- The direct palm OM path was rejected because raw outputs and end-to-end behavior did not meet accuracy requirements.
- Lite palm/landmark combinations can run but remain report-only.
- The full OM rebuilt on the 20T board was numerically identical to the current default full OM output, so a separate 20T-specific OM is not kept.
- Mix-precision palm OM results are promising for speed, but they are still
  documented as a candidate until the deployment default is changed consistently.
- Board-side results should always identify the board, SoC, model pair, and pipeline mode.

## 7. Rebuild and Revalidation Notes

When rebuilding OM files:

1. Run ATC on the target board or in a matching CANN environment.
2. Capture raw model outputs for fixed inputs.
3. Compare raw outputs against the reference ONNX/OM path.
4. Run the end-to-end dataset evaluation before promoting the file.
5. Do not keep duplicate OM files when raw outputs are identical.

Typical full palm ATC input:

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
```

Typical full landmark ATC input:

```text
models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx
```

## 8. Related Reports

- AMCT INT8 PTQ details are in [07_amct_int8_quantization_8t_20t_results.md](07_amct_int8_quantization_8t_20t_results.md).
- Tracking behavior and ROI state-machine details are in [06_tracking_algorithm.md](06_tracking_algorithm.md).
- PianoVAM video tracking results are in [08_pianovam_video_tracking_performance.md](08_pianovam_video_tracking_performance.md) and [09_pianovam_mediapipe_tasks_gpu_tracking.md](09_pianovam_mediapipe_tasks_gpu_tracking.md).
