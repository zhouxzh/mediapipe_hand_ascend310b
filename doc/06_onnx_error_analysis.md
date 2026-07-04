# ONNX 模型误差分析

本文件只讨论 TFLite -> ONNX 阶段，包括 ONNX 导出口径、ONNX raw-output 对齐，以及 ONNX 在 `palm_datasets/test` 上的端到端两阶段验证。TFLite baseline 见 [TFLite 模型误差分析](04_tflite_error_analysis.md)，Ascend 310B OM 误差见 [OM 模型误差分析](07_om_error_analysis.md)。

当前数值来自：

```text
models/onnx/export_report_all.json
runs/onnx_two_stage/summary.md
runs/onnx_two_stage/{legacy_full,legacy_lite,task_full}/summary.json
```

## 1. 导出口径

本工程使用 `tf2onnx` 将 TFLite 模型导出为 ONNX：

```text
opset = 11
input = input_1
layout = NHWC
```

导出后会移除 ATC 不需要的 `ai.onnx.ml` opset，只保留默认 domain 的 opset 11。因此当前 ONNX 模型不是 opset 16，而是 opset 11。

当前 6 个 ONNX 模型：

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| `mediapipe_legacy_0_10_14_palm_detection_full.onnx` | `1,192,192,3` | `[1,2016,18]`, `[1,2016,1]` |
| `mediapipe_legacy_0_10_14_hand_landmark_full.onnx` | `1,224,224,3` | `[1,63]`, `[1,1]`, `[1,1]`, `[1,63]` |
| `mediapipe_legacy_0_10_14_palm_detection_lite.onnx` | `1,192,192,3` | `[1,2016,18]`, `[1,2016,1]` |
| `mediapipe_legacy_0_10_14_hand_landmark_lite.onnx` | `1,224,224,3` | `[1,63]`, `[1,1]`, `[1,1]`, `[1,63]` |
| `mediapipe_task_hand_detector_full.onnx` | `1,192,192,3` | `[1,2016,18]`, `[1,2016,1]` |
| `mediapipe_task_hand_landmark_full.onnx` | `1,224,224,3` | `[1,63]`, `[1,1]`, `[1,1]`, `[1,63]` |

## 2. Raw-output 对齐方法

raw-output 对齐使用固定随机输入 tensor：

```text
fixed random tensor
  -> TFLite reference output
  -> ONNX Runtime output
  -> compare each output tensor
```

这个测试只回答一个问题：同一份输入 tensor 进入 TFLite 和 ONNX 后，模型 raw outputs 是否一致。它不包含图像前处理、decode、NMS、ROI crop 或最终 21 点反投影。

## 3. ONNX vs TFLite Raw-output 结果

| 模型 | role | 每个输出 mean_abs / max_abs | 最大 mean_abs | 最大 max_abs |
| --- | --- | --- | ---: | ---: |
| `legacy_full_palm` | detector | `1.27e-05/1.34e-04`, `4.70e-06/2.48e-05` | `1.27e-05` | `1.34e-04` |
| `legacy_full_landmark` | landmark | `2.78e-05/9.16e-05`, `1.96e-08/1.96e-08`, `4.77e-07/4.77e-07`, `9.22e-08/2.41e-07` | `2.78e-05` | `9.16e-05` |
| `legacy_lite_palm` | detector | `1.13e-05/1.16e-04`, `3.44e-06/1.86e-05` | `1.13e-05` | `1.16e-04` |
| `legacy_lite_landmark` | landmark | `2.98e-05/1.22e-04`, `8.38e-09/8.38e-09`, `8.05e-07/8.05e-07`, `5.66e-08/1.92e-07` | `2.98e-05` | `1.22e-04` |
| `task_full_palm` | detector | `1.45e-05/1.56e-04`, `4.91e-06/2.19e-05` | `1.45e-05` | `1.56e-04` |
| `task_full_landmark` | landmark | `2.61e-05/9.16e-05`, `2.33e-09/2.33e-09`, `5.36e-07/5.36e-07`, `7.25e-08/2.80e-07` | `2.61e-05` | `9.16e-05` |

结论：

- TFLite 到 ONNX 的 raw-output 转换误差在 `1e-4` 量级以内。
- detector 和 landmark 的输入输出 shape 保持 NHWC，与 TFLite 模型一致。
- 这个误差量级远小于像素级关键点误差，因此不能解释 OM 端到端约 2px 级误差。

## 4. ONNX 端到端测试方法

ONNX 端到端验证使用 `scripts/eval_two_stage_onnx.py`，链路与 TFLite 两阶段脚本一致，只把 detector 和 landmark 推理后端替换为 ONNX Runtime：

```text
image
  -> detector ONNX
  -> decode_raw_palm
  -> weighted NMS
  -> make_hand_roi
  -> preprocess_landmark_tflite
  -> landmark ONNX
  -> landmarks_to_original
  -> compare with matching TFLite / legacy graph / current Tasks
```

测试数据固定为：

```text
data/palm_datasets/test
images = 1859
```

三组 coherent pipeline：

| 组合 | detector | landmark | TFLite reference |
| --- | --- | --- | --- |
| `legacy_full` | legacy full palm ONNX | legacy full landmark ONNX | `runs/baseline/two_stage_vs_legacy_graph/predictions.json` |
| `legacy_lite` | legacy lite palm ONNX | legacy lite landmark ONNX | `runs/baseline/tflite_matrix/det_legacy_lite__lm_legacy_lite/predictions.json` |
| `task_full` | task full detector ONNX | task full landmark ONNX | `runs/baseline/tflite_matrix/det_task_full__lm_task_full/predictions.json` |

## 5. ONNX 端到端结果

| ONNX 组合 | Hands | Palm P/R | AP50 | mAP | vs 同组 TFLite mean/p95/max | PCK@0.05 | vs legacy graph mean/p95 | detector 推理 | landmark 推理 | 端到端 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `legacy_full` | `2213` | `0.967/0.972` | `0.9777` | `0.5825` | `0.0048/0.0149/0.5550 px` | `1.0000` | `0.0250/0.0364 px` | `16.70 ms` | `10.65 ms` | `35.65 ms` |
| `legacy_lite` | `2131` | `0.948/0.929` | `0.9510` | `0.5013` | `0.0048/0.0144/0.5120 px` | `1.0000` | `4.1623/11.0707 px` | `14.21 ms` | `6.47 ms` | `27.94 ms` |
| `task_full` | `2213` | `0.967/0.972` | `0.9777` | `0.5825` | `0.0062/0.0151/4.2572 px` | `1.0000` | `0.0250/0.0364 px` | `16.12 ms` | `11.30 ms` | `35.74 ms` |

端到端结果说明：

- ONNX 与同组 TFLite 的 mean 误差只有 `0.0048 px` 到 `0.0062 px`，p95 约 `0.015 px`。
- `legacy_full` 和 `task_full` 对 legacy graph 仍然只有约 `0.025 px` mean，说明 ONNX Runtime 跑完整 pipeline 后没有破坏 PC 基线。
- `legacy_lite` 对 legacy graph 的误差较大，是 lite 模型本身与 full/legacy graph 的差异；它对同组 TFLite 仍然接近 0。
- `task_full` 的 max 误差有一个较大的离群值，但 mean、p95 和 `PCK@0.05` 都正常，不影响“ONNX 与 TFLite 已对齐”的结论。

## 6. 复现命令

legacy full：

```bash
conda activate mediapipe_legacy
python scripts/eval_two_stage_onnx.py \
  --split test \
  --output-dir runs/onnx_two_stage/legacy_full \
  --save-vis 0
```

legacy lite：

```bash
python scripts/eval_two_stage_onnx.py \
  --split test \
  --detector models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --landmark models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx \
  --reference-tflite runs/baseline/tflite_matrix/det_legacy_lite__lm_legacy_lite/predictions.json \
  --output-dir runs/onnx_two_stage/legacy_lite \
  --save-vis 0
```

task full：

```bash
python scripts/eval_two_stage_onnx.py \
  --split test \
  --detector models/onnx/mediapipe_task_hand_detector_full.onnx \
  --landmark models/onnx/mediapipe_task_hand_landmark_full.onnx \
  --reference-tflite runs/baseline/tflite_matrix/det_task_full__lm_task_full/predictions.json \
  --output-dir runs/onnx_two_stage/task_full \
  --save-vis 0
```

汇总输出：

```text
runs/onnx_two_stage/summary.md
runs/onnx_two_stage/summary.csv
runs/onnx_two_stage/{legacy_full,legacy_lite,task_full}/summary.json
```

## 7. 当前判断

ONNX 阶段已经通过：

| 项目 | 判断 |
| --- | --- |
| opset | opset 11，适合作为当前 ATC 输入 |
| raw-output | TFLite vs ONNX 在 `1e-4` 量级以内 |
| 端到端 | ONNX vs 同组 TFLite mean 小于 `0.01 px` |
| 模型选择 | `legacy_full` / `task_full` 可作为 OM 移植基线，`legacy_lite` 用于速度探索 |
| 下一阶段重点 | ONNX -> OM 后的 palm detector 数值偏差和 ACL runner 稳定性 |


