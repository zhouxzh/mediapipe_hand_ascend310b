# 模型资产

本文说明当前仓库保留的模型资产。已经证明输出错误或与正式模型数值完全重复的 OM 不再保留。

## 正式 OM

| 角色 | 文件 | 状态 |
| --- | --- | --- |
| full palm detector | `models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om` | 正式默认 |
| full hand landmark | `models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om` | 正式默认 |

正式 full palm OM 来自 optimized ONNX：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
```

## Lite 候选 OM

| 角色 | 文件 | 状态 |
| --- | --- | --- |
| lite palm detector | `models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om` | report-only 候选 |
| lite hand landmark | `models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om` | report-only 候选 |

lite 不作为正式默认模型。它可用于速度探索和数据集对照报告。

## 已删除的 OM 类型

| 类型 | 删除原因 |
| --- | --- |
| direct full palm OM | raw-output 不满足正式精度要求 |
| direct lite palm OM | raw-output 和视频端到端结果错误 |
| `*_ascend310b1.om` full 20T 重编译版本 | 与现有正式 OM raw-output 完全一致，重复 |
| task full OM | 当前部署链路不使用，未作为正式模型验收 |

## ONNX

当前保留 ONNX 的目的：

- 作为 ATC 输入。
- 作为 OM raw-output 和视频端到端对比的参考。
- 支持 full/lite palm optimized ONNX 复现。

关键 ONNX：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx
```

## TFLite

TFLite 模型保留为原始来源和必要的转换参考，不是板端部署入口。

```text
models/tflite/mediapipe_legacy_0_10_14_palm_detection_full.tflite
models/tflite/mediapipe_legacy_0_10_14_hand_landmark_full.tflite
models/tflite/mediapipe_legacy_0_10_14_palm_detection_lite.tflite
models/tflite/mediapipe_legacy_0_10_14_hand_landmark_lite.tflite
```

## 模型选择

| 场景 | 模型 |
| --- | --- |
| 正式部署、WebRTC 默认、数据集验收 | full palm optimized OM + full landmark OM |
| 速度探索、对照报告 | lite optimized palm candidate + lite landmark OM |
| ATC 复现 | optimized ONNX + `scripts/build_optimized_palm_om.py` |

不要只根据单模型耗时选择模型。正式选择必须同时看 detector AP/recall、21 点误差、端到端速度和视频/数据集回归结果。
