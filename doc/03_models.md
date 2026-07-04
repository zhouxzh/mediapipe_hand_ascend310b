# 模型资产与模型特点

本文件说明当前工程中的 TFLite 模型、full/lite 的差异，以及最新 baseline 下的速度和精度取舍。

## 1. 当前模型资产

`models/tflite/` 保留 6 个模型：

| 文件 | 角色 | 版本 |
| --- | --- | --- |
| `mediapipe_task_hand_detector_full.tflite` | palm detector | 当前 MediaPipe Tasks full |
| `mediapipe_task_hand_landmark_full.tflite` | hand landmark | 当前 MediaPipe Tasks full |
| `mediapipe_legacy_0_10_14_palm_detection_full.tflite` | palm detector | legacy 0.10.14 full |
| `mediapipe_legacy_0_10_14_palm_detection_lite.tflite` | palm detector | legacy 0.10.14 lite |
| `mediapipe_legacy_0_10_14_hand_landmark_full.tflite` | hand landmark | legacy 0.10.14 full |
| `mediapipe_legacy_0_10_14_hand_landmark_lite.tflite` | hand landmark | legacy 0.10.14 lite |

重复别名模型已经删除。文件名保留来源和版本，避免把 task、legacy、full、lite 混在一起。

## 2. 模型接口检查

模型文件变更后运行：

```bash
python scripts/inspect_tflite.py --model-dir models/tflite --output runs/baseline/model_info.json
```

输出包含：

- 输入 tensor shape、dtype、量化参数。
- 输出 tensor shape、dtype、量化参数。
- 文件大小和 SHA256。

如果需要 MiB：

```text
MiB = size_bytes / 1024 / 1024
```

判断模型是否为同一份权重时，优先比较 SHA256；若 SHA256 不同，再比较输入输出 shape、metadata 和 raw output。

## 3. 最新人工 GT Landmark 对比

数据来源：

```text
data/handlm_datasets/annotations.json
1210 images / 1210 hands / 25410 visible points
```

评估方式：直接把 `224x224` 手部图像输入 landmark 模型，与人工校正 21 点 GT 比较。

| 模型 | Mean px | Median px | P95 px | NME | PCK@0.05 | PCK@0.10 | total_mean_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task_full` | 5.940073 | 4.045608 | 16.996678 | 0.026518 | 0.864738 | 0.980992 | 24.759624 |
| `legacy_full` | 5.940073 | 4.045608 | 16.996678 | 0.026518 | 0.864738 | 0.980992 | 13.005933 |
| `legacy_lite` | 6.602299 | 4.575089 | 18.822053 | 0.029475 | 0.837741 | 0.971035 | 6.107360 |

结论：

- `task_full` 与 `legacy_full` 在当前人工 GT 数据集上输出一致，说明这两份 full landmark 模型在当前评估链路中等价。
- `legacy_lite` 速度明显更快，`total_mean_ms` 约为 full 的一半。
- `legacy_lite` 精度下降：mean 从 `5.94 px` 增加到 `6.60 px`，PCK@0.05 从 `0.8647` 降到 `0.8377`。

## 4. Palm Detector 当前结果

`data/palm_datasets` 是人工校验 palm box 和 7 点的数据集。当前 full detector 结果：

| Metric | Value |
| --- | ---: |
| images | 1859 |
| GT palms | 2207 |
| predictions | 2219 |
| precision | 0.967102 |
| recall | 0.972361 |
| AP@0.50 | 0.977699 |
| mAP@0.50:0.95 | 0.582451 |
| total_mean_ms | 18.926058 |

Palm bbox 的 mAP@0.50:0.95 不如 AP@0.50 高，是预期现象：严格高 IoU 对 palm box 很敏感，而 MediaPipe palm detector 的主要作用是生成稳定 ROI。

## 5. Full 与 Lite 的部署取舍

full/lite 不应该只看文件大小或单模型耗时。完整部署要同时看：

- detector 的 precision、recall、AP、mAP。
- landmark 在人工 GT 上的 mean/median/P95、NME、PCK。
- 完整两阶段链路相对 legacy graph 的误差。
- 端到端耗时。

当前建议：

```text
PC/310B 精度基线: full detector + full landmark
性能优化阶段: 在 full 链路逐层对齐后，再评估 lite landmark 或 INT8
```

lite 不是无损替换。它适合在速度优先场景中使用，但要接受 21 点误差上升。
