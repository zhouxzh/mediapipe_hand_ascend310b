# Lite Palm 状态

本文合并 lite palm 相关结论。旧的 20T lite 转换、operator diff、split ONNX、8T lite 优化等单次记录已删除。

## 当前结论

lite 不是当前正式默认模型。

| 路径 | 状态 |
| --- | --- |
| direct lite palm OM | raw-output 与 ONNX 不一致，端到端结果错误，已删除 |
| 20T optimized lite palm OM | ATC/TBE 未生成可用 OM |
| 8T optimized lite palm OM | 可运行，作为 report-only 候选保留 |
| lite 两阶段数据集评估 | 可运行，速度略快，但精度明显低于 full，默认 report-only |

当前保留的 lite OM：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om
```

## Full 与 Lite 的结构差异

legacy full palm 和 lite palm 没有使用完全不同的算子集合。lite 没有新增 full 不存在的 op_type；主要差异是 backbone 更浅，每个尺度少一个 residual bottleneck，总计少：

```text
10 Conv + 5 Add + 5 PRelu
```

两者都包含同类敏感结构：

- downsample residual tail channel `Pad + Add`
- bilinear `Resize(linear, half_pixel)`
- fixed `MaxPool`

因此 lite direct OM 的错误不能简单解释为“lite 有 full 没有的新算子”。更准确的结论是：lite 使用同一类敏感结构，但 ATC/OM 的数值问题需要单独验证，不能直接套用 full 的成功结果。

## Direct Lite Palm 为什么删除

direct lite palm OM 曾经能生成并执行，但有两个问题：

1. 复用同一个 ACL 模型句柄时，raw-output 会漂移到 `655xx` 量级。
2. 每个样本 fresh-load 后异常漂移消失，但 raw-output 仍明显偏离原始 lite ONNX，视频端到端不一致。

典型 fresh-load raw-output 误差：

| output | max_abs | mean_abs | p95_abs |
| ---: | ---: | ---: | ---: |
| boxes | `22.685806` | `1.314462` | `4.596044` |
| scores | `3.369934` | `0.353768` | `1.303941` |

因此 direct lite palm OM 虽快，但输出错误，已从 `models/om/` 删除。

## Optimized Lite Palm

lite palm 已创建与 full 类似的 optimized ONNX：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

ONNX vs ONNX 等价性：

| output | max_abs | mean_abs | p95_abs |
| ---: | ---: | ---: | ---: |
| boxes | `5.340576e-05` | `3.211731e-06` | `1.001358e-05` |
| scores | `8.583069e-06` | `1.155112e-06` | `2.861023e-06` |

20T 上该 optimized ONNX 未能生成可用 OM；8T 上生成了当前保留的 report-only 候选：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om
```

8T raw-output 对原始 lite ONNX：

| 指标 | 值 |
| --- | ---: |
| boxes mean_rel | `0.829947%` |
| scores mean_rel | `0.014741%` |
| boxes p95_rel | `1.046142%` |
| execute mean | `24.799 ms` |

该结果接近目标，但速度收益有限，所以只作为候选和 report-only 模型保留。

## 数据集表现

Portable HaGRIDv2 MediaPipe test `1663` 张图片：

| 模型 | Precision | Recall | AP50 | AP75 | mAP50:95 | Full21 mean px | Full21 P95 px | Total mean 20T | Total mean 8T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | `0.996399` | `0.997596` | `0.994933` | `0.994933` | `0.994755` | `0.127865` | `0.247230` | `21.353 ms` | `39.553 ms` |
| lite | `0.978877` | `0.974760` | `0.983212` | `0.750811` | `0.645321` | `1.318512` | `2.481823` | `19.900 ms` | `36.300 ms` |

lite 的速度只比 full 快约 `6-8%`，但 AP75、mAP 和关键点误差明显变差。

## 使用建议

- 正式部署使用 full。
- lite 可以用于速度探索、回归对照和报告，不作为默认模型。
- 如果后续继续优化 lite，需要以数据集指标为准，而不是只看单模型 execute。
- 任何新 lite palm OM 进入 `models/om/` 前，必须通过 ONNX/OM raw-output 和数据集端到端验证。
