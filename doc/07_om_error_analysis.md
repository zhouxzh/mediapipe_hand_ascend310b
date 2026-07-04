# OM 模型误差分析

本文只讨论 Ascend 310B 上的 OM 误差，重点是 `mediapipe_legacy_0_10_14_palm_detection_full` 从普通 ONNX 转 OM 后的数值偏差、定位过程、已验证的算子改写、最终误差，以及 ATC/算子兼容性记录。PC 侧 TFLite 对齐见 [04_tflite_error_analysis.md](04_tflite_error_analysis.md)，ONNX 对齐见 [06_onnx_error_analysis.md](06_onnx_error_analysis.md)。

## 1. 测试对象

| 项目 | 值 |
| --- | --- |
| 开发板 | Ascend 310B，SSH 名称 `310` |
| 工程路径 | `~/Documents/artemis/mediapipe_hand_ascend310b` |
| 数据集 | `palm_datasets/test` 的前 200 张 reference |
| reference | TFLite `mediapipe_legacy_0_10_14_palm_detection_full.tflite` |
| 原始 ONNX | `models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx` |
| 优化 ONNX | `models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx` |
| 优化 OM | `models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om` |

核心验证脚本：

```text
scripts/analyze_palm_om.py
scripts/build_optimized_palm_om.py
scripts/rewrite_palm_downsample_residual.py
scripts/rewrite_palm_bilinear_resize.py
scripts/rewrite_palm_maxpool_slices.py
```

## 2. 原始问题

普通 ONNX 直接转出的 `legacy_full_palm` OM 在同一批 TFLite reference 上误差很大：

| 项目 | 原始 OM vs TFLite |
| --- | ---: |
| raw boxes mean_abs | `2.7714` |
| raw score logits mean_abs | `0.9354` |
| positive-anchor center mean | `15.6008 px` |
| positive-anchor palm7 mean | `16.3980 px` |
| NMS match center mean | `17.0703 px` |
| top20 score overlap | `0.6967` |
| top100 score overlap | `0.6532` |

这说明问题不是单纯的后处理或 anchor decode，而是 OM raw boxes 与 raw score logits 同时出现了严重偏移。由于同一 ONNX 在 ONNX Runtime 中与 TFLite 基本一致，问题范围被缩小到 ONNX -> OM / ATC / ACL 执行阶段。

ONNX 对 TFLite 的参考误差：

| 项目 | mean_abs | max_abs |
| --- | ---: | ---: |
| ONNX raw boxes vs TFLite | `1.293e-05` | `4.79e-04` |
| ONNX raw scores vs TFLite | `4.717e-06` | `4.77e-05` |

因此 ONNX 不是主要误差来源。

## 3. 定位方法

定位阶段曾采用逐层 probe，而不是只看最终输出；这些临时 probe 脚本和中间数据已经在工程整理时删除，保留分析结论和最终复现脚本：

1. 用 `scripts/analyze_palm_om.py make-reference` 保存真实图片对应的输入 tensor、TFLite raw boxes、raw scores、letterbox 和 decode 结果。
2. 给 ONNX 追加中间层输出，并在 PC 侧用 ONNX Runtime 生成中间层 reference。
3. 在 310B 上编译同一 probe ONNX 为 OM。
4. 对比每个中间层 mean_abs、p95_abs、max_abs。

这一路径把误差定位到两个明确的 ONNX 算子模式：

```text
channel-padding residual: Pad + Add
FPN upsample: Resize(linear, half_pixel, scale=2)
```

## 4. 第一个问题：Pad + Add 下采样残差

legacy palm detector 中有三处通道数翻倍的下采样残差：

```text
Add(Pad(pool, channel_tail_zeros), conv_branch)
```

在 310B OM 中，`Pad` 的 channel tail 不稳定，导致 shortcut 分支被污染。最小 probe 证明：

| 输出 | mean_abs |
| --- | ---: |
| shortcut_pool | `5.34e-05` |
| shortcut_pad | `0.155077` |
| main_depthwise | `0.000747` |
| main_pointwise | `0.000298` |
| residual_add | `0.155265` |
| prelu15 | `0.120361` |

修复方法是把：

```text
Add(Pad(pool), branch)
```

改写为数学等价的：

```text
first = Add(pool, Slice(branch, channels 0:C))
tail  = Slice(branch, channels C:2C)
out   = Concat(first, tail, axis=1)
```

复现脚本：

```text
scripts/rewrite_palm_downsample_residual.py
```

局部修复效果：

| 模型片段 | 修复前 mean_abs | 修复后 mean_abs |
| --- | ---: | ---: |
| prelu14 -> prelu15 | `0.067093` | `0.000228` |

整图应用该改写后，`prelu19` 之前误差恢复到 `0.001` 以内，但最终 raw 输出仍然很差，说明还有第二个问题。

## 5. 第二个问题：Resize(linear, half_pixel)

FPN 细粒度 probe 显示，downsample split 后首个大误差出现在第一个上采样：

| 输出 | mean_abs |
| --- | ---: |
| prelu19 | `0.000694` |
| max_pooling2d_3 | `0.000912` |
| prelu24 | `0.008689` |
| Resize__279:0 | `0.912137` |
| prelu27 | `0.294106` |
| Resize__302:0 | `0.385329` |
| prelu30 | `0.629410` |

ONNX 中两个 Resize 节点的属性：

```text
mode = linear
coordinate_transformation_mode = half_pixel
scales = [1.0, 1.0, 2.0, 2.0]
layout = NCHW
```

这说明 CANN/ATC 对该 Resize 模式的 OM 结果与 ONNX Runtime 不一致。修复方式是把 2x half-pixel 双线性上采样展开为显式算术：

```text
Slice rows/cols
Mul by 0.75 or 0.25
Add blended neighbors
Concat back to 2x height/width
```

复现脚本：

```text
scripts/rewrite_palm_bilinear_resize.py
```

改写后的 ONNX 与原 ONNX 最终输出等价：

| 验证 | images | mean_abs | max_abs |
| --- | ---: | ---: | ---: |
| smoke | `20` | `2.3647517e-06` | `3.9815903e-05` |
| 200 reference | `200` | `2.3497785e-06` | `4.5776367e-05` |

## 6. 第三个问题：MaxPool 与高精度编译

`Pad + Add` 和 `Resize` 改写后，普通 `force_fp16` OM 已经没有明显的算子语义错误，但 raw boxes/keypoints 的相对平均误差仍约 `0.158%`。进一步尝试 `must_keep_origin_dtype` 时，ATC 在 `MaxPoolV3` 上失败：

```text
Optype [MaxPoolV3] of Ops kernel [AIcoreEngine] is unsupported.
data type DT_FLOAT of input [x] is not supported.
All supported data type and format of tensor input0.x is:
Data Type: {DT_FLOAT16} Format:{NC1HWC0}.
```

legacy palm detector 中 4 个 MaxPool 都是固定模式：

```text
kernel_shape = [2, 2]
strides = [2, 2]
pads = [0, 0, 0, 0]
layout = NCHW
```

因此可以把每个 MaxPool 改写为四个交错采样分支再取最大值：

```text
s00 = Slice(x, h=0::2, w=0::2)
s01 = Slice(x, h=0::2, w=1::2)
s10 = Slice(x, h=1::2, w=0::2)
s11 = Slice(x, h=1::2, w=1::2)
out = Max(s00, s01, s10, s11)
```

复现脚本：

```text
scripts/rewrite_palm_maxpool_slices.py
```

MaxPool 改写前后 ONNX 等价验证：

| 验证 | images | mean_abs | p99_abs | max_abs |
| --- | ---: | ---: | ---: | ---: |
| smoke | `20` | `3.8338449e-06` | `1.8119812e-05` | `5.4121017e-05` |
| 200 reference | `200` | `3.8343796e-06` | `1.8119812e-05` | `9.0122223e-05` |

改写后 `must_keep_origin_dtype` 编译成功，ATC 命令由 `scripts/build_optimized_palm_om.py` 生成，并自动使用：

```text
taskset -c 0 nice -n 19 atc ...
```

## 7. 最终优化结果

当前最优 OM：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
```

对应 ONNX：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
```

200 张 reference 上的 raw 输出误差：

| 输出 | mean_abs | median_abs | p95_abs | p99_abs | max_abs | ref_abs_mean | relative_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw boxes/keypoints | `0.007474` | `0.004939` | `0.023107` | `0.039187` | `0.376629` | `11.929138` | `0.062652%` |
| raw score logits | `0.002589` | `0.001923` | `0.007274` | `0.011180` | `0.040539` | `10.126442` | `0.025567%` |

因此，当前优化结果已经满足 raw boxes/keypoints 与 raw score logits 平均相对误差均小于 `0.1%` 的目标。

decode 后误差已经很小：

| 项目 | mean |
| --- | ---: |
| sigmoid probability abs | `4.5417e-06` |
| positive-anchor center | `0.013455 px` |
| positive-anchor palm7 | `0.027383 px` |
| NMS match IoU | `0.999597` |
| NMS match center | `0.014242 px` |
| NMS match palm7 mean | `0.024093 px` |

检测指标几乎与 TFLite 重合：

| Backend | predictions | precision | recall | AP50 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| TFLite | `201` | `0.970149` | `0.975000` | `0.984348` | `0.604634` |
| 优化 OM | `201` | `0.970149` | `0.975000` | `0.984348` | `0.604666` |

50 张 smoke 中，`allow_mix_precision` 和 `high_precision_for_all` 与普通优化版数值相同：

| ATC 模式 | raw box mean_abs | raw score mean_abs | 结论 |
| --- | ---: | ---: | --- |
| `force_fp16` | `0.0186839` | `0.0066565` | 旧优化方案，未达 raw `<0.1%` |
| `allow_mix_precision` | `0.0186839` | `0.0066565` | 无改善 |
| `force_fp16 + high_precision_for_all` | `0.0186839` | `0.0066565` | 无改善 |
| `maxpool_slices + must_keep_origin_dtype` | `0.0071785` | `0.0025563` | 50 张 smoke 明显改善 |

## 8. 剩余误差来源判断

修复后 layer probe 显示，已经不存在 `Pad` 或 `Resize` 那种单点灾难误差。剩余误差更像 Ascend 默认 FP16 算子路径的稳定累计误差：

| 输出 | mean_abs | mean_abs/ref_abs_mean |
| --- | ---: | ---: |
| prelu19 | `0.000694` | `0.1938%` |
| resize_slices_0_height_up | `0.007728` | `0.1764%` |
| Resize__279:0 | `0.006902` | `0.1582%` |
| add_24 | `0.001490` | `0.1944%` |
| regressor_palm_16 head | `0.024438` | `0.1305%` |
| classifier_palm_16 head | `0.006830` | `0.0581%` |
| Resize__302:0 | `0.002155` | `0.1919%` |
| regressor_palm_8 head | `0.014369` | `0.2069%` |
| classifier_palm_8 head | `0.006526` | `0.0710%` |

也就是说，`force_fp16` 版本的剩余偏差不再来自明显的算子语义错误，而是多数层保持在 `0.1%~0.2%` 附近的数值累计。MaxPool 改写后可以切到 `must_keep_origin_dtype`，raw 输出相对误差进一步降到 `0.1%` 以内。

## 9. ATC 兼容性与失败记录

本轮尝试过的 ATC 编译模式如下：

| 模式 | 结果 | 说明 |
| --- | --- | --- |
| 默认 `force_fp16` | 成功 | 旧优化方案，误差高于最终方案 |
| `allow_mix_precision` | 成功 | 误差与默认相同 |
| `force_fp16 + high_precision_for_all` | 成功 | 误差与默认相同 |
| `force_fp32` | 失败 | `PluginManager InvokeAll failed` |
| `cube_fp16in_fp32out` | 失败 | `PluginManager InvokeAll failed` |
| 未改写 MaxPool 的 `must_keep_origin_dtype` | 失败 | `MaxPoolV3` 不支持 FP32 输入 |
| `maxpool_slices + must_keep_origin_dtype` | 成功 | 当前正式方案 |

`must_keep_origin_dtype` 的关键错误：

```text
Optype [MaxPoolV3] of Ops kernel [AIcoreEngine] is unsupported.
data type DT_FLOAT of input [x] is not supported.
All supported data type and format of tensor input0.x is:
Data Type: {DT_FLOAT16} Format:{NC1HWC0}.
```

这说明当前 310B CANN 环境下，palm detector 中的 MaxPoolV3 只能走 FP16。把 MaxPool 改写为 `Slice + Max` 后，`must_keep_origin_dtype` 可以成功编译并显著降低剩余 FP16 累计误差。

## 10. 运行时稳定性记录

Python ACL runner 曾观察到 palm detector OM 常驻复用时 raw output 漂移。探针现象：

| Tensor | repeat mean_abs vs first repeat | p95_abs | max_abs |
| --- | ---: | ---: | ---: |
| boxes | `2205.9204` | `5388.7081` | `6039.4111` |
| scores | `1849.4375` | `4720.1341` | `5323.6758` |

因此当前正式 raw-output 指标使用“每张图重新加载 detector 模型”的稳定模式。后续 C++ ACL 部署必须重新验证：

- 模型常驻是否稳定；
- input/output dataset buffer 是否每次正确刷新；
- output host/device copy 是否完整；
- context、stream、model descriptor 生命周期是否正确；
- 多次执行同一输入是否 bitwise 或近似稳定。

如果 C++ 常驻模型没有漂移，端到端速度会显著优于当前 Python 验证脚本。

## 11. 当前验收判断

当前优化 OM 已经达到可移植开发状态：

| 项目 | 判断 |
| --- | --- |
| ONNX 等价性 | 通过，普通 ONNX 与最终优化 ONNX mean_abs `3.9e-6` 量级 |
| OM raw score | 平均相对误差 `0.025567%`，低于 `0.1%` |
| OM raw box/keypoint | 平均相对误差 `0.062652%`，低于 `0.1%` |
| decode / NMS | 几何误差小，NMS `201/201` 匹配 |
| AP50 | 与 TFLite 相同，均为 `0.984348` |
| mAP | 与 TFLite 差异约 `0.000032` |

因此，本轮优化已经解决了 OM 的主要语义错误，并把 raw tensor 平均相对误差压到 `0.1%` 以内；当前版本可以作为 legacy full palm detector 的正式 310B 移植基线。



