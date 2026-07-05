# Palm Full/Lite ONNX 算子结构差异

本文记录 2026-07-05 对 `mediapipe==0.10.14` legacy palm detector full 与 lite ONNX 图的算子级对比。目的不是判断精度，而是先弄清楚 lite palm 与 full palm 在图结构和算子使用上到底有什么不同。

## 1. 对比命令

对比脚本：

```bash
python scripts/compare_onnx_ops.py \
  --left models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx \
  --right models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --left-name full_palm \
  --right-name lite_palm \
  --output-dir runs/onnx_op_compare/full_vs_lite_palm_original
```

改写后图也做了一次对比：

```bash
python scripts/compare_onnx_ops.py \
  --left models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx \
  --right models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx \
  --left-name full_palm_optimized \
  --right-name lite_palm_optimized \
  --output-dir runs/onnx_op_compare/full_vs_lite_palm_optimized
```

输出文件：

```text
runs/onnx_op_compare/full_vs_lite_palm_original/report.md
runs/onnx_op_compare/full_vs_lite_palm_original/summary.json
runs/onnx_op_compare/full_vs_lite_palm_original/left_nodes.csv
runs/onnx_op_compare/full_vs_lite_palm_original/right_nodes.csv
runs/onnx_op_compare/full_vs_lite_palm_original/conv_pairs.csv
runs/onnx_op_compare/full_vs_lite_palm_optimized/report.md
runs/onnx_op_compare/full_vs_lite_palm_optimized/summary.json
```

## 2. 总体算子数量

原始 ONNX 图：

| op_type | full | lite | lite-full |
| --- | ---: | ---: | ---: |
| `Conv` | `63` | `53` | `-10` |
| `Add` | `30` | `25` | `-5` |
| `PRelu` | `31` | `26` | `-5` |

节点总数：

| 模型 | node_count |
| --- | ---: |
| full palm | `144` |
| lite palm | `124` |

这说明 lite 没有引入 full 不存在的新 op_type；它主要是删减了 5 个 repeated residual bottleneck。每少一个 bottleneck，对应少：

```text
1 x depthwise Conv
1 x pointwise Conv
1 x Add
1 x PRelu
```

所以合计少 `10 Conv + 5 Add + 5 PRelu`。

## 3. 已知敏感结构对比

full 和 lite 的这些结构数量和属性是一致的：

| 结构 | full | lite | 说明 |
| --- | ---: | ---: | --- |
| `Resize` | `2` | `2` | 都是 `mode=linear`, `coordinate_transformation_mode=half_pixel`, `scales=[1,1,2,2]` |
| tail channel `Pad + Add` residual | `3` | `3` | 都是通道尾部补零后 residual add |
| `MaxPool` | `4` | `4` | 都是 `kernel_shape=[2,2]`, `strides=[2,2]`, `pads=[0,0,0,0]` |
| `Slice` | `0` | `0` | 原始图都没有 Slice |

具体 shape 也一致：

### Resize

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| full/lite | `1x256x6x6` | `1x256x12x12` |
| full/lite | `1x256x12x12` | `1x256x24x24` |

### Tail Channel Pad + Add

| 模型 | Pad 输入 | Pad 输出 |
| --- | --- | --- |
| full/lite | `1x32x48x48` | `1x64x48x48` |
| full/lite | `1x64x24x24` | `1x128x24x24` |
| full/lite | `1x128x12x12` | `1x256x12x12` |

### MaxPool

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| full/lite | `1x32x96x96` | `1x32x48x48` |
| full/lite | `1x64x48x48` | `1x64x24x24` |
| full/lite | `1x128x24x24` | `1x128x12x12` |
| full/lite | `1x256x12x12` | `1x256x6x6` |

因此，lite direct OM 的大误差不能简单解释为“lite 使用了 full 没有的新算子”。更准确的说法是：lite 仍使用同一类敏感结构，但 backbone 深度和层编号不同，ATC/OM 的出错位置需要重新定位，不能直接套用 full 的结论。

## 4. Conv 结构对比

Conv 汇总：

| 指标 | full | lite |
| --- | ---: | ---: |
| Conv total | `63` | `53` |
| depthwise Conv | `28` | `23` |
| grouped non-depthwise Conv | `0` | `0` |
| pointwise `1x1` Conv | `34` | `29` |
| `5x5` Conv | `29` | `24` |
| stride `2x2` Conv | `5` | `5` |

Conv 按 shape/核/stride 聚合后，lite 在每个主干尺度基本比 full 少一个 residual block：

| 输出尺度 | full | lite | 差异 |
| --- | ---: | ---: | ---: |
| `1x32x96x96` residual Add/PRelu | `4 Add / 5 PRelu` | `3 Add / 4 PRelu` | `-1 block` |
| `1x64x48x48` residual Add/PRelu | `5 / 5` | `4 / 4` | `-1 block` |
| `1x128x24x24` residual Add/PRelu | `8 / 8` | `7 / 7` | `-1 block` |
| `1x256x12x12` residual Add/PRelu | `8 / 8` | `7 / 7` | `-1 block` |
| `1x256x6x6` residual Add/PRelu | `5 / 5` | `4 / 4` | `-1 block` |

## 5. 检测头对比

两者的 palm detection head shape 一致：

| head | full 输入/输出 | lite 输入/输出 |
| --- | --- | --- |
| `regressor_palm_16` | `1x256x12x12 -> 1x108x12x12 -> 1x864x18` | 同 full |
| `classifier_palm_16` | `1x256x12x12 -> 1x6x12x12 -> 1x864x1` | 同 full |
| `regressor_palm_8` | `1x128x24x24 -> 1x36x24x24 -> 1x1152x18` | 同 full |
| `classifier_palm_8` | `1x128x24x24 -> 1x2x24x24 -> 1x1152x1` | 同 full |

所以 lite/full 的输出 tensor shape 完全一致：

```text
boxes:  1x2016x18
scores: 1x2016x1
```

## 6. 改写后图的差异

对 full/lite 都应用 `downsample + resize + maxpool_slices` 改写后：

| 结构 | full optimized | lite optimized |
| --- | ---: | ---: |
| `Resize` | `0` | `0` |
| `Pad` | `0` | `0` |
| tail Pad residual | `0` | `0` |
| `MaxPool` | `0` | `0` |
| `Slice` | `158` | `158` |

改写消除了 full 中已经确认过的 Resize/Pad/MaxPool 敏感 op；但 lite optimized ATC 仍失败，说明下一步应重点排查改写后图中的：

- `Slice + Max` 替代 MaxPool 后的编译形态；
- `PRelu` 节点与不同层编号/不同通道分布的交互；
- ATC 日志中明确报错的 `Conv2D`、`ConcatD`、`StridedSliceD`、`PRelu` 等算子编译失败位置。

## 7. 当前判断

1. lite palm 和 full palm 使用的是同一套算子族，没有发现 lite 独有的新 op_type。
2. lite palm 是更浅的 backbone：每个主干尺度少一个 repeated residual bottleneck，总计少 `5` 个 block。
3. full 中已知需要处理的 `Resize`、tail channel `Pad + Add`、`MaxPool` 在 lite 中仍然存在，数量和 shape 与 full 一致。
4. 由于 lite direct OM 的 raw-output 误差极大，后续不能只看算子数量；当前问题定位、优化 lite ONNX 和 ATC 状态记录在 `doc/12_lite_palm_issue_localization.md`。
