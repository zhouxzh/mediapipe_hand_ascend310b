# 板端验证结果

本文合并记录 8T/20T 开发板上的 full/lite OM 精度和速度结果。旧的单独 20T benchmark、8T benchmark、lite 20T、lite 8T、20T 数据集文档已删除，统一以本文为准。

## 测试对象

正式 full：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

report-only lite：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om
```

数据集：

```text
data/portable-hagridv2-mediapipe-hand/test-00000.parquet
1663 images / 1664 GT hands
```

正式评估命令：

```bash
python scripts/eval_hf_hand_dataset_om.py --model-set full,lite
```

## 数据集精度对比

8T 和 20T 使用同一套 OM 与同一套 Python 后处理，精度结果一致；差异主要体现在速度。

| 板卡 | 模型 | 状态 | Precision | Recall | AP50 | AP75 | mAP50:95 | Full21 mean px | Full21 P95 px | unmatched GT rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20T | full | passed | `0.996399` | `0.997596` | `0.994933` | `0.994933` | `0.994755` | `0.127865` | `0.247230` | `0.002404` |
| 8T | full | passed | `0.996399` | `0.997596` | `0.994933` | `0.994933` | `0.994755` | `0.127865` | `0.247230` | `0.002404` |
| 20T | lite | report-only | `0.978877` | `0.974760` | `0.983212` | `0.750811` | `0.645321` | `1.318512` | `2.481823` | `0.009615` |
| 8T | lite | report-only | `0.978877` | `0.974760` | `0.983212` | `0.750811` | `0.645321` | `1.318512` | `2.481823` | `0.009615` |

full 正式阈值：

| 阈值 | 要求 | full 结果 |
| --- | ---: | ---: |
| recall | `>= 0.95` | `0.997596` |
| AP50 | `>= 0.95` | `0.994933` |
| full21 mean px | `<= 2.0` | `0.127865` |
| full21 p95 px | `<= 5.0` | `0.247230` |
| unmatched GT rate | `<= 0.05` | `0.002404` |

结论：full 通过正式验收；lite 仅作为对照报告，不影响退出码。

## 数据集端到端速度

下表为真实图片完整两阶段 pipeline 的平均耗时，单位 `ms/image`。它包含图像预处理、detector OM、decode、NMS、ROI crop、landmark OM 和后处理。

| 板卡 | 模型 | preprocess | detector | decode | roi | landmark | post | total mean | total p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20T | full | `1.974` | `12.338` | `1.542` | `2.436` | `2.836` | `0.175` | `21.353` | `26.098` |
| 8T | full | `2.130` | `27.825` | `1.601` | `2.591` | `5.104` | `0.191` | `39.553` | `46.372` |
| 20T | lite | `1.987` | `11.408` | `1.579` | `2.416` | `2.282` | `0.172` | `19.900` | `20.170` |
| 8T | lite | `2.157` | `25.798` | `1.602` | `2.552` | `3.881` | `0.189` | `36.300` | `36.764` |

速度结论：

- 20T full 真实链路平均 `21.353 ms/image`，约为 8T full 的 `1.85x`。
- 20T lite 平均 `19.900 ms/image`，仅比 20T full 快约 `6.8%`。
- 8T lite 平均 `36.300 ms/image`，仅比 8T full 快约 `8.2%`。
- 当前 lite 速度收益有限，且精度明显低于 full。

## 单模型 execute benchmark

单模型 benchmark 只统计 warmed-up `acl.mdl.execute`，不包含完整 pipeline 的 CPU 后处理。

| 板卡 | 模型 | execute mean | execute p95 | 结论 |
| --- | --- | ---: | ---: | --- |
| 20T | full palm optimized | `11.532 ms` | `11.574 ms` | 正式模型 |
| 20T | full landmark | `1.891 ms` | `1.937 ms` | 正式模型 |
| 8T | full palm optimized | `26.931 ms` | `26.972 ms` | 正式模型 |
| 8T | full landmark | `3.889 ms` | `3.920 ms` | 正式模型 |
| 8T | optimized lite palm | `24.799 ms` | `24.842 ms` | report-only 候选 |
| 20T | direct lite palm | `1.562 ms` | `1.581 ms` | 输出错误，模型已删除 |

direct lite palm 单模型很快，但 raw-output 和端到端结果不正确，不作为有效部署速度。

## 20T 重编译 OM 对比

20T 上使用 `Ascend310B1` 重新 ATC 编译 full palm 和 full landmark 后，与当前正式 OM 的 raw-output 完全一致：

```text
palm output[0] max_abs=0 mean_abs=0
palm output[1] max_abs=0 mean_abs=0
landmark output[0] max_abs=0 mean_abs=0
landmark output[1] max_abs=0 mean_abs=0
landmark output[2] max_abs=0 mean_abs=0
landmark output[3] max_abs=0 mean_abs=0
```

20 组随机输入汇总：

```text
palm     samples=20 outputs_checked=40 max_abs=0.0 mean_abs_avg=0.0
landmark samples=20 outputs_checked=80 max_abs=0.0 mean_abs_avg=0.0
```

因此当前仓库不保留 `*_ascend310b1.om` 重复模型。

## 报告路径

20T 正式数据集报告：

```text
runs/hf_hand_dataset_om/20260705_201621/summary.md
runs/hf_hand_dataset_om/20260705_201621/summary.json
```

8T 正式数据集报告：

```text
runs/hf_hand_dataset_om/20260705_194253/summary.md
runs/hf_hand_dataset_om/20260705_194253/summary.json
```

## 最终判断

- full OM 是当前唯一正式默认部署模型。
- lite OM 可以保留为 report-only 对照，但不能替代 full。
- 20T 专用 full OM 与现有 full OM 没有数值差异，已删除。
- direct lite palm OM 输出错误，已删除。
