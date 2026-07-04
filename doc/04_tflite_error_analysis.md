# TFLite 模型误差分析

本文件只讨论 PC 侧 TFLite 模型和 TFLite 两阶段复刻链路。ONNX 转换误差放在 [ONNX 模型误差分析](06_onnx_error_analysis.md)，Ascend 310B OM 误差放在 [OM 模型误差分析](07_om_error_analysis.md)。

当前数值来自：

```text
runs/baseline/verification_summary.json
runs/baseline/verification_summary.md
runs/baseline/tflite_matrix/summary.json
```

## 1. 数据和参考对象

| 数据/参考 | 用途 |
| --- | --- |
| `data/palm_datasets/test` | 人工校验 palm box 和 7 个 palm keypoints，评估 palm detector 和完整两阶段链路 |
| `data/handlm_datasets` | 人工校正 21 个手指关键点，评估 landmark 模型本身 |
| `mediapipe==0.10.14` legacy graph | 旧版 MediaPipe 官方 graph 输出，用作本工程两阶段复刻链路的主要对齐目标 |
| current MediaPipe Tasks | 新版 Tasks 输出，用于观察新旧 MediaPipe 版本差异 |

`palm_datasets` 和 `handlm_datasets` 的任务不同，不能把两个数据集的指标直接混在一起解释：

- `palm_datasets/test` 检查 detector、palm-to-rect、ROI crop、landmark 和反投影组成的完整 pipeline。
- `handlm_datasets` 固定使用人工裁剪/标注的 hand crop，只检查 landmark 模型输出是否接近人工 21 点。
- legacy graph 和 current Tasks 是官方 pipeline 参考，不等同于人工 GT。

## 2. Palm Detector 计算方法

palm detector 的评估链路：

```text
image
  -> 192x192 ImageToTensor warpPerspective
  -> detector TFLite
  -> decode_raw_palm
  -> weighted_nms
  -> palm detections
```

TP/FP/FN 计算：

```text
1. 保留 score >= operating_conf 的预测。
2. 按 score 从高到低排序。
3. 在同一张图内与人工 GT palm box 做一对一匹配。
4. 匹配条件：IoU(pred_box, gt_box) >= threshold。
5. 一个 GT 最多匹配一个 prediction，一个 prediction 最多匹配一个 GT。
```

```text
TP = 成功匹配的预测数
FP = 没有匹配到 GT 的预测数
FN = 没有被任何预测匹配到的 GT 数
Precision = TP / (TP + FP)
Recall = TP / GT_total
Miss rate = FN / GT_total
```

最新 `palm_datasets/test` 结果：

| Metric | Value |
| --- | ---: |
| images | `1859` |
| GT palms | `2207` |
| predictions | `2219` |
| precision | `0.967102` |
| recall | `0.972361` |
| AP@0.50 | `0.977699` |
| mAP@0.50:0.95 | `0.582451` |
| total_mean_ms | `17.676557` |

## 3. AP 与 mAP 计算方法

AP 不是固定阈值下的单点 precision/recall，而是在候选预测集合内按 score 从高到低扫描得到 precision-recall 曲线。

对每个 IoU 阈值：

```text
1. 按 score 降序排列预测。
2. 逐个 prediction 做一对一 GT 匹配。
3. 累计 TP_cumsum 和 FP_cumsum。
4. 计算 recall 和 precision。
5. 对 precision 做单调包络。
6. 在 [0, 1] 上采样 101 个 recall 点并积分。
```

```text
recall_i = TP_cumsum_i / GT_total
precision_i = TP_cumsum_i / (TP_cumsum_i + FP_cumsum_i)
```

`AP@0.50` 表示 IoU 阈值为 `0.50` 的 AP。`mAP@0.50:0.95` 表示对 `0.50, 0.55, ..., 0.95` 的 AP 取平均。

## 4. 21 点误差、NME 与 PCK

对 21 点评估，逐点计算像素欧氏距离：

```text
err_j = sqrt((pred_x_j - ref_x_j)^2 + (pred_y_j - ref_y_j)^2), j = 0..20
```

统计时把所有手的 21 个点摊平成一个点集：

```text
point_count = matched_hands * 21
mean_px = mean(err)
median_px = median(err)
p95_px = percentile(err, 95)
max_px = max(err)
```

NME 和 PCK 使用参考 box 的面积尺度归一化：

```text
w = max(ref_box_width, 1)
h = max(ref_box_height, 1)
norm = max(sqrt(w * h), 1)
normalized_err_j = err_j / norm
NME = mean(normalized_err_j)
PCK@t = count(normalized_err_j <= t) / point_count
```

## 5. Landmark 人工 GT 结果

`handlm_manual_gt` 直接评估 landmark 模型本身，不经过 palm detector：

```text
224x224 hand crop
  -> landmark TFLite
  -> 21 points
  -> compare with manually corrected GT
```

最新结果：

| Model | Mean px | Median px | P95 px | NME | PCK@0.05 | PCK@0.10 | total_mean_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task_full` | `5.940073` | `4.045608` | `16.996678` | `0.026518` | `0.864738` | `0.980992` | `23.646644` |
| `legacy_full` | `5.940073` | `4.045608` | `16.996678` | `0.026518` | `0.864738` | `0.980992` | `13.065471` |
| `legacy_lite` | `6.602299` | `4.575089` | `18.822053` | `0.029475` | `0.837741` | `0.971035` | `6.203049` |

结论：

- full landmark 在人工 GT 上优于 lite。
- lite 的推理速度明显更快，但精度有可见损失。
- `task_full` 和 `legacy_full` 的 21 点精度相同，说明这两份 full landmark TFLite 在当前数据上输出一致；耗时差异主要来自运行路径和读图/封装开销。

## 6. TFLite 两阶段链路对齐

| 验证项 | Mean px | P95 px | PCK@0.05 | total_mean_ms | 含义 |
| --- | ---: | ---: | ---: | ---: | --- |
| `two_stage_vs_current_tasks` | `3.630226` | `9.486280` | `0.908602` | `33.715122` | 当前复刻链路对齐 current Tasks |
| `two_stage_vs_legacy_graph` | `0.024968` | `0.035854` | `0.999289` | `35.437393` | 当前复刻链路对齐 legacy graph |
| `legacy_rect_landmark` | `0.016921` | `0.027788` | `0.999526` | NA | 使用 legacy 官方 rect 验证 landmark 子链路 |

`legacy_rect_landmark` 接近 0，说明 landmark TFLite、ROI crop 和 landmark 反投影基本正确。完整两阶段链路相对 legacy graph 的 mean 误差为 `0.024968 px`，说明当前 PC TFLite 复刻链路可以作为 ONNX/OM 迁移前的基准。

此前 1px 级误差的主要原因是 detector 输入没有复刻 MediaPipe `ImageToTensorCalculator` 的连续 ROI 采样：

```text
普通 resize + pad
  -> palm detector raw output 发生细微变化
  -> palm_detections / hand_rect_from_palm 偏移
  -> 最终 21 点出现像素级误差
```

当前 TFLite 链路使用 MediaPipe 风格 `warpPerspective` 后，这一层误差已经被压到接近 0。

## 7. TFLite 三组模型矩阵

正式移植重点看 coherent pipeline，也就是 detector 和 landmark 来自同一组模型：

| TFLite 组合 | Hands | Palm P/R | AP50 | mAP | vs legacy graph mean/p95 | PCK@0.05 | total_mean_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task_full` | `2219` | `0.967/0.972` | `0.977699` | `0.582451` | `0.024968/0.035864 px` | `0.999289` | `34.983148` |
| `legacy_full` | `2219` | `0.967/0.972` | `0.977699` | `0.582451` | `0.024968/0.035864 px` | `0.999289` | `35.569386` |
| `legacy_lite` | `2162` | `0.948/0.929` | `0.950769` | `0.501329` | `4.217669/11.257708 px` | `0.885352` | `23.829546` |

结论：

- `legacy_full` 是对齐 `mediapipe==0.10.14` 的第一基准。
- `task_full` 在当前 TFLite 结果上与 `legacy_full` 基本一致，可以作为新版模型来源的对照。
- `legacy_lite` 明显更快，但 palm recall、mAP 和最终 21 点对齐误差都更差，不建议作为第一版精度基线。

## 8. PC 侧回归检查

`scripts/summarize_baseline.py` 当前执行以下基础检查：

| 检查 | 默认阈值 | 当前结果 |
| --- | ---: | ---: |
| legacy 官方 rect + landmark 子链路平均误差 | `<= 0.05 px` | `0.016921 px` |
| legacy 官方 rect + landmark 子链路 `PCK@0.05` | `>= 0.999` | `0.999526` |
| palm 路径两阶段法对齐 legacy graph 平均误差 | `<= 1.0 px` | `0.024968 px` |

这些检查通过后，才应该继续讨论 ONNX 转换和 OM 板端误差。

