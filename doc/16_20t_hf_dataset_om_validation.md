# 20T Portable HaGRIDv2 OM 数据集精度与速度验证

本文记录 2026-07-05 在 Orange Pi AI Pro 20T 开发板 `ascend20t` 上，对当前仓库中的 full 与 lite 两组 OM 模型进行 Portable HaGRIDv2 MediaPipe test 集全量评估的结果。

## 1. 测试目的

本次验证目标是确认仓库迁移到 20T 版 Ascend 310B 开发板后：

- full OM 两阶段链路在真实图片数据集上仍满足正式验收阈值。
- lite OM 两阶段链路的精度与速度有完整记录，作为速度优先场景的参考。
- 推理速度按真实图片端到端 pipeline 统计，而不只看单模型 `acl.mdl.execute`。

## 2. 测试环境

| 项目 | 值 |
| --- | --- |
| 板卡 | Orange Pi AI Pro 20T |
| SSH host | `ascend20t` |
| hostname | `orangepiaipro-20t` |
| 运行路径 | `~/Documents/mediapipe_hand_ascend310b` |
| Python | `/usr/local/miniconda3/bin/python` |
| Conda env | `base` |
| CANN env | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 数据集 | `data/portable-hagridv2-mediapipe-hand/test-00000.parquet` |
| 图片数 | `1663` |
| GT hands | `1664` |

板端环境：

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 3. 测试命令

```bash
python scripts/eval_hf_hand_dataset_om.py \
  --model-set full,lite \
  --progress-interval 100
```

输出目录：

```text
runs/hf_hand_dataset_om/20260705_201621
```

本地同步后的报告：

```text
runs/hf_hand_dataset_om/20260705_201621/summary.md
runs/hf_hand_dataset_om/20260705_201621/summary.json
runs/hf_hand_dataset_om/20260705_201621/full/om/report.md
runs/hf_hand_dataset_om/20260705_201621/lite/om/report.md
```

## 4. 模型文件

full：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

lite：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype_ascend310b4_singlethread.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om
```

说明：lite palm 使用的是此前在 8T 板 `ascend8t` 上生成并接受的 optimized lite palm OM。此次测试验证它在 20T 板上可运行，并记录它在真实数据集上的精度和速度。

## 5. 精度结果

| 模型 | enforced | passed | precision | recall | AP50 | AP75 | mAP50:95 | unmatched GT rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | true | true | `0.996399` | `0.997596` | `0.994933` | `0.994933` | `0.994755` | `0.002404` |
| lite | false | null | `0.978877` | `0.974760` | `0.983212` | `0.750811` | `0.645321` | `0.009615` |

full 明细：

| 指标 | 值 |
| --- | ---: |
| images | `1663` |
| gt_hands | `1664` |
| pred_hands | `1666` |
| matched_hands | `1660` |
| unmatched_gt | `4` |
| unmatched_pred | `6` |
| full21 mean px | `0.127865` |
| full21 median px | `0.107629` |
| full21 p95 px | `0.247230` |
| full21 max px | `5.014464` |
| full21 NME mean | `0.001707` |
| full21 PCK@0.01 mean | `0.996644` |
| full21 PCK@0.05 mean | `0.999684` |
| palm7 mean px | `0.214910` |
| palm7 p95 px | `0.527395` |

lite 明细：

| 指标 | 值 |
| --- | ---: |
| images | `1663` |
| gt_hands | `1664` |
| pred_hands | `1657` |
| matched_hands | `1648` |
| unmatched_gt | `16` |
| unmatched_pred | `9` |
| full21 mean px | `1.318512` |
| full21 median px | `1.170327` |
| full21 p95 px | `2.481823` |
| full21 max px | `11.601822` |
| full21 NME mean | `0.016958` |
| full21 PCK@0.01 mean | `0.309379` |
| full21 PCK@0.05 mean | `0.977144` |
| palm7 mean px | `5.949103` |
| palm7 p95 px | `14.319945` |

## 6. 速度结果

下表是对真实图片运行完整两阶段 pipeline 的分阶段耗时，单位为 ms/image。

| 模型 | preprocess mean | detector mean | decode mean | roi mean | landmark mean | post mean | total mean | total median | total p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | `1.974` | `12.338` | `1.542` | `2.436` | `2.836` | `0.175` | `21.353` | `21.059` | `26.098` |
| lite | `1.987` | `11.408` | `1.579` | `2.416` | `2.282` | `0.172` | `19.900` | `19.655` | `20.170` |

对比此前 `doc/09_20t_om_benchmark.md` 中单模型 benchmark：

- full palm 纯 `acl.mdl.execute` 约 `11.5 ms`，在真实图片两阶段评估中 detector 阶段均值为 `12.338 ms`。
- full landmark 纯 `acl.mdl.execute` 约 `1.9 ms`，在真实图片两阶段评估中 landmark 阶段均值为 `2.836 ms`。
- 数据集评估的 `total_ms` 还包括图像预处理、SSD decode、NMS、ROI crop、landmark 后处理等 CPU 侧工作，因此更接近实际部署链路耗时。

## 7. 验收结论

full 使用正式 pass/fail 阈值：

| 阈值 | 要求 | full 结果 |
| --- | ---: | ---: |
| recall | `>= 0.95` | `0.997596` |
| AP50 | `>= 0.95` | `0.994933` |
| full21 mean px | `<= 2.0` | `0.127865` |
| full21 p95 px | `<= 5.0` | `0.247230` |
| unmatched GT rate | `<= 0.05` | `0.002404` |

full 在 20T 板端全量 test 集上通过正式验收。

lite 默认只报告、不决定进程退出码，因为数据集标注来自 MediaPipe legacy/full 风格链路。此次 lite 的总体速度比 full 快约 `6.8%`，但 AP75、mAP50:95、palm7 误差和 full21 误差均明显弱于 full。

## 8. 脚本修复记录

首次在同一进程中运行 `--model-set full,lite` 时，full 完整跑完后加载 lite OM 报错：

```text
acl.mdl.load_from_file ... failed, ret=145001
GeExecutor has not been initialized
```

原因是数据集评估脚本在每个模型组结束时调用 `acl.finalize()`，导致同一 Python 进程中后续重新加载第二组 OM 时 GE runtime 不能稳定重建。

已修复：

- `hand_pipeline/two_stage.py`：`OmHandPipeline` 增加 `finalize_on_release` 参数。
- `scripts/eval_hf_hand_dataset_om.py`：数据集评估入口创建 OM pipeline 时传入 `finalize_on_release=False`。

修复后 `full,lite` 可在同一进程内连续完成评估，并生成统一报告目录 `runs/hf_hand_dataset_om/20260705_201621`。

## 9. 最终结论

- 20T 版 Ascend 310B 开发板上，当前 full OM 两阶段部署链路在 Portable HaGRIDv2 MediaPipe test 集 `1663` 张图片上通过正式验收。
- full 真实图片端到端平均耗时为 `21.353 ms/image`，约等价于 `46.8 FPS` 的纯推理链路上限。
- lite 可在 20T 上运行，真实图片端到端平均耗时为 `19.900 ms/image`，但精度明显低于 full；默认仍不作为正式验收模型。
- 当前正式部署仍建议使用 full palm optimized OM + full landmark OM。
