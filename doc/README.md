# MediaPipe Hand Ascend 310B 文档

本目录只保留当前部署需要的说明和少量可追溯记录。旧的单次调试文档、失败实验日志和已经被合并的结果文档已删除；当前可执行命令以仓库根目录 `README.md` 和 `scripts/README.md` 为准。

## 当前状态

正式部署模型：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

正式结论：

- 当前默认部署使用 legacy full palm optimized OM + legacy full landmark OM。
- full 模型在 `ascend20t` 上完成 Portable HaGRIDv2 MediaPipe test 全量 `1663` 张图片评估，并通过正式阈值。
- 20T 上重新 ATC 编译得到的 full OM 与现有 full OM 原始输出完全一致，当前不需要保留 20T 专用 full OM。
- lite 组合可以运行并生成报告，但默认是 `report-only`，不作为正式验收模型。
- 失败或明显错误的 direct palm OM 不再保留在 `models/om/` 中。

20T 全量数据集结果：

| 模型 | 状态 | Precision | Recall | AP50 | mAP50:95 | Full21 mean px | Full21 P95 px | Total mean ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | passed | `0.996399` | `0.997596` | `0.994933` | `0.994755` | `0.127865` | `0.247230` | `21.353` |
| lite | report-only | `0.978877` | `0.974760` | `0.983212` | `0.645321` | `1.318512` | `2.481823` | `19.900` |

完整 8T/20T 精度和速度对比见 [板端验证结果](05_board_validation_results.md)。

## 文档列表

| 文档 | 作用 |
| --- | --- |
| [01_pipeline_graph.md](01_pipeline_graph.md) | MediaPipe Hand 两阶段链路、legacy graph 对应关系和验证分层 |
| [02_data_structures.md](02_data_structures.md) | 预处理、anchor、palm detection、ROI、landmark 输出等核心数据结构 |
| [03_models.md](03_models.md) | 当前保留的模型资产、正式模型和 lite 候选模型 |
| [04_webrtc_runtime.md](04_webrtc_runtime.md) | WebRTC 实时运行、摄像头参数、VENC/DVPP 边界 |
| [05_board_validation_results.md](05_board_validation_results.md) | 8T/20T、full/lite、精度/速度统一对比记录 |
| [06_lite_palm_status.md](06_lite_palm_status.md) | lite palm 的算子差异、失败路径、8T 候选和当前部署判断 |
| [07_om_conversion_reproduction.md](07_om_conversion_reproduction.md) | full palm OM 为什么需要 downsample/resize/maxpool 改写，以及如何复现 ATC |

## 维护规则

- 新的正式验收结果写入 [05_board_validation_results.md](05_board_validation_results.md)，并同步更新本 README 的摘要表。
- 如果新增 OM 模型，必须说明它是 `正式默认`、`report-only`、`候选` 还是 `历史失败路径`。
- 不能通过精度验证的 palm OM 不放入 `models/om/`。
- 如果不同开发板 ATC 输出没有数值差异，不保留按板卡命名的重复 OM。
- 历史脚本和旧路径不再写入 `doc/README.md`；需要运行命令时看根目录 `README.md` 和 `scripts/README.md`。
