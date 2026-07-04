# MediaPipe Hand Ascend 310B

这是一个独立的 MediaPipe Hand 复刻、验证和 Ascend 310B 移植子工程。工程目标不是封装成 pip 包，而是保留一套可以直接复制到开发板上的源码、模型、脚本和教程。

当前源码分为两层：

| 路径 | 作用 |
| --- | --- |
| `hand_pipeline/` | 可复用核心库：ImageToTensor 预处理、SSD anchor、palm decode、weighted NMS、ROI、landmark 反投影、TFLite 推理封装 |
| `scripts/` | 可直接运行的 Python 程序：模型检查、palm 评估、两阶段评估、legacy graph 导出、baseline 汇总 |
| `models/tflite/` | 当前 Tasks full 与 legacy full/lite TFLite 模型 |
| `models/onnx/` | ONNX 转换输出目录，作为 ATC 输入 |
| `models/om/` | Ascend 310B ATC 转换输出目录 |
| `references/current_tasks/` | 当前 MediaPipe Tasks 参考输出，文件较大，默认不纳入版本管理 |
| `runs/baseline/` | 当前验证结果输出目录，重新运行脚本即可生成 |
| `doc/` | 教程和深度分析文档 |

`runs/` 只保存生成结果，不作为源码维护。文档里不再手工维护历史数值表，所有速度和精度数据都以 `runs/baseline/verification_summary.md` 为准。

当前使用两类人工校正数据：

- `data/palm_datasets`：人工校验过的 palm box 和 7 个 palm keypoints，用于 palm detector 检测精度。
- `data/handlm_datasets`：人工校正的 21 个手指关键点，用于 hand landmark 模型精度。

## 模型

`models/tflite/` 当前保留带来源和版本信息的模型名：

| 模型 | 角色 | 输入 |
| --- | --- | --- |
| `mediapipe_task_hand_detector_full.tflite` | 当前 Tasks palm detector full | `1x192x192x3` |
| `mediapipe_task_hand_landmark_full.tflite` | 当前 Tasks hand landmark full | `1x224x224x3` |
| `mediapipe_legacy_0_10_14_palm_detection_full.tflite` | legacy palm detector full | `1x192x192x3` |
| `mediapipe_legacy_0_10_14_palm_detection_lite.tflite` | legacy palm detector lite | `1x192x192x3` |
| `mediapipe_legacy_0_10_14_hand_landmark_full.tflite` | legacy hand landmark full | `1x224x224x3` |
| `mediapipe_legacy_0_10_14_hand_landmark_lite.tflite` | legacy hand landmark lite | `1x224x224x3` |

重复别名模型已经删除。模型清单在 `models/tflite/model_version_manifest.json`。

## 运行方式

不需要安装本工程。建议在 `mediapipe_legacy` 环境中运行；这个环境安装了 `mediapipe==0.10.14` 和 `ai_edge_litert`，两步法 TFLite 推理和官方 legacy graph 对比可以在同一个环境完成：

```bash
conda activate mediapipe_legacy
cd mediapipe_hand_ascend310b
python scripts/run_baseline.py --split test --run-matrix
```

快速抽样验证：

```bash
python scripts/run_baseline.py --split test --run-matrix --max-images 300 --output-root runs/baseline_smoke
```

生成 full/lite 组合矩阵：

```bash
python scripts/run_baseline.py --run-matrix
```

核心输出：

```text
runs/baseline/
  model_info.json
  palm_detector/
  two_stage_vs_current_tasks/
  handlm_manual_gt/
  legacy_graph/
  two_stage_vs_legacy_graph/
  legacy_rect_landmark/
  verification_summary.json
  verification_summary.md
```

## 速度和精度怎么看

`verification_summary.md` 会汇总当前 run 的关键指标：

| 模块 | 精度指标 | 速度指标 |
| --- | --- | --- |
| palm detector | TP/FP/FN、precision、recall、AP、mAP | `total_mean_ms` |
| two-stage vs current Tasks | 21 点 mean/median/p95 px、NME、PCK | `total_mean_ms` |
| two-stage vs legacy graph | 复刻链路相对 legacy graph 的 21 点误差 | `total_mean_ms` |
| legacy rect landmark | 排除 palm-to-rect 后的 landmark 子链路误差 | landmark 子链路验证 |
| handlm manual GT | 对人工校正 21 点 GT 的 landmark full/lite 对比 | landmark 单模型耗时 |
| tflite matrix | full/lite detector 与 landmark 组合对比 | 同一次 run 内横向比较 |

判断误差来源时优先看两层：

1. `legacy_rect_landmark` 接近 0，说明 ROI crop、landmark TFLite、反投影基本正确。
2. `two_stage_vs_legacy_graph` 当前也已经接近 0；如果后续在 310B 上回归，优先检查 detector 输入 tensor 是否仍使用 MediaPipe 风格的连续 ROI `warpPerspective` 采样，而不是普通 `resize + pad`。

## 310B 移植路线

建议先固定 full 模型作为精度基线，再移植到 310B：

1. 在 PC 上运行 `python scripts/run_baseline.py --split test --run-matrix`，确认 TFLite 参考链路稳定。
2. 在 PC 的 `mediapipe_legacy` 环境中运行 `python scripts/export_onnx.py --group legacy_full`。
3. 在 310B 上 source CANN 环境后运行 `python scripts/export_ascend_om.py --group legacy_full`；该脚本固定单线程 ATC。
4. 逐层对齐输入 tensor、raw output、decoded palm、NMS、hand rect、ROI crop、landmark raw output 和最终 21 点。
5. full 链路稳定后，再评估 lite 或 INT8 的速度收益。

详细教程见 [doc/README.md](doc/README.md)。
