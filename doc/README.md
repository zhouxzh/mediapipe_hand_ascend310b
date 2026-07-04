# MediaPipe Hand Ascend 310B 文档

这组文档面向 `mediapipe_hand_ascend310b` 独立子工程，解释 MediaPipe Hand 的算法、数据结构、模型特点、PC 侧 TFLite baseline、ONNX 转换误差、Ascend 310B OM 误差和移植路线。

当前 baseline 已同时纳入两类人工校正数据：

- `data/palm_datasets`：人工校验的 palm box 和 7 个 palm keypoints，用于 palm detector 检测精度。
- `data/handlm_datasets`：人工校正的 21 个手指关键点，用于 hand landmark 模型精度。

## 阅读顺序

1. [Pipeline 与 MediaPipe Graph](01_pipeline_graph.md)：完整两阶段链路、legacy graph 对应关系和验证分层。
2. [核心数据结构](02_data_structures.md)：`LetterboxInfo`、`Anchor`、`PalmDetection`、`HandRoi`、人工 21 点 GT 等结构。
3. [模型资产与模型特点](03_models.md)：full/lite 模型、模型接口、人工 GT 上的速度与精度对比。
4. [TFLite 模型误差分析](04_tflite_error_analysis.md)：PC 侧 TFLite baseline、TFLite 链路对齐和三组 TFLite 模型矩阵。
5. [Ascend 310B 移植路线](05_ascend310b_migration.md)：板端实现顺序、逐层对齐和模型选择建议。
6. [ONNX 模型误差分析](06_onnx_error_analysis.md)：ONNX opset、raw-output 对齐、ONNX 端到端 test 集验证。
7. [OM 模型误差分析](07_om_error_analysis.md)：OM raw-output 对齐、OM 端到端误差、板端运行时现象和后续优化方向。
8. [WebRTC 实时运行方案](08_webrtc_runtime.md)：从 case8 移植的 WebRTC/H.264/DVPP 方案，以及手部 OM 两阶段实时入口。

## 最新核心结论

| 验证项 | 数据/参考 | 关键结果 |
| --- | --- | --- |
| palm detector | 人工校验 `palm_datasets/test` | precision `0.967102`，recall `0.972361`，AP@0.50 `0.977699` |
| 两阶段链路 vs current Tasks | 当前 MediaPipe Tasks 参考输出 | mean `3.630226 px`，PCK@0.05 `0.908602` |
| 两阶段链路 vs legacy graph | `mediapipe==0.10.14` graph | mean `0.024968 px`，PCK@0.05 `0.999289` |
| legacy rect landmark | legacy graph 导出的官方 rect | mean `0.016921 px`，PCK@0.05 `0.999526` |
| landmark full vs 人工 GT | 人工校正 `handlm_datasets` | mean `5.940073 px`，PCK@0.05 `0.864738` |
| landmark lite vs 人工 GT | 人工校正 `handlm_datasets` | mean `6.602299 px`，PCK@0.05 `0.837741` |
| ONNX opset | `models/onnx/export_report_all.json` | 当前 ONNX 模型为 opset `11` |
| ONNX vs TFLite raw output | 固定随机输入 tensor | 所有模型最大 mean_abs 小于 `3e-5`，最大 max_abs 小于 `1.6e-4` |
| ONNX 端到端 `legacy_full` | `palm_datasets/test` | vs 同组 TFLite mean `0.0048 px`，p95 `0.0149 px` |
| 310B 优化 OM `legacy_full_palm` | `palm_datasets/test` 前 200 张 reference | AP50 与 TFLite 相同 `0.984348`，mAP `0.604666` vs TFLite `0.604634` |
| 310B 优化 OM raw output | `downsample_resize_maxpool_slices_origin_dtype.om` | raw box mean_abs `0.007474`，raw score mean_abs `0.002589` |

结论很清楚：

- MediaPipe Hand 是 palm detector + hand landmark 的两阶段 pipeline，不是单个端到端模型。
- PC 侧 TFLite 复刻链路已经基本对齐 `mediapipe==0.10.14` legacy graph。
- TFLite 转 ONNX 的 raw-output 和端到端误差都很小，ONNX 可以作为 ATC 转 OM 的输入基线。
- 原始 310B OM 的 palm detector raw output 偏差很大；通过改写 `Pad+Add` 下采样残差、`Resize(linear, half_pixel)` 和 `MaxPool`，当前优化 OM 已将 AP50 拉回到与 TFLite 一致，mAP 差异约 `0.000032`。
- 当前 raw box/keypoint 平均相对误差为 `0.062652%`，raw score 平均相对误差为 `0.025567%`，均低于 `0.1%`。
- 迁移时必须同时复刻模型推理和几何后处理：MediaPipe 风格的 detector `warpPerspective` 输入采样、SSD anchor、decode、weighted NMS、hand rect、旋转 ROI、landmark 反投影。

## 复现实验命令

本工程不需要 pip 安装，脚本会把项目根目录加入 `sys.path`。PC 侧 baseline 使用 `mediapipe_legacy` 环境：

```bash
conda activate mediapipe_legacy
python scripts/run_baseline.py --split test --run-matrix
```

快速验证部分图片：

```bash
python scripts/run_baseline.py --split test --max-images 300
```

保存部分可视化：

```bash
python scripts/run_baseline.py --split test --save-vis 8
```

显式传入数据：

```bash
python scripts/run_baseline.py \
  --data /path/to/palm_datasets \
  --handlm-data /path/to/handlm_datasets \
  --current-reference /path/to/mediapipe_predictions.json
```

PC baseline 输出目录：

```text
runs/baseline/model_info.json
runs/baseline/palm_detector/
runs/baseline/two_stage_vs_current_tasks/
runs/baseline/handlm_manual_gt/
runs/baseline/legacy_graph/
runs/baseline/two_stage_vs_legacy_graph/
runs/baseline/legacy_rect_landmark/
runs/baseline/tflite_matrix/
runs/baseline/verification_summary.md
runs/baseline/verification_summary.json
```

单项脚本可以直接调用：

```bash
python scripts/inspect_tflite.py --model-dir models/tflite
python scripts/eval_palm_tflite.py --split test --data ../data/palm_datasets --official-mediapipe references/current_tasks/mediapipe_predictions.json
python scripts/eval_handlm_tflite.py --data ../data/handlm_datasets --output-dir runs/baseline/handlm_manual_gt
python scripts/eval_two_stage_tflite.py --split test --data ../data/palm_datasets --official-mediapipe references/current_tasks/mediapipe_predictions.json
python scripts/eval_tflite_matrix.py --split test --data ../data/palm_datasets
python scripts/summarize_baseline.py --output-root runs/baseline
```

ONNX 导出与 raw-output 对齐：

```bash
conda activate mediapipe_legacy
python scripts/export_onnx.py --group all --output-report models/onnx/export_report_all.json
```

ONNX 端到端 test 集验证：

```bash
python scripts/eval_two_stage_onnx.py \
  --split test \
  --output-dir runs/onnx_two_stage/legacy_full \
  --save-vis 0

python scripts/eval_two_stage_onnx.py \
  --split test \
  --detector models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --landmark models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx \
  --reference-tflite runs/baseline/tflite_matrix/det_legacy_lite__lm_legacy_lite/predictions.json \
  --output-dir runs/onnx_two_stage/legacy_lite \
  --save-vis 0

python scripts/eval_two_stage_onnx.py \
  --split test \
  --detector models/onnx/mediapipe_task_hand_detector_full.onnx \
  --landmark models/onnx/mediapipe_task_hand_landmark_full.onnx \
  --reference-tflite runs/baseline/tflite_matrix/det_task_full__lm_task_full/predictions.json \
  --output-dir runs/onnx_two_stage/task_full \
  --save-vis 0
```

ONNX 端到端关键输出：

```text
runs/onnx_two_stage/summary.md
runs/onnx_two_stage/summary.csv
runs/onnx_two_stage/legacy_full/summary.json
runs/onnx_two_stage/legacy_lite/summary.json
runs/onnx_two_stage/task_full/summary.json
```

`legacy_full_palm` 专项分析：

```bash
# PC / mediapipe_legacy 环境：生成真实图片输入的 TFLite reference
python scripts/analyze_palm_om.py make-reference \
  --split test \
  --max-images 200 \
  --output-dir runs/palm_om/legacy_full_palm

# 从 PC/WSL 同步到 310B
rsync -av runs/palm_om/legacy_full_palm/ \
  310:~/Documents/artemis/mediapipe_hand_ascend310b/runs/palm_om/legacy_full_palm/

# 310B / base 环境：对同一批输入运行 OM 并逐层对齐
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/analyze_palm_om.py compare-om \
  --model models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om \
  --reference-dir runs/palm_om/legacy_full_palm \
  --output-dir runs/palm_om/legacy_full_palm/downsample_resize_maxpool_slices_origin_dtype_om_compare
```

310B 上的 OM 端到端 smoke：

```bash
python scripts/eval_two_stage_om.py \
  --data ~/Documents/artemis/data/palm_datasets \
  --split test \
  --max-images 20 \
  --output-dir runs/om_two_stage_smoke/legacy_full
```

310B 上的正式 full OM：

```bash
python scripts/eval_two_stage_om.py \
  --data ~/Documents/artemis/data/palm_datasets \
  --split test \
  --output-dir runs/om_two_stage/legacy_full
```

310B 上的正式 lite OM：

```bash
python scripts/eval_two_stage_om.py \
  --data ~/Documents/artemis/data/palm_datasets \
  --split test \
  --detector models/om/mediapipe_legacy_0_10_14_palm_detection_lite.om \
  --landmark models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om \
  --reference-tflite runs/baseline/tflite_matrix/det_legacy_lite__lm_legacy_lite/predictions.json \
  --output-dir runs/om_two_stage/legacy_lite
```

310B 上的正式 task full OM：

```bash
python scripts/eval_two_stage_om.py \
  --data ~/Documents/artemis/data/palm_datasets \
  --split test \
  --detector models/om/mediapipe_task_hand_detector_full.om \
  --landmark models/om/mediapipe_task_hand_landmark_full.om \
  --reference-tflite runs/baseline/tflite_matrix/det_task_full__lm_task_full/predictions.json \
  --output-dir runs/om_two_stage/task_full
```

OM 端到端关键输出：

```text
runs/palm_om/legacy_full_palm/downsample_resize_maxpool_slices_origin_dtype_om_compare_report.remote.md
runs/palm_om/legacy_full_palm/downsample_resize_maxpool_slices_origin_dtype_om_compare_summary.remote.json
runs/om_two_stage/legacy_full/summary.json
runs/om_two_stage/legacy_full/report.md
runs/om_two_stage/legacy_lite/summary.json
runs/om_two_stage/task_full/summary.json
runs/om_two_stage/summary.md
runs/om_two_stage/summary.csv
```

当前正式 OM 端到端验证不使用 `--keep-detector-loaded`。板端探针已经发现复用同一个 palm detector OM 模型句柄会导致 raw output 漂移，进而触发 decode/NMS 异常。


