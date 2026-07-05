# MediaPipe Hand Ascend 310B 文档

这组文档面向 `mediapipe_hand_ascend310b` 独立子工程，解释 MediaPipe Hand 的算法、数据结构、模型特点、PC 侧 TFLite baseline、ONNX 转换误差、Ascend 310B OM 误差和移植路线。

注意：本目录保留的是迁移和误差分析历史记录，其中部分 PC 侧 baseline、TFLite 评估和临时探针脚本已不再放入正式 310B 部署包。当前可运行入口以仓库根目录 `README.md` 和 `scripts/README.md` 为准。

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
9. [20T 开发板 OM 重编译与推理时间记录](09_20t_om_benchmark.md)：在 Orange Pi AI Pro 20T 上重编译 `Ascend310B1` OM、对比旧 OM 和 20T OM 的推理时间与输出一致性。
10. [Lite 模型 20T OM 转换、误差与速度验证](10_lite_om_20t_validation.md)：lite palm/landmark 在 20T 上的 ATC 转换、ONNX vs OM raw-output、视频端到端对比和速度测试。
11. [Palm Full/Lite ONNX 算子结构差异](11_palm_full_lite_operator_diff.md)：对比 legacy palm full/lite 的 op_type、Conv block、Resize/Pad/MaxPool 敏感结构和检测头。
12. [Lite Palm 问题定位与优化 ONNX](12_lite_palm_issue_localization.md)：定位 direct lite palm OM 的复用漂移和 fresh-load 精度问题，并记录可用于 ATC 的 full-style lite palm 优化 ONNX。
13. [Lite Palm Split ONNX 与 ATC 后续定位](13_lite_palm_split_atc_followup.md)：记录 identity-bridge split ONNX、split ATC 失败结果，以及当前板端 ATC/TBE 环境复测结论。
14. [8T 开发板 Full OM 推理时间记录](14_8t_full_om_benchmark.md)：在 `ascend8t` 上按 20T 相同流程测试 full palm/landmark OM 的推理时间。
15. [Lite Palm 8T OM 优化、误差与速度](15_lite_palm_8t_om_optimization.md)：在 `ascend8t` 上无 graph parallel 编译 optimized lite palm OM，并记录 ONNX vs OM raw-output 误差和推理速度。
16. [20T Portable HaGRIDv2 OM 数据集精度与速度验证](16_20t_hf_dataset_om_validation.md)：在 `ascend20t` 上用 `1663` 张真实图片全量评估 full/lite OM 的精度和真实链路速度。

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
| 20T 重编译 OM | `Ascend310B1` ATC + 20 组随机输入 | 与旧 OM 原始输出完全一致，`max_abs=0` |
| 20T OM 推理时间 | `warmup=20, iterations=200` | palm execute mean `11.532 ms`，landmark execute mean `1.891 ms`，未比旧 OM 更快 |
| 8T full OM 推理时间 | `ascend8t`, `warmup=20, iterations=200` | palm execute mean `26.931 ms`，landmark execute mean `3.889 ms` |
| 20T lite OM | `legacy_lite` ONNX vs OM | lite palm direct OM raw-output 误差极大，视频端到端不一致；lite landmark OM 可单独运行 |
| palm full/lite 算子差异 | ONNX graph-level compare | lite 没有新增 full 不存在的 op_type，主要是每个尺度少一个 residual bottleneck，总计少 `10 Conv + 5 Add + 5 PRelu` |
| lite palm 优化 ONNX | 原始 lite ONNX vs full-style optimized lite ONNX | boxes max_abs `5.340576e-05`，scores max_abs `8.583069e-06`；ONNX 语义等价，并已在 8T 上生成可执行优化 OM |
| lite palm split ONNX | 原始 lite ONNX vs identity-bridge stage1+stage2 | 本地 20 组随机输入输出误差全为 `0`；但 20T ATC 编译 stage1/stage2 均 returncode `139`，未生成 split OM |
| 8T optimized lite palm OM | `ascend8t` CANN 8.3, no graph parallel, `must_keep_origin_dtype` | 100 组随机输入：boxes mean_rel `0.829947%`，scores mean_rel `0.014741%`，boxes p95_rel `1.046142%`；execute mean `24.799 ms` |
| 20T full OM 数据集验证 | Portable HaGRIDv2 MediaPipe test `1663` 张 | full passed；precision `0.996399`，recall `0.997596`，AP50 `0.994933`，full21 mean `0.127865 px`，total mean `21.353 ms` |
| 20T lite OM 数据集验证 | Portable HaGRIDv2 MediaPipe test `1663` 张 | lite report-only；precision `0.978877`，recall `0.974760`，AP50 `0.983212`，full21 mean `1.318512 px`，total mean `19.900 ms` |

结论很清楚：

- MediaPipe Hand 是 palm detector + hand landmark 的两阶段 pipeline，不是单个端到端模型。
- PC 侧 TFLite 复刻链路已经基本对齐 `mediapipe==0.10.14` legacy graph。
- TFLite 转 ONNX 的 raw-output 和端到端误差都很小，ONNX 可以作为 ATC 转 OM 的输入基线。
- 原始 310B OM 的 palm detector raw output 偏差很大；通过改写 `Pad+Add` 下采样残差、`Resize(linear, half_pixel)` 和 `MaxPool`，当前优化 OM 已将 AP50 拉回到与 TFLite 一致，mAP 差异约 `0.000032`。
- 当前 raw box/keypoint 平均相对误差为 `0.062652%`，raw score 平均相对误差为 `0.025567%`，均低于 `0.1%`。
- 在 Orange Pi AI Pro 20T 上用 `Ascend310B1` 重新 ATC 编译的 OM 与旧正式 OM 原始输出完全一致，推理速度基本持平；当前不需要按 8T/20T 硬件版本维护多套 OM。
- `legacy_lite` 的 direct palm OM 虽可生成并执行，但 ONNX vs OM raw-output 和 `video/test.mp4` 端到端结果均不一致，不能用于正式部署。
- `legacy_lite` direct palm OM 复用模型句柄会出现 `655xx` 级 raw-output 漂移；每次 fresh-load 可消除该漂移，但 raw-output 仍不足以通过视频端到端一致性验证。
- `legacy_lite` palm 与 full 使用同一套敏感结构：`Resize`、tail channel `Pad+Add`、`MaxPool` 的数量和 shape 一致；lite 主要是 backbone 更浅。
- 已创建 `mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx`，它与原始 lite ONNX 等价，可作为后续 ATC/TBE 问题定位输入。
- 已创建 identity-bridge split lite palm ONNX，stage1/stage2 在本地严格等价；20T 板端 ATC/TBE 未能生成 split OM。
- 在 `ascend8t` 上，full-style optimized lite palm ONNX 已成功生成 `must_keep_origin_dtype` OM；raw-output 平均相对误差低于 `1%`，但 boxes p95_rel `1.046142%`，速度接近 full palm，收益有限。
- 当前板端复测中，原始 lite palm ONNX 使用旧 `build_20t_om_models.py` 重新 ATC 也返回 `139`，说明阻塞不只来自新 split/optimized ONNX。
- 在 `ascend20t` 上，full OM 使用 Portable HaGRIDv2 MediaPipe test 全量 `1663` 张图片通过正式验收；真实两阶段链路平均 `21.353 ms/image`。lite OM 可运行，平均 `19.900 ms/image`，但精度明显低于 full，默认仍作为 report-only。
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


