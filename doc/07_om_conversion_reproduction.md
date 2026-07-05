# OM 转换与复现

本文记录当前正式 full palm OM 的转换策略和复现命令。更早的 raw-output 调试和失败日志已合并到本文，不再单独保留。

## 正式模型

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

`mediapipe_legacy_0_10_14_palm_detection_full.om` 这类 direct palm OM 曾出现明显 raw-output 偏差，已从模型目录删除，不作为部署或参考模型保留。

## 为什么 palm ONNX 要改写

full palm direct ONNX 转 OM 后，palm detector 的 raw boxes/scores 与 ONNX reference 不一致，进而影响 decode、NMS、ROI 和最终 landmark。当前正式 palm OM 使用以下等价图改写降低 310B ATC/runtime 误差：

| 改写 | 目的 |
| --- | --- |
| downsample residual `Pad + Add` 改写 | 避免下采样残差分支在 ATC 后输出污染 |
| bilinear `Resize(linear, half_pixel)` 改写 | 避免 half-pixel 双线性上采样在 OM 中与 ONNX 不一致 |
| `MaxPool` slices 改写 | 让 `must_keep_origin_dtype` 可用于相关子图，稳定 raw-output |

转换链路：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx
  -> models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
  -> models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
```

## 复现 optimized ONNX

在 PC 或板端 Python 环境中运行：

```bash
python scripts/build_optimized_palm_om.py \
  --input-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
```

## 板端 ATC 编译

在 Ascend 310B 开发板：

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

编译正式 full palm OM：

```bash
python scripts/build_optimized_palm_om.py \
  --skip-rewrite \
  --compile-om \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx \
  --om-output models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype \
  --atc-log runs/palm_om/atc_logs/downsample_resize_maxpool_slices_origin_dtype.log \
  --precision-mode must_keep_origin_dtype
```

## 20T 重编译结论

在 Orange Pi AI Pro 20T 上按 `Ascend310B1` 重新 ATC 编译 full palm 和 full landmark 后，与现有正式 OM raw-output 完全一致：

```text
palm     samples=20 outputs_checked=40 max_abs=0.0 mean_abs_avg=0.0
landmark samples=20 outputs_checked=80 max_abs=0.0 mean_abs_avg=0.0
```

因此当前不保留 `*_ascend310b1.om` 这类重复版本。只有在未来出现加载/执行不兼容，或专用 SoC OM 实测更快且数值一致时，才重新引入按板卡命名的 OM。

## 验证入口

视频 ONNX/OM 回归：

```bash
python scripts/run_video_onnx_om_checks.py --video video/test.mp4 --max-frames 5 --save-vis 0
```

数据集正式评估：

```bash
python scripts/eval_hf_hand_dataset_om.py --model-set full,lite
```

详细板端结果见 [05_board_validation_results.md](05_board_validation_results.md)。
