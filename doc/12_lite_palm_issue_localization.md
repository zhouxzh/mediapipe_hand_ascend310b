# Lite Palm 问题定位与优化 ONNX

本文记录 2026-07-05 对 legacy lite palm detector 的定位结论，以及为 ATC 转 OM 准备的优化 ONNX。

## 1. 问题定位

当前有两个独立问题：

1. direct lite palm OM 复用同一个 ACL 模型句柄时，raw output 会漂移到 `655xx` 量级。这个现象不是 ONNX 模型本身的问题，而是 OM 运行时复用路径的问题。`scripts/compare_onnx_om_raw.py` 已新增 `--reload-om-each-sample` 用于区分这个问题。
2. 每个样本重新加载 direct lite palm OM 后，`655xx` 异常消失，但 raw-output 仍明显偏离原始 lite ONNX。boxes `mean_abs=1.314462`，scores `mean_abs=0.353768`，在 `video/test.mp4` 上仍然导致端到端不一致。

因此，不能把问题简单归结为“测试脚本复用模型句柄”。即使规避复用漂移，direct lite palm OM 的数值误差仍会破坏 detector decode/NMS/ROI 链路。

## 2. Lite 和 Full 的结构关系

`doc/11_palm_full_lite_operator_diff.md` 已确认：

- lite 没有引入 full 不存在的新 op_type。
- lite 主要是 backbone 更浅：每个主干尺度少一个 repeated residual bottleneck，总计少 `10 Conv + 5 Add + 5 PRelu`。
- full 中已经证明敏感的 `Resize`、tail channel `Pad + Add`、`MaxPool` 在 lite 中数量和 shape 都一致。

所以 lite palm 的合理优化方向是先复用 full palm 已验证的三类图改写。

## 3. 已创建的优化 ONNX

正式保留的 lite palm 优化链路：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx
  -> models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample.onnx
  -> models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize.onnx
  -> models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

改写内容：

| 阶段 | 脚本 | 数量 | 目的 |
| --- | --- | ---: | --- |
| downsample residual | `scripts/rewrite_palm_downsample_residual.py` | `3` | 消除 `Pad(pool)+branch` 对尾部零通道 Pad 的依赖 |
| resize | `scripts/rewrite_palm_bilinear_resize.py` | `2` | 用显式 Slice/Add/Concat 复刻 `Resize(linear, half_pixel)` |
| maxpool | `scripts/rewrite_palm_maxpool_slices.py` | `4` | 用 Slice + Max 复刻 `MaxPool` |

生成命令：

```bash
python scripts/build_optimized_palm_om.py \
  --input-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --downsample-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample.onnx \
  --resize-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize.onnx \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

## 4. ONNX 等价性

优化 ONNX 已经和原始 lite ONNX 做 raw-output 对比：

```bash
python scripts/compare_onnx_raw.py \
  --left models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --right models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx \
  --shape 1,192,192,3 \
  --samples 10 \
  --output-dir runs/onnx_raw_compare/lite_palm_original_vs_full_style_optimized
```

结果：

| output | shape | max_abs | mean_abs | p95_abs |
| ---: | --- | ---: | ---: | ---: |
| `0` | `1x2016x18` | `5.340576e-05` | `3.211731e-06` | `1.001358e-05` |
| `1` | `1x2016x1` | `8.583069e-06` | `1.155112e-06` | `2.861023e-06` |

这个量级和浮点重排噪声一致，说明 `mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx` 可以作为 ATC 输入候选。

## 5. ATC 现状

当前 20T 板端 ATC 没有成功把该优化 ONNX 编译成可用 OM：

- `downsample.onnx`：ATC 段错误，无 OM。
- `downsample_resize.onnx`：ATC 段错误，无 OM。
- `downsample_resize_maxpool_slices.onnx`：默认精度日志出现大量 `ProcessAllFailedCompileTasks`，涉及 `resize_slices_*`、`maxpool_slices_*`、`depthwise_conv2d_*`、检测头 `Conv2D`，随后 ATC 段错误。
- `must_keep_origin_dtype`、`allow_mix_precision`、`force_fp16` 均未生成可用优化 OM。
- direct lite ONNX 额外尝试 `precision_mode=force_fp32`、`precision_mode_v2=origin`、`op_select_implmode=high_precision_for_all`、`disable_reuse_memory=1`、`enable_single_stream=true`，均没有生成新的可用 OM。
- 追加尝试了 `downsample_splitop` 候选：用 ONNX `Split` 替代两个 `Slice` 来降低下采样 residual 改写复杂度。ONNX 侧可运行，但 ATC 以 `free(): invalid next size (fast)` 退出，未生成 OM。

相关日志在：

```text
runs/atc_20t/logs/legacy_lite_palm_downsample_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_downsample_resize_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_optimized_default_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_optimized_origin_dtype_single_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_origin_dtype_nopyc_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_allow_mix_precision_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_force_fp16_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_high_precision_env_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_origin_v2_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_no_reuse_single_stream_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_downsample_splitop_ascend310b1.log
```

## 6. 排查脚本

本次保留的有用脚本：

| 脚本 | 用途 |
| --- | --- |
| `scripts/compare_onnx_raw.py` | ONNX vs ONNX raw-output 等价性检查 |
| `scripts/compare_onnx_om_raw.py` | ONNX vs OM raw-output 检查，支持 `--reload-om-each-sample` |
| `scripts/make_onnx_debug_outputs.py` | 为 ONNX 临时增加或替换 debug graph output |
| `scripts/run_lite_palm_debug_sweep.py` | 批量生成单输出 debug ONNX 并尝试 ATC |
| `scripts/inspect_lite_palm_om_outputs.py` | 检查 direct lite palm OM 输出范围、NaN、FP16 饱和和大值异常 |
| `scripts/run_clean_atc.py` | 用可控环境、单线程参数和额外 ATC 选项编译单个 ONNX |
| `scripts/fit_lite_palm_output_calibration.py` | 验证 direct lite OM 输出端 affine 校准是否能把误差压到目标内 |
| `scripts/om_infer_once.py` | 单次 OM 推理子进程，用于规避 ACL 释放阶段崩溃对采样的影响 |
| `scripts/rewrite_palm_downsample_split_op.py` | 生成 Split-op 版 downsample residual 改写候选 |

已删除不适合正式保留的 Pad-only 实验改写脚本，因为 ONNX raw-output 检查证明它们不是严格等价变换。

## 7. 输出端校准结果

尝试对 direct lite palm OM 输出做 affine 校准：

```bash
python scripts/fit_lite_palm_output_calibration.py \
  --train-samples 16 \
  --eval-samples 8 \
  --mode element \
  --om-subprocess \
  --output runs/lite_palm_calibration/element_subproc_16_8_summary.json
```

held-out 结果：

| output | before mean_abs/mean_target_abs | after mean_abs/mean_target_abs | 结论 |
| ---: | ---: | ---: | --- |
| `0` boxes | `14.96%` | `7.04%` | 仍高于 1% |
| `1` scores | `4.35%` | `2.08%` | 仍高于 1% |

因此误差不是简单固定输出 scale/bias，输出端校准不能满足“OM 相对原始 ONNX 误差小于 1%”这个目标。

## 8. 当前结论

已经创建了可用于 ATC 的 full-style lite palm 优化 ONNX：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

它与原始 lite ONNX 等价；当前阻塞点不在 ONNX 改写语义，而在 20T 板端 ATC/TBE 对 lite 优化图的编译稳定性。direct lite OM 和输出端校准均未达到 1% 误差目标，正式部署仍使用 full 优化 palm OM。

