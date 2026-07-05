# Lite 模型 20T OM 转换、误差与速度验证

本文记录 2026-07-05 在 Orange Pi AI Pro 20T 开发板上，对 `mediapipe==0.10.14` legacy lite 模型进行 ATC 转换、ONNX vs OM raw-output 对比、`video/test.mp4` 端到端对比，以及 NPU 推理时间测试的结果。

## 1. 结论

当前 lite 组合不能作为正式部署模型切换目标：

- `hand_landmark_lite_ascend310b1.om` 可以成功编译，raw-output 误差较小，单模型 NPU execute mean 约 `1.351 ms`。
- `palm_detection_lite_ascend310b1.om` 虽然生成了 OM 文件并可在 NPU 上执行，但检测结果不能和原始 lite ONNX 对齐。
- lite palm 的 full-style 优化 ONNX 已经生成，且 ONNX vs ONNX raw-output 基本等价；但该优化 ONNX 在当前 20T 板端 ATC 环境中没有生成可用 OM。

因此当前正式部署仍应使用已经完成精度验证的 full palm 优化 OM + full landmark OM。

## 2. 测试环境

| 项目 | 值 |
| --- | --- |
| 板卡 | Orange Pi AI Pro 20T |
| runtime SoC | `Ascend310B1` |
| Python | `/usr/local/miniconda3/bin/python` |
| ATC | `/usr/local/Ascend/ascend-toolkit/latest/bin/atc` |
| CANN env | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |

板端前置环境：

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 3. 20T ATC 转换结果

使用脚本：

```bash
python scripts/build_20t_om_models.py --soc-version auto --model-set legacy_lite
```

本次自动检测到：

```text
--soc_version=Ascend310B1
```

转换产物：

| 模型 | returncode | 耗时 | size_bytes | sha256 | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| `mediapipe_legacy_0_10_14_palm_detection_lite_ascend310b1.om` | `139` | `388.3 s` | `5372958` | `08cccc6dea0bda04ebe7653739c430a66dfb66ed3012a353170eeb4295fdc50d` | ATC 末尾段错误，但 OM 文件存在；后续精度验证失败 |
| `mediapipe_legacy_0_10_14_hand_landmark_lite_ascend310b1.om` | `0` | `277.8 s` | `5970361` | `d576cd85362d9b15d747a1d4b24c3086baa9c7b21f9202f8a9fb1f49a309fb29` | 编译成功 |

lite palm 日志末尾为：

```text
ATC run success
/usr/local/Ascend/ascend-toolkit/latest/bin/atc: line 17: 8100 Segmentation fault (core dumped) ${PKG_PATH}/bin/atc.bin "$@"
```

## 4. Lite Palm 优化 ONNX

已经按 full palm 正式 OM 的同款策略，生成了 lite palm 优化 ONNX：

```bash
python scripts/build_optimized_palm_om.py \
  --input-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --downsample-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample.onnx \
  --resize-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize.onnx \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

改写结果：

```text
downsample_rewrites: 3
resize_rewrites: 2
maxpool_rewrites: 4
```

正式保留的优化 ONNX：

| 文件 | SHA256 |
| --- | --- |
| `models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample.onnx` | `EB95AFEB7CB85F6EEF98CCECD842ECCAA6EC5B1F27C5D33AB21C46FAF6DC6305` |
| `models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize.onnx` | `5C725A13019B5CB715379B7AE166EFC7C2E9AB034BFDFC55119625FB4CF85FC1` |
| `models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx` | `CBA34C7B4EFDCAD2BABF8354CBF7FD0E33402239BB08224F5A25D69EC1A1BEF0` |

ONNX 等价性验证：

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

结论：优化 ONNX 本身没有改变 lite palm 语义，可以作为 ATC 输入候选。

## 5. 优化 Lite Palm ATC 状态

尝试将 `mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx` 转为 OM，当前没有得到可用 OM。

相关日志：

```text
runs/atc_20t/logs/legacy_lite_palm_downsample_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_downsample_resize_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_optimized_default_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_optimized_origin_dtype_single_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_origin_dtype_nopyc_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_allow_mix_precision_ascend310b1.log
runs/atc_20t/logs/legacy_lite_palm_force_fp16_ascend310b1.log
```

观察到的失败类型：

- `downsample` 和 `downsample_resize` 中间图均以 ATC `Segmentation fault` 结束，未生成 OM。
- full-style optimized 默认精度日志中出现 `ProcessAllFailedCompileTasks`，涉及 `resize_slices_*`、`maxpool_slices_*`、`depthwise_conv2d_*`、检测头 `Conv2D` 等，再以 ATC 段错误结束。
- `must_keep_origin_dtype`、`allow_mix_precision`、`force_fp16` 都没有生成可用优化 lite palm OM。

## 6. ONNX vs OM Raw-Output 对比

对 direct lite palm OM 的 raw-output 分两种方式测：

```bash
python scripts/compare_onnx_om_raw.py \
  --onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --om models/om/mediapipe_legacy_0_10_14_palm_detection_lite_ascend310b1.om \
  --shape 1,192,192,3 \
  --samples 10 \
  --warmup 2 \
  --output-dir runs/onnx_om_raw_compare/legacy_lite_palm_ascend310b1_reuse_check
```

复用同一个 OM 模型句柄时，输出会出现大幅漂移：

| output | max_abs | mean_abs | p95_abs | 判断 |
| ---: | ---: | ---: | ---: | --- |
| `0` | `65526.460938` | `6678.914197` | `65431.636719` | 失败 |
| `1` | `38839.199219` | `3979.297321` | `26202.860352` | 失败 |

加入 `--reload-om-each-sample` 后，漂移消失，但误差仍明显大于 full 优化 OM：

```bash
python scripts/compare_onnx_om_raw.py \
  --onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --om models/om/mediapipe_legacy_0_10_14_palm_detection_lite_ascend310b1.om \
  --shape 1,192,192,3 \
  --samples 10 \
  --warmup 2 \
  --reload-om-each-sample \
  --output-dir runs/onnx_om_raw_compare/legacy_lite_palm_ascend310b1_reload_check
```

| output | max_abs | mean_abs | p95_abs | 判断 |
| ---: | ---: | ---: | ---: | --- |
| `0` | `22.685806` | `1.314462` | `4.596044` | 仍需失败处理 |
| `1` | `3.369934` | `0.353768` | `1.303941` | 仍需失败处理 |

单次 fresh-load 检查没有发现 NaN、FP16 饱和或 `655xx` 级异常：

```text
runs/onnx_om_raw_compare/legacy_lite_palm_ascend310b1/value_stats.json
```

## 7. `video/test.mp4` 端到端对比

命令：

```bash
python scripts/compare_video_onnx_om.py \
  --video video/test.mp4 \
  --onnx-detector models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx \
  --onnx-landmark models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx \
  --om-detector models/om/mediapipe_legacy_0_10_14_palm_detection_lite_ascend310b1.om \
  --om-landmark models/om/mediapipe_legacy_0_10_14_hand_landmark_lite_ascend310b1.om \
  --output-dir runs/video_onnx_om_compare/legacy_lite_ascend310b1_reload_detector_first_50 \
  --reload-detector-each-frame \
  --save-vis 4 \
  --max-frames 50
```

关键结果：

| 指标 | 值 |
| --- | ---: |
| processed_frames | `50` |
| matched_hands | `36` |
| onnx_unmatched_hands | `23` |
| om_unmatched_hands | `1` |
| count_mismatch_rate | `0.44` |
| box_mean_abs_px_mean | `59.9548 px` |
| palm7_mean_px_mean | `83.4680 px` |
| hand21_mean_px_mean | `78.0445 px` |
| consistent | `false` |

即使每帧重载 detector，direct lite palm OM 仍不能和原始 lite ONNX 端到端对齐。

## 8. 推理速度

测速命令：

```bash
python scripts/benchmark_om_inference.py \
  --warmup 20 \
  --iterations 200 \
  --model models/om/mediapipe_legacy_0_10_14_palm_detection_lite_ascend310b1.om \
  --model models/om/mediapipe_legacy_0_10_14_hand_landmark_lite_ascend310b1.om
```

| 模型 | execute mean | execute p95 | h2d+execute mean | full mean | full p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lite palm direct OM | `1.562 ms` | `1.581 ms` | `1.652 ms` | `2.192 ms` | `2.218 ms` |
| lite landmark OM | `1.351 ms` | `1.378 ms` | `1.468 ms` | `2.045 ms` | `2.082 ms` |

lite palm 的速度很快，但因为精度链路失败，不能作为有效部署性能数据使用。

## 9. 部署建议

- 不要把 `mediapipe_legacy_0_10_14_palm_detection_lite_ascend310b1.om` 放入正式 WebRTC/视频部署默认路径。
- 可以保留 `hand_landmark_lite_ascend310b1.om` 作为后续组合实验材料，但在缺少可用 lite palm detector OM 前，不建议切换 lite 两阶段组合。
- 当前生产配置继续使用 full 优化 palm OM，因为它已经在 raw-output、人工数据和视频链路上完成一致性验证。
- lite palm 的下一步应聚焦 ATC/TBE 编译问题或更小子图复现，而不是继续直接使用已生成的 direct lite palm OM。
