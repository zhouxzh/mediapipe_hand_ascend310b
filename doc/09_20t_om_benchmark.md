# 20T 开发板 OM 重编译与推理时间记录

本文记录 2026-07-05 在 Orange Pi AI Pro 20T 开发板上，对当前部署 OM 进行板端 ATC 重编译、推理时间 benchmark、以及旧 OM 与 20T 重编译 OM 的输出一致性验证。

## 1. 测试目的

已有正式部署模型是在另一块 Ascend 310B 开发板上编译得到，编译记录中的目标 SoC 为 `Ascend310B4`。当前测试板为 Orange Pi AI Pro 20T，运行时 `acl.get_soc_name()` 返回：

```text
Ascend310B1
```

本次验证回答两个问题：

- `Ascend310B4` 目标编译出的旧 OM 是否能在 20T 板上稳定运行。
- 用当前 20T 板端 ATC 按 `Ascend310B1` 重新编译后，推理结果或速度是否有实质差异。

## 2. 测试环境

| 项目 | 值 |
| --- | --- |
| 板卡 | Orange Pi AI Pro 20T |
| 系统 | Ubuntu 22.04.3 LTS, aarch64 |
| runtime SoC | `Ascend310B1` |
| ATC 路径 | `/usr/local/Ascend/ascend-toolkit/latest/bin/atc` |
| Python | `/usr/local/miniconda3/bin/python` |
| ACL Python | `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/acl.so` |

板端运行前置环境：

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 3. 模型文件

旧部署 OM：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

20T 板端重编译 OM：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype_ascend310b1.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full_ascend310b1.om
```

20T OM 当前同步到本地后的实际 hash：

| 模型 | size_bytes | sha256 |
| --- | ---: | --- |
| `palm_detection_full_downsample_resize_maxpool_slices_origin_dtype_ascend310b1.om` | `12787491` | `bdf3ad98feb39b55c43947e2273bae93bdd582074731277bac1c26417a1b30ce` |
| `hand_landmark_full_ascend310b1.om` | `11668928` | `efeafb129b0ec514d6d78c961f55c630a7ddce793041c9f409dc5d221fd81965` |

同步清单：

```text
runs/atc_20t/synced_20t_om_manifest_20260705.json
```

## 4. 20T ATC 编译

编译脚本：

```bash
python scripts/build_20t_om_models.py --soc-version auto
```

该脚本读取 `acl.get_soc_name()`，本次自动使用：

```text
--soc_version=Ascend310B1
```

编译报告：

```text
runs/atc_20t/compile_20t_om_20260705_102254.json
```

编译日志：

```text
runs/atc_20t/logs/optimized_palm_ascend310b1.log
runs/atc_20t/logs/legacy_full_landmark_ascend310b1.log
```

编译耗时：

| 模型 | ATC 目标 | 耗时 |
| --- | --- | ---: |
| optimized palm | `Ascend310B1` | `533.7 s` |
| legacy full landmark | `Ascend310B1` | `272.0 s` |

说明：

- 编译过程中 ATC 报出多条 `W18888 Check output param size failed` warning，但最终 `ATC run success`。
- 首次失败原因是 ATC 内部调用 `/usr/bin/python3` 时找不到可用 numpy。脚本已将当前 conda Python 的 `bin`、`lib` 和 `site-packages` 加入 ATC 子进程环境，避免安装额外包。

## 5. 推理时间 Benchmark

测试脚本：

```bash
python scripts/benchmark_om_inference.py --warmup 20 --iterations 200
```

指定 20T OM：

```bash
python scripts/benchmark_om_inference.py \
  --warmup 20 \
  --iterations 200 \
  --model models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype_ascend310b1.om \
  --model models/om/mediapipe_legacy_0_10_14_hand_landmark_full_ascend310b1.om
```

指标含义：

| 指标 | 含义 |
| --- | --- |
| `execute_ms` | 只计 warmed-up `acl.mdl.execute` |
| `h2d_execute_ms` | 输入 host-to-device 拷贝 + `acl.mdl.execute` |
| `full_ms` | 当前 Python runner 路径，包括输入拷贝、execute、输出拷贝和 numpy 转换 |

### 5.1 旧 OM 结果

报告：

```text
runs/om_inference_benchmark/om_benchmark_20260705_100107.json
```

| 模型 | 输入 | execute mean | execute p95 | full mean | full p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| old optimized palm | `1x192x192x3 fp32` | `11.523 ms` | `11.564 ms` | `12.204 ms` | `12.257 ms` |
| old landmark full | `1x224x224x3 fp32` | `1.865 ms` | `1.899 ms` | `2.565 ms` | `2.607 ms` |

### 5.2 20T OM 结果

报告：

```text
runs/om_inference_benchmark/om_benchmark_20260705_102640.json
```

| 模型 | 输入 | execute mean | execute p95 | full mean | full p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `ascend310b1` optimized palm | `1x192x192x3 fp32` | `11.532 ms` | `11.574 ms` | `12.219 ms` | `12.279 ms` |
| `ascend310b1` landmark full | `1x224x224x3 fp32` | `1.891 ms` | `1.937 ms` | `2.586 ms` | `2.629 ms` |

## 6. 旧 OM 与 20T OM 输出一致性

在 20T 板上使用同一批随机输入，分别运行旧 OM 和 `ascend310b1` OM，直接比较原始输出 tensor。

单个随机输入结果：

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

说明：20 组比较完成并打印结果后，Python ACL 进程在释放阶段出现一次 `Segmentation fault`。输出比较已经完成，差异为 0；该问题属于 Python ACL 资源释放稳定性现象，不影响本次数值结论。

## 7. 结论

| 问题 | 结论 |
| --- | --- |
| 旧 OM 能否在 20T 板运行 | 可以，`Ascend310B1` runtime 上加载和执行正常 |
| 20T 重编译 OM 是否输出不同 | 当前测试下原始输出完全一致，`max_abs=0` |
| 20T 重编译 OM 是否更快 | 没有，速度与旧 OM 基本持平 |
| 是否需要按 8T/20T 维护多套 OM | 当前不需要 |

当前建议：

- 默认生产部署继续使用已经完成 ONNX/OM 精度验证的旧正式 OM。
- 20T 重编译 OM 可保留为归档和兼容性验证材料，但不作为默认切换目标。
- 只有在某块板子出现 OM 加载/执行不兼容、或专用 SoC OM 实测明显更快时，才需要按硬件版本维护多套 OM。

当前主要推理瓶颈仍是 palm detector，纯 `acl.mdl.execute` 约 `11.5 ms`；landmark full 约 `1.9 ms`。
