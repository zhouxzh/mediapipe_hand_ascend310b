# 8T 开发板 Full OM 推理时间记录

本文记录 2026-07-05 在 8T 版 Ascend 310B 开发板 `ascend8t` 上，按 20T 开发板相同流程测试 full palm detector 和 full hand landmark OM 的推理时间。

## 1. 测试目的

在继续优化 lite palm OM 之前，先建立 8T 开发板上的 full 模型性能基线，用来和 20T 开发板结果对照。

测试对象：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

## 2. 测试环境

| 项目 | 值 |
| --- | --- |
| SSH host | `ascend8t` |
| hostname | `orangepiaipro` |
| runtime SoC | `Ascend310B4` |
| Python | `/usr/local/miniconda3/bin/python` |
| ACL Python | `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/acl.so` |
| ATC | `/usr/local/Ascend/ascend-toolkit/latest/bin/atc` |
| warmup | `20` |
| iterations | `200` |
| input fill | `zeros` |

`npu-smi` 显示 `Health=Alarm`，该项按已确认的板端硬件告警现象忽略，不作为过热或性能异常依据。

## 3. 测试命令

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python scripts/benchmark_om_inference.py \
  --warmup 20 \
  --iterations 200 \
  --model models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om \
  --model models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

输出报告：

```text
runs/om_inference_benchmark/om_benchmark_20260705_153034.json
```

## 4. 模型信息

| 模型 | compile_soc_version | size_bytes | sha256 |
| --- | --- | ---: | --- |
| optimized full palm | `Ascend310B4` | `12787549` | `0e1b3499955c4e25d69f975961c144eae5494b7fc34cbedd57f504ee4dcdff24` |
| full landmark | `Ascend310B4` | `11669346` | `2cba28875b224be8ddfb88a5364641f04eb24d094db9658070ba3d0c4ac2f3f6` |

## 5. 8T 推理时间

| 模型 | 输入 | execute mean | execute p95 | h2d+execute mean | full mean | full p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| optimized full palm | `1x192x192x3 fp32` | `26.931 ms` | `27.010 ms` | `27.018 ms` | `27.738 ms` | `27.847 ms` |
| full landmark | `1x224x224x3 fp32` | `3.889 ms` | `3.930 ms` | `4.015 ms` | `4.641 ms` | `4.738 ms` |

指标含义：

| 指标 | 含义 |
| --- | --- |
| `execute_ms` | warmed-up `acl.mdl.execute` |
| `h2d_execute_ms` | 输入 host-to-device copy + execute |
| `full_ms` | 当前 Python runner 路径，包括输入 copy、execute、输出 copy 和 numpy 转换 |

## 6. 与 20T 测试结果对照

20T 结果见 `doc/09_20t_om_benchmark.md`。同一 benchmark 流程下：

| 模型 | 8T execute mean | 20T execute mean | 倍率 |
| --- | ---: | ---: | ---: |
| optimized full palm | `26.931 ms` | `11.523 ms` | `2.34x` |
| full landmark | `3.889 ms` | `1.865 ms` | `2.09x` |

8T 板的 full 模型推理时间明显慢于 20T 板，符合硬件算力差异预期。当前 full 模型性能基线为：

```text
palm detector full: execute mean 26.931 ms
hand landmark full: execute mean 3.889 ms
```

后续 lite palm OM 优化应在 8T 板上继续，因为该板与最初 full optimized OM 的编译环境一致，优先排除 20T 板端 ATC/TBE 环境不稳定对 lite 优化的干扰。

