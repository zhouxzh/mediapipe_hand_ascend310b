# Lite Palm Split ONNX 与 ATC 后续定位

本文记录 2026-07-05 对 lite palm detector 的进一步定位：目标是创建一个类似 full 优化路径、但仍严格等价于原始 lite ONNX 的候选图，用于 ATC 转 OM 后和原始 ONNX 对比精度与速度。

## 1. 当前目标

要验证的是：

- 原始模型：`models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx`
- 候选 OM：由优化 lite ONNX 或 split lite ONNX 在 20T 开发板上 ATC 转换得到
- 验证标准：OM raw output 对原始 ONNX 的误差小于 1%，并测量 OM 推理耗时

截至本记录，ONNX 候选图已经完成并验证等价，但 ATC 没有生成可用 OM。因此还不能给出“优化 lite OM vs 原始 lite ONNX”的精度和速度结论。

## 2. Full-Style 优化 ONNX

保留的 full-style lite 优化图：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
```

本地图等价验证结果：

| output | shape | max_abs | mean_abs | p95_abs |
| ---: | --- | ---: | ---: | ---: |
| `0` boxes | `1x2016x18` | `5.340576e-05` | `3.211731e-06` | `1.001358e-05` |
| `1` scores | `1x2016x1` | `8.583069e-06` | `1.155112e-06` | `2.861023e-06` |

这说明 full-style 改写没有改变 lite ONNX 的语义，可以作为 ATC 输入候选。

20T 板端 ATC 结果：

```text
runs/atc_20t/legacy_lite_palm_optimized_clean_ascend310b1.json
runs/atc_20t/lite_palm_fullstyle_ascend310b1_pycache.json
```

关键结论：

- `returncode=139` 或 `returncode=255`
- 未生成 `mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_*.om`
- 新日志中出现过 TBE 初始化错误：`source code string cannot contain null bytes`
- 使用隔离 pycache 后，该 Python 错误消失，但 `atc.bin` 仍会段错误

## 3. Split ONNX 候选

为了避开 ATC 对完整 lite 图的处理路径，新增了按 FPN 特征切分的候选图。

切分点：

| clean name | original tensor | shape |
| --- | --- | --- |
| `fpn_24x24` | `model_1/model/p_re_lu_11/.../mul1` | `1x128x24x24` |
| `fpn_12x12` | `model_1/model/p_re_lu_15/.../mul1` | `1x256x12x12` |
| `fpn_6x6` | `model_1/model/p_re_lu_19/.../mul1` | `1x256x6x6` |

生成的 identity-bridge split ONNX：

```text
runs/onnx_split_debug/identity_bridge/lite_stage1.onnx
runs/onnx_split_debug/identity_bridge/lite_stage2.onnx
runs/onnx_split_debug/identity_bridge/lite_stage1.split_manifest.json
runs/onnx_split_debug/identity_bridge/lite_stage2.split_manifest.json
```

本地 `mediapipe_legacy` 环境验证结果：

```text
runs/onnx_raw_compare/lite_palm_identity_bridge_local/summary.json
runs/onnx_split_debug/lite_palm_identity_bridge_local/summary.json
```

20 个随机输入下：

- stage1 边界张量误差：全部 `0`
- stage2 使用 full 边界输入的误差：全部 `0`
- stage1 + stage2 pipeline 对原始 lite ONNX 的输出误差：全部 `0`

这说明 split ONNX 本身是严格等价的。

## 4. Split ONNX 的 ATC 结果

使用 20T 板端 ATC 编译 identity split：

```bash
python scripts/build_lite_palm_split_om.py \
  --soc-version Ascend310B1 \
  --suffix ascend310b1 \
  --variant identity \
  --stage1-model runs/onnx_split_debug/identity_bridge/lite_stage1.onnx \
  --stage2-model runs/onnx_split_debug/identity_bridge/lite_stage2.onnx \
  --env-mode python_runtime \
  --cache-mode force
```

结果：

| stage | report | returncode | OM |
| --- | --- | ---: | --- |
| stage1 identity | `runs/atc_20t/lite_palm_split_stage1_identity_ascend310b1.json` | `139` | 未生成 |
| stage2 identity | `runs/atc_20t/lite_palm_split_stage2_identity_ascend310b1.json` | `139` | 未生成 |
| stage1 identity + pycache isolation | `runs/atc_20t/lite_palm_split_stage1_identity_ascend310b1_pycache.json` | `139` | 未生成 |
| stage2 identity + pycache isolation | `runs/atc_20t/lite_palm_split_stage2_identity_ascend310b1_pycache.json` | `139` | 未生成 |

相关日志：

```text
runs/atc_20t/logs/lite_palm_split_stage1_identity_ascend310b1.log
runs/atc_20t/logs/lite_palm_split_stage2_identity_ascend310b1.log
runs/atc_20t/logs/lite_palm_split_stage1_identity_ascend310b1_pycache.log
runs/atc_20t/logs/lite_palm_split_stage2_identity_ascend310b1_pycache.log
```

结论：split ONNX 在本地严格等价，但当前 20T 板端 ATC 没有生成 split OM，因此不能继续做 split OM 的 raw-output 精度和推理速度测试。

## 5. 板端 ONNX Runtime 现象

板端 ONNX Runtime 对拆分图和 full-style 改写图的输出与本地 ONNX Runtime 不一致。例如 identity split 在板端 smoke 中出现 boxes `1e20` 级 max_abs。

因此：

- 本地 ONNX Runtime 可用于判断 rewrite/split 语义是否正确
- 板端 ONNX Runtime 当前不适合作为 split ONNX 的精度基准
- 最终验收仍应使用原始 ONNX reference 对比 OM 输出

## 6. ATC/TBE 环境复测

为了区分“新 ONNX 图问题”和“当前 ATC/TBE 环境问题”，重新用旧构建脚本复测原始 lite palm ONNX：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/usr/local/miniconda3/bin/python scripts/build_20t_om_models.py \
  --soc-version Ascend310B1 \
  --model-set legacy_lite \
  --suffix ascend310b1_retest_build20t_env
```

结果：

```text
runs/atc_20t/compile_20t_om_20260705_143909.json
runs/atc_20t/logs/legacy_lite_palm_ascend310b1_retest_build20t_env.log
```

原始 lite palm ONNX 也返回：

```text
returncode=139
om.exists=false
```

这说明当前板端 ATC/TBE 状态下，同一个原始 lite palm ONNX 也不能稳定重编译。继续只修改 lite 优化 ONNX 不能解决当前阻塞。

## 7. 脚本变更

本次保留的有用脚本：

| 脚本 | 用途 |
| --- | --- |
| `scripts/split_onnx_at_tensor.py` | 历史调试脚本：按指定 tensor 切分 ONNX，支持 clean name 和 identity bridge；已从正式部署脚本集合移除 |
| `scripts/debug_split_onnx_pipeline.py` | 历史调试脚本：调试 full/stage1/stage2 边界张量和 pipeline 等价性；已从正式部署脚本集合移除 |
| `scripts/compare_split_onnx_raw.py` | 历史调试脚本：比较原始 ONNX 与 stage1+stage2 ONNX pipeline；已从正式部署脚本集合移除 |
| `scripts/build_lite_palm_split_om.py` | 历史调试脚本：在板端用 ATC 编译 split lite palm stage1/stage2；已从正式部署脚本集合移除 |
| `scripts/compare_split_onnx_om_raw.py` | 历史调试脚本：比较原始 ONNX 与 split OM pipeline；已从正式部署脚本集合移除 |
| `scripts/run_clean_atc.py` | 统一 ATC 环境、日志、report、cache/pycache 控制 |
| `hand_pipeline/om_runtime.py` | `PersistentAclModel.infer` 支持多输入 OM，用于 stage2 split OM |

## 8. 当前结论

1. lite palm 与 full palm 的算子族没有本质差异，lite 主要是 backbone 更浅。
2. full-style lite 优化 ONNX 与原始 lite ONNX 基本等价。
3. identity-bridge split ONNX 与原始 lite ONNX 在本地严格等价。
4. direct lite palm OM 仍不满足 1% 误差目标。
5. full-style lite OM 和 split lite OM 当前都没有在 20T 板端成功生成。
6. 当前阻塞点是 ATC/TBE 在板端对 lite palm 图的编译稳定性，而不是 ONNX rewrite/split 语义。
7. 正式部署仍应使用已验证的 full optimized palm OM + full landmark OM。


