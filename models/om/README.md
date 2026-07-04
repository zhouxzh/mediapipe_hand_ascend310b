# Ascend OM Models

本目录保存 Ascend 310B 环境中通过 ATC 生成的 OM 模型。

在 310B 上重新生成时，必须顺序、单线程运行 ATC：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/export_ascend_om.py --group all --output-report models/om/export_report_all.json
```

`scripts/export_ascend_om.py` 会固定这些单线程参数：

```text
TE_PARALLEL_COMPILER=1
TBE_PARALLEL_COMPILER=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
taskset -c 0
nice -n 19
--enable_graph_parallel=0
--ac_parallel_enable=0
```

当前已生成的 OM 文件：

| 模型 | 大小 | SHA256 |
| --- | ---: | --- |
| `mediapipe_legacy_0_10_14_palm_detection_full.om` | 6384138 bytes | `9851225e082175b3b2f0b3ccb76f078b20c7f0c2d2bed803a4d61e9843b639a5` |
| `mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om` | 12787549 bytes | 最终优化 full palm，正式部署优先使用 |
| `mediapipe_legacy_0_10_14_hand_landmark_full.om` | 11669346 bytes | `2cba28875b224be8ddfb88a5364641f04eb24d094db9658070ba3d0c4ac2f3f6` |
| `mediapipe_legacy_0_10_14_palm_detection_lite.om` | 5373375 bytes | `08f1793300127aad40710abe1cfbd5ffcfc525223aa3a3051b832bb4674deac8` |
| `mediapipe_legacy_0_10_14_hand_landmark_lite.om` | 5970779 bytes | `cee00e002cc5a9ce436d2593f6253fe4eee4a4cb25d6a65292e1371a4c8f9c26` |
| `mediapipe_task_hand_detector_full.om` | 6384094 bytes | `c5d856498995775524c6275bdedc9c3c008abb1d4f0b8acaa70fa7e00f12d586` |
| `mediapipe_task_hand_landmark_full.om` | 11669306 bytes | `09e05ebf02c0ca2718e5fb8095eef8bfe91e97c0e6215802bab7769d3533ce59` |

优化后的 legacy full palm raw-output 对齐结果在 `runs/palm_om/legacy_full_palm/downsample_resize_maxpool_slices_origin_dtype_om_compare_report.remote.md` 和 `runs/palm_om/legacy_full_palm/downsample_resize_maxpool_slices_origin_dtype_om_compare_summary.remote.json`。原始 `mediapipe_legacy_0_10_14_palm_detection_full.om` 保留为转换对照，不建议作为 full palm 正式部署模型。

OM 验收顺序：

```text
1. detector input tensor 与 PC 参考完全一致
2. detector OM raw output 接近 TFLite raw output
3. palm decode + NMS 后的 box/7 点接近 PC 参考
4. landmark OM raw output 接近 TFLite raw output
5. projected 21 points 对齐 legacy graph
```

当前 PC baseline 中 `two_stage_vs_legacy_graph` mean 为 `0.024968 px`，310B 端后续应以这个链路作为端到端对齐参考。

