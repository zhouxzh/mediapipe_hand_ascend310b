# ONNX Models

本目录保存从 `models/tflite/` 转出的 ONNX 模型，用于验证 TFLite 到 ONNX 的数值一致性。

重新生成全部模型：

```bash
conda activate mediapipe_legacy
python scripts/export_onnx.py --group all --output-report models/onnx/export_report_all.json
python scripts/make_raw_alignment_inputs.py --group all --output-dir runs/onnx_alignment_inputs
```

当前已验证的 ONNX 文件：

| 模型 | 输入 | SHA256 |
| --- | --- | --- |
| `mediapipe_legacy_0_10_14_palm_detection_full.onnx` | `input_1:1,192,192,3` | `67abbcd98ef6c96d45222781cd912ca25b03bd87c0cc6f2552d1ff4d7b671476` |
| `mediapipe_legacy_0_10_14_hand_landmark_full.onnx` | `input_1:1,224,224,3` | `a3b1f390d34f124cb43cd860418031a5056cc378ed941e24d45e4233dabd463a` |
| `mediapipe_legacy_0_10_14_palm_detection_lite.onnx` | `input_1:1,192,192,3` | `4a55a7fe906b10978180a849c6abd3c8de23ccaa9d1625cabef82e893ec2e39d` |
| `mediapipe_legacy_0_10_14_hand_landmark_lite.onnx` | `input_1:1,224,224,3` | `df8230684582f4aa03d4d027a9185452340c96be67c8c6988065939e8c986ba4` |

`runs/onnx_alignment_inputs/reference_report.json` 记录了 ONNX 与 TFLite raw output 的误差；当前 legacy full/lite 模型均在 `1e-4` 量级以内。

注意：detector 前处理不是 ONNX 模型的一部分。任何推理入口都必须单独复刻 `ImageToTensorCalculator` 的 full-image ROI、normalized padding 和 `warpPerspective` 采样。
