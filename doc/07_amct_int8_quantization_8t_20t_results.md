# AMCT INT8 Quantization Results on Ascend 310B Boards

Date: 2026-07-08

This note records the AMCT ONNX PTQ results for the optimized legacy full palm
detector on two Ascend 310B boards:

- `ascend20t`: Orange Pi AI Pro 20T, reported SoC `Ascend310B1`
- `ascend8t`: Orange Pi AI Pro 8T, reported SoC `Ascend310B4`

The AMCT comparison baseline is the non-AMCT mix-precision palm detector:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_allow_mix_precision.om
```

The AMCT quantized candidates are:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_amct_int8_allow_mix_precision.om
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_amct_int8_skip_heads_allow_mix_precision.om
```

`skip_heads` keeps the four final palm classifier/regressor heads out of AMCT
quantization:

```text
classifier_palm_8_NO_PRUNING
regressor_palm_8_NO_PRUNING
classifier_palm_16_NO_PRUNING
regressor_palm_16_NO_PRUNING
```

## Conversion Notes

AMCT generated deploy ONNX models with Ascend quantization operators. ATC could
not compile them with `must_keep_origin_dtype` because `AscendDequant` FP32
output is unsupported by the current 310B OPP package. The successful conversion
used:

```text
--precision_mode=allow_mix_precision
```

On `ascend20t`, ATC needed the conda Python runtime in the environment because
the system Python did not provide a compatible NumPy for TBE initialization.

## Palm OM Latency

Measured with `scripts/benchmark_om_inference.py`, `warmup=20`,
`iterations=200`, `fill=zeros`. `execute_ms` is the primary metric; it measures
the warmed `acl.mdl.execute` call with persistent device buffers.

| Board | Model | execute mean ms | full mean ms |
| --- | --- | ---: | ---: |
| ascend20t | origin dtype | 11.557 | 12.256 |
| ascend20t | mix precision | 6.672 | 7.311 |
| ascend20t | AMCT INT8 full | 6.745 | 7.384 |
| ascend20t | AMCT INT8 skip-heads | 6.729 | 7.355 |
| ascend8t | origin dtype | 27.037 | 27.842 |
| ascend8t | mix precision | 15.428 | 16.121 |
| ascend8t | AMCT INT8 full | 15.772 | 16.466 |
| ascend8t | AMCT INT8 skip-heads | 15.682 | 16.403 |

Result: neither AMCT INT8 candidate is faster than the non-AMCT mix-precision
comparison OM on either board.

## HaGRIDv2 Accuracy

Measured with `scripts/eval_hf_hand_dataset_om.py` on
`data/portable-hagridv2-mediapipe-hand/test-00000.parquet`, 1663 images. The
cascade uses the candidate palm detector plus the existing full landmark OM.

| Board | Palm detector | Recall | AP50 | full21 mean px | detector mean ms | total mean ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ascend20t | mix precision | 0.997596 | 0.994933 | 0.128635 | 7.462 | 17.570 |
| ascend20t | AMCT INT8 full | 0.989784 | 0.994728 | 0.630227 | 7.514 | 17.703 |
| ascend20t | AMCT INT8 skip-heads | 0.990385 | 0.994927 | 0.613581 | 7.529 | 18.047 |
| ascend8t | mix precision | 0.997596 | 0.994933 | 0.128635 | 16.315 | 28.863 |
| ascend8t | AMCT INT8 full | 0.989784 | 0.994728 | 0.630227 | 16.558 | 29.118 |
| ascend8t | AMCT INT8 skip-heads | 0.990385 | 0.994927 | 0.613581 | 16.573 | 28.941 |

Result: AMCT INT8 passes the broad evaluation thresholds, but its palm geometry
accuracy drops substantially relative to the mix-precision comparison OM.
Skip-head quantization recovers a small amount of accuracy but remains far from
that comparison baseline.

## Interpretation

The INT8 candidates do not improve speed because the compiled graph is not a
continuous INT8 execution path. AMCT inserts Ascend quant/dequant boundaries, and
the optimized palm detector contains many small non-convolution nodes such as
`Slice`, `Max`, `Concat`, `Reshape`, and `Transpose`. These nodes interrupt INT8
fusion and introduce internal tensor format conversion and on-chip memory
traffic.

The model is also depthwise-separable and small-batch. On 310B, the current FP16
mix precision path already maps well to supported kernels, while INT8 saves some
weight bandwidth but pays extra quant/dequant and scheduling overhead.

## Recommendation

Do not promote either AMCT INT8 candidate. For this AMCT experiment, the best
non-AMCT comparison point remains:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_allow_mix_precision.om
```

This does not change the repository-wide deployment default, which remains the
`origin_dtype` full palm OM plus the full landmark OM unless the code defaults
and deployment documents are updated together. The AMCT INT8 models are useful
as reproducible artifacts, but they should not replace either the current
default palm detector or the mix-precision comparison candidate unless a future
CANN/OPP release or a different graph rewrite produces a fused INT8 path with
lower `acl.mdl.execute` latency.

## Report Artifacts

Local report copies:

```text
runs/remote_reports/ascend20t/
runs/remote_reports/ascend8t/
```

Remote report roots:

```text
ascend20t:/home/HwHiAiUser/Documents/mediapipe_hand_ascend310b/runs/amct_palm_int8_benchmark/
ascend20t:/home/HwHiAiUser/Documents/mediapipe_hand_ascend310b/runs/amct_palm_int8_eval_20t/
ascend20t:/home/HwHiAiUser/Documents/mediapipe_hand_ascend310b/runs/amct_palm_int8_skip_heads_eval_20t/
ascend8t:/home/HwHiAiUser/Documents/mediapipe_hand_ascend310b/runs/amct_palm_int8_benchmark_8t/
ascend8t:/home/HwHiAiUser/Documents/mediapipe_hand_ascend310b/runs/mix_precision_eval_8t/
ascend8t:/home/HwHiAiUser/Documents/mediapipe_hand_ascend310b/runs/amct_palm_int8_eval_8t/
ascend8t:/home/HwHiAiUser/Documents/mediapipe_hand_ascend310b/runs/amct_palm_int8_skip_heads_eval_8t/
```
