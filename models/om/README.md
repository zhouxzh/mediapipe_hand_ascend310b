# Ascend OM Models

This directory keeps only the OM models that are useful for deployment or current comparison.

## Production

| Role | Model |
| --- | --- |
| full palm detector | `mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om` |
| full hand landmark | `mediapipe_legacy_0_10_14_hand_landmark_full.om` |

These two files are the default accuracy baseline.

## Recommended Mixed Precision

| Role | Model | Status |
| --- | --- | --- |
| full palm detector | `mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_allow_mix_precision.om` | Recommended speedup candidate. Full dataset pass is unchanged; detector execute latency is much lower on 20T. Use it with the official full landmark OM. |

## Report-Only Lite Baseline

| Role | Model |
| --- | --- |
| lite palm detector | `mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om` |
| lite hand landmark | `mediapipe_legacy_0_10_14_hand_landmark_lite.om` |

The lite pair can be evaluated with `scripts/eval_hf_hand_dataset_om.py`, but it is not the default production pair.

## Historical Results

Other mixed precision and FP16 I/O OM files were tested but are not kept here because they were slower, had no measurable speed gain, or are only useful as historical comparisons. Their results remain recorded in `doc/05_board_validation_results.md`.

Known failed paths are not deployable: `force_fp16`, all-FP16 palm ONNX, lite all-FP16 landmark ONNX, and `precision_mode_v2=mixed_float16` failed ATC on the tested 20T board.
