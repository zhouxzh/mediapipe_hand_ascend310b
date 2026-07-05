# Ascend OM Models

This directory keeps only deployable or reportable Ascend 310B OM models.

## Production

| Role | Model |
| --- | --- |
| full palm detector | `mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om` |
| full hand landmark | `mediapipe_legacy_0_10_14_hand_landmark_full.om` |

These two files are the default deployment pair.

## Report-Only Lite Candidate

| Role | Model |
| --- | --- |
| lite palm detector | `mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om` |
| lite hand landmark | `mediapipe_legacy_0_10_14_hand_landmark_lite.om` |

The lite pair can be evaluated with `scripts/eval_hf_hand_dataset_om.py`, but it is not the default production pair.

## Cleanup Rule

Do not keep OM files that are known to produce wrong palm raw outputs. Do not keep hardware-suffixed duplicates when the 20T/8T ATC outputs are numerically identical to the existing production model.
