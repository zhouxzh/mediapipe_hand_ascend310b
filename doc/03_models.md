# Model Assets

This document describes the model assets currently kept in the repository.
Failed OM variants and numerically duplicate board-specific builds are not kept
as active deployment models.

## Default OM Models

| Role | File | Status |
| --- | --- | --- |
| full palm detector | `models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om` | default |
| full hand landmark | `models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om` | default |

The default full palm OM is built from:

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
```

## Lite Candidate OM Models

| Role | File | Status |
| --- | --- | --- |
| lite palm detector | `models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om` | report-only candidate |
| lite hand landmark | `models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om` | report-only candidate |

Lite models are not default deployment models. They are useful for speed
experiments and comparison reports.

## Speed Comparison Candidate

| Role | File | Status |
| --- | --- | --- |
| full palm detector, mix precision | `models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_allow_mix_precision.om` | speed comparison candidate, not default |

The mix-precision full palm OM is a useful speed comparison candidate. It is not
the repository-wide default unless the code defaults and deployment docs are
updated together.

## Removed OM Categories

| Category | Reason |
| --- | --- |
| direct full palm OM | Raw outputs did not satisfy production accuracy requirements. |
| direct lite palm OM | Raw outputs and end-to-end video results were incorrect. |
| `*_ascend310b1.om` full 20T rebuild | Raw outputs matched the existing default OM exactly, so the file was duplicate. |
| task full OM | Not used by the current deployment path and not accepted as a default model. |

## ONNX Assets

ONNX files are kept for:

- ATC input.
- Raw-output comparison against OM models.
- Reproducing the optimized full/lite palm conversion path.

Important ONNX files:

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx
models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx
models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx
```

TFLite models are kept as original sources and conversion references. They are
not the board deployment entry point.

## Model Selection

| Scenario | Model choice |
| --- | --- |
| Default deployment, WebRTC default, dataset acceptance | `origin_dtype` full optimized palm OM + full landmark OM |
| Speed exploration and comparison reports | lite optimized palm candidate + lite landmark OM |
| Full-model speed comparison | mix-precision full palm candidate + full landmark OM |
| ATC reproduction | optimized ONNX + build scripts |

Model choice should not be based on single-model latency alone. Formal
selection must consider detector AP/recall, 21-point error, end-to-end speed,
and video/dataset regression results.
