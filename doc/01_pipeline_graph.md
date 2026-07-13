# Pipeline and MediaPipe Graph

This document describes the MediaPipe Hands pipeline, the legacy MediaPipe graph
nodes that matter for this project, and the validation layers used to isolate
errors during the Ascend 310B port.

## 1. Target Pipeline

MediaPipe Hands is not a single end-to-end model. It is a two-stage pipeline:

```text
BGR/RGB input image
  -> ImageToTensor preprocessing
       - full-image ROI
       - keep-aspect-ratio padding
       - warpPerspective sampling to 192x192
  -> palm detector TFLite
  -> SSD anchor decode
  -> score sigmoid
  -> remove normalized padding
  -> weighted NMS
  -> palm box + 7 palm keypoints
  -> palm detection to rotated hand rect
  -> rotated ROI crop to 224x224
  -> hand landmark TFLite
  -> 21 landmarks + hand score + handedness + world landmarks
  -> project landmarks back to the source image
```

The Ascend 310B port must reproduce three things: model inference, geometric
post-processing, and validation. Converting the TFLite models alone is not
enough.

## 2. Code Mapping

| Module | Role |
| --- | --- |
| `hand_pipeline/preprocess.py` | Reproduces `ImageToTensorCalculator` using continuous ROI sampling and normalized padding. |
| `hand_pipeline/decode.py` | Generates 2016 SSD anchors, decodes raw palm outputs, and runs weighted NMS. |
| `hand_pipeline/roi.py` | Builds hand ROIs from palm detections, performs rotated crops, and projects landmarks back. |
| `hand_pipeline/inference.py` | Wraps local TFLite/LiteRT-style model inference. |
| `hand_pipeline/eval.py` | Loads palm ground truth and computes IoU, AP, precision, and recall. |
| `hand_pipeline/tracking.py` | Implements MediaPipe-style ROI reuse for video tracking. |
| `scripts/*.py` | Runnable validation, annotation, conversion, and benchmark tools. |

`hand_pipeline/` is the reusable core library. `scripts/` contains runnable
engineering tools and is kept outside the library package so the same logic can
be used on PC, ace2, and Ascend boards.

## 3. Legacy Graph Reference

The legacy `mediapipe==0.10.14` graph can still run:

```text
mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb
```

The simplified main graph path is:

```text
image
  -> PalmDetectionCpu
  -> ClipDetectionVectorSizeCalculator
  -> ImagePropertiesCalculator
  -> BeginLoopDetectionCalculator
  -> PalmDetectionDetectionToRoi
  -> RectTransformationCalculator
  -> HandLandmarkCpu
  -> EndLoop... calculators
  -> multi_hand_landmarks / handedness / world_landmarks
```

Key graph-node mapping:

| MediaPipe node | Role | Local implementation |
| --- | --- | --- |
| `PalmDetectionCpu` | Full-image palm detector subgraph, including image-to-tensor, TFLite, decode, and NMS. | `preprocess.py`, `decode.py`, `scripts/eval_palm_tflite.py` |
| `ImageToTensorCalculator` | Samples the detector tensor from a continuous full-image ROI and produces normalized padding. | `preprocess.image_to_tensor()` |
| `PalmDetectionDetectionToRoi` | Converts palm boxes and palm keypoints into rotated hand rects. | `roi.normalized_rect_from_palm_detection()` |
| `RectTransformationCalculator` | Applies shift, scale, and square-long transformations to rects. | `roi.transform_normalized_rect()` |
| `HandLandmarkCpu` | Runs the landmark model inside a single-hand ROI. | `hand_pipeline.two_stage` |
| `LandmarkProjectionCalculator` | Projects ROI-space landmarks back to image coordinates. | `roi.project_landmarks_with_normalized_rect()` |
| `HandLandmarkLandmarksToRoi` | Builds the next tracking ROI from previous landmarks. | `tracking.HandTracker`, `roi.landmarks_to_tracking_roi()` |

Video and WebRTC use the tracking graph by default. Independent dataset image
evaluation uses image mode to avoid carrying ROI state across unrelated samples.

## 4. Validation Layers

The official Tasks API generally exposes final 21 landmarks, handedness, and
limited hand-level outputs. It does not expose `palm_detections`,
`hand_rects_from_palm_detections`, or `letterbox_padding`. This project keeps
separate validation layers so errors can be localized.

| Output area | Validation goal | Current summary |
| --- | --- | --- |
| `palm_detector/` | Validate detector accuracy on manually checked palm boxes and 7 keypoints. | precision `0.967102`, recall `0.972361` |
| `handlm_manual_gt/` | Validate the landmark model on manually corrected 21-point crops. | full mean `5.940073 px`, lite mean `6.602299 px` |
| `legacy_graph/` | Export legacy graph intermediate rects. | Legacy graph is used as calculator reference. |
| `legacy_rect_landmark/` | Validate the landmark subpath using official legacy rects. | mean `0.016921 px` |
| `two_stage_vs_legacy_graph/` | Validate full palm-to-rect-to-landmark reproduction. | mean `0.024968 px` |
| `two_stage_vs_current_tasks/` | Compare against current Tasks outputs. | mean `3.630226 px` |

The important diagnostic pattern is:

```text
legacy_rect_landmark ~= 0
  -> landmark TFLite, ROI crop, and projection are aligned
two_stage_vs_legacy_graph ~= 0
  -> detector tensor sampling, palm decode, NMS, hand rects, and projection
     are aligned with the legacy graph
```

`ImageToTensorCalculator` must be treated as its own operator. The legacy CPU
graph does not simply resize and copy-border the input. It builds a padded
full-image ROI and samples directly into the detector tensor. A mismatch here
changes raw palm outputs and propagates into boxes, 7 palm keypoints, hand
rects, and final 21 landmarks.

## 5. Ascend 310B Porting Principle

Keep the same validation boundaries on the board:

```text
camera/video frame
  -> detector input tensor
  -> palm detector OM raw outputs
  -> palm decode and NMS
  -> hand ROI
  -> landmark input tensor
  -> landmark OM raw outputs
  -> projected 21 landmarks
  -> tracking ROI state
```

Save intermediate data at each layer and compare it against the PC reference
path. The detector input tensor and normalized padding must be captured first;
if they differ, later decode and ROI formulas can be correct while final
landmarks still diverge.

Detailed tracking behavior is documented in
[06_tracking_algorithm.md](06_tracking_algorithm.md).
