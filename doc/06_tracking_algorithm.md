# Tracking Algorithm

This document records the video/WebRTC tracking path, its relationship to the
legacy MediaPipe graph, and the key debugging conclusions from the `demo1`
tracking investigation. Formal still-image dataset acceptance remains based on
`image` mode; tracking is for continuous frame sequences.

## 1. Scope

`tracking` mode is used for video and WebRTC:

- If the current frame has enough valid ROIs from the previous frame, skip the
  palm detector and run the landmark model directly on those ROIs.
- If tracked ROIs are missing or insufficient, run the palm detector again and
  merge palm ROIs with previous tracking ROIs.
- Each hand with sufficient hand-presence score produces a next-frame tracking
  ROI.

`image` mode is used for independent-image evaluation:

- Every image runs the palm detector.
- ROI state is not reused between unrelated samples.

## 2. Legacy Graph Mapping

The implementation follows the core state loop of:

```text
mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb
```

Reference graph streams:

| Stream | Meaning |
| --- | --- |
| `palm_detections` | Raw palm detector outputs. |
| `hand_rects_from_palm_detections` | Hand ROIs created from palm detections. |
| `multi_hand_landmarks` | Final 21 image landmarks. |
| `hand_rects_from_landmarks` | Next-frame tracking ROIs created from landmarks. |

MediaPipe's `AssociationNormRectCalculator` keeps the later input rect when
overlapping rects are associated. The local implementation therefore orders
palm ROIs before previous-landmark ROIs so previous landmarks can override
overlapping palm detections.

## 3. Implementation

Core code:

- `hand_pipeline/tracking.py`: `TrackingConfig`, `HandTracker`, rect
  association, and tracking-state updates.
- `hand_pipeline/roi.py`: palm ROI, landmark ROI, projection, and next tracking
  ROI computation.
- `hand_pipeline/two_stage.py`: shared two-stage entry point for OM and ONNX
  pipelines.
- `scripts/eval_video_mediapipe_om.py`: compares video reference annotations
  against OM image/tracking outputs.

Default configuration keeps pure MediaPipe-style behavior:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_hands` | `2` | Keep at most two hands. |
| `association_min_similarity` | `0.5` | MediaPipe-style association threshold. |
| `min_hand_score` | `0.5` | Landmark hand-presence threshold. |
| `max_tracking_lost_frames` | `0` | Do not retain lost-hand ROIs by default. |
| `roi_precision` | `float32` | Match the float path used by MediaPipe calculators. |
| `projection_precision` | `float32` | Use float projection for landmark reprojection. |
| `tracking_rect_smooth_alpha` | disabled | No additional ROI smoothing by default. |
| `max_tracking_rejected_frames` | `0` | No abnormal ROI rejection by default. |

Per-frame flow:

1. Read previous `next_tracking_roi` values from the tracker.
2. If the previous ROI count reaches `max_hands`, skip palm detection.
3. Otherwise run the palm detector and build `hand_rects_from_palm_detections`.
4. Merge palm ROIs and previous ROIs with MediaPipe-style association.
5. Crop each ROI, run landmark inference, and project landmarks back.
6. Build `next_tracking_roi` for results with `hand_score >= min_hand_score`.
7. Store valid next ROIs as the tracker state for the next frame.

## 4. ROI Computation

Palm detection to landmark input ROI:

- Use the palm bbox and palm keypoints to estimate hand orientation.
- Match the rotation definition used by `PalmDetectionDetectionToRoi`.
- Apply `RectTransformation` with `scale_x=2.6`, `scale_y=2.6`,
  `shift_y=-0.5`, and `square_long=true`.

Landmarks to next-frame tracking ROI:

- Use a stable subset of the 21 landmarks to compute a rotated bounding rect.
- Estimate rotation mainly from wrist-to-finger-base directions.
- Apply `RectTransformation` with `scale_x=2.0`, `scale_y=2.0`,
  `shift_y=-0.1`, and `square_long=true`.

The default `float32` path is intentional. A previous double-precision path
amplified tiny closed-loop differences and caused an early second-hand loss in
`demo1`. Using `float32` fixed that early divergence.

## 5. Debug Fields

`FrameResult.predictions` keeps common evaluation fields:

```text
box
palm7
hand21
hand_score
handedness
```

Tracking-specific fields:

- `source_roi`: ROI used for the current landmark inference.
- `hand_roi`: compatibility alias for the source ROI.
- `subgraph_next_tracking_roi`: next ROI computed directly from current
  landmarks.
- `next_tracking_roi`: actual tracker ROI after optional smoothing/rejection.
- `palm_detector_skipped`: whether palm detection was skipped for this frame.
- `tracking_state_rejected` and `tracking_reject_reason`: optional rejection
  diagnostics.

## 6. Optional Robustness Parameters

These controls are engineering/debug options, not part of strict MediaPipe
reproduction:

- `--tracking-rect-smooth-alpha`
- `--max-tracking-rejected-frames`
- `--max-tracking-rotation-delta`
- `--min-tracking-size-ratio`
- `--max-tracking-size-ratio`
- `--max-tracking-center-shift`
- `--max-tracking-lost-frames`

They can improve hand-count stability in some clips, but they also change the
strict reproduction semantics and can increase landmark pixel error. They are
disabled in the default MediaPipe-style path.

## 7. `demo1` Debug Conclusion

The `demo1` investigation found:

- Palm detector single-step output, palm ROI, landmark crop, landmark output
  parsing, projection, and next-ROI formulas can align with MediaPipe.
- Given MediaPipe's true ROI, the local TFLite/ONNX subpath produces a next ROI
  within roughly `0.02 px`.
- `roi_precision=float32` and `projection_precision=float32` fixed the earlier
  second-hand loss around frame `33`.
- The remaining loss after frame `55` is closed-loop ROI drift, not a landmark
  model failure on that individual frame.

The important frame-55 evidence was that MediaPipe's second-hand input ROI was
about `584 px` wide with hand score `0.905`, while the drifted local ROI was
about `346 px` wide and produced a very low hand score even when passed through
MediaPipe's own landmark subgraph.

## 8. Validation Commands

Generate MediaPipe video reference annotations:

```bash
python scripts/annotate_pianovam_mediapipe_legacy.py \
  --data-root data/PianoVAM_v1 \
  --split test \
  --frame-stride 1 \
  --max-frames 600 \
  --save-vis 2 \
  --force
```

Compare image and tracking streams:

```bash
python scripts/compare_pianovam_handskeleton_mediapipe.py \
  --data-root data/PianoVAM_v1 \
  --annotation-root data/PianoVAM_v1/mediapipe_legacy_annotations \
  --split test \
  --streams image,tracking \
  --output-dir runs/pianovam_handskeleton_mediapipe_compare/test_first600_tracking_analysis
```

Run OM tracking against saved references:

```bash
python scripts/eval_video_mediapipe_om.py \
  --video data/PianoVAM_v1/Video/<record_time>.mp4 \
  --annotations data/PianoVAM_v1/mediapipe_legacy_annotations/<record_time>/mediapipe_annotations.json \
  --pipeline-mode tracking \
  --reference-stream tracking \
  --model-set full \
  --max-frames 600 \
  --output-dir runs/pianovam_om_tracking/<record_time>
```

## 9. Current Conclusion

The code reproduces the main legacy MediaPipe tracking graph structure and the
single-step ROI/projection formulas. Remaining difficult-video differences come
from closed-loop ROI drift: small landmark deviations in critical poses can
change the next crop, which then changes hand score and future state.

The default strategy remains pure MediaPipe-style tracking. Smoothing and
rejection options are debug/product controls, not part of strict graph
reproduction unless explicitly enabled in a report.
