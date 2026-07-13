# PianoVAM MediaPipe Tasks GPU Tracking Evaluation

## Scope

- Dataset split: `test`.
- Videos: `9`.
- Frames: `363309`.
- Reference labels: PianoVAM `Handskeleton` 21-point annotations, treated as
  MediaPipe-style pseudo labels rather than independent human ground truth.
- Model/API: MediaPipe Tasks `HandLandmarker` with `running_mode=VIDEO` and `delegate=gpu` on ace2.
- Output stream: `tracking` only. MediaPipe Tasks does not expose raw palm detector outputs, so palm metrics are intentionally absent.

## Overall Results

| Metric | Value |
| --- | ---: |
| Handskeleton hands | 709951 |
| MediaPipe hands | 719867 |
| Matched hands | 709512 |
| Precision | 0.985615 |
| Recall | 0.999382 |
| Miss rate | 0.000618 |
| Count mismatch frames | 10764 (2.96%) |
| Hand21 mean error | 0.545 px |
| Hand21 mean error P95 | 1.030 px |
| Hand21 P95-point error mean | 0.979 px |
| Hand bbox IoU mean | 0.992174 |
| Exposed palm detection frames | 0 |

## Speed

| Metric | Value |
| --- | ---: |
| Total annotation wall time | 33.40 min |
| Effective FPS | 181.30 |
| Mean tracking call latency | 4.184 ms |
| Mean tracking P95 latency | 4.816 ms |

## Per-Video Results

| Record | Frames | Precision | Recall | Mismatch | Extra frames | Miss frames | Mean px | P95 px | Max px | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-02-14_19-55-17 | 45319 | 0.9926 | 0.9977 | 1.88% | 656 | 195 | 0.793 | 1.309 | 149.702 | 180.97 |
| 2024-02-14_20-10-08 | 77992 | 0.9887 | 0.9995 | 2.34% | 1747 | 80 | 0.730 | 1.311 | 106.220 | 180.89 |
| 2024-02-15_20-17-26 | 32185 | 0.9098 | 0.9991 | 17.93% | 5724 | 46 | 0.769 | 1.218 | 177.835 | 182.67 |
| 2024-02-15_20-47-59 | 18451 | 0.9644 | 0.9997 | 6.95% | 1273 | 10 | 0.700 | 1.238 | 59.997 | 180.47 |
| 2024-02-17_21-44-37 | 10672 | 0.9855 | 0.9982 | 3.05% | 291 | 34 | 0.766 | 1.201 | 140.717 | 181.62 |
| 2024-09-02_14-10-41 | 17588 | 0.9950 | 1.0000 | 0.99% | 174 | 0 | 0.550 | 1.012 | 66.004 | 180.67 |
| 2024-09-02_21-04-45 | 28426 | 0.9954 | 0.9996 | 1.01% | 261 | 25 | 0.442 | 0.747 | 89.676 | 179.76 |
| 2024-09-03_00-07-46 | 96493 | 0.9991 | 0.9999 | 0.18% | 165 | 10 | 0.287 | 0.433 | 41.762 | 181.40 |
| 2024-09-03_00-44-45 | 36183 | 0.9993 | 0.9997 | 0.20% | 50 | 23 | 0.302 | 0.414 | 137.867 | 182.99 |

## Interpretation

- Tracking coverage is very high: recall is `99.938%`, with only `439` unmatched Handskeleton instances out of `709951`.
- Precision is lower than recall because MediaPipe produces extra hands in some frames: `10355` unmatched MediaPipe instances. Most count mismatches are `gt=1, mp=2`, so the dominant error is over-detection rather than missed tracking.
- Landmark agreement is strong for normal frames: mean 21-point error is about `0.545 px`, and the 95th percentile of per-hand mean error is about `1.030 px`.
- Large maximum errors are sparse outliers. The worst cases have low bbox IoU or ambiguous matching, usually near occlusion, hand crossing, or labeling/tracking identity instability.
- `exposed palm detection frames=0` is expected for this run because MediaPipe
  Tasks does not expose palm detector internals through the Python result API.
- Precision and recall here measure agreement with PianoVAM `Handskeleton`
  pseudo labels, not absolute accuracy against independently corrected human
  annotations.

## Worst Landmark Outliers

| Record | Frame | GT | MP idx | MP hand | IoU | Mean px | P95 px | Max px |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2024-02-15_20-17-26 | 3318 | Left | 0 | Left | 0.1920 | 177.835 | 245.373 | 246.680 |
| 2024-02-15_20-17-26 | 31872 | Left | 0 | Left | 0.1016 | 159.664 | 278.169 | 290.436 |
| 2024-02-14_19-55-17 | 32603 | Left | 0 | Left | 0.1438 | 149.702 | 291.118 | 293.846 |
| 2024-02-17_21-44-37 | 9191 | Left | 0 | Left | 0.1649 | 140.717 | 188.580 | 204.362 |
| 2024-09-03_00-44-45 | 36121 | Left | 1 | Right | 0.1602 | 137.867 | 215.653 | 246.815 |
| 2024-09-03_00-44-45 | 36122 | Left | 1 | Right | 0.1400 | 137.693 | 248.575 | 253.523 |
| 2024-09-03_00-44-45 | 36119 | Left | 1 | Right | 0.2963 | 137.445 | 217.437 | 225.214 |
| 2024-02-14_19-55-17 | 4005 | Left | 0 | Left | 0.1661 | 132.989 | 237.823 | 253.213 |
| 2024-09-03_00-44-45 | 36120 | Left | 1 | Right | 0.2668 | 130.073 | 205.831 | 220.268 |
| 2024-02-14_19-55-17 | 4007 | Left | 0 | Left | 0.2153 | 128.854 | 230.901 | 246.744 |

## Files

- Annotations: `data/PianoVAM_v1/mediapipe_tasks_gpu_tracking_annotations/`.
- Comparison summary: `runs/pianovam_handskeleton_mediapipe_tasks_gpu_tracking_test/summary.json`.
- Per-frame comparison: `runs/pianovam_handskeleton_mediapipe_tasks_gpu_tracking_test/frames.csv`.
- Matched-hand errors: `runs/pianovam_handskeleton_mediapipe_tasks_gpu_tracking_test/matches.csv`.
