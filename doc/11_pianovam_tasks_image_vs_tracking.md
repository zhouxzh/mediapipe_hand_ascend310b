# PianoVAM Image Mode vs Tracking Mode

## Scope

This report compares the full PianoVAM `test` split results from two MediaPipe
Tasks runs:

| Run | Delegate | Running mode | Output stream | Local result directory |
| --- | --- | --- | --- | --- |
| per-frame image detection | CPU | `IMAGE` | `image` | `runs/pianovam_handskeleton_mediapipe_tasks_cpu_image_test/` |
| video tracking | GPU | `VIDEO` | `tracking` | `runs/pianovam_handskeleton_mediapipe_tasks_gpu_tracking_test/` |

Both runs are compared against PianoVAM `Handskeleton` 21-point annotations,
which are treated as MediaPipe-style pseudo labels rather than independent
human ground truth.

Important limitation: this is not a pure CPU-vs-GPU or pure mode-only
experiment. Accuracy differences mainly reflect `IMAGE` versus `VIDEO`
tracking behavior, while speed differences also include the CPU/GPU delegate
difference.

## Overall Difference

| Metric | Image mode | Tracking mode | Tracking - image |
| --- | ---: | ---: | ---: |
| Processed frames | 363309 | 363309 | 0 |
| Handskeleton hands | 709951 | 709951 | 0 |
| MediaPipe hands | 690328 | 719867 | +29539 |
| Matched hands | 682178 | 709512 | +27334 |
| Unmatched Handskeleton hands | 27773 | 439 | -27334 |
| Unmatched MediaPipe hands | 8150 | 10355 | +2205 |
| Precision | 0.988194 | 0.985615 | -0.002579 |
| Recall | 0.960880 | 0.999382 | +0.038501 |
| Count mismatch rate | 8.70% | 2.96% | -5.73 pp |
| Hand21 mean error | 3.566 px | 0.545 px | -3.021 px |
| Hand21 mean error P95 | 6.656 px | 1.030 px | -5.626 px |
| Hand bbox IoU mean | 0.943344 | 0.992174 | +0.048829 |

Tracking recovers almost all image-mode misses: unmatched Handskeleton hands
drop from `27773` to `439`. The cost is `2205` additional unmatched MediaPipe
hands, so precision decreases slightly while recall improves substantially.

## Frame-Level Count Behavior

| Frame category | Frames |
| --- | ---: |
| Both modes have correct hand count | 328460 |
| Tracking count correct, image count wrong | 24085 |
| Image count correct, tracking count wrong | 3254 |
| Both modes count wrong | 7510 |

Matched-hand delta by frame:

| Tracking matched - image matched | Frames |
| ---: | ---: |
| -2 | 1 |
| -1 | 132 |
| 0 | 337759 |
| +1 | 23366 |
| +2 | 2051 |

The net matched-hand gain is:

```text
23366 * 1 + 2051 * 2 - 132 * 1 - 1 * 2 = 27334
```

The dominant image-mode failure is under-detection:

| Mode | Handskeleton hands | MediaPipe hands | Frames |
| --- | ---: | ---: | ---: |
| image | 2 | 1 | 21315 |
| image | 2 | 0 | 2082 |
| image | 1 | 0 | 1172 |
| tracking | 2 | 1 | 383 |
| tracking | 2 | 0 | 3 |
| tracking | 1 | 0 | 37 |

Tracking largely removes the repeated one-hand-missing cases that appear in
per-frame image mode.

## Same-Hand Landmark Error

For `682026` Handskeleton labels that are matched in both modes:

| Metric | Value |
| --- | ---: |
| Tracking has lower mean keypoint error | 681451 hands |
| Image mode has lower mean keypoint error | 575 hands |
| Mean tracking-minus-image error | -3.043 px |
| Median tracking-minus-image error | -2.301 px |
| P05 tracking-minus-image error | -5.824 px |
| P95 tracking-minus-image error | -1.297 px |

This means tracking is not only finding more hands. For almost every hand that
both modes detect, tracking also produces landmarks closer to the PianoVAM
`Handskeleton` pseudo labels.

## Per-Video Difference

| Record | Image recall | Tracking recall | Recall gain | Image mismatch | Tracking mismatch | Mean px image | Mean px tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-02-14_19-55-17 | 0.9588 | 0.9977 | +0.0389 | 8.39% | 1.88% | 4.368 | 0.793 |
| 2024-02-14_20-10-08 | 0.9809 | 0.9995 | +0.0186 | 5.15% | 2.34% | 3.772 | 0.730 |
| 2024-02-15_20-17-26 | 0.8197 | 0.9991 | +0.1793 | 40.59% | 17.93% | 6.003 | 0.769 |
| 2024-02-15_20-47-59 | 0.7846 | 0.9997 | +0.2151 | 40.00% | 6.95% | 5.103 | 0.700 |
| 2024-02-17_21-44-37 | 0.8897 | 0.9982 | +0.1085 | 16.14% | 3.05% | 4.198 | 0.766 |
| 2024-09-02_14-10-41 | 0.9953 | 1.0000 | +0.0047 | 1.81% | 0.99% | 3.084 | 0.550 |
| 2024-09-02_21-04-45 | 0.9913 | 0.9996 | +0.0082 | 2.43% | 1.01% | 3.470 | 0.442 |
| 2024-09-03_00-07-46 | 0.9986 | 0.9999 | +0.0014 | 0.37% | 0.18% | 2.651 | 0.287 |
| 2024-09-03_00-44-45 | 0.9969 | 0.9997 | +0.0028 | 0.67% | 0.20% | 2.602 | 0.302 |

The largest tracking gains occur in:

- `2024-02-15_20-47-59`: recall gain `+21.51 pp`, mismatch rate drops from
  `40.00%` to `6.95%`;
- `2024-02-15_20-17-26`: recall gain `+17.93 pp`, mismatch rate drops from
  `40.59%` to `17.93%`;
- `2024-02-17_21-44-37`: recall gain `+10.85 pp`, mismatch rate drops from
  `16.14%` to `3.05%`.

## Interpretation

Per-frame image mode repeatedly loses hands that are still present in the
PianoVAM `Handskeleton` stream, especially in the February videos with more
occlusion or difficult hand poses. Tracking carries forward a stable ROI from
previous frames, so the landmark model can continue to evaluate the hand even
when the current frame's independent detector would not reliably recover it.

The tradeoff is extra predictions. Tracking produces `29539` more MediaPipe
hands than image mode, and unmatched MediaPipe hands increase by `2205`.
However, this cost is small relative to the `27334` recovered Handskeleton hand
instances.

The landmark-error gap is also decisive. The same-label comparison shows that
tracking has lower mean keypoint error for `681451 / 682026` commonly matched
hands. This strongly suggests that PianoVAM `Handskeleton` was generated by a
video-tracking style process, not by independent per-frame detection.

## Practical Conclusion

Use tracking-mode references for video tracking validation. Per-frame image
mode is useful for isolating independent detector behavior, but it should not
be used as the main reference for PianoVAM video tracking performance.

For a strict mode-only benchmark, run one of these additional controls:

1. MediaPipe Tasks CPU `VIDEO` tracking, compared against CPU `IMAGE`.
2. MediaPipe Tasks GPU `IMAGE`, compared against GPU `VIDEO`.

Until then, the current full-test comparison supports the qualitative
conclusion that tracking mode is much closer to PianoVAM `Handskeleton` than
independent frame-by-frame image mode.

## Files

Comparison output:

```text
runs/pianovam_tasks_image_vs_tracking_test/
```

Key files:

- `summary.json`: overall, frame-level, and same-label error summaries.
- `per_video_comparison.csv`: per-record comparison table.
- `frame_comparison.csv`: frame-level count and match deltas.
- `same_label_error_delta_extremes.csv`: largest tracking improvements and
  regressions for labels matched in both modes.
