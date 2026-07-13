# PianoVAM MediaPipe Tasks CPU Image Evaluation

## Scope

- Dataset split: `test`.
- Videos: `9`.
- Frames: `363309`.
- Reference labels: PianoVAM `Handskeleton` 21-point annotations, treated as
  MediaPipe-style pseudo labels rather than independent human ground truth.
- Model/API: MediaPipe Tasks `HandLandmarker` with `running_mode=IMAGE` and
  `delegate=cpu` on ace2.
- Output stream: `image` only. Each frame is processed independently.
- Palm metrics are absent because MediaPipe Tasks does not expose raw palm
  detector boxes or keypoints through the Python result API.

Command used on ace2:

```bash
python scripts/annotate_pianovam_mediapipe_tasks.py \
  --data-root data/PianoVAM_v1 \
  --split test \
  --running-mode image \
  --delegate cpu \
  --output-root data/PianoVAM_v1/mediapipe_tasks_cpu_image_annotations \
  --save-vis 2 \
  --force
```

Comparison command:

```bash
python scripts/compare_pianovam_handskeleton_mediapipe.py \
  --data-root data/PianoVAM_v1 \
  --annotation-root data/PianoVAM_v1/mediapipe_tasks_cpu_image_annotations \
  --split test \
  --streams image \
  --output-dir runs/pianovam_handskeleton_mediapipe_tasks_cpu_image_test
```

## Overall Results

| Metric | Value |
| --- | ---: |
| Handskeleton hands | 709951 |
| MediaPipe hands | 690328 |
| Matched hands | 682178 |
| Precision | 0.988194 |
| Recall | 0.960880 |
| Miss rate | 0.039120 |
| Count mismatch frames | 31595 (8.70%) |
| Hand21 mean error | 3.566 px |
| Hand21 mean error P95 | 6.656 px |
| Hand21 P95-point error mean | 7.153 px |
| Hand bbox IoU mean | 0.943344 |
| Exposed palm detection frames | 0 |

The dominant count-mismatch pattern is under-detection in independent image
mode:

| Handskeleton hands | MediaPipe hands | Frames |
| ---: | ---: | ---: |
| 2 | 1 | 21315 |
| 1 | 2 | 7015 |
| 2 | 0 | 2082 |
| 1 | 0 | 1172 |

## Speed

| Metric | Value |
| --- | ---: |
| Total annotation wall time | 92.34 min |
| Effective FPS | 65.58 |
| Weighted mean image call latency | 13.773 ms |
| Weighted image P95 call latency | 14.996 ms |
| Video frame rate | 60 fps |

CPU image mode is slightly faster than real time on ace2 for this dataset, but
it is much slower than the earlier GPU `VIDEO` tracking run. The GPU tracking
run processed the same `363309` frames at about `181.30 FPS` with mean tracking
call latency about `4.184 ms`.

This is not a pure CPU-vs-GPU comparison because the modes differ:

- this run uses CPU delegate with independent `IMAGE` mode;
- the previous run used GPU delegate with temporal `VIDEO` mode.

## Per-Video Results

| Record | Frames | Precision | Recall | Mismatch | Mean px | P95 px | Max px | FPS | Mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-02-14_19-55-17 | 45319 | 0.9943 | 0.9588 | 8.39% | 4.368 | 8.705 | 335.678 | 65.85 | 13.740 |
| 2024-02-14_20-10-08 | 77992 | 0.9916 | 0.9809 | 5.15% | 3.772 | 7.578 | 286.658 | 64.54 | 14.016 |
| 2024-02-15_20-17-26 | 32185 | 0.9118 | 0.8197 | 40.59% | 6.003 | 10.218 | 318.726 | 71.01 | 12.643 |
| 2024-02-15_20-47-59 | 18451 | 0.9671 | 0.7846 | 40.00% | 5.103 | 11.862 | 191.247 | 72.67 | 12.313 |
| 2024-02-17_21-44-37 | 10672 | 0.9837 | 0.8897 | 16.14% | 4.198 | 7.872 | 232.658 | 69.63 | 12.885 |
| 2024-09-02_14-10-41 | 17588 | 0.9954 | 0.9953 | 1.81% | 3.084 | 4.862 | 154.883 | 64.44 | 14.028 |
| 2024-09-02_21-04-45 | 28426 | 0.9948 | 0.9913 | 2.43% | 3.470 | 6.149 | 198.497 | 64.25 | 14.026 |
| 2024-09-03_00-07-46 | 96493 | 0.9994 | 0.9986 | 0.37% | 2.651 | 3.935 | 304.260 | 64.16 | 14.113 |
| 2024-09-03_00-44-45 | 36183 | 0.9995 | 0.9969 | 0.67% | 2.602 | 4.068 | 234.349 | 64.18 | 14.072 |

## Comparison With GPU Tracking

| Metric | CPU image | GPU video/tracking |
| --- | ---: | ---: |
| Processed frames | 363309 | 363309 |
| Precision | 0.988194 | 0.985615 |
| Recall | 0.960880 | 0.999382 |
| Count mismatch rate | 8.70% | 2.96% |
| Hand21 mean error | 3.566 px | 0.545 px |
| Hand21 mean error P95 | 6.656 px | 1.030 px |
| Effective FPS | 65.58 | 181.30 |
| Mean call latency | 13.773 ms | 4.184 ms |

The CPU image run has slightly higher precision, but recall and landmark
agreement are much worse. This supports the earlier conclusion that PianoVAM
`Handskeleton` aligns with MediaPipe-style video tracking rather than with
independent per-frame image inference.

The most difficult records for independent image mode are:

- `2024-02-15_20-17-26`: recall `0.8197`, mismatch rate `40.59%`;
- `2024-02-15_20-47-59`: recall `0.7846`, mismatch rate `40.00%`;
- `2024-02-17_21-44-37`: recall `0.8897`, mismatch rate `16.14%`.

## Worst Landmark Outliers

| Record | Frame | GT | MP idx | MP hand | IoU | Mean px | P95 px | Max px |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2024-02-14_19-55-17 | 25467 | Left | 0 | Left | 0.1104 | 193.830 | 319.737 | 334.925 |
| 2024-02-14_19-55-17 | 25466 | Left | 0 | Left | 0.1203 | 186.118 | 315.660 | 334.479 |
| 2024-02-14_19-55-17 | 25581 | Left | 0 | Left | 0.1002 | 179.777 | 303.975 | 318.527 |
| 2024-02-15_20-17-26 | 17089 | Left | 0 | Right | 0.1520 | 177.280 | 299.517 | 318.726 |
| 2024-02-15_20-17-26 | 17082 | Left | 0 | Right | 0.1054 | 176.427 | 296.966 | 306.711 |

These outliers have low matching IoU and often involve hand ambiguity or
handedness disagreement, so they should be inspected visually before being
treated as pure keypoint regression failures.

## Files

Local compact annotation artifacts:

```text
data/PianoVAM_v1/mediapipe_tasks_cpu_image_annotations/
```

Local comparison artifacts:

```text
runs/pianovam_handskeleton_mediapipe_tasks_cpu_image_test/
```

This local run directory contains `summary.json`, `per_video_summary.json`,
`annotation_speed_by_video.json`, `frames.csv`, `matches.csv`, and `report.md`.

The full raw per-frame `mediapipe_annotations.json` files are large
approximately `6.6 GB` total and are kept on ace2:

```text
ace2:~/Documents/mediapipe_hand_ascend310b/data/PianoVAM_v1/mediapipe_tasks_cpu_image_annotations/
```
