# PianoVAM Legacy MediaPipe Video Tracking Evaluation

## Goal

This evaluation checks how closely PianoVAM `Handskeleton` labels match legacy
MediaPipe Hands outputs on video. It compares two legacy graph modes:

- `image`: independent per-frame detection with `use_prev_landmarks=False`.
- `tracking`: video tracking with `use_prev_landmarks=True`.

`Handskeleton` is treated as a MediaPipe-style pseudo label distributed with
PianoVAM, not as independent human ground truth.

## Environment

- Repository: `D:\Github\mediapipe_hand_ascend310b`
- Conda environment: `mediapipe_legacy`
- MediaPipe version: `0.10.14`
- Video resolution: `1920x1080`
- Video frame rate: `60 fps`

## Data

This run used the first 600 continuous frames from each of the 9 PianoVAM
`test` videos. Continuous frames were used so tracking state remains valid.

| Item | Value |
| --- | ---: |
| videos | 9 |
| frames per video | 600 |
| total frames | 5400 |
| Handskeleton hand instances | 8278 |
| annotation output | `data/PianoVAM_v1/mediapipe_legacy_annotations/` |
| comparison output | `runs/pianovam_handskeleton_mediapipe_compare/test_first600_tracking_analysis/` |

Annotation command:

```bash
python scripts/annotate_pianovam_mediapipe_legacy.py \
  --data-root data/PianoVAM_v1 \
  --split test \
  --frame-stride 1 \
  --max-frames 600 \
  --save-vis 2 \
  --force
```

Comparison command:

```bash
python scripts/compare_pianovam_handskeleton_mediapipe.py \
  --data-root data/PianoVAM_v1 \
  --annotation-root data/PianoVAM_v1/mediapipe_legacy_annotations \
  --split test \
  --streams image,tracking \
  --output-dir runs/pianovam_handskeleton_mediapipe_compare/test_first600_tracking_analysis
```

## Output Streams

Each video directory contains:

- `mediapipe_annotations.json`: combined `image` and `tracking` streams.
- `image_mediapipe_annotations.json`: image-mode stream.
- `tracking_mediapipe_annotations.json`: tracking stream.
- `frames.csv`: per-frame hand counts and timing.
- `hands.csv`: hand landmark and ROI summary.
- `palm_detections.csv`: palm detector outputs.

## Overall Results

| Metric | image | tracking |
| --- | ---: | ---: |
| processed frames | 5400 | 5400 |
| Handskeleton hands | 8278 | 8278 |
| MediaPipe hands | 7486 | 8623 |
| matched hands | 7255 | 8264 |
| precision | 96.91% | 95.84% |
| recall | 87.64% | 99.83% |
| miss rate | 12.36% | 0.17% |
| count mismatch rate | 18.07% | 6.83% |
| hand21 mean px | 4.46 px | 0.48 px |
| hand21 mean px P95 | 14.65 px | 1.67 px |
| hand21 P95 px mean | 8.71 px | 1.01 px |
| hand bbox IoU mean | 0.9293 | 0.9933 |
| frames with palm detections | 4303 | 558 |
| non-empty palm detection frame rate | 79.69% | 10.33% |

The conclusion is clear: PianoVAM `Handskeleton` aligns much more closely with
legacy MediaPipe `tracking` than with per-frame `image` mode. On these 5400
frames, tracking misses only 14 Handskeleton hand instances and has a mean
21-point error of about `0.48 px`.

## Tracking Speed

The timing below is from the local PC CPU legacy MediaPipe graph. It is not
Ascend OM model timing. The script runs `image` and `tracking` graphs during the
same video read, so absolute values are local-reference timings; the relative
difference is still useful.

| Video | image mean ms | tracking mean ms | image P95 ms | tracking P95 ms |
| --- | ---: | ---: | ---: | ---: |
| 2024-02-14_19-55-17 | 84.04 | 65.14 | 117.06 | 90.38 |
| 2024-02-14_20-10-08 | 91.82 | 65.35 | 127.91 | 88.49 |
| 2024-02-15_20-17-26 | 84.24 | 69.45 | 122.20 | 96.71 |
| 2024-02-15_20-47-59 | 88.13 | 69.84 | 132.70 | 95.30 |
| 2024-02-17_21-44-37 | 92.21 | 66.94 | 127.57 | 92.88 |
| 2024-09-02_14-10-41 | 89.65 | 70.38 | 128.98 | 96.13 |
| 2024-09-02_21-04-45 | 94.98 | 71.86 | 144.91 | 100.16 |
| 2024-09-03_00-07-46 | 95.26 | 70.29 | 141.20 | 99.74 |
| 2024-09-03_00-44-45 | 92.61 | 68.70 | 147.11 | 92.61 |
| **average** | **90.33** | **68.66** | **132.29** | **94.71** |

Tracking is about `24.0%` faster than per-frame image mode, or about `1.32x`.
The main reason is that the tracking stream reuses ROIs generated from previous
landmarks. The saved graph output contains non-empty palm detections on `10.33%`
of frames; this is useful evidence of ROI reuse, but it should not be treated as
complete detector-invocation telemetry.

## Per-Video Consistency

| Video | image recall | image mean px | image mismatch | tracking recall | tracking mean px | tracking mismatch | tracking palm frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-02-14_19-55-17 | 94.63% | 4.15 | 19.00% | 99.57% | 0.63 | 6.17% | 8 |
| 2024-02-14_20-10-08 | 93.01% | 4.51 | 20.83% | 99.91% | 0.42 | 5.17% | 2 |
| 2024-02-15_20-17-26 | 78.33% | 7.31 | 35.17% | 99.83% | 0.80 | 21.67% | 77 |
| 2024-02-15_20-47-59 | 85.39% | 5.46 | 21.83% | 100.00% | 0.57 | 6.50% | 104 |
| 2024-02-17_21-44-37 | 76.49% | 3.75 | 28.83% | 100.00% | 0.30 | 1.00% | 108 |
| 2024-09-02_14-10-41 | 91.52% | 3.52 | 16.67% | 100.00% | 0.20 | 2.00% | 99 |
| 2024-09-02_21-04-45 | 89.22% | 5.31 | 14.17% | 100.00% | 0.71 | 7.00% | 5 |
| 2024-09-03_00-07-46 | 96.14% | 2.41 | 4.83% | 99.72% | 0.26 | 9.17% | 64 |
| 2024-09-03_00-44-45 | 94.96% | 2.73 | 1.33% | 99.64% | 0.52 | 2.83% | 91 |

Per-frame `image` mode has noticeably lower recall on several videos. This
supports the interpretation that PianoVAM skeletons were generated by a
MediaPipe-style video tracking process rather than by independent detection on
each frame.

## Implications

1. Use the `tracking` reference stream when evaluating video tracking behavior.
2. Treat `Handskeleton` as a MediaPipe-style pseudo label, not independent
   human-labeled ground truth.
3. Do not evaluate tracking with skipped frames unless the goal is specifically
   to test robustness to sparse input. Skipping frames changes ROI state.
4. Use `image` mode to inspect palm detector robustness, but not as a substitute
   for tracking evaluation.
5. End-to-end tracking speed is dominated by the landmark ROI path once tracking
   is stable; exact palm detector invocation counts require explicit runtime
   instrumentation, not just non-empty palm output counts.

## Limitations

- This run used only the first 600 frames per test video.
- Timing comes from the local CPU legacy MediaPipe graph, not from Ascend 310B.
- PianoVAM `Handskeleton` is generated by a MediaPipe-style process and should
  not be interpreted as absolute human keypoint ground truth.

## Next Step

For a stable full-test conclusion, regenerate annotations without
`--max-frames 600`. For Ascend 310B evaluation, run the board-side OM pipeline
in tracking mode against the same `tracking` reference stream and report
keypoint error, miss rate, and end-to-end latency.
