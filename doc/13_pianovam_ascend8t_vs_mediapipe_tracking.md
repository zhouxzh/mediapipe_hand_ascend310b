# PianoVAM Ascend 8T Tracking vs MediaPipe Tracking

## Purpose

This document compares three result sources on PianoVAM video tracking:

1. PianoVAM `Handskeleton` labels, used as the reference annotation stream.
2. MediaPipe Tasks GPU tracking, used as the upstream MediaPipe tracking baseline.
3. The Ascend 310B `ascend8t` OM tracking pipeline, used as the deployment target result.

The main comparison uses the same complete test video:

```text
2024-02-15_20-47-59
```

This video is useful because the full record has already been evaluated by both
MediaPipe Tasks GPU tracking and the Ascend 8T OM tracking pipeline.

## Important Ground-Truth Qualification

PianoVAM `Handskeleton` should be treated as a MediaPipe-style pseudo-label
reference, not as independently corrected human ground truth. High agreement
with `Handskeleton` means the pipeline is consistent with the dataset's
MediaPipe-style tracking labels. It does not prove absolute hand landmark
accuracy against manual annotation.

This distinction matters for conclusions:

- It is valid to use `Handskeleton` to check whether the Ascend 310B pipeline
  preserves MediaPipe-compatible tracking behavior.
- It is valid to compare miss rate, extra hand rate, keypoint error, and speed
  against the same reference stream.
- It is not valid to claim that either MediaPipe or the Ascend OM pipeline is
  more accurate than a human labeler from this comparison alone.

## Evidence Sources

| Source | Path |
| --- | --- |
| Ascend 8T full-video summary | `runs/pianovam_hand_pipeline/ascend8t_tracking_full_2024-02-15_20-47-59_20260712_202019/summary.json` |
| Ascend 8T full-video frames | `runs/pianovam_hand_pipeline/ascend8t_tracking_full_2024-02-15_20-47-59_20260712_202019/frames.csv` |
| Ascend 8T full-video matches | `runs/pianovam_hand_pipeline/ascend8t_tracking_full_2024-02-15_20-47-59_20260712_202019/matches.csv` |
| MediaPipe Tasks GPU tracking comparison | `runs/pianovam_handskeleton_mediapipe_tasks_gpu_tracking_test/` |
| Full MediaPipe tracking report | `doc/09_pianovam_mediapipe_tasks_gpu_tracking.md` |
| Legacy MediaPipe tracking sample report | `doc/08_pianovam_video_tracking_performance.md` |

## Same-Video Comparison

### Test Video

| Item | Value |
| --- | ---: |
| Record | `2024-02-15_20-47-59` |
| Split | `test` |
| Resolution | `1920x1080` |
| Video FPS | `60.0` |
| Processed frames | `18451` |
| Duration | `307.52 s` |
| Reference hand instances | `34488` |

### Detection and Matching

Both systems are compared against the same PianoVAM `Handskeleton` reference
hands. Matching uses hand-landmark bounding-box IoU with threshold `0.1`, then
landmark error is computed in original video pixels.

| Metric | MediaPipe Tasks GPU tracking | Ascend 8T OM tracking | Difference, Ascend - MediaPipe |
| --- | ---: | ---: | ---: |
| Processed frames | `18451` | `18451` | `0` |
| Reference hands | `34488` | `34488` | `0` |
| Predicted hands | `35751` | `35692` | `-59` |
| Matched hands | `34478` | `34482` | `+4` |
| Unmatched reference hands | `10` | `6` | `-4` |
| Unmatched predicted hands | `1273` | `1210` | `-63` |
| Precision | `0.964393` | `0.966099` | `+0.001706` |
| Recall | `0.999710` | `0.999826` | `+0.000116` |
| Miss rate | `0.000290` | `0.000174` | `-0.000116` |
| Count mismatch frames | `1283` | `1216` | `-67` |
| Count mismatch rate | `0.069536` | `0.065904` | `-0.003631` |

The Ascend 8T pipeline is slightly better than the MediaPipe Tasks GPU tracking
baseline on this video under the `Handskeleton` reference: it has fewer extra
hands, fewer missed reference hands, and a lower mismatch-frame rate.

The differences are small in absolute terms. The practical conclusion is not
that the Ascend model is fundamentally more accurate, but that the OM conversion,
coordinate mapping, ROI loopback, and tracking state are consistent with the
MediaPipe-style reference.

### Landmark Agreement

| Metric | MediaPipe Tasks GPU tracking | Ascend 8T OM tracking | Difference, Ascend - MediaPipe |
| --- | ---: | ---: | ---: |
| Hand21 mean error, mean | `0.700303 px` | `0.456947 px` | `-0.243356 px` |
| Hand21 mean error, median | `0.574071 px` | `0.351750 px` | `-0.222321 px` |
| Hand21 mean error, P95 | `1.237754 px` | `0.966269 px` | `-0.271486 px` |
| Hand21 mean error, max | `59.996849 px` | `28.036276 px` | `-31.960573 px` |
| BBox IoU mean | `0.990518` | `0.993351` | `+0.002833` |

The Ascend 8T output is very close to `Handskeleton`: median mean-landmark error
is about `0.35 px`, and the P95 per-hand mean error is below `1 px`. This is
strong evidence that the landmark coordinate transform and ROI reconstruction
are correct for normal tracking frames.

The largest Ascend 8T landmark outliers are sparse:

| Frame | Hand | ROI source | Mean px | Max px | Match IoU |
| ---: | --- | --- | ---: | ---: | ---: |
| `11942` | Left | `previous_landmarks` | `28.036` | `43.680` | `0.683` |
| `7` | Left | `palm_detection` | `24.081` | `36.007` | `0.702` |
| `509` | Left | `previous_landmarks` | `17.568` | `43.088` | `0.783` |
| `5481` | Right | `previous_landmarks` | `16.340` | `32.278` | `0.913` |
| `5480` | Right | `previous_landmarks` | `13.477` | `29.354` | `0.876` |

Most outliers come from `previous_landmarks`, which points to transient tracking
drift or identity/ROI ambiguity rather than a systematic palm detector failure.

### Speed

| Metric | MediaPipe Tasks GPU tracking | Ascend 8T OM tracking |
| --- | ---: | ---: |
| Hardware | ace2 NVIDIA GPU | Ascend 310B on `ascend8t` |
| Effective / pipeline FPS | `180.47` | `51.29` |
| Relative to 60 FPS video | `3.01x realtime` | `0.85x realtime` |
| Mean total pipeline latency | not directly comparable | `19.50 ms` |
| Median total pipeline latency | not directly comparable | `18.60 ms` |
| P95 total pipeline latency | not directly comparable | `20.44 ms` |

The speed comparison is hardware-dependent. MediaPipe Tasks GPU on ace2 is about
`3.52x` faster than the current Ascend 8T Python OM pipeline for this video.

The Ascend 8T pipeline is close to but still below real-time for a `60 fps`
video. To reach 60 FPS, the current `51.29 FPS` pipeline needs roughly a `17%`
speedup.

Ascend 8T timing breakdown:

| Stage | Mean latency |
| --- | ---: |
| Total pipeline | `19.496 ms` |
| Palm detector | `1.247 ms` |
| ROI | `5.478 ms` |
| Landmark | `9.660 ms` |
| Postprocess | `2.918 ms` |

Because tracking reuses previous landmarks for most frames, the palm detector
mean is low even though a detector invocation is expensive when it occurs. The
main runtime cost in stable tracking is the ROI path plus landmark inference.

### Tracking Behavior

| Metric | Ascend 8T OM tracking |
| --- | ---: |
| Frames with palm detector run | `818` |
| Frames reusing tracking state | `17633` |
| Palm detector run rate | `4.43%` |
| Tracking reuse frame rate | `95.57%` |
| Matched hands from palm detection ROI | `20` |
| Matched hands from previous landmarks ROI | `34462` |

This confirms that the tested Ascend 8T run is actually operating as a tracking
pipeline. Most frames reuse `previous_landmarks`; only a small number require
fresh palm detection.

MediaPipe Tasks GPU tracking does not expose palm detector invocation details
through the Python result API, so equivalent detector-run telemetry is not
available for the MediaPipe Tasks baseline.

## Dataset-Level MediaPipe Tracking Baseline

The full PianoVAM test split MediaPipe Tasks GPU tracking run gives the broader
reference context:

| Metric | Full test split MediaPipe Tasks GPU tracking |
| --- | ---: |
| Videos | `9` |
| Processed frames | `363309` |
| Handskeleton hands | `709951` |
| MediaPipe hands | `719867` |
| Matched hands | `709512` |
| Precision | `0.985615` |
| Recall | `0.999382` |
| Miss rate | `0.000618` |
| Count mismatch rate | `0.029628` |
| Hand21 mean error | `0.545076 px` |
| Hand21 mean error P95 | `1.030125 px` |
| Hand bbox IoU mean | `0.992174` |
| Effective FPS | `181.30` |

The selected video `2024-02-15_20-47-59` is harder than the full-test average
for count consistency: MediaPipe Tasks GPU tracking has a `6.95%` count mismatch
rate on this video versus `2.96%` over the full test split. That makes it a
reasonable stress case for checking tracking robustness.

## First-600-Frame Ascend 8T Smoke Test

Before the full-video run, the Ascend 8T pipeline was tested on the first 600
continuous frames of `2024-02-14_19-55-17`.

| Metric | Value |
| --- | ---: |
| Processed frames | `600` |
| Reference hands | `1164` |
| Predicted hands | `1195` |
| Matched hands | `1159` |
| Precision | `0.969874` |
| Recall | `0.995704` |
| Miss rate | `0.004296` |
| Count mismatch rate | `0.063333` |
| Estimated pipeline FPS | `53.32` |
| Hand21 mean error | `0.978991 px` |
| Hand21 mean error P95 | `3.602713 px` |

This run was mainly a correctness smoke test. It exposed startup-frame misses
and short tracking transients, but the pipeline completed normally and produced
valid tracking outputs.

## Interpretation

The Ascend 8T OM tracking pipeline is accurate enough to use PianoVAM
`Handskeleton` as a reference for board-side tracking validation.

For the complete same-video comparison, the Ascend 8T pipeline is at least as
consistent with PianoVAM `Handskeleton` as the MediaPipe Tasks GPU tracking
baseline:

- It matches `4` more reference hands on the same `18451` frames.
- It misses only `6` reference hands across the full video.
- It produces `63` fewer unmatched predicted hands.
- It has lower mean and P95 landmark error.
- Its mismatch-frame rate is slightly lower.

The main accuracy weakness is not missing real reference hands. The dominant
error is extra predicted hands, which lowers precision and creates count
mismatch frames. This is the same failure pattern seen in MediaPipe tracking,
so it is likely tied to ambiguous hands, temporary duplicate tracking, or
reference/prediction count instability rather than a simple landmark-stage
failure.

The main deployment weakness is speed. The current Python OM tracking pipeline
runs at about `51.29 FPS` on a `60 FPS` video. It is close to real-time, but not
yet real-time at the source frame rate. Optimization should focus on the
landmark path, ROI generation, and Python postprocessing before changing model
accuracy thresholds.

## Recommended Next Steps

1. Run the same Ascend 8T tracking command over all 9 PianoVAM test videos.
   This will determine whether the full-test precision stays close to the
   MediaPipe Tasks tracking baseline.
2. Keep `Handskeleton` as the primary pseudo-label reference for tracking
   evaluation, but avoid describing it as manual ground truth.
3. Record palm detector run rate, tracking reuse rate, extra predicted hands,
   missed reference hands, and landmark error for every board-side run.
4. Optimize speed only after full-test accuracy is stable. The first targets
   should be ROI processing, postprocessing, and landmark inference overhead.
5. Use a separate manually checked sample only if the goal changes from
   MediaPipe compatibility to absolute human-label accuracy.
