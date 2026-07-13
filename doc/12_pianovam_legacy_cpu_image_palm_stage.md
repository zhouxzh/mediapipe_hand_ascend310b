# PianoVAM Legacy CPU Image Palm-Stage Evaluation

## Scope

- Dataset split: `test`.
- Videos: `9`.
- Frames: `363309`.
- Reference labels: PianoVAM `Handskeleton` 21-point annotations, treated as
  MediaPipe-style pseudo labels rather than independent human ground truth.
- Model/API: legacy MediaPipe `0.10.14` hand graph on CPU.
- Graph mode: `use_prev_landmarks=False`, exported as the `image` stream.
- Purpose: separate full-frame detection failures into palm-stage failures and
  hand-landmark-stage failures.

This report is different from the MediaPipe Tasks CPU image report. MediaPipe
Tasks does not expose raw palm detections. The legacy graph does expose
`palm_detections`, `hand_rects_from_palm_detections`, final hand landmarks, and
`hand_rects_from_landmarks`, so it can be used for stage-level diagnosis.

Annotation command used on ace2:

```bash
python scripts/annotate_pianovam_mediapipe_legacy.py \
  --data-root data/PianoVAM_v1 \
  --split test \
  --running-mode image \
  --output-root data/PianoVAM_v1/mediapipe_legacy_cpu_image_annotations \
  --save-vis 2 \
  --force
```

Comparison command used on ace2:

```bash
python scripts/compare_pianovam_handskeleton_mediapipe.py \
  --data-root data/PianoVAM_v1 \
  --annotation-root data/PianoVAM_v1/mediapipe_legacy_cpu_image_annotations \
  --split test \
  --streams image \
  --output-dir runs/pianovam_handskeleton_mediapipe_legacy_cpu_image_test
```

## Output Files

Per-video annotation directories contain:

| File | Meaning |
| --- | --- |
| `mediapipe_annotations.json` | Full frame-level legacy graph output. |
| `image_mediapipe_annotations.json` | Image-stream-only annotation view. |
| `frames.csv` | Per-frame palm count, ROI count, hand count, and timing. |
| `palm_detections.csv` | Raw palm detector boxes, scores, labels, and seven palm keypoints. |
| `palm_rois.csv` | Landmark ROIs generated from palm detections. |
| `hands.csv` | Final 21-point hand landmarks and associated palm/ROI metadata. |
| `tracking_rois.csv` | ROIs generated from hand landmarks; useful for tracking-loop analysis. |
| `summary.json` | Per-video run metadata and timing. |

Comparison outputs are under:

```text
runs/pianovam_handskeleton_mediapipe_legacy_cpu_image_test
```

## Overall Accuracy

| Metric | Value |
| --- | ---: |
| Handskeleton hands | 709951 |
| MediaPipe hands | 690735 |
| Matched hands | 682627 |
| Unmatched Handskeleton hands | 27324 |
| Unmatched MediaPipe hands | 8108 |
| Precision | 0.988262 |
| Recall | 0.961513 |
| Miss rate | 0.038487 |
| Count mismatch frames | 31196 |
| Count mismatch frame rate | 0.085866 |
| Hand21 mean error mean | 2.408 px |
| Hand21 mean error median | 1.964 px |
| Hand21 mean error P95 | 4.345 px |
| Hand21 mean error max | 194.969 px |
| Hand21 per-hand P95-point error mean | 4.893 px |
| Hand21 per-hand P95-point error P95 | 9.961 px |
| Hand bbox IoU mean | 0.953676 |
| Hand bbox IoU median | 0.960420 |
| Hand bbox IoU P95 | 0.984665 |

The overall image-mode accuracy is close to the earlier MediaPipe Tasks CPU
image result, but this run provides palm-stage visibility. Compared with the
Tasks CPU image run, the legacy graph matched slightly more hands and reduced
the mean hand21 error:

| Run | Matched hands | Precision | Recall | Hand21 mean error |
| --- | ---: | ---: | ---: | ---: |
| Tasks CPU image | 682178 | 0.988194 | 0.960880 | 3.566 px |
| Legacy CPU image | 682627 | 0.988262 | 0.961513 | 2.408 px |

## Palm-to-Landmark Flow

| Metric | Value |
| --- | ---: |
| Frames with at least one palm detection | 359185 |
| Palm detections | 692276 |
| Final hand landmarks | 690735 |
| Palm-to-landmark ratio | 0.997774 |

The palm-to-landmark ratio is very high. Once the legacy palm detector produces
a candidate, the landmark stage almost always produces a final hand. The main
recall loss is therefore concentrated in the palm detector stage rather than in
the hand landmark model.

## Stage Diagnosis

The stage diagnosis is frame-level. For image mode, a frame with fewer palm
detections than Handskeleton hands is counted as a palm-stage miss. A frame with
enough palms but too few final hand landmarks is counted as a landmark-stage
miss after palm detection.

| Diagnosis | Frames | Share of frames |
| --- | ---: | ---: |
| `all_hands_matched` | 329824 | 90.783% |
| `palm_under_detected` | 20227 | 5.567% |
| `extra_mediapipe_hands` | 7004 | 1.928% |
| `palm_no_detection` | 2994 | 0.824% |
| `no_reference_hands` | 1189 | 0.327% |
| `landmark_localization_or_matching_error` | 1100 | 0.303% |
| `landmark_under_detected_after_palm` | 971 | 0.267% |

Grouped failure interpretation:

| Group | Frames | Share of frames |
| --- | ---: | ---: |
| Palm-stage misses (`palm_under_detected` + `palm_no_detection`) | 23221 | 6.391% |
| Landmark-stage issues after palm detection | 2071 | 0.570% |

The palm-stage miss count is more than ten times the landmark-stage issue
count. This supports the conclusion that the full-frame image-mode misses are
primarily caused by palm detection failing to produce enough hand candidates.

## Runtime

| Metric | Value |
| --- | ---: |
| Completed records | 9 / 9 |
| Failed records | 0 |
| Total annotation wall time | 92.87 min |
| Effective FPS | 65.20 |
| Video frame rate | 60 fps |

Per-video annotation time:

| Record time | Elapsed seconds |
| --- | ---: |
| `2024-02-14_19-55-17` | 692.965 |
| `2024-02-14_20-10-08` | 1210.844 |
| `2024-02-15_20-17-26` | 456.292 |
| `2024-02-15_20-47-59` | 253.249 |
| `2024-02-17_21-44-37` | 153.410 |
| `2024-09-02_14-10-41` | 276.539 |
| `2024-09-02_21-04-45` | 444.228 |
| `2024-09-03_00-07-46` | 1516.220 |
| `2024-09-03_00-44-45` | 568.714 |

## Conclusion

The full PianoVAM test split legacy CPU image evaluation confirms that
independent per-frame misses are dominated by the palm detector stage. The hand
landmark stage is not the primary bottleneck in this run: final landmarks are
produced for `99.777%` of palm detections, and landmark-stage issues account
for only `0.570%` of all frames.

For further model debugging, prioritize palm-stage analysis: missed hands,
low-score palms, palm NMS behavior, and frame conditions where only one palm is
detected while Handskeleton contains two hands.
