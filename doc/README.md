# MediaPipe Hand Ascend 310B Documentation

This directory keeps the current deployment notes, validation records, and
PianoVAM evaluation reports for the MediaPipe hand pipeline on Ascend 310B.
Historical one-off debug logs are intentionally kept out of this index. Use the
repository root `README.md` and `scripts/README.md` for runnable command
references.

## Current Status

Default deployment models:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

Current conclusions:

- The default deployment uses the legacy full optimized `origin_dtype` palm
  detector OM plus the legacy full hand landmark OM.
- Video and WebRTC default to MediaPipe-style tracking. Dataset acceptance
  still uses independent image-mode evaluation unless the report explicitly
  states that it is a video tracking evaluation.
- Full-class models have been evaluated on both the 8T and 20T boards using the
  Portable HaGRIDv2 MediaPipe test set with 1663 images. The board-results
  table includes current `origin_dtype` defaults and mix-precision full-model
  comparison candidates.
- The full OM rebuilt on the 20T board matches the current full OM raw outputs,
  so a separate 20T-specific full OM is not required.
- Full PianoVAM legacy CPU image evaluation shows that independent-frame misses
  are dominated by palm-stage under-detection rather than hand-landmark failure.
- Ascend 8T full-video PianoVAM tracking evaluation shows that the OM tracking
  pipeline is highly consistent with the PianoVAM `Handskeleton` pseudo labels
  and the MediaPipe Tasks GPU tracking baseline, but it is still below real-time
  for 60 FPS video.
- Lite models can run and generate reports, but they are report-only candidates,
  not default acceptance models.
- Failed or clearly incorrect direct palm OM variants are not kept in
  `models/om/`.

## Documents

| Document | Purpose |
| --- | --- |
| [01_pipeline_graph.md](01_pipeline_graph.md) | MediaPipe Hand two-stage pipeline, legacy graph mapping, and validation layers. |
| [02_data_structures.md](02_data_structures.md) | Core preprocessing, anchor, palm detection, ROI, and landmark data structures. |
| [03_models.md](03_models.md) | Current model assets, default models, speed candidates, lite candidates, and removed OM variants. |
| [04_webrtc_runtime.md](04_webrtc_runtime.md) | WebRTC runtime design, camera ingestion, VENC/DVPP boundaries, and OM integration. |
| [05_board_validation_results.md](05_board_validation_results.md) | Unified 8T/20T model validation, speed, lite status, and OM rebuild records. |
| [06_tracking_algorithm.md](06_tracking_algorithm.md) | Video/WebRTC tracking state machine, ROI loopback behavior, and debug conclusions. |
| [07_amct_int8_quantization_8t_20t_results.md](07_amct_int8_quantization_8t_20t_results.md) | AMCT INT8 quantization results on 8T and 20T boards. |
| [08_pianovam_video_tracking_performance.md](08_pianovam_video_tracking_performance.md) | Legacy MediaPipe tracking analysis on PianoVAM sample frames. |
| [09_pianovam_mediapipe_tasks_gpu_tracking.md](09_pianovam_mediapipe_tasks_gpu_tracking.md) | Full PianoVAM test split tracking analysis using MediaPipe Tasks GPU on ace2. |
| [10_pianovam_mediapipe_tasks_cpu_image.md](10_pianovam_mediapipe_tasks_cpu_image.md) | Full PianoVAM test split image-mode analysis using MediaPipe Tasks CPU on ace2. |
| [11_pianovam_tasks_image_vs_tracking.md](11_pianovam_tasks_image_vs_tracking.md) | Direct full-test comparison between MediaPipe Tasks image mode and video tracking mode. |
| [12_pianovam_legacy_cpu_image_palm_stage.md](12_pianovam_legacy_cpu_image_palm_stage.md) | Full PianoVAM test split legacy CPU image analysis with palm-stage and landmark-stage diagnosis. |
| [13_pianovam_ascend8t_vs_mediapipe_tracking.md](13_pianovam_ascend8t_vs_mediapipe_tracking.md) | Full-video comparison between Ascend 8T OM tracking, MediaPipe Tasks GPU tracking, and PianoVAM Handskeleton pseudo labels. |
| [14_ascend20t_system_instability_record.md](14_ascend20t_system_instability_record.md) | Ascend 20T system instability record for the `opiaipro_20t_ubuntu22.04_desktop_aarch64_20250211.img` image, ext4 errors, conda crashes, and recovery criteria. |

## Maintenance Rules

- New formal board validation results should update
  [05_board_validation_results.md](05_board_validation_results.md) and this
  index if they change the deployment recommendation.
- New OM files must be classified as default, report-only, candidate, or
  historical failed path.
- Palm OM variants that fail accuracy validation should not be added to
  `models/om/`.
- If ATC outputs from different boards are numerically identical, do not keep
  duplicate board-specific OM files.
- Documentation in this directory must be written in English.
