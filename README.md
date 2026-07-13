# MediaPipe Hand Ascend 310B

This repository is the Ascend 310B deployment package for the MediaPipe Hand
two-stage pipeline.

The deployed runtime is:

```text
image/video frame
  -> MediaPipe-style ImageToTensor preprocessing
  -> palm detector OM
  -> SSD decode + weighted NMS
  -> MediaPipe-style NormalizedRect hand ROI crop
  -> hand landmark OM
  -> LandmarkProjection back to the original frame
```

For videos and WebRTC streams the default runtime also uses MediaPipe-style
hand tracking: landmark results from the previous frame generate the next hand
ROI, and the palm detector is skipped when enough hands are already tracked.
Dataset evaluation keeps independent image mode so every image is validated
from the palm detector.

## Runtime Models

Production OM models:

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

The palm detector OM is generated from the legacy full palm model through
mathematically equivalent ONNX graph rewrites for 310B stability:

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx
  -> models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
  -> models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
```

Direct palm OM files that do not pass raw-output and end-to-end validation are
not kept in `models/om/`.

## Board Environment

On the Ascend 310B board:

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Install Python dependencies manually when needed:

```bash
python -m pip install -r requirements.txt
```

If the board has already installed `huggingface_hub==1.4.1` and pip reports a
conflict with `tokenizers==0.19.1`, downgrade HF Hub to the pinned compatible
version:

```bash
python -m pip install --upgrade --force-reinstall "huggingface_hub==0.36.2"
python -m pip check
```

## Portable HaGRIDv2 MediaPipe Dataset

The validation dataset is hosted as a Hugging Face dataset:

```text
zhouxzh/portable-hagridv2-mediapipe-hand
```

It is a Parquet dataset with embedded JPEG images. The exported metadata records
`9754` images split as `train=7246`, `valid=845`, and `test=1663`. Labels include
MediaPipe-style `palm_bbox_xyxy`, `palm7_keypoints`, and `full21_keypoints` in
the `normalized_exported_image` coordinate system.

On the Ascend 310B board, activate `base` and expose the user-local HF CLI:

```bash
cd ~/Documents/mediapipe_hand_ascend310b
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
```

Download only the test split for OM validation:

```bash
hf download zhouxzh/portable-hagridv2-mediapipe-hand \
  test-00000.parquet dataset.json keypoints.json summary.json \
  --repo-type dataset \
  --local-dir data/portable-hagridv2-mediapipe-hand
```

Download the full dataset only when the board has stable storage and power:

```bash
hf download zhouxzh/portable-hagridv2-mediapipe-hand \
  --repo-type dataset \
  --local-dir data/portable-hagridv2-mediapipe-hand
```

If `hf` is not installed in the active environment, install the pinned versions
from `requirements.txt` manually and rerun the commands above.

Evaluate the downloaded test split with the deployed OM models:

```bash
python scripts/eval_hf_hand_dataset_om.py --model-set full --max-images 20 --save-vis 2
```

Run the full `1663` image test split for both full and lite OM models:

```bash
python scripts/eval_hf_hand_dataset_om.py --model-set full,lite
```

By default the evaluator reports three components in the same run:

- `cascade`: full deployed two-stage pipeline, detector output drives landmark ROI.
- `palm`: palm detector OM only, evaluated against dataset `palm_bbox_xyxy` and `palm7_keypoints`.
- `landmark`: landmark OM only, using dataset palm bbox/palm7 to build the ROI.

Use `--components cascade`, `--components palm`, or `--components landmark` for
targeted debugging. The default is `--components cascade,palm,landmark`.

Add `--run-onnx` only when you also need ONNX reference metrics. Reports are
written under `runs/hf_hand_dataset_om/`.

## PianoVAM Dataset

Download PianoVAM v1 with the Hugging Face CLI:

```bash
hf download PianoVAM/PianoVAM_v1 \
  --repo-type dataset \
  --local-dir data/PianoVAM_v1
```

On Windows PowerShell:

```powershell
hf download PianoVAM/PianoVAM_v1 `
  --repo-type dataset `
  --local-dir data/PianoVAM_v1
```

If the download fails with `cas-server.xethub.hf.co` or `401 Unauthorized`,
disable the Xet/CAS backend and rerun the same command. Existing files in
`data/PianoVAM_v1` will be reused.

```powershell
$env:HF_HUB_DISABLE_XET="1"
hf download PianoVAM/PianoVAM_v1 `
  --repo-type dataset `
  --local-dir data/PianoVAM_v1
```

For Linux or the Ascend board:

```bash
export HF_HUB_DISABLE_XET=1
hf download PianoVAM/PianoVAM_v1 \
  --repo-type dataset \
  --local-dir data/PianoVAM_v1
```

## Validate ONNX vs OM

Run the video conversion checks:

```bash
python scripts/run_video_onnx_om_checks.py --video data/eval_videos/test.mp4
```

The video checks default to `--pipeline-mode tracking`, matching MediaPipe's
streaming graph. Use `--pipeline-mode image` to force a palm detector pass on
every processed frame.

This runs two checks:

- `optimized`: optimized palm ONNX vs final optimized palm OM, checking the
  exact ATC input graph against the deployed OM.
- `original`: original legacy full palm ONNX vs final optimized palm OM,
  checking that the final OM has no end-to-end regression relative to the
  original model graph.

Quick smoke test:

```bash
python scripts/run_video_onnx_om_checks.py --video data/eval_videos/test.mp4 --max-frames 5 --save-vis 2
```

Outputs are written under `runs/video_onnx_om_compare/`.

Dedicated evaluation videos live under `data/eval_videos/`. Generate
MediaPipe legacy graph annotations on the PC with:

```bash
python scripts/annotate_mediapipe_video.py --video data/eval_videos/test.mp4
```

MediaPipe annotation output is stored by video name. For
`data/eval_videos/test.mp4`, the folder is `data/eval_videos/annotations/test/`.

## Reproduce The OM Model

To rebuild the production palm OM on the board:

```bash
python scripts/build_optimized_palm_om.py \
  --skip-rewrite \
  --compile-om \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx \
  --om-output models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype \
  --atc-log runs/palm_om/atc_logs/downsample_resize_maxpool_slices_origin_dtype.log \
  --precision-mode must_keep_origin_dtype
```

To regenerate the optimized ONNX graph before ATC:

```bash
python scripts/build_optimized_palm_om.py \
  --input-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
```

The individual graph rewrite scripts are kept in `scripts/` so the conversion
can be debugged and reproduced.

## Realtime WebRTC Demo

```bash
python scripts/webrtc_hand_om_app.py \
  --source /dev/video0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --camera-backend opencv \
  --camera-fourcc MJPG \
  --encoder-mode cpu \
  --pipeline-mode tracking \
  --port 8080
```

Open the printed URL from a browser.
Use `--pipeline-mode image` or the page's Pipeline selector when you need to
debug frame-by-frame palm detection without tracking.

The WebRTC server defaults to LAN-only ICE candidates (`--ice-servers ""`).
This avoids the public STUN path in aiortc, which has triggered native
`Illegal instruction` crashes on the 20T board's base environment. Add
`--ice-servers stun:...` only when the browser must connect across NAT.

## Repository Layout

```text
hand_pipeline/   Core preprocessing, decode, ROI, OM runtime, visualization
scripts/         Board deployment, validation, and OM reproduction entry points
models/onnx/     ONNX reference graphs
models/om/       Ascend 310B OM models
data/eval_videos/ Dedicated video evaluation assets
web/             WebRTC browser client
webrtc_app/      Camera, DVPP, and CANN encoder helpers
doc/             Current deployment, validation, and algorithm notes
runs/            Generated validation outputs
```

The code is intended to run directly from the copied repository; installing the
repository itself as a package is not required.
