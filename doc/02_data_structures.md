# Core Data Structures

This document explains the main data structures used to reproduce the
MediaPipe Hands pipeline and how the validation datasets enter the baseline.

## 1. `LetterboxInfo`

The palm detector input is not produced by stretching the original image to a
square, and it is not the same as a simple `resize + copyMakeBorder` path. The
legacy `PalmDetectionCpu` subgraph uses `ImageToTensorCalculator`: it builds a
full-image ROI with keep-aspect-ratio padding and samples directly into the
`192x192` detector tensor with continuous coordinates.

```text
source image HxW
  -> full-image ROI(center_x, center_y, roi_width, roi_height)
  -> keep-aspect-ratio ROI expansion and normalized padding
  -> warpPerspective sampling
  -> RGB float32 NHWC [1, 192, 192, 3], range [0, 1]
```

`LetterboxInfo` stores two kinds of values:

| Field | Purpose |
| --- | --- |
| `orig_width / orig_height` | Source image size for mapping normalized coordinates back to pixels. |
| `resized_width / resized_height` | Integer-equivalent resized dimensions, used only for debugging and compatibility notes. |
| `pad_left / pad_top / pad_right / pad_bottom` | Integer-equivalent padding, used only for debugging and compatibility notes. |
| `normalized_padding_values` | The actual continuous MediaPipe padding used to unpad detector output coordinates. |

Palm detector coordinates are expressed in the padded normalized tensor
coordinate system. Decode must remove `normalized_padding_values` before
mapping detections back to source pixels. Using integer padding here introduces
systematic pixel-level drift in the final 21 landmarks.

## 2. Palm Anchors

The MediaPipe palm detector is SSD-style. The model predicts offsets relative
to fixed anchors rather than direct boxes.

Current anchor parameters:

| Parameter | Value |
| --- | ---: |
| input size | `192x192` |
| strides | `8, 16, 16, 16` |
| feature map sizes | `24x24`, `12x12`, `12x12`, `12x12` |
| anchors per location | `2` |
| total anchors | `2016` |
| fixed anchor size | `1.0` |

The total anchor count is:

```text
24 * 24 * 2 + 12 * 12 * 2 * 3 = 2016
```

## 3. Palm Detection Output

Palm detector outputs:

| Output | Shape | Meaning |
| --- | --- | --- |
| `Identity` | `[1, 2016, 18]` | Box plus 7 palm keypoints for each anchor. |
| `Identity_1` | `[1, 2016, 1]` | Classification logit for each anchor. |

The 18 values are:

```text
[x_center, y_center, w, h,
 kp0_x, kp0_y, ..., kp6_x, kp6_y]
```

Decode steps:

1. Apply `sigmoid` to scores.
2. Filter low-score anchors.
3. Remove ImageToTensor normalized padding with `LetterboxInfo`.
4. Map normalized coordinates back to source image pixels.
5. Merge overlapping detections with weighted NMS.

The `data/palm_datasets` ground truth contains manually checked palm boxes and
7 palm keypoints and is the main acceptance set for palm detector accuracy.

## 4. Hand ROI

MediaPipe does not feed the palm box directly into the landmark model. It
builds a rotated hand ROI from the palm box and palm keypoints.

`HandRoi` stores:

| Field | Meaning |
| --- | --- |
| `center` | ROI center in source-image pixels. |
| `size` | Square ROI side length in pixels. |
| `rotation` | ROI rotation in radians. |
| `matrix` | Affine transform from source image to `224x224` crop. |
| `inverse` | Inverse affine transform from crop coordinates back to source image. |
| `crop` | Landmark model input image. |

Current palm-to-hand-rect parameters:

| Parameter | Value |
| --- | ---: |
| shift x | `0.0` |
| shift y | `-0.5` |
| scale x | `2.6` |
| scale y | `2.6` |
| square long | enabled |

Cropping uses `cv2.warpAffine(..., borderMode=cv2.BORDER_REPLICATE)`. Border
mode affects landmark inputs when hands are near image edges.

## 5. Hand Landmark Output

Hand landmark model input:

```text
RGB float32 NHWC [1, 224, 224, 3], range [0, 1]
```

Outputs:

| Output | Shape | Meaning |
| --- | --- | --- |
| `Identity` | `[1, 63]` | 21 ROI-space landmarks, each with `x, y, z`. |
| hand score output | scalar | Hand-presence score. |
| handedness output | scalar/classification | Left/right hand classification. |
| world landmark output | `[1, 63]` | 21 world-space landmarks. |

`Identity` is reshaped to `[21, 3]`. The `x, y` values are ROI-space
coordinates and are projected back to source pixels with `HandRoi.inverse`.

## 6. Manual 21-Point Ground Truth

`data/handlm_datasets/annotations.json` is a COCO-style keypoint annotation set:

```json
{
  "images": [...],
  "annotations": [
    {
      "image_id": 1,
      "bbox": [x, y, w, h],
      "keypoints": [x0, y0, v0, ...]
    }
  ]
}
```

The keypoint order matches MediaPipe's 21-point hand topology:

```text
0 wrist
1-4 thumb
5-8 index
9-12 middle
13-16 ring
17-20 pinky
```

This dataset is treated as manually corrected ground truth for evaluating the
landmark model itself in `224x224` crop coordinates. It bypasses palm detection
and therefore isolates landmark-model accuracy and speed.
