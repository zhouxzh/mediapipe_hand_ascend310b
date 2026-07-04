# 核心数据结构

本文件解释本工程复刻 MediaPipe Hand 时用到的核心数据结构，以及两个人工校正数据集如何进入 baseline。

## 1. `LetterboxInfo`

Palm detector 的输入不是直接把原图拉伸到正方形，也不是简单的 `resize + copyMakeBorder`。legacy `PalmDetectionCpu` 内部使用 `ImageToTensorCalculator`：先构造带 padding 的 full-image ROI，再用连续坐标 `warpPerspective` 直接采样到 `192x192`。

```text
原图 HxW
  -> full-image ROI(center_x, center_y, roi_width, roi_height)
  -> keep_aspect_ratio 扩展 ROI，得到 normalized padding
  -> warpPerspective 采样
  -> RGB float32 NHWC [1, 192, 192, 3], range [0, 1]
```

`LetterboxInfo` 保存两类信息：

| 字段 | 作用 |
| --- | --- |
| `orig_width / orig_height` | 原图尺寸，用于把归一化坐标映射回像素坐标 |
| `resized_width / resized_height` | 等效缩放后的整数尺寸，仅用于调试和兼容说明 |
| `pad_left / pad_top / pad_right / pad_bottom` | 等效整数 padding，仅用于调试和兼容说明 |
| `normalized_padding_values` | MediaPipe 连续 ROI 产生的真实 padding，用于 detector 输出坐标反变换 |

Palm detector 输出坐标处在带 padding 的归一化 tensor 坐标系中。解码时必须使用 `normalized_padding_values` 去掉 padding，再映射回原图像素坐标。这里如果用整数 padding 近似，最终 21 点会出现 1px 级系统误差。

## 2. `Anchor`

MediaPipe palm detector 是 SSD 风格检测器。模型不是直接输出框，而是相对固定 anchor 输出偏移量。

本工程使用的 anchor 参数：

| 参数 | 值 |
| --- | --- |
| input size | `192` |
| num layers | `4` |
| strides | `[8, 16, 16, 16]` |
| min/max scale | `0.1484375 / 0.75` |
| aspect ratios | `[1.0]` |
| interpolated scale aspect ratio | `1.0` |
| fixed anchor size | `true` |
| anchor count | `2016` |

anchor 数量来自：

```text
stride 8:  24 x 24 x 2 = 1152
stride 16: 12 x 12 x 6 = 864
total: 2016
```

## 3. `PalmDetection`

Palm detector 输出：

| 输出 | shape | 含义 |
| --- | --- | --- |
| `Identity` | `[1, 2016, 18]` | 每个 anchor 的 box + 7 个 palm keypoints |
| `Identity_1` | `[1, 2016, 1]` | 每个 anchor 的分类 logit |

`18` 维结构：

```text
[x_center, y_center, width, height,
 kp0_x, kp0_y,
 ...
 kp6_x, kp6_y]
```

解码之后执行：

1. `sigmoid` 得到 score。
2. 过滤低分 anchor。
3. 使用 `LetterboxInfo.normalized_padding` 去掉 ImageToTensor padding。
4. 映射回原图像素坐标。
5. weighted NMS 合并重叠检测。

`data/palm_datasets` 中的 GT 是人工校验过的 palm box 和 7 个 palm keypoints，因此它是 palm detector 精度的主要验收数据。

## 4. `HandRoi`

MediaPipe 不会把 palm box 直接送给 landmark 模型，而是根据 palm box 和 palm keypoints 构造旋转手部 ROI。

`HandRoi` 保存：

| 字段 | 含义 |
| --- | --- |
| `center` | ROI 中心点，像素坐标 |
| `size` | 正方形 ROI 边长，像素 |
| `rotation` | ROI 旋转角，弧度 |
| `matrix` | 原图到 `224x224` crop 的仿射矩阵 |
| `inverse` | `224x224` crop 到原图的逆仿射矩阵 |
| `crop` | landmark 模型输入图像 |

当前 palm-to-hand-rect 参数：

| 参数 | 值 |
| --- | --- |
| rotation start keypoint | `0` |
| rotation end keypoint | `2` |
| target angle | `90 deg` |
| scale | `2.6` |
| shift_y | `-0.5` |
| landmark input | `224x224` |

裁剪使用 `cv2.warpAffine(..., borderMode=cv2.BORDER_REPLICATE)`。边界模式不同会影响手在图像边缘时的 landmark 输入。

## 5. Hand Landmark 输出

Hand landmark 模型输入：

```text
RGB float32 NHWC [1, 224, 224, 3], range [0, 1]
```

输出：

| 输出 | shape | 含义 |
| --- | --- | --- |
| `Identity` | `[1, 63]` | 21 个 ROI 内 landmark，每点 `x, y, z` |
| `Identity_1` | `[1, 1]` | hand presence / hand score |
| `Identity_2` | `[1, 1]` | handedness |
| `Identity_3` | `[1, 63]` | world landmarks |

`Identity` reshape 为 `[21, 3]`。其中 `x, y` 位于 ROI 坐标系，再通过 `HandRoi.inverse` 反投影回原图。

## 6. 人工 21 点 GT 数据

`data/handlm_datasets/annotations.json` 是 COCO 风格关键点标注：

```json
{
  "images": [
    {"id": 1, "file_name": "hand_1_0.jpg", "width": 224, "height": 224}
  ],
  "annotations": [
    {
      "image_id": 1,
      "keypoints": [x0, y0, v0, ..., x20, y20, v20],
      "num_keypoints": 21,
      "bbox": [0, 0, 224, 224],
      "handedness": "Left"
    }
  ]
}
```

关键点顺序与 MediaPipe 21 点一致：

```text
wrist,
thumb_cmc, thumb_mcp, thumb_ip, thumb_tip,
index_mcp, index_pip, index_dip, index_tip,
middle_mcp, middle_pip, middle_dip, middle_tip,
ring_mcp, ring_pip, ring_dip, ring_tip,
pinky_mcp, pinky_pip, pinky_dip, pinky_tip
```

当前 baseline 将它视为人工校正 GT，在 `224x224` crop 坐标系内直接评估 landmark 模型。这个评估不经过 palm detector，因此用于衡量 landmark 模型本身的精度和速度。
