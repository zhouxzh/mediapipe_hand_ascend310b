# Pipeline 与 MediaPipe Graph

本文件解释 MediaPipe Hand 的完整 pipeline、legacy MediaPipe graph 的关键节点，以及本工程如何把验证拆成可定位误差的层级。

## 1. 复刻对象

MediaPipe Hand 不是单个端到端模型，而是一条两阶段链路：

```text
原图 BGR/RGB
  -> ImageToTensor 预处理算子
       - full-image ROI
       - keep_aspect_ratio padding
       - warpPerspective 采样到 192x192
  -> palm detector TFLite
  -> SSD anchor decode
  -> score sigmoid
  -> remove normalized padding
  -> weighted NMS
  -> palm box + 7 个 palm keypoints
  -> palm detection to hand rect
  -> rotated ROI crop, 224x224
  -> hand landmark TFLite
  -> 21 个 landmark + hand score + handedness + world landmarks
  -> landmark 反投影回原图
```

迁移到 Ascend 310B 时，真正要移植的是“模型 + 几何后处理 + 验证方法”三件事。只转换 TFLite 模型是不够的。

## 2. 代码对应关系

| 模块 | 作用 |
| --- | --- |
| `hand_pipeline/preprocess.py` | 复刻 `ImageToTensorCalculator` 预处理算子，用连续 ROI `warpPerspective` 生成 detector 输入 tensor，并记录 normalized padding |
| `hand_pipeline/decode.py` | 生成 2016 个 SSD anchor，解码 raw palm 输出，执行 weighted NMS |
| `hand_pipeline/roi.py` | 从 palm detection 生成 hand ROI，完成旋转 crop 和 landmark 反投影 |
| `hand_pipeline/inference.py` | 封装 LiteRT/TFLite interpreter |
| `hand_pipeline/eval.py` | palm GT 加载、IoU、AP、precision/recall 计算 |
| `scripts/*.py` | 可直接运行的验证和分析程序 |

`hand_pipeline/` 是可复用核心库。`scripts/` 是工程工具入口，不再放在库包内部，便于后续移植到 310B 或接入 `webrtc/`。

## 3. Legacy Graph 的关键节点

legacy `mediapipe==0.10.14` 中仍可以运行：

```text
mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb
```

简化后的 graph 主链路：

```text
IMAGE:image
  -> PalmDetectionCpu
  -> PalmDetectionDetectionToRoi
  -> AssociationNormRectCalculator
  -> HandLandmarkCpu
  -> HandLandmarkLandmarksToRoi
  -> PreviousLoopbackCalculator
```

关键节点对应关系：

| MediaPipe 节点 | 作用 | 本工程对应实现 |
| --- | --- | --- |
| `PalmDetectionCpu` | 整图 palm detector，包括 image-to-tensor、TFLite、decode、NMS | `preprocess.py`、`decode.py`、`scripts/eval_palm_tflite.py` |
| `ImageToTensorCalculator` | `PalmDetectionCpu` 子图内部预处理算子：把整图按连续 ROI 采样成 `192x192` detector tensor，同时产生 padding 信息 | `preprocess.image_to_tensor()` |
| `PalmDetectionDetectionToRoi` | 根据 palm box 和 palm keypoints 生成旋转 hand rect | `roi.make_hand_roi()` |
| `RectTransformationCalculator` | 对 rect 做 shift、scale、square_long 等变换 | `roi.make_hand_roi()` |
| `HandLandmarkCpu` | 在单手 ROI 内运行 landmark 模型 | `scripts/eval_two_stage_tflite.py` |
| `HandLandmarkLandmarksToRoi` | 根据上一帧 landmark 生成 tracking ROI | 当前静态图验证关闭 tracking |

## 4. 当前验证分层

官方 Tasks API 通常只暴露最终 21 点、handedness 和部分 box 信息，不暴露 `palm_detections`、`hand_rects_from_palm_detections`、`letterbox_padding` 等中间流。因此本工程把 baseline 拆成多个层级：

| 输出目录 | 验证目标 | 最新结果摘要 |
| --- | --- | --- |
| `palm_detector/` | 人工校验 palm box/7 点上的 detector 精度 | precision `0.967102`，recall `0.972361` |
| `handlm_manual_gt/` | 人工校正 21 点 GT 上的 landmark 模型精度 | full mean `5.940073 px`，lite mean `6.602299 px` |
| `legacy_graph/` | 运行 legacy graph 并导出中间 rect | legacy graph 可作为 calculator 参考 |
| `legacy_rect_landmark/` | 使用 legacy 官方 rect 验证 landmark 子链路 | mean `0.016921 px`，接近 0 |
| `two_stage_vs_legacy_graph/` | 验证完整 palm-to-rect-to-landmark 复刻链路 | mean `0.024968 px` |
| `two_stage_vs_current_tasks/` | 对齐当前 Tasks 参考输出 | mean `3.630226 px` |

这套分层有一个重要判断：

```text
legacy_rect_landmark 接近 0
  -> landmark TFLite + ROI crop + 反投影基本正确
two_stage_vs_legacy_graph 也接近 0
  -> detector 输入采样、palm decode、NMS、hand rect 和 projection 已经基本对齐 legacy graph
```

其中 `ImageToTensorCalculator` 必须单独作为一个算子看待。旧版 MediaPipe CPU graph 不是简单地先 `resize` 再 `copyMakeBorder`，而是先构造带 padding 的 full-image ROI，再通过 `warpPerspective` 直接采样到 detector 输入 tensor。这个差异会改变 palm detector 的 raw output，进而影响 palm box、7 个 palm keypoints、hand rect 和最终 21 点。

## 5. 面向 310B 的拆分原则

310B 移植时建议保持同样的层级边界：

```text
ImageToTensor preprocess
detector OM
decode + NMS
hand rect
ROI crop
landmark OM
projection
```

每层都保存中间数据，与 PC Python 参考链路逐项比较。第一层必须保存 detector input tensor 和 normalized padding；如果这一层和 PC 参考不一致，后面的 palm decode 和 ROI 即使公式正确，也会出现像素级误差。这样比直接看最终 21 点更容易定位误差，也更适合后续接入视频流、WebRTC 和板端推理调度。
