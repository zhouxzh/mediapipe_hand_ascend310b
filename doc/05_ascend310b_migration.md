# Ascend 310B 移植与复现流程

本文记录把 MediaPipe Hand 子工程迁移到 Ascend 310B 的推荐路径，以及本轮 `mediapipe_legacy_0_10_14_palm_detection_full` OM 精度优化后的可复现转换流程。ONNX 数值对齐见 [06_onnx_error_analysis.md](06_onnx_error_analysis.md)，OM 误差定位和算子兼容性记录见 [07_om_error_analysis.md](07_om_error_analysis.md)。

## 1. 当前迁移结论

PC 侧 TFLite 和 ONNX 链路已经对齐：

| 项目 | 当前结果 | 含义 |
| --- | ---: | --- |
| two-stage TFLite vs legacy graph | mean `0.024968 px` | Python 两步法复刻基本对齐 MediaPipe 0.10.14 |
| legacy rect landmark | mean `0.016921 px` | ROI crop + landmark + 反投影链路正确 |
| ONNX vs TFLite raw output | mean_abs `1e-5` 量级 | ONNX 可作为 ATC 输入基线 |
| ONNX two-stage legacy_full | mean `0.0048 px` | ONNX 端到端 test 集通过 |

310B 侧的关键结论：

| OM 模型 | 状态 | raw box mean_abs | raw score mean_abs | AP50 | mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| 原始 full palm OM | 不建议继续使用 | `2.7714` | `0.9354` | 低于 TFLite | 明显下降 |
| 优化 full palm OM | 当前最优 | `0.007474` | `0.002589` | `0.984348` | `0.604666` |
| TFLite reference | 参考 | `0` | `0` | `0.984348` | `0.604634` |

当前最优 OM 已经使 AP50 与 TFLite 完全一致，mAP 差异约 `0.000032`。raw box/keypoint 平均相对误差 `0.062652%`，raw score 平均相对误差 `0.025567%`，均低于 `0.1%`。

## 2. 最优模型产物

当前建议用于后续移植的 palm detector 产物是：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
```

它来自普通 ONNX：

```text
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx
```

通过三类等价改写得到：

1. `Pad + Add` 下采样残差改写为 `Slice + Add + Slice + Concat`。
2. `Resize(linear, half_pixel, scale=2)` 改写为显式 `Slice + Mul + Add + Concat` 双线性插值。
3. 固定 `2x2/stride2/pad0` 的 `MaxPool` 改写为四个步长 `Slice` 后接 `Max`，从而避开 `MaxPoolV3` 在 `must_keep_origin_dtype` 下不支持 FP32 输入的问题。

这些改写只改变图的算子表达，不改变 ONNX 数学结果。20 张 smoke 验证中，普通 ONNX 与最终优化 ONNX 的最终输出：

```text
all_outputs_abs_mean = 3.8517878e-06
all_outputs_abs_max  = 6.1035156e-05
```

200 张 reference 验证中，MaxPool 改写前后 ONNX：

```text
all_outputs_abs_mean = 3.8343796e-06
all_outputs_abs_max  = 9.0122223e-05
```

这些差异属于浮点运算重排量级。

## 3. 为什么不能直接 ONNX 转 OM

最容易想到的流程是：

```text
TFLite -> ONNX -> ATC -> OM
```

这个流程在形式上是通的，但对 `mediapipe_legacy_0_10_14_palm_detection_full` 不够准确。这里要先明确一个原则：本工程做模型转换的目标不是“把模型格式转成功”，而是让板端 OM 的 raw tensor 尽量等于 MediaPipe 0.10.14 使用的 TFLite raw tensor。

所以每一步转换都必须回答两个问题：

1. 这一层的输出是否还等价于 TFLite reference？
2. 如果不等价，问题是模型数学语义变了，还是后端执行某个算子模式时数值不一致？

根据这个原则，直接 `ONNX -> OM` 被排除。原因不是 TFLite 转 ONNX 出错，而是 CANN/ATC 在 310B 上执行某些 ONNX 算子模式时，结果和 ONNX Runtime/TFLite 不一致。

这个判断来自三层对比：

| 层级 | 观察 | 结论 |
| --- | --- | --- |
| TFLite vs ONNX | raw output mean_abs 在 `1e-5` 量级 | ONNX 本身可以作为正确参考 |
| 原始 ONNX -> 原始 OM | raw boxes mean_abs `2.7714`，raw score mean_abs `0.9354` | 直接 ATC 后数值严重偏移 |
| 改写 ONNX -> 优化 OM | raw box 相对误差 `0.062652%`，raw score 相对误差 `0.025567%` | 问题来自特定算子表达和精度路径 |

因此，本工程没有采用“原始 ONNX 直接 ATC”的方案，而是先做数学等价的 ONNX 图改写，再交给 ATC 编译。

换句话说，这里的改写不是为了改变模型，而是为了把 CANN 不稳定或不兼容的高层算子表达，换成 310B 更容易稳定执行的基础算子表达。判断改写是否成立，只看一件事：改写后的 ONNX 必须继续对齐改写前 ONNX/TFLite。

## 4. 三类改写分别解决什么问题

### 4.1 Pad + Add 改写：解决残差分支污染

palm detector 的 backbone 有几处通道数翻倍的下采样残差结构。原始 ONNX 用 `Pad` 给 shortcut 分支补零，再和主分支相加：

```text
shortcut = Pad(pool, channel_tail_zeros)
out = Add(shortcut, conv_branch)
```

这个表达在 ONNX Runtime 中是正确的，但在 310B OM 中，`Pad` 后补出来的 channel tail 曾出现不稳定，导致后续残差输出被污染。直接后果是中间层误差从 `1e-4` 级别跳到 `0.1` 级别。

改写后的结构是：

```text
first = Add(pool, Slice(conv_branch, channels 0:C))
tail  = Slice(conv_branch, channels C:2C)
out   = Concat(first, tail, axis=1)
```

这和 `Add(Pad(pool), conv_branch)` 数学等价，但不再依赖 channel 维度补零的 `Pad`。它把“补零再相加”变成了“前半通道相加，后半通道直接保留主分支”，语义更明确，也更适合 ATC。

### 4.2 Resize 改写：解决 half-pixel 双线性上采样不一致

FPN 中有两个 2 倍双线性上采样：

```text
Resize(mode=linear, coordinate_transformation_mode=half_pixel, scales=[1,1,2,2])
```

ONNX Runtime 和 TFLite 对这个 half-pixel 坐标规则是一致的，但 310B OM 对该 `Resize` 模式的输出出现明显偏差，第一处上采样后 mean_abs 曾达到 `0.912137`。

改写方法是把固定 2 倍 half-pixel 双线性插值展开成显式算术：

```text
Slice rows/cols
Mul by 0.75 or 0.25
Add blended neighbors
Concat back to 2x height/width
```

这样做的目的不是改变插值算法，而是把一个高层 `Resize` 算子拆成 CANN 更稳定的基础算子组合。改写后的 ONNX 与原始 ONNX 最终输出 mean_abs 仍是 `1e-6` 量级，说明数学语义没有变。

### 4.3 MaxPool 改写：让高精度编译模式可用

前两步改写后，普通 `force_fp16` OM 已经基本可用，但 raw box/keypoint 的平均相对误差仍约 `0.158%`，没有达到 `<0.1%`。继续降低误差需要尝试 `must_keep_origin_dtype`，让更多计算保持原始 FP32 精度。

问题是原始图里的 `MaxPoolV3` 在当前 310B CANN 环境下不支持 FP32 输入：

```text
MaxPoolV3: data type DT_FLOAT of input x is not supported.
supported: DT_FLOAT16
```

由于 palm detector 里的 MaxPool 都是固定的 `2x2/stride2/pad0`，可以用四个步长切片加逐元素最大值代替：

```text
s00 = Slice(x, h=0::2, w=0::2)
s01 = Slice(x, h=0::2, w=1::2)
s10 = Slice(x, h=1::2, w=0::2)
s11 = Slice(x, h=1::2, w=1::2)
out = Max(s00, s01, s10, s11)
```

这一步的核心价值是让 `must_keep_origin_dtype` 编译通过。最终 raw box 相对误差从约 `0.158%` 降到 `0.062652%`，raw score 相对误差降到 `0.025567%`。

## 5. 最终转换策略

最终采用的是：

```text
原始 TFLite
  -> tf2onnx 导出普通 ONNX
  -> Pad + Add 等价改写
  -> Resize half-pixel 等价改写
  -> MaxPool 等价改写
  -> ATC must_keep_origin_dtype 单线程编译
  -> OM 与同批 TFLite reference 对齐验证
```

这个顺序不能随便调整，原因如下：

| 步骤 | 为什么放在这里 |
| --- | --- |
| `tf2onnx` 导出普通 ONNX | 先得到一个可被 ONNX Runtime 验证的中间表示；如果这一步已经不对，后面所有 OM 分析都没有意义 |
| Pad + Add 改写 | 它解决最早出现的大幅中间层偏移；不先处理这个问题，后面的 Resize/MaxPool 误差会叠在一起，难以定位 |
| Resize 改写 | FPN 上采样会影响最终检测头特征；它属于主要数值偏移来源之一，必须在进入高精度编译前固定 |
| MaxPool 改写 | 它主要不是为了降低 ONNX 误差，而是为了让 `must_keep_origin_dtype` 能在 310B 上编译通过 |
| `must_keep_origin_dtype` 编译 | 前面三步把不兼容/不稳定算子模式替换掉之后，才有条件让更多计算保持 FP32 精度 |
| OM vs TFLite reference 验证 | 最终验收必须回到同一批 TFLite raw 输出，不能只看模型能否执行 |

这个策略有几个边界：

| 问题 | 当前选择 |
| --- | --- |
| 为什么不改 TFLite 模型 | TFLite 是 MediaPipe 0.10.14 reference，不能动 |
| 为什么不手改后处理抵消误差 | 原始 OM raw tensor 已经严重偏移，后处理补偿不可靠 |
| 为什么不直接 `force_fp32` | 当前 ATC 编译失败，`PluginManager InvokeAll failed` |
| 为什么不用普通 `force_fp16` | 端到端指标接近，但 raw box 相对误差仍高于 `<0.1%` |
| 为什么选择 `must_keep_origin_dtype` | MaxPool 改写后可编译，并且 raw tensor 平均相对误差低于 `<0.1%` |

所以，这套转换不是为了“让模型能跑”，而是为了让 OM 的 raw 输出尽可能接近 TFLite reference，后续 C++/ACL 部署时才有稳定的数值基线。

可以把这套方案理解成一个逐层收敛过程：

```text
普通 ONNX 已经正确
  -> 原始 OM 不正确
  -> 定位到 Pad/Resize/MaxPool 相关算子模式
  -> 保持 ONNX 数学等价，只替换算子表达
  -> 让 ATC 走更稳定的执行路径
  -> 用同一批 TFLite reference 验收 raw tensor 和 decode/NMS 结果
```

## 6. 一键复现脚本

正式复现入口：

```text
scripts/build_optimized_palm_om.py
```

该脚本串联以下步骤：

```text
普通 full palm ONNX
  -> rewrite_palm_downsample_residual.py
  -> rewrite_palm_bilinear_resize.py
  -> rewrite_palm_maxpool_slices.py
  -> 优化 ONNX
  -> 可选 ONNX 等价验证
  -> 可选板端 ATC 编译 OM
```

底层改写脚本仍然保留，便于单独调试：

```text
scripts/rewrite_palm_downsample_residual.py
scripts/rewrite_palm_bilinear_resize.py
scripts/rewrite_palm_maxpool_slices.py
```

### 6.1 PC 侧生成优化 ONNX

在本地 `mediapipe_legacy` 环境中运行：

```bash
cd D:/Piano/artemis/mediapipe_hand_ascend310b
conda activate mediapipe_legacy

python scripts/build_optimized_palm_om.py \
  --verify-onnx \
  --max-images 200 \
  --verify-dir runs/palm_om/legacy_full_palm/onnx_optimized_compare
```

输出文件：

```text
runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_split.onnx
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx
models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.build.json
runs/palm_om/legacy_full_palm/onnx_optimized_compare/summary.json
```

## 7. 同步到 310B

同步工程到开发板：

```bash
rsync -av --delete \
  D:/Piano/artemis/mediapipe_hand_ascend310b/ \
  310:~/Documents/artemis/mediapipe_hand_ascend310b/
```

如果在 Windows PowerShell 中操作，也可以通过 WSL 调用 `rsync`。不要用多线程 ATC 编译；310B 开发板容易被打满。

## 8. 板端编译 OM

在 310B 上进入项目：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/Documents/artemis/mediapipe_hand_ascend310b
```

推荐用 Python 复现入口编译：

```bash
python scripts/build_optimized_palm_om.py \
  --skip-rewrite \
  --compile-om \
  --optimized-onnx models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx \
  --om-output models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype \
  --atc-log runs/palm_om/atc_logs/downsample_resize_maxpool_slices_origin_dtype.log \
  --precision-mode must_keep_origin_dtype
```

脚本会使用单核低优先级方式调用 ATC：

```text
taskset -c 0 nice -n 19 atc ...
```

也可以使用已经沉淀的 shell 脚本：

```bash
runs/palm_om/compile_downsample_resize_maxpool_slices_origin_dtype.sh
```

这些 shell 脚本仅用于板端调试和复现，正式工程入口以 Python 脚本为准。

## 9. 板端验证命令

生成或同步 TFLite reference：

```bash
# PC / mediapipe_legacy 环境
python scripts/analyze_palm_om.py make-reference \
  --split test \
  --max-images 200 \
  --output-dir runs/palm_om/legacy_full_palm
```

同步到板端后验证优化 OM：

```bash
python scripts/analyze_palm_om.py compare-om \
  --model models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om \
  --reference-dir runs/palm_om/legacy_full_palm \
  --output-dir runs/palm_om/legacy_full_palm/downsample_resize_maxpool_slices_origin_dtype_om_compare \
  --max-images 200
```

当前 200 张 reference 的验收结果：

| 指标 | TFLite | 优化 OM |
| --- | ---: | ---: |
| predictions | `201` | `201` |
| AP50 | `0.984348` | `0.984348` |
| mAP | `0.604634` | `0.604666` |
| NMS matched | - | `201/201` |
| NMS unmatched | - | `0` |

raw 输出误差：

| 输出 | mean_abs | ref_abs_mean | relative_mean | p95_abs | max_abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw boxes/keypoints | `0.007474` | `11.929138` | `0.062652%` | `0.023107` | `0.376629` |
| raw score logits | `0.002589` | `10.126442` | `0.025567%` | `0.007274` | `0.040539` |

decode 后几何误差：

| 项目 | mean |
| --- | ---: |
| positive anchor center | `0.013455 px` |
| positive anchor palm7 | `0.027383 px` |
| NMS match IoU | `0.999597` |
| NMS match center | `0.014242 px` |
| NMS match palm7 mean | `0.024093 px` |

## 10. 推荐移植顺序

1. 先在 PC 侧确认 `run_baseline.py --split test --run-matrix` 通过。
2. 固定 `legacy_full` 的 TFLite reference。
3. 导出 ONNX，并确认 ONNX 与 TFLite raw-output 在 `1e-4` 量级内。
4. 对 full palm ONNX 执行本文的三类算子改写。
5. 在 PC 侧验证优化 ONNX 与普通 ONNX 等价。
6. 在 310B 上单线程 ATC 编译优化 ONNX。
7. 用同一批 TFLite reference 对比 OM raw boxes、raw scores、decode、NMS。
8. 再接入 landmark OM 和完整两阶段链路。
9. C++/ACL 部署时重新验证模型常驻复用是否稳定。

## 11. 板端必须保留的数据结构

```c
typedef struct {
  int input_size;
  int orig_width;
  int orig_height;
  int resized_width;
  int resized_height;
  int pad_left;
  int pad_top;
  int pad_right;
  int pad_bottom;
  float padding_left;
  float padding_top;
  float padding_right;
  float padding_bottom;
} LetterboxInfo;

typedef struct {
  float x_center;
  float y_center;
  float width;
  float height;
} Anchor;

typedef struct {
  float score;
  float box[4];
  float palm7[7][2];
} PalmDetection;

typedef struct {
  float center[2];
  float size;
  float rotation;
  float affine[2][3];
  float inverse[2][3];
} HandRoi;
```

这些结构必须先与 Python reference 对齐，再做性能优化。

## 12. 注意事项

- ATC 必须单线程低优先级运行：`taskset -c 0 nice -n 19 atc ...`。
- CANN 可能打印 `/sys/fs/cgroup/memory/usermemory/tasks: Permission denied`，当前观察为非致命提示。
- 当前最优编译模式是 `must_keep_origin_dtype`，前提是使用 MaxPool 改写后的 ONNX。
- 未改写 MaxPool 时，`must_keep_origin_dtype` 会因为 `MaxPoolV3` 不支持 FP32 输入而失败；`force_fp32` / `cube_fp16in_fp32out` 仍不可作为正式方案，具体错误见 07 文档。
- 当前 Python ACL runner 曾观察到 palm detector 模型常驻复用漂移。正式 C++ ACL 部署时必须单独验证常驻模型、多次执行、输出 buffer 拷贝和 context 生命周期。






