# Ascend 310B Portable Hand Pipeline

这个目录就是可直接复制到昇腾 310B 开发板上的移植包。它把核心代码、模型目录、转换脚本和运行脚本都收在同一个文件夹内，不依赖本仓库其它历史目录。

## 目录结构

```text
ascend310b_portable/
  hand_pipeline/               # 核心代码：预处理、decode、NMS、ROI、landmark 投影、tracking
  hand_pipeline/runtimes/
    ascend.py                  # 310B OM/ais_bench 适配层
    onnx.py                    # PC 验证用 ONNX Runtime 适配层
  models/
    onnx/                      # 已带 ONNX，供转换 OM 或 PC 验证
    ascend/                    # 放转换后的 .om 模型
  scripts/
    convert_onnx_to_om.sh      # ONNX -> OM 转换
    run_image_ascend.py        # 310B 单图推理
    run_video_ascend_tracking.py
    run_image_onnx.py          # PC 单图对照验证
    run_video_onnx_tracking.py
  data/smoke_images/           # 小型测试图片
  runs/                        # 默认输出目录
```

## 板端环境

先确认 CANN/AIT 环境可用：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 - <<'PY'
import aclruntime
from ais_bench.infer.interface import InferSession
print("Ascend runtime OK")
PY
```

Python 依赖只需要：

```bash
python3 -m pip install -r requirements-board.txt
```

如果 `ais_bench` 或 `aclruntime` 没有预装，请用 CANN/AIT 提供的本机架构 wheel 安装。

## 生成 OM

默认转换 full 模型，输入保持 NHWC float32，不使用 AIPP，因为预处理已经在 `hand_pipeline` 中完成。

```bash
cd ascend310b_portable
bash scripts/convert_onnx_to_om.sh full Ascend310B1
```

如果你的板卡 SOC 不是 `Ascend310B1`，把第二个参数改成板端实际值，例如：

```bash
bash scripts/convert_onnx_to_om.sh full Ascend310B4
```

转换完成后应得到：

```text
models/ascend/mediapipe_legacy_0_10_14_palm_detection_full.om
models/ascend/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

## 310B 单图运行

```bash
python3 scripts/run_image_ascend.py \
  data/smoke_images/images/train_palm_ac44e9bd-97a1-4f28-8398-f825842fc59d.jpg \
  --output runs/ascend_image_result.json
```

输出 JSON 里包含：

- `palms`: palm bbox + 7 个 palm keypoint
- `hands`: 21 个 hand landmark、ROI、分数等

## 310B 视频 Tracking

```bash
python3 scripts/run_video_ascend_tracking.py /path/to/video.mp4 \
  --max-frames 100 \
  --output runs/ascend_video_predictions.jsonl
```

## PC 对照验证

在 PC 上可先验证相同 pipeline 的 ONNX 路径：

```bash
python -m pip install -r requirements-pc.txt
python scripts/run_image_onnx.py \
  data/smoke_images/images/train_palm_ac44e9bd-97a1-4f28-8398-f825842fc59d.jpg \
  --output runs/onnx_image_result.json
```

## 模型输入输出契约

Palm detector:

- input: `1x192x192x3`, NHWC, float32, range `[0, 1]`
- outputs: `1x2016x18`, `1x2016x1`

Hand landmark:

- input: `1x224x224x3`, NHWC, float32, range `[0, 1]`
- outputs: `1x63`, `1x1`, `1x1`, `1x63`

如果 ATC/OM 输出顺序和 ONNX 不一致，可在运行脚本里传输出索引重排：

```bash
python3 scripts/run_image_ascend.py image.jpg \
  --detector-output-indices 0,1 \
  --landmark-output-indices 0,1,2,3
```

核心逻辑只依赖 `ModelRunner.__call__(tensor) -> list[np.ndarray]`，后续如果你不用 `ais_bench`，只需要替换 `hand_pipeline/runtimes/ascend.py`，不要改 `pipeline.py`、`tracking.py`、`decode.py`、`roi.py`。

