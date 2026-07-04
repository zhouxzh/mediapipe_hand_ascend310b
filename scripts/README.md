# Scripts

`scripts/` 只放可直接运行的 Python 程序，不放 PowerShell 或 shell wrapper。这样后续复制到 Ascend 310B Linux 环境时不需要额外适配入口脚本。

统一 baseline：

```bash
conda activate mediapipe_legacy
python scripts/run_baseline.py --split test --run-matrix
```

当前环境已经满足依赖时：

```bash
python scripts/run_baseline.py
```

常用参数：

```bash
python scripts/run_baseline.py --split valid --run-matrix --output-root runs/baseline_valid
python scripts/run_baseline.py --max-images 300
python scripts/run_baseline.py --save-vis 8
python scripts/run_baseline.py --run-matrix
python scripts/run_baseline.py --data /path/to/palm_datasets --handlm-data /path/to/handlm_datasets --current-reference /path/to/mediapipe_predictions.json
```

单项脚本也可以直接运行，例如：

```bash
python scripts/inspect_tflite.py --model-dir models/tflite --output runs/baseline/model_info.json
python scripts/eval_palm_tflite.py --data ../data/palm_datasets --official-mediapipe references/current_tasks/mediapipe_predictions.json
python scripts/eval_handlm_tflite.py --data ../data/handlm_datasets --output-dir runs/baseline/handlm_manual_gt
python scripts/summarize_baseline.py --output-root runs/baseline
```

模型移植入口：

```bash
# PC / mediapipe_legacy 环境
python scripts/export_onnx.py --group legacy_full

# Ascend 310B，先 source CANN 环境；脚本内部会固定单线程 ATC
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/export_ascend_om.py --group legacy_full
```

palm detector OM 专项分析：

```bash
# PC / mediapipe_legacy 环境：生成真实图片输入的 TFLite reference
python scripts/analyze_palm_om.py make-reference --split test --max-images 200 --output-dir runs/palm_om/legacy_full_palm

# 同步 reference 到 310B 后，在板端 base 环境运行
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/analyze_palm_om.py compare-om --reference-dir runs/palm_om/legacy_full_palm --output-dir runs/palm_om/legacy_full_palm/om_compare
```

WebRTC 实时手部关键点入口：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
python -m pip install --user -r requirements.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python scripts/webrtc_hand_om_app.py \
  --source /dev/video0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --camera-backend opencv \
  --camera-fourcc MJPG \
  --port 8080
```

浏览器打开脚本打印的 `http://<310B-ip>:8080`。详细说明见 `doc/08_webrtc_runtime.md`。

VENC 诊断入口：

```bash
# 只读状态检查，不创建 VENC channel
python scripts/check_venc_runtime.py

# ACLLite 风格 C++ 最小探针，默认只编译到 build/，不创建 VENC channel
python scripts/probe_venc_acllite_cpp.py
```

真实创建 VENC channel 的 `--probe` / `--run` 都需要显式风险确认参数，避免在 CANN 8.0 失败路径上反复触发驱动侧内存压力。

`hand_pipeline/` 是库代码；这些脚本在启动时会自动把项目根目录加入 `sys.path`，因此不需要安装本工程。

