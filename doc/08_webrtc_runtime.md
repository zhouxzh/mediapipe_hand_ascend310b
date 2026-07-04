# WebRTC 实时运行方案

本文记录从 `D:\Github\Ascend310\samples\case8` 移植过来的 WebRTC 方案，以及它在 `mediapipe_hand_ascend310b` 中的手部两阶段 OM 接入方式。

## 1. 移植内容

从 `case8` 直接复用了和传输、采集、硬件编码相关的通用模块：

| 目标路径 | 来源 | 作用 |
| --- | --- | --- |
| `webrtc_app/cann_encoder.py` | `case8/webrtc_app/cann_encoder.py` + 板端 ACLLite VENC sample | 将 aiortc 的 H.264 编码器替换为 CANN VENC；CANN 不可用时直接报错，不自动回退 |
| `webrtc_app/dvpp_jpegd.py` | `case8/webrtc_app/dvpp_jpegd.py` | 使用 DVPP JPEGD 把 MJPEG 摄像头帧解码为 NV12 |
| `webrtc_app/v4l2_capture.py` | `case8/webrtc_app/v4l2_capture.py` | 通过 PyAV 读取 V4L2 MJPEG |
| `webrtc_app/v4l2_raw.py` | `case8/webrtc_app/v4l2_raw.py` | 通过 ioctl + mmap 直接读取 V4L2 MJPEG |

没有复用 `case8` 的 YOLO 检测器和 YOLO 后处理。手部推理入口是本工程自己的：

```text
scripts/webrtc_hand_om_app.py
hand_pipeline/om_runtime.py
hand_pipeline/visualization.py
web/webrtc_index.html
web/webrtc_client.js
web/webrtc_styles.css
```

## 2. 实时链路

板端实时链路如下：

```text
USB camera / V4L2
  -> OpenCV BGR 或 V4L2 MJPEG + DVPP JPEGD
  -> MediaPipe ImageToTensor palm 输入采样
  -> palm detector OM
  -> SSD anchor decode
  -> weighted NMS
  -> palm box + palm7 转 rotated hand ROI
  -> landmark OM
  -> 21 点反投影回原图
  -> OpenCV 叠加 palm/landmark 可视化
  -> NV12
  -> 页面选择 CPU libx264 或 CANN VENC H.264
  -> aiortc WebRTC
  -> 浏览器播放
```

这里的推理链路复用离线验证中的算法实现：`image_to_tensor`、`decode_raw_palm`、`weighted_nms`、`make_hand_roi`、`preprocess_landmark_tflite`、`landmarks_to_original`。因此 WebRTC 展示出来的结果和 `eval_two_stage_om.py` 是同一套几何逻辑。

## 3. 默认模型

默认使用当前最适合作为 full palm 正式部署基线的模型：

```text
models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om
models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om
```

第一个是经过 Pad/Resize/MaxPool 改写并使用 `must_keep_origin_dtype` 编译的优化 palm OM；第二个是 legacy full landmark OM。

## 4. 板端依赖

310B 的 conda `base` 环境需要能 import：

```text
numpy
cv2
av
aiohttp
aiortc
acl
```

CANN 环境必须可用：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which atc
```

WebRTC 运行不调用 ATC，但 CANN VENC、DVPP、ACL OM 推理都依赖 CANN runtime。

当前板端检查结果：

```text
cv2     FOUND
numpy   FOUND
av      FOUND
aiohttp FOUND
acl     FOUND after source CANN
aiortc  MISSING
```

因此首次运行 WebRTC 入口前，需要在 310B 的 `base` 环境安装 `aiortc`。不要直接安装最新版 `aiortc`，因为 `aiortc 1.13` 会要求 `av>=14,<15`，而 310B/aarch64 上可能没有可用 wheel，`pip` 会退回源码编译 PyAV。

本工程的 `requirements.txt` 已经把 WebRTC 版本锁到可复用现有 `av==10.0.0` 的组合：

```bash
python -m pip install --user -r requirements.txt
```

如果板端已经安装了 `numpy`、`opencv-python`、`av==10.0.0`、`aiohttp`，只想补 WebRTC 缺失包，可以使用：

```bash
python -m pip install --user "aiortc==1.5.0" "aioice>=0.9,<1" "pylibsrtp>=0.8,<1"
```

## 5. 启动命令

在 310B 上：

```bash
cd ~/Documents/artemis/mediapipe_hand_ascend310b
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python scripts/webrtc_hand_om_app.py \
  --source /dev/video0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --encoder-mode cpu \
  --camera-backend opencv \
  --camera-fourcc MJPG \
  --port 8080
```

启动后浏览器打开脚本打印的 URL，例如：

```text
http://<310B-ip>:8080
```

如果浏览器或网络环境下 H.264 协商失败，需要换用支持 H.264 的浏览器，或者通过 SSH 端口转发访问。

## 6. DVPP JPEGD 模式

如果摄像头支持 MJPEG，并且希望减少 CPU 解码压力，可以尝试：

```bash
python scripts/webrtc_hand_om_app.py \
  --source /dev/video0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --camera-backend dvpp \
  --camera-fourcc MJPG
```

该模式会优先使用 `V4l2RawCapture` 直接拿 MJPEG bitstream，再交给 `DvppJpegDecoder` 解码为 NV12。若摄像头或驱动不支持 MJPEG，先回退到 `--camera-backend opencv`。

## 7. 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--detector` | 优化 full palm OM | palm detector OM 路径 |
| `--landmark` | legacy full landmark OM | hand landmark OM 路径 |
| `--score-threshold` | `0.5` | palm detector sigmoid 后阈值 |
| `--nms-iou` | `0.3` | weighted NMS IoU 阈值 |
| `--max-hands` | `2` | 每帧最多输出手数 |
| `--min-hand-score` | `0.5` | landmark hand presence 分数阈值 |
| `--infer-every-n` | `1` | 每 N 帧推理一次，其余帧复用上次结果 |
| `--encoder-mode` | `cpu` | `cpu` 使用 libx264；`cann` 使用 CANN VENC H.264 |
| `--cann-venc-retry-seconds` | `300` | CANN VENC 创建失败后的冷却时间，避免页面反复重试造成驱动侧内存压力 |
| `--reload-detector-each-call` | 关闭 | 调试 palm OM 常驻复用漂移时使用 |

Web 页面上也可以选择 detector、landmark、摄像头、分辨率、编码器、阈值和推理间隔。

编码器选择遵循显式策略：

- 选择 `CPU libx264` 时，只使用 CPU 编码。
- 选择 `CANN VENC H.264` 时，服务端会先创建一次 VENC channel 做预检。
- 如果 CANN VENC 不可用，`/offer` 会直接返回错误，页面日志显示失败原因；程序不会自动退回 CPU，避免误判性能。
- 如果一次 CANN VENC 创建失败，服务端会在 `--cann-venc-retry-seconds` 时间内拒绝新的 CANN VENC 请求，避免重复触发 `venc_create_channel()` 失败路径导致 NPU/CMA 内存持续升高。
- CANN 8.0 的 Python 封装按板端 `/opt/opi_test/ACLLite/DVPPLite/src/VencHelper.cpp` 对齐：`venc_send_frame` 第三个参数是保留参数，传 `None`；编码输出只从回调的 `output_stream_desc` 读取。

## 8. CANN 8.0 VENC 诊断

当前 310B 板端 CANN 路径为：

```text
/usr/local/Ascend/ascend-toolkit/8.0.0
/usr/local/Ascend/ascend-toolkit/latest -> 8.0.0
```

板端安装版 Python ACLLite 位于：

```text
/usr/local/Ascend/thirdpart/aarch64/acllite
```

这个目录提供了 ACL resource、DVPP image、camera、VDEC 等 Python 封装，但没有 VENC/VideoWrite Python 封装；VENC 只能参考 C++ DVPPLite 封装。板端可参考的 VENC sample 不在 toolkit 的 `samples` 目录，而在：

```text
/opt/opi_test/ACLLite/DVPPLite/src/VencHelper.cpp
```

该 sample 的关键点：

```text
entype = H264_MAIN_LEVEL
pic_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420
key_frame_interval = 16
rc_mode = 2
max_bit_rate = 10000
aclvencSendFrame(..., nullptr, frame_config, user_data)
```

为排除 Python wrapper 和 WebRTC 的影响，本工程增加了一个 ACLLite 风格的 C++ 最小探针：

```bash
python scripts/probe_venc_acllite_cpp.py
```

默认只生成并编译 `build/venc_acllite_cpp_probe/venc_acllite_probe`，不会创建 VENC channel。需要真正验证官方 C++ 创建路径时，才加风险确认参数：

```bash
python scripts/probe_venc_acllite_cpp.py \
  --run \
  --i-understand-venc-probe-risk \
  --join-usermemory \
  --width 640 \
  --height 480 \
  --fps 30 \
  --bitrate-kbps 10000 \
  --profile main \
  --key-frame-interval 16 \
  --rc-mode 2
```

2026-06-25 板端验证结果：该 C++ 探针已经完全绕过 WebRTC 和 Python VENC wrapper，但仍在 `aclvencCreateChannel()` 返回 `507018`。因此当前 CANN VENC 不可用的边界已经收窄到板端 CANN 8.0 VENC 驱动/受保护内存映射，而不是 WebRTC 页面、H.264 协商、CPU 推流、摄像头读取或 OM 推理。

如果页面选择 `CANN VENC H.264` 后返回：

```text
venc_create_channel failed: 507018 (0x7bc8a)
```

不要只看 ACL 错误码。必须同时看内核日志：

```bash
dmesg | grep -iE 'venc|h264e|h265e|encoder node|rc_' | tail -40
```

当前板端实际看到的关键错误是：

```text
dvpp_prot_mem_map ... iommu_map failed -34
media_prot_mem_malloc ... dvpp_prot_mem_map failed ret:-34
h264e_create_chn ... alloc encoder node buffer failed
Error 0xa008800c: venc user function create chn error
```

其中 `0xa008800c` 对应 VENC 驱动侧 `HI_ERR_VENC_NO_MEM`。这表示错误发生在 `venc_create_channel()` / `aclvencCreateChannel()` 阶段，VENC 驱动没有成功映射受保护 DVPP 内存，也没有成功分配内部编码器节点 buffer。它不是浏览器 H.264 协商问题，也不是 WebRTC 前端卡住；在这种情况下，即使官方风格的最小 640x480 VENC channel 创建也会失败。

本工程提供只读诊断脚本：

```bash
python scripts/check_venc_runtime.py
```

需要显式创建一次 VENC channel 时再加 `--probe` 和风险确认参数：

```bash
python scripts/check_venc_runtime.py --join-usermemory --probe --i-understand-venc-probe-risk --width 640 --height 480 --fps 30 --bitrate-kbps 4000
```

注意：失败的 VENC create 可能让 `npu-smi info` 中的 NPU Memory-Usage 临时升高。当前 CANN 8.0 板端已经观察到 `h264e_create_chn ... alloc encoder node buffer failed` 后，用户态进程退出但驱动侧内存短时间不释放的情况；如果多次失败后内存长期不释放，先重启板子或重载驱动，再继续验证。不要用循环反复 probe。

## 9. 当前边界

- WebRTC 入口是实时展示和板端联调工具，不替代 `eval_two_stage_om.py` 的正式精度报告。
- 当前 Python ACL runner 已经提供常驻输入输出 buffer，适合实时流；如果发现 palm detector 常驻复用仍然漂移，可临时加 `--reload-detector-each-call` 排查。
- H.264 硬件编码失败时不会自动回退 CPU；页面会显示失败原因。需要 CPU 编码时必须显式选择 `CPU libx264`。
- DVPP JPEGD 依赖摄像头 MJPEG 输出；普通 YUYV 摄像头建议先用 OpenCV 后端。
- 正式产品化时，建议把 Python WebRTC 链路中的 OM runner 和图像后处理迁移到 C++/ACL，并保留本文的浏览器前端作为调试入口。
