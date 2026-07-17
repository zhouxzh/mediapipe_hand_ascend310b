const elements = {
  detector: document.querySelector("#detector"),
  landmark: document.querySelector("#landmark"),
  modelInput: document.querySelector("#modelInput"),
  pipelineMode: document.querySelector("#pipelineMode"),
  threadingMode: document.querySelector("#threadingMode"),
  pipelineQueueSize: document.querySelector("#pipelineQueueSize"),
  sourceModeCamera: document.querySelector("#sourceModeCamera"),
  sourceModeVideo: document.querySelector("#sourceModeVideo"),
  cameraSourceField: document.querySelector("#cameraSourceField"),
  videoSourceField: document.querySelector("#videoSourceField"),
  videoSource: document.querySelector("#videoSource"),
  videoSourceMeta: document.querySelector("#videoSourceMeta"),
  source: document.querySelector("#source"),
  resolution: document.querySelector("#resolution"),
  fps: document.querySelector("#fps"),
  bitrateKbps: document.querySelector("#bitrateKbps"),
  cameraBackend: document.querySelector("#cameraBackend"),
  cameraFourcc: document.querySelector("#cameraFourcc"),
  encoderMode: document.querySelector("#encoderMode"),
  scoreThreshold: document.querySelector("#scoreThreshold"),
  nmsIou: document.querySelector("#nmsIou"),
  maxHands: document.querySelector("#maxHands"),
  minHandScore: document.querySelector("#minHandScore"),
  inferEvery: document.querySelector("#inferEvery"),
  start: document.querySelector("#start"),
  stop: document.querySelector("#stop"),
  clearLog: document.querySelector("#clearLog"),
  remoteVideo: document.querySelector("#remoteVideo"),
  videoDimensions: document.querySelector("#videoDimensions"),
  serverStatus: document.querySelector("#serverStatus"),
  runtimeStatus: document.querySelector("#runtimeStatus"),
  pipelineStatus: document.querySelector("#pipelineStatus"),
  peerStatus: document.querySelector("#peerStatus"),
  bitrateStatus: document.querySelector("#bitrateStatus"),
  npuStatus: document.querySelector("#npuStatus"),
  inferStatus: document.querySelector("#inferStatus"),
  handStatus: document.querySelector("#handStatus"),
  trackFpsStatus: document.querySelector("#trackFpsStatus"),
  fpsOverlay: document.querySelector("#fpsOverlay"),
  fpsStatus: document.querySelector("#fpsStatus"),
  codecStatus: document.querySelector("#codecStatus"),
  logOutput: document.querySelector("#logOutput"),
};

let activeConnection = null;
let activeAttemptId = 0;
let pendingOfferController = null;
let statsTimer = null;
let serverStatsTimer = null;
let lastStats = null;
let lastFpsStats = null;
let lastServerErrorText = "";
let startInProgress = false;
let fpsTrackingId = null;
let fpsTimestamps = [];
let controlsBusy = false;
let sourceMode = "camera";
let datasetVideos = [];
let cameraSourceValue = "/dev/video0";
let cameraBackendValue = "dvpp";
let cameraFourccValue = "MJPG";

function log(message) {
  const timestamp = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  elements.logOutput.textContent += `[${timestamp}] ${message}\n`;
  elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
}

function setText(element, text) {
  element.textContent = text;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function parseResolution(value) {
  const [width, height] = value.split("x").map(Number);
  return { width, height };
}

function positiveInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} 必须是正整数。`);
  }
  return parsed;
}

function numericRange(value, name, lower, upper) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < lower || parsed > upper) {
    throw new Error(`${name} 必须在 ${lower} 到 ${upper} 之间。`);
  }
  return parsed;
}

function parseBitrateKbps(value) {
  if (!value || !String(value).trim()) {
    return null;
  }
  return positiveInteger(value, "H.264 码率");
}

function setSelectValue(select, value, label = null) {
  const stringValue = String(value);
  const exists = Array.from(select.options).some((option) => option.value === stringValue);
  if (!exists) {
    const option = document.createElement("option");
    option.value = stringValue;
    option.textContent = label ?? stringValue;
    select.append(option);
  }
  select.value = stringValue;
}

function fillSelect(select, items, fallbackName) {
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.name;
    select.append(option);
  }
  if (!items.length) {
    const option = document.createElement("option");
    option.value = fallbackName;
    option.textContent = fallbackName;
    select.append(option);
  }
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KiB`;
  }
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  }
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

function findDatasetVideo(source) {
  const normalized = String(source ?? "").replaceAll("\\", "/");
  return datasetVideos.find((item) => normalized === item.path || normalized.endsWith(`/${item.path}`));
}

function updateVideoSourceMeta() {
  const selected = datasetVideos.find((item) => item.path === elements.videoSource.value);
  if (!selected) {
    elements.videoSourceMeta.textContent = datasetVideos.length ? `${datasetVideos.length} 个视频` : "未找到视频";
    return;
  }
  elements.videoSourceMeta.textContent = `${datasetVideos.length} 个视频 · ${formatBytes(selected.size_bytes)} · 循环播放`;
}

function updateSourceControls() {
  const videoMode = sourceMode === "video";
  elements.sourceModeCamera.classList.toggle("active", !videoMode);
  elements.sourceModeVideo.classList.toggle("active", videoMode);
  elements.sourceModeCamera.setAttribute("aria-pressed", String(!videoMode));
  elements.sourceModeVideo.setAttribute("aria-pressed", String(videoMode));
  elements.cameraSourceField.hidden = videoMode;
  elements.videoSourceField.hidden = !videoMode;
  elements.videoSourceMeta.hidden = !videoMode;
  elements.sourceModeCamera.disabled = controlsBusy;
  elements.sourceModeVideo.disabled = controlsBusy || !datasetVideos.length;
  elements.source.disabled = controlsBusy || videoMode;
  elements.videoSource.disabled = controlsBusy || !videoMode || !datasetVideos.length;
  elements.cameraBackend.disabled = controlsBusy || videoMode;
  elements.cameraFourcc.disabled = controlsBusy || videoMode;
}

function setSourceMode(mode, announce = false) {
  const nextMode = mode === "video" && datasetVideos.length ? "video" : "camera";
  if (sourceMode === "camera" && nextMode === "video") {
    cameraSourceValue = elements.source.value.trim() || cameraSourceValue;
    cameraBackendValue = elements.cameraBackend.value;
    cameraFourccValue = elements.cameraFourcc.value;
  }
  sourceMode = nextMode;
  if (sourceMode === "video") {
    elements.cameraBackend.value = "opencv";
    elements.cameraFourcc.value = "DEFAULT";
    elements.source.value = elements.videoSource.value;
    updateVideoSourceMeta();
  } else {
    elements.source.value = cameraSourceValue;
    elements.cameraBackend.value = cameraBackendValue;
    elements.cameraFourcc.value = cameraFourccValue;
  }
  updateSourceControls();
  if (announce) {
    log(sourceMode === "video" ? `已选择视频: ${elements.videoSource.options[elements.videoSource.selectedIndex]?.textContent ?? ""}` : "已切换到摄像头输入");
  }
}

async function loadVideos() {
  const data = await fetchJson("/videos");
  datasetVideos = data.videos ?? [];
  elements.videoSource.innerHTML = "";
  for (const item of datasetVideos) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = `${item.name} (${formatBytes(item.size_bytes)})`;
    elements.videoSource.append(option);
  }
  if (!datasetVideos.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "未找到 PianoVAM 视频";
    elements.videoSource.append(option);
  }
  updateVideoSourceMeta();
  updateSourceControls();
}

async function loadModels() {
  const data = await fetchJson("/models");
  fillSelect(elements.detector, data.detectors ?? [], "mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om");
  fillSelect(elements.landmark, data.landmarks ?? [], "mediapipe_legacy_0_10_14_hand_landmark_full.om");
  elements.detector.value = data.default_detector ?? elements.detector.options[0]?.value ?? "";
  elements.landmark.value = data.default_landmark ?? elements.landmark.options[0]?.value ?? "";
}

async function checkHealth() {
  const data = await fetchJson("/health");
  setText(elements.serverStatus, data.status === "ok" ? "在线" : "异常");
  setText(elements.runtimeStatus, `${data.runtime_target ?? "unknown"} / ${data.transport ?? "unknown"}`);
  setText(elements.codecStatus, (data.video_codec ?? "h264").toUpperCase());
  if (data.defaults?.encoder_mode) {
    setSelectValue(elements.encoderMode, data.defaults.encoder_mode);
  }
  if (data.defaults?.pipeline_mode) {
    setSelectValue(elements.pipelineMode, data.defaults.pipeline_mode);
  }
  if (data.defaults?.threading_mode) {
    setSelectValue(elements.threadingMode, data.defaults.threading_mode);
  }
  if (data.defaults?.pipeline_queue_size) {
    setSelectValue(
      elements.pipelineQueueSize,
      data.defaults.pipeline_queue_size,
      Number(data.defaults.pipeline_queue_size) === 1 ? "Latest only" : `${data.defaults.pipeline_queue_size} frames`
    );
  }
  setText(elements.pipelineStatus, `${data.defaults?.camera_backend ?? "opencv"} / ${data.encoder ?? "unknown"}`);
  if (data.default_source) {
    elements.source.value = data.default_source;
  }
  const defaults = data.defaults ?? {};
  if (defaults.width && defaults.height) {
    setSelectValue(elements.resolution, `${defaults.width}x${defaults.height}`, `${defaults.width} x ${defaults.height}`);
  }
  if (defaults.fps) {
    setSelectValue(elements.fps, defaults.fps, `${defaults.fps} fps`);
  }
  if (defaults.infer_every_n) {
    elements.inferEvery.value = defaults.infer_every_n;
  }
  if (defaults.score_threshold) {
    elements.scoreThreshold.value = defaults.score_threshold;
  }
  if (defaults.nms_iou) {
    elements.nmsIou.value = defaults.nms_iou;
  }
  if (defaults.max_hands) {
    elements.maxHands.value = defaults.max_hands;
  }
  if (defaults.min_hand_score) {
    elements.minHandScore.value = defaults.min_hand_score;
  }
  if (defaults.bitrate_kbps) {
    setSelectValue(elements.bitrateKbps, defaults.bitrate_kbps, `${defaults.bitrate_kbps} kbps`);
  }
  if (defaults.camera_backend) {
    setSelectValue(elements.cameraBackend, defaults.camera_backend);
  }
  if (defaults.camera_fourcc) {
    setSelectValue(elements.cameraFourcc, defaults.camera_fourcc);
  }
  cameraBackendValue = elements.cameraBackend.value;
  cameraFourccValue = elements.cameraFourcc.value;
  const defaultVideo = findDatasetVideo(data.default_source);
  if (defaultVideo) {
    cameraSourceValue = "/dev/video0";
    elements.videoSource.value = defaultVideo.path;
    setSourceMode("video");
  } else {
    cameraSourceValue = data.default_source || cameraSourceValue;
    setSourceMode("camera");
  }
}

function setControlsBusy(isBusy) {
  controlsBusy = isBusy;
  elements.start.disabled = isBusy;
  elements.detector.disabled = isBusy;
  elements.landmark.disabled = isBusy;
  elements.pipelineMode.disabled = isBusy;
  elements.threadingMode.disabled = isBusy;
  elements.pipelineQueueSize.disabled = isBusy;
  elements.resolution.disabled = isBusy;
  elements.fps.disabled = isBusy;
  elements.bitrateKbps.disabled = isBusy;
  elements.encoderMode.disabled = isBusy;
  elements.scoreThreshold.disabled = isBusy;
  elements.nmsIou.disabled = isBusy;
  elements.maxHands.disabled = isBusy;
  elements.minHandScore.disabled = isBusy;
  elements.inferEvery.disabled = isBusy;
  updateSourceControls();
}

function resetRemoteVideoSize() {
  elements.videoDimensions.textContent = "-";
}

function updateRemoteVideoSize() {
  const sourceWidth = elements.remoteVideo.videoWidth;
  const sourceHeight = elements.remoteVideo.videoHeight;
  if (!sourceWidth || !sourceHeight) {
    return;
  }
  const displayWidth = Math.round(elements.remoteVideo.getBoundingClientRect().width);
  const displayHeight = Math.round(elements.remoteVideo.getBoundingClientRect().height);
  elements.videoDimensions.textContent = `${sourceWidth}x${sourceHeight} / ${displayWidth}x${displayHeight}`;
}

function setFpsDisplay(fps) {
  const text = `${fps} fps`;
  elements.fpsOverlay.textContent = text;
  elements.fpsStatus.textContent = text;
}

function resetFpsDisplay() {
  elements.fpsOverlay.textContent = "- fps";
  elements.fpsStatus.textContent = "-";
}

function formatNumber(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return number.toFixed(digits);
}

function rvfcCallback() {
  if (fpsTrackingId === null) return;
  const now = performance.now();
  fpsTimestamps.push(now);
  while (fpsTimestamps.length > 0 && fpsTimestamps[0] <= now - 1000) {
    fpsTimestamps.shift();
  }
  setFpsDisplay(fpsTimestamps.length);
  fpsTrackingId = elements.remoteVideo.requestVideoFrameCallback(rvfcCallback);
}

function startFpsTracking() {
  stopFpsTracking();
  fpsTimestamps = [];
  lastFpsStats = null;
  if (typeof elements.remoteVideo.requestVideoFrameCallback === "function") {
    fpsTrackingId = elements.remoteVideo.requestVideoFrameCallback(rvfcCallback);
  }
}

function stopFpsTracking() {
  if (fpsTrackingId !== null) {
    try {
      elements.remoteVideo.cancelVideoFrameCallback(fpsTrackingId);
    } catch (_) {}
    fpsTrackingId = null;
  }
  fpsTimestamps = [];
  resetFpsDisplay();
}

function closeConnection(connection) {
  try {
    connection.getReceivers().forEach((receiver) => {
      if (receiver.track) {
        receiver.track.stop();
      }
    });
    connection.close();
  } catch (error) {
    log(`关闭连接警告: ${error.message}`);
  }
}

function stopStats() {
  if (statsTimer) {
    clearInterval(statsTimer);
    statsTimer = null;
  }
  if (serverStatsTimer) {
    clearInterval(serverStatsTimer);
    serverStatsTimer = null;
  }
  lastStats = null;
  lastFpsStats = null;
  lastServerErrorText = "";
  setText(elements.bitrateStatus, "-");
  setText(elements.npuStatus, "-");
  setText(elements.inferStatus, "-");
  setText(elements.handStatus, "-");
  setText(elements.trackFpsStatus, "-");
}

function cancelPendingOffer() {
  if (pendingOfferController) {
    pendingOfferController.abort();
    pendingOfferController = null;
  }
}

function teardownActiveConnection() {
  cancelPendingOffer();
  stopStats();
  stopFpsTracking();
  if (activeConnection) {
    closeConnection(activeConnection);
    activeConnection = null;
  }
  try {
    elements.remoteVideo.pause();
  } catch (_) {}
  elements.remoteVideo.srcObject = null;
  elements.remoteVideo.removeAttribute("src");
  try {
    elements.remoteVideo.load();
  } catch (_) {}
  resetRemoteVideoSize();
  setText(elements.peerStatus, "未建立");
}

function stopConnection({ logMessage = true } = {}) {
  const hadActiveWork = Boolean(activeConnection || pendingOfferController || startInProgress);
  activeAttemptId += 1;
  startInProgress = false;
  setControlsBusy(false);
  teardownActiveConnection();
  if (logMessage && hadActiveWork) {
    log("连接已关闭。");
  }
}

function isActiveAttempt(connection, attemptId) {
  return activeConnection === connection && activeAttemptId === attemptId;
}

function assertActiveAttempt(connection, attemptId) {
  if (!isActiveAttempt(connection, attemptId)) {
    const error = new Error("Connection attempt was superseded.");
    error.name = "AbortError";
    throw error;
  }
}

async function readInboundStats(connection) {
  if (activeConnection !== connection) {
    return;
  }
  const stats = await connection.getStats();
  for (const report of stats.values()) {
    if (report.type !== "inbound-rtp" || report.kind !== "video") {
      continue;
    }
    if (!lastStats) {
      lastStats = { bytesReceived: report.bytesReceived, timestamp: report.timestamp };
      lastFpsStats = { framesDecoded: report.framesDecoded ?? report.framesReceived ?? 0, timestamp: report.timestamp };
      return;
    }
    const bytesDelta = report.bytesReceived - lastStats.bytesReceived;
    const timeDeltaMs = report.timestamp - lastStats.timestamp;
    if (timeDeltaMs > 0) {
      const bitrateKbps = ((bytesDelta * 8) / timeDeltaMs).toFixed(1);
      setText(elements.bitrateStatus, `${bitrateKbps} kbps`);
    }
    const framesDecoded = report.framesDecoded ?? report.framesReceived ?? 0;
    const framesDelta = framesDecoded - lastFpsStats.framesDecoded;
    const fpsTimeDeltaMs = report.timestamp - lastFpsStats.timestamp;
    if (fpsTimeDeltaMs > 0 && framesDelta > 0) {
      setFpsDisplay(Math.round((framesDelta * 1000) / fpsTimeDeltaMs));
    }
    lastStats = { bytesReceived: report.bytesReceived, timestamp: report.timestamp };
    lastFpsStats = { framesDecoded, timestamp: report.timestamp };
    return;
  }
}

function startStats(connection) {
  stopStats();
  statsTimer = window.setInterval(() => {
    readInboundStats(connection).catch((error) => log(`Stats failed: ${error.message}`));
  }, 1000);
  serverStatsTimer = window.setInterval(() => {
    readServerStats().catch((error) => log(`Server stats failed: ${error.message}`));
  }, 500);
  readServerStats().catch((error) => log(`Server stats failed: ${error.message}`));
}

async function readServerStats() {
  if (!activeConnection) {
    return;
  }
  const data = await fetchJson("/stats");
  if (!data || Object.keys(data).length === 0) {
    return;
  }
  const serverErrors = [
    data.capture_error ? `capture: ${data.capture_error}` : "",
    data.render_error ? `render: ${data.render_error}` : "",
    data.infer_error ? `infer: ${data.infer_error}` : "",
  ].filter(Boolean).join(" / ");
  if (serverErrors) {
    if (serverErrors !== lastServerErrorText) {
      log(`服务端错误: ${serverErrors}`);
      lastServerErrorText = serverErrors;
    }
    setText(elements.pipelineStatus, serverErrors);
  } else {
    lastServerErrorText = "";
  }
  if (Number.isFinite(Number(data.npu_latency_ms))) {
    setText(elements.npuStatus, `${Number(data.npu_latency_ms).toFixed(1)} ms`);
  }
  if (Number.isFinite(Number(data.track_fps))) {
    setText(elements.trackFpsStatus, `${Number(data.track_fps).toFixed(1)} fps`);
  }
  const npuMs = formatNumber(data.npu_latency_ms);
  const totalMs = formatNumber(data.infer_total_ms);
  const inferFps = formatNumber(data.infer_fps);
  const predictionAgeMs = formatNumber(data.prediction_age_ms);
  const detPreMs = formatNumber(data.det_pre_ms);
  const detNpuMs = formatNumber(data.det_npu_ms);
  const detPostMs = formatNumber(data.det_post_ms);
  const roiMs = formatNumber(data.roi_ms);
  const cropMs = formatNumber(data.crop_ms);
  const landmarkNpuMs = formatNumber(data.landmark_npu_ms);
  const landmarkPostMs = formatNumber(data.landmark_post_ms);
  setText(
    elements.inferStatus,
    `1/${data.infer_every_n ?? "?"} / NPU ${npuMs} ms / total ${totalMs} ms / ${inferFps} fps / age ${predictionAgeMs} ms / det ${detPreMs}+${detNpuMs}+${detPostMs} / roi ${roiMs}+${cropMs} / lm ${landmarkNpuMs}+${landmarkPostMs}`
  );
  setText(elements.handStatus, `${data.hands ?? 0}`);
  const captureFps = formatNumber(data.capture_fps);
  const captureMs = formatNumber(data.capture_ms);
  const pipelineMs = formatNumber(data.pipeline_ms);
  const nv12Ms = formatNumber(data.nv12_ms);
  const latestAgeMs = formatNumber(data.latest_frame_age_ms);
  const droppedFrames = Number(data.dropped_frames ?? 0);
  const fourcc = data.actual_fourcc || data.camera_fourcc || "?";
  if (!serverErrors) {
    setText(
      elements.pipelineStatus,
      `${data.pipeline_mode ?? elements.pipelineMode.value} ${data.threading_mode ?? elements.threadingMode.value} q${data.pipeline_queue_size ?? elements.pipelineQueueSize.value} drop ${droppedFrames} / ${data.camera_backend ?? "?"}/${fourcc} / cap ${captureFps} fps/${captureMs} ms age ${latestAgeMs} ms / nv12 ${nv12Ms} ms / frame ${pipelineMs} ms / ${data.encoder ?? elements.encoderMode.value}`
    );
  }
}

function bindConnectionEvents(connection, attemptId) {
  connection.ontrack = (event) => {
    if (!isActiveAttempt(connection, attemptId)) {
      return;
    }
    elements.remoteVideo.srcObject = event.streams[0];
    startFpsTracking();
    log(`收到远端视频轨: ${event.track.id}`);
  };
  connection.onconnectionstatechange = () => {
    if (!isActiveAttempt(connection, attemptId)) {
      return;
    }
    setText(elements.peerStatus, connection.connectionState);
    log(`PeerConnection: ${connection.connectionState}`);
  };
  connection.oniceconnectionstatechange = () => {
    if (!isActiveAttempt(connection, attemptId)) {
      return;
    }
    log(`ICE: ${connection.iceConnectionState}`);
  };
}

function bindVideoEvents() {
  elements.remoteVideo.addEventListener("loadedmetadata", () => {
    updateRemoteVideoSize();
    log(`视频尺寸: ${elements.remoteVideo.videoWidth}x${elements.remoteVideo.videoHeight}`);
  });
  elements.remoteVideo.addEventListener("resize", updateRemoteVideoSize);
  elements.remoteVideo.addEventListener("error", () => {
    const error = elements.remoteVideo.error;
    log(`Video error: code=${error?.code ?? "unknown"} message=${error?.message ?? ""}`);
  });
}

function getReceiverCodecs(mimeType) {
  const capabilities = RTCRtpReceiver.getCapabilities?.("video");
  return (capabilities?.codecs ?? []).filter((codec) => codec.mimeType.toLowerCase() === mimeType.toLowerCase());
}

function applyH264Preference(transceiver) {
  const h264Codecs = getReceiverCodecs("video/H264");
  if (!h264Codecs.length) {
    throw new Error("当前浏览器没有 video/H264 WebRTC 接收能力。");
  }
  transceiver.setCodecPreferences(h264Codecs);
}

function logAppliedSourceSettings(sourceSettings) {
  if (!sourceSettings?.applied) {
    return;
  }
  const applied = sourceSettings.applied;
  const bitrate = applied.bitrate_kbps ? `${applied.bitrate_kbps} kbps` : "auto";
  const threading = applied.threading_mode ?? sourceSettings.threading_mode ?? "pipeline";
  const queueSize = applied.pipeline_queue_size ?? sourceSettings.pipeline_queue_size ?? 1;
  log(
    `服务端 detector=${sourceSettings.detector ?? "unknown"} landmark=${sourceSettings.landmark ?? "unknown"} source=${sourceSettings.source ?? "unknown"} mode=${applied.pipeline_mode ?? sourceSettings.pipeline_mode ?? "tracking"} threading=${threading} queue=${queueSize} capture=${applied.width ?? "?"}x${applied.height ?? "?"}@${applied.fps ?? "?"} bitrate=${bitrate} backend=${applied.camera_backend ?? "?"} infer=1/${applied.infer_every_n ?? "?"} fourcc=${applied.actual_fourcc || applied.camera_fourcc || "?"}`
  );
  setText(elements.pipelineStatus, `${applied.pipeline_mode ?? sourceSettings.pipeline_mode ?? "tracking"} ${threading} q${queueSize} / ${applied.camera_backend ?? "?"}/${applied.actual_fourcc || applied.camera_fourcc || "?"} / ${sourceSettings.encoder ?? elements.encoderMode.value}`);
}

async function startConnection() {
  if (startInProgress) {
    log("连接正在建立。");
    return;
  }

  startInProgress = true;
  setControlsBusy(true);
  const attemptId = activeAttemptId + 1;
  activeAttemptId = attemptId;
  teardownActiveConnection();

  const { width, height } = parseResolution(elements.resolution.value);
  const fps = positiveInteger(elements.fps.value, "采集帧率");
  const bitrateKbps = parseBitrateKbps(elements.bitrateKbps.value);
  const inferEvery = positiveInteger(elements.inferEvery.value, "推理间隔");
  const maxHands = positiveInteger(elements.maxHands.value, "最大手数");
  const scoreThreshold = numericRange(elements.scoreThreshold.value, "Palm 阈值", 0.01, 0.99);
  const nmsIou = numericRange(elements.nmsIou.value, "NMS IoU", 0.01, 0.99);
  const minHandScore = numericRange(elements.minHandScore.value, "Landmark 阈值", 0, 1);
  const selectedSource = sourceMode === "video" ? elements.videoSource.value : elements.source.value.trim();
  if (!selectedSource) {
    startInProgress = false;
    setControlsBusy(false);
    throw new Error("请选择输入来源。");
  }

  let connection = null;
  try {
    connection = new RTCPeerConnection();
    activeConnection = connection;
    const transceiver = connection.addTransceiver("video", { direction: "recvonly" });
    applyH264Preference(transceiver);
    bindConnectionEvents(connection, attemptId);

    const offer = await connection.createOffer();
    assertActiveAttempt(connection, attemptId);
    await connection.setLocalDescription(offer);
    assertActiveAttempt(connection, attemptId);

    pendingOfferController = new AbortController();
    const answer = await fetchJson("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: pendingOfferController.signal,
      body: JSON.stringify({
        sdp: connection.localDescription.sdp,
        type: connection.localDescription.type,
        detector: elements.detector.value,
        landmark: elements.landmark.value,
        source: selectedSource,
        width,
        height,
        fps,
        bitrate_kbps: bitrateKbps,
        encoder_mode: elements.encoderMode.value,
        camera_backend: sourceMode === "video" ? "opencv" : elements.cameraBackend.value,
        camera_fourcc: sourceMode === "video" ? "DEFAULT" : elements.cameraFourcc.value,
        pipeline_mode: elements.pipelineMode.value,
        threading_mode: elements.threadingMode.value,
        pipeline_queue_size: positiveInteger(elements.pipelineQueueSize.value, "Pipeline queue"),
        infer_every_n: inferEvery,
        score_threshold: scoreThreshold,
        nms_iou: nmsIou,
        max_hands: maxHands,
        min_hand_score: minHandScore,
      }),
    });
    pendingOfferController = null;
    assertActiveAttempt(connection, attemptId);

    await connection.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    assertActiveAttempt(connection, attemptId);

    startStats(connection);
    log(`WebRTC offer 已发送: ${width}x${height}@${fps} H.264 bitrate=${bitrateKbps ?? "auto"}`);
    logAppliedSourceSettings(answer.source_settings);
  } catch (error) {
    pendingOfferController = null;
    if (error.name === "AbortError") {
      if (connection) closeConnection(connection);
      return;
    }
    const message = error.message || String(error);
    if (connection && isActiveAttempt(connection, attemptId)) {
      teardownActiveConnection();
      setText(elements.peerStatus, "启动失败");
      setText(elements.pipelineStatus, message);
      log(`启动失败: ${message}`);
    } else if (connection) {
      closeConnection(connection);
      setText(elements.peerStatus, "启动失败");
      setText(elements.pipelineStatus, message);
      log(`启动失败: ${message}`);
    } else {
      teardownActiveConnection();
      setText(elements.peerStatus, "启动失败");
      setText(elements.pipelineStatus, message);
      log(`启动失败: ${message}`);
    }
    startInProgress = false;
    setControlsBusy(false);
  } finally {
    if (activeAttemptId === attemptId) {
      startInProgress = false;
      setControlsBusy(false);
    }
  }
}

function bindEvents() {
  elements.source.addEventListener("input", () => {
    if (sourceMode === "camera") {
      cameraSourceValue = elements.source.value;
    }
  });
  elements.cameraBackend.addEventListener("change", () => {
    if (sourceMode === "camera") {
      cameraBackendValue = elements.cameraBackend.value;
    }
  });
  elements.cameraFourcc.addEventListener("change", () => {
    if (sourceMode === "camera") {
      cameraFourccValue = elements.cameraFourcc.value;
    }
  });
  elements.sourceModeCamera.addEventListener("click", () => setSourceMode("camera", true));
  elements.sourceModeVideo.addEventListener("click", () => setSourceMode("video", true));
  elements.videoSource.addEventListener("change", () => {
    elements.source.value = elements.videoSource.value;
    updateVideoSourceMeta();
    log(`已选择视频: ${elements.videoSource.options[elements.videoSource.selectedIndex]?.textContent ?? ""}`);
  });
  elements.start.addEventListener("click", () => {
    startConnection().catch((error) => log(`启动失败: ${error.message}`));
  });
  elements.stop.addEventListener("click", () => stopConnection());
  elements.clearLog.addEventListener("click", () => {
    elements.logOutput.textContent = "";
  });
  bindVideoEvents();
  window.addEventListener("beforeunload", () => {
    stopConnection({ logMessage: false });
  });
}

async function init() {
  bindEvents();
  resetRemoteVideoSize();
  resetFpsDisplay();
  setText(elements.peerStatus, "未建立");
  setText(elements.bitrateStatus, "-");
  setText(elements.npuStatus, "-");
  setText(elements.inferStatus, "-");
  setText(elements.handStatus, "-");
  setText(elements.trackFpsStatus, "-");
  await Promise.all([loadModels(), loadVideos()]);
  await checkHealth();
  log("页面就绪。");
}

init().catch((error) => {
  setText(elements.serverStatus, "不可用");
  log(`初始化失败: ${error.message}`);
});
