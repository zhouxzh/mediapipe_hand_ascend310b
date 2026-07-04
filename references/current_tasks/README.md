# Current MediaPipe Tasks Reference

本目录用于放置当前 MediaPipe Tasks API 生成的参考输出：

```text
mediapipe_predictions.json
```

它只用于 `two_stage_vs_current_tasks` 观察当前 Tasks 与本工程两步法之间的差异。legacy 对齐的权威参考不是这个文件，而是 `mediapipe_legacy` 环境中 `mediapipe==0.10.14` graph 生成的：

```text
runs/baseline/legacy_graph/legacy_hand_predictions.json
```

如果脱离父工程单独复制 `mediapipe_hand_ascend310b`，需要把 `mediapipe_predictions.json` 放到本目录；否则 `scripts/run_baseline.py` 会按默认查找顺序从父目录或历史 runs 中寻找 current Tasks 参考输出。
