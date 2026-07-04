"""Inspect MediaPipe TFLite models with LiteRT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_edge_litert.interpreter import Interpreter


def tensor_info(tensor: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("name", "index", "shape", "shape_signature", "dtype", "quantization", "quantization_parameters"):
        value = tensor.get(key)
        if key == "dtype":
            value = str(value)
        elif hasattr(value, "tolist"):
            value = value.tolist()
        elif isinstance(value, dict):
            value = {
                k: v.tolist() if hasattr(v, "tolist") else str(v) if k == "dtype" else v
                for k, v in value.items()
            }
        elif str(type(value)).startswith("<class 'numpy."):
            value = str(value)
        result[key] = value
    return result


def inspect_model(path: Path) -> dict[str, Any]:
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "inputs": [tensor_info(item) for item in interpreter.get_input_details()],
        "outputs": [tensor_info(item) for item in interpreter.get_output_details()],
    }


def inspect_model_dir(model_dir: Path) -> list[dict[str, Any]]:
    models = [
        model_dir / "mediapipe_task_hand_detector_full.tflite",
        model_dir / "mediapipe_task_hand_landmark_full.tflite",
        model_dir / "mediapipe_legacy_0_10_14_palm_detection_full.tflite",
        model_dir / "mediapipe_legacy_0_10_14_palm_detection_lite.tflite",
        model_dir / "mediapipe_legacy_0_10_14_hand_landmark_full.tflite",
        model_dir / "mediapipe_legacy_0_10_14_hand_landmark_lite.tflite",
    ]
    return [inspect_model(path) for path in models if path.exists()]


def write_model_info(model_dir: Path, output: Path) -> list[dict[str, Any]]:
    info = inspect_model_dir(model_dir)
    output.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info
