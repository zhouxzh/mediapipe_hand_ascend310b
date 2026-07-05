"""ONNX Runtime adapter for PC validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxModel:
    """Small ONNX Runtime wrapper for one-input MediaPipe ONNX models."""

    def __init__(self, model_path: str | Path, num_threads: int | None = None) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        sess_options = ort.SessionOptions()
        if num_threads is not None and num_threads > 0:
            sess_options.intra_op_num_threads = int(num_threads)
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        input_name = self.inputs[0].name
        output_names = [item.name for item in self.outputs]
        return [np.asarray(item) for item in self.session.run(output_names, {input_name: tensor})]

