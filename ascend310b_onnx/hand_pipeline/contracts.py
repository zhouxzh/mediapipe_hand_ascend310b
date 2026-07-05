"""Deployment-facing constants for the MediaPipe hand pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContract:
    key: str
    input_shape: tuple[int, ...]
    input_dtype: str
    input_layout: str
    input_range: tuple[float, float]
    output_shapes: tuple[tuple[int, ...], ...]


DETECTOR_CONTRACT = ModelContract(
    key="palm_detector",
    input_shape=(1, 192, 192, 3),
    input_dtype="float32",
    input_layout="NHWC",
    input_range=(0.0, 1.0),
    output_shapes=((1, 2016, 18), (1, 2016, 1)),
)

LANDMARK_CONTRACT = ModelContract(
    key="hand_landmark",
    input_shape=(1, 224, 224, 3),
    input_dtype="float32",
    input_layout="NHWC",
    input_range=(0.0, 1.0),
    output_shapes=((1, 63), (1, 1), (1, 1), (1, 63)),
)


DEPLOYABLE_MODULES = (
    "hand_pipeline.contracts",
    "hand_pipeline.preprocess",
    "hand_pipeline.decode",
    "hand_pipeline.roi",
    "hand_pipeline.outputs",
    "hand_pipeline.pipeline",
    "hand_pipeline.tracking",
    "hand_pipeline.runtime",
)


PC_ONLY_MODULES = (
    "hand_pipeline.runtimes.onnx",
)
