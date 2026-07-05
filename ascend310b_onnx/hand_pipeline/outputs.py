"""Model output parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LandmarkOutputs:
    landmarks: np.ndarray
    hand_score: float
    handedness: float
    world_landmarks: np.ndarray | None = None


def pick_landmark_outputs(outputs: list[np.ndarray]) -> LandmarkOutputs:
    """Parse MediaPipe hand landmark model outputs.

    The bundled legacy models expose image landmarks, hand-presence score,
    handedness, and world landmarks. Shape checks keep this helper robust
    across TFLite and ONNX Runtime wrappers while still failing loudly if a
    converter changes the output contract.
    """
    landmarks: np.ndarray | None = None
    world: np.ndarray | None = None
    scalar_outputs: list[np.ndarray] = []
    for value in outputs:
        arr = np.asarray(value)
        if arr.size == 63 and landmarks is None:
            landmarks = arr.reshape(21, 3)
        elif arr.size == 63:
            world = arr.reshape(21, 3)
        elif arr.size == 1:
            scalar_outputs.append(arr.reshape(-1))
    if landmarks is None:
        raise ValueError(f"Could not find 63-value landmark output: {[np.asarray(x).shape for x in outputs]}")
    hand_score = float(scalar_outputs[0][0]) if len(scalar_outputs) >= 1 else math.nan
    handedness = float(scalar_outputs[1][0]) if len(scalar_outputs) >= 2 else math.nan
    return LandmarkOutputs(
        landmarks=landmarks.astype(np.float32),
        hand_score=hand_score,
        handedness=handedness,
        world_landmarks=None if world is None else world.astype(np.float32),
    )

