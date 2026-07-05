"""Runtime-neutral model runner protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class ModelRunner(Protocol):
    """One-input model adapter used by deployable pipeline code."""

    model_path: Path

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        ...

