"""Project IO helpers."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_onnx_model_dir() -> Path:
    return project_root() / "models" / "onnx"


def default_ascend_model_dir() -> Path:
    return project_root() / "models" / "ascend"
