"""Project IO helpers."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_tflite_model_dir() -> Path:
    return project_root() / "models" / "tflite"
