#!/usr/bin/env python3
"""Inspect extracted MediaPipe TFLite models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.io import default_tflite_model_dir
from hand_pipeline.tflite_inspect import inspect_model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(default_tflite_model_dir()))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    info = inspect_model_dir(Path(args.model_dir))
    text = json.dumps(info, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

