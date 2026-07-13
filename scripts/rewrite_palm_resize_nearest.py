#!/usr/bin/env python3
"""Rewrite legacy palm bilinear Resize nodes to nearest-neighbor Resize.

This is an accuracy/speed experiment for Ascend 310B. The exact bilinear
half-pixel rewrite is numerically safest but expands into many small
Slice/Mul/Add nodes. Nearest 2x upsample keeps the graph compact and may be
faster if the detector tolerates the FPN approximation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = "runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_split.onnx"
DEFAULT_OUTPUT = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_nearest_resize.onnx"


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def initializer_arrays(model: Any) -> dict[str, np.ndarray]:
    import onnx

    return {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}


def node_attributes(node: Any) -> dict[str, Any]:
    import onnx

    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def is_target_resize(node: Any, arrays: dict[str, np.ndarray]) -> bool:
    if node.op_type != "Resize" or len(node.input) < 3:
        return False
    attrs = node_attributes(node)
    if attrs.get("mode") != b"linear" or attrs.get("coordinate_transformation_mode") != b"half_pixel":
        return False
    scales_name = node.input[2]
    if scales_name not in arrays:
        return False
    scales = np.asarray(arrays[scales_name], dtype=np.float32).reshape(-1)
    return scales.shape == (4,) and np.allclose(scales, np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32))


def rewrite_model(
    input_model: Path,
    output_model: Path,
    coordinate_transformation_mode: str,
    nearest_mode: str,
) -> dict[str, Any]:
    import onnx
    from onnx import helper, shape_inference

    model = onnx.load(str(input_model))
    arrays = initializer_arrays(model)
    rewrites: list[dict[str, Any]] = []

    for index, node in enumerate(model.graph.node):
        if not is_target_resize(node, arrays):
            continue
        old_attrs = {
            key: (value.decode("utf-8") if isinstance(value, bytes) else value)
            for key, value in node_attributes(node).items()
        }
        del node.attribute[:]
        node.attribute.extend(
            [
                helper.make_attribute("coordinate_transformation_mode", coordinate_transformation_mode),
                helper.make_attribute("mode", "nearest"),
                helper.make_attribute("nearest_mode", nearest_mode),
            ]
        )
        rewrites.append(
            {
                "node_index": index,
                "node_name": node.name,
                "outputs": list(node.output),
                "old_attrs": old_attrs,
                "new_attrs": {
                    "coordinate_transformation_mode": coordinate_transformation_mode,
                    "mode": "nearest",
                    "nearest_mode": nearest_mode,
                },
            }
        )

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))
    manifest = {
        "task": "rewrite_palm_resize_nearest",
        "source_model": str(input_model),
        "output_model": str(output_model),
        "coordinate_transformation_mode": coordinate_transformation_mode,
        "nearest_mode": nearest_mode,
        "rewrites": rewrites,
        "rewritten_nodes": len(rewrites),
    }
    write_json(output_model.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", default=DEFAULT_INPUT)
    parser.add_argument("--output-model", default=DEFAULT_OUTPUT)
    parser.add_argument("--coordinate-transformation-mode", default="asymmetric")
    parser.add_argument("--nearest-mode", default="floor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = rewrite_model(
        resolve_path(args.input_model),
        resolve_path(args.output_model),
        args.coordinate_transformation_mode,
        args.nearest_mode,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
