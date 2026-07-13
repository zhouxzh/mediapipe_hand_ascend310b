#!/usr/bin/env python3
"""Rewrite legacy palm 2x bilinear Resize nodes to depthwise ConvTranspose.

The legacy MediaPipe palm detector has two static NCHW Resize nodes:

    Resize(mode=linear, coordinate_transformation_mode=half_pixel, scales=2)

The existing Ascend-safe rewrite expands each Resize into many Slice/Mul/Add
nodes. This script keeps the same ONNX numerics but uses separable depthwise
ConvTranspose passes:

    edge-replicate pad -> ConvTranspose stride-2 -> Slice crop

for height and width. It is intended as a speed candidate for Ascend 310B; the
result must still be compiled and validated on the board.
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
DEFAULT_OUTPUT = "runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_deconv.onnx"


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tensor_shape(value_info: Any) -> list[int]:
    shape: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ValueError(f"Non-static shape in {value_info.name!r}")
        shape.append(int(dim.dim_value))
    return shape


def collect_shapes(model: Any) -> dict[str, list[int]]:
    from onnx import shape_inference

    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, list[int]] = {}
    for value_info in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if value_info.type.HasField("tensor_type") and value_info.type.tensor_type.HasField("shape"):
            shapes[value_info.name] = tensor_shape(value_info)
    return shapes


def initializer_arrays(model: Any) -> dict[str, np.ndarray]:
    import onnx

    return {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}


def add_i64_initializer(model: Any, name: str, values: list[int]) -> str:
    from onnx import numpy_helper

    model.graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    return name


def add_f32_initializer(model: Any, name: str, values: np.ndarray) -> str:
    from onnx import numpy_helper

    model.graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name))
    return name


def node_attributes(node: Any) -> dict[str, Any]:
    import onnx

    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def is_target_resize(node: Any, shapes: dict[str, list[int]], arrays: dict[str, np.ndarray]) -> tuple[bool, dict[str, Any]]:
    if node.op_type != "Resize" or len(node.input) < 3 or not node.output:
        return False, {}

    attrs = node_attributes(node)
    if attrs.get("mode") != b"linear" or attrs.get("coordinate_transformation_mode") != b"half_pixel":
        return False, {}

    input_name = node.input[0]
    output_name = node.output[0]
    scales_name = node.input[2]
    if input_name not in shapes or output_name not in shapes or scales_name not in arrays:
        return False, {}

    scales = np.asarray(arrays[scales_name], dtype=np.float32).reshape(-1)
    input_shape = shapes[input_name]
    output_shape = shapes[output_name]
    if scales.shape != (4,) or not np.allclose(scales, np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32)):
        return False, {}
    if len(input_shape) != 4 or len(output_shape) != 4:
        return False, {}
    expected = [input_shape[0], input_shape[1], input_shape[2] * 2, input_shape[3] * 2]
    if output_shape != expected:
        raise ValueError(f"Resize shape mismatch for {node.name}: expected {expected}, got {output_shape}")

    return True, {
        "input_name": input_name,
        "output_name": output_name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "attrs": {key: (value.decode("utf-8") if isinstance(value, bytes) else value) for key, value in attrs.items()},
    }


def make_slice(model: Any, prefix: str, source: str, axis: int, start: int, end: int, output: str) -> Any:
    from onnx import helper

    starts = add_i64_initializer(model, f"{prefix}_starts", [start])
    ends = add_i64_initializer(model, f"{prefix}_ends", [end])
    axes = add_i64_initializer(model, f"{prefix}_axes", [axis])
    steps = add_i64_initializer(model, f"{prefix}_steps", [1])
    return helper.make_node("Slice", inputs=[source, starts, ends, axes, steps], outputs=[output], name=prefix)


def edge_concat_nodes(model: Any, prefix: str, source: str, axis: int, size: int) -> tuple[list[Any], str]:
    from onnx import helper

    first = f"{prefix}_edge_first"
    last = f"{prefix}_edge_last"
    padded = f"{prefix}_edge_padded"
    nodes = [
        make_slice(model, f"{prefix}_slice_first", source, axis, 0, 1, first),
        make_slice(model, f"{prefix}_slice_last", source, axis, size - 1, size, last),
        helper.make_node("Concat", inputs=[first, source, last], outputs=[padded], name=f"{prefix}_concat_edges", axis=axis),
    ]
    return nodes, padded


def convtranspose_weight(channels: int, axis: int) -> np.ndarray:
    kernel = np.asarray([0.25, 0.75, 0.75, 0.25], dtype=np.float32)
    if axis == 2:
        weight = np.zeros((channels, 1, 4, 1), dtype=np.float32)
        weight[:, 0, :, 0] = kernel
        return weight
    if axis == 3:
        weight = np.zeros((channels, 1, 1, 4), dtype=np.float32)
        weight[:, 0, 0, :] = kernel
        return weight
    raise ValueError(f"Unsupported axis: {axis}")


def axis_deconv_nodes(
    model: Any,
    prefix: str,
    source: str,
    source_shape: list[int],
    axis: int,
    output: str,
) -> tuple[list[Any], list[int]]:
    from onnx import helper

    if axis not in (2, 3):
        raise ValueError(f"Only NCHW spatial axes are supported, got axis={axis}")
    channels = int(source_shape[1])
    size = int(source_shape[axis])
    nodes, padded = edge_concat_nodes(model, prefix, source, axis, size)

    deconv = f"{prefix}_deconv"
    weight_name = add_f32_initializer(model, f"{prefix}_weight", convtranspose_weight(channels, axis))
    if axis == 2:
        kernel_shape = [4, 1]
        strides = [2, 1]
        pads = [1, 0, 1, 0]
    else:
        kernel_shape = [1, 4]
        strides = [1, 2]
        pads = [0, 1, 0, 1]
    nodes.append(
        helper.make_node(
            "ConvTranspose",
            inputs=[padded, weight_name],
            outputs=[deconv],
            name=f"{prefix}_convtranspose",
            group=channels,
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
        )
    )
    nodes.append(make_slice(model, f"{prefix}_crop", deconv, axis, 2, 2 + size * 2, output))

    output_shape = list(source_shape)
    output_shape[axis] = size * 2
    return nodes, output_shape


def build_resize_replacement(model: Any, rewrite_id: int, info: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    prefix = f"resize_deconv_{rewrite_id}"
    height_output = f"{prefix}_height_up"
    height_nodes, height_shape = axis_deconv_nodes(
        model,
        f"{prefix}_h",
        info["input_name"],
        info["input_shape"],
        2,
        height_output,
    )
    width_nodes, output_shape = axis_deconv_nodes(
        model,
        f"{prefix}_w",
        height_output,
        height_shape,
        3,
        info["output_name"],
    )
    rewrite = {
        **info,
        "rewrite_prefix": prefix,
        "height_intermediate_shape": height_shape,
        "replacement_output_shape": output_shape,
        "replacement_nodes": len(height_nodes) + len(width_nodes),
    }
    return height_nodes + width_nodes, rewrite


def rewrite_model(input_model: Path, output_model: Path) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(input_model))
    shapes = collect_shapes(model)
    arrays = initializer_arrays(model)

    new_nodes = []
    rewrites: list[dict[str, Any]] = []
    rewrite_id = 0
    for index, node in enumerate(model.graph.node):
        matched, info = is_target_resize(node, shapes, arrays)
        if not matched:
            new_nodes.append(node)
            continue
        replacement, rewrite = build_resize_replacement(model, rewrite_id, info)
        rewrite["node_index"] = index
        rewrite["node_name"] = node.name
        rewrites.append(rewrite)
        new_nodes.extend(replacement)
        rewrite_id += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))

    manifest = {
        "task": "rewrite_palm_bilinear_resize_deconv",
        "source_model": str(input_model),
        "output_model": str(output_model),
        "rewrites": rewrites,
        "rewritten_nodes": len(rewrites),
    }
    write_json(output_model.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", default=DEFAULT_INPUT)
    parser.add_argument("--output-model", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = rewrite_model(resolve_path(args.input_model), resolve_path(args.output_model))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
