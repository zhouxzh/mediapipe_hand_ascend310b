#!/usr/bin/env python3
"""Rewrite legacy palm FPN bilinear Resize nodes to explicit arithmetic.

The legacy MediaPipe palm detector has two FPN upsample nodes:

    Resize(mode=linear, coordinate_transformation_mode=half_pixel, scales=2)

ONNX Runtime matches TFLite for these nodes, but Ascend 310B ATC/OM showed a
large drift at the first Resize output. This script rewrites only the static
NCHW 2x bilinear half-pixel pattern into Slice/Mul/Add/Concat nodes. The
rewritten graph is mathematically equivalent for the fixed palm-detector
shapes and avoids relying on OM Resize semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = "runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_split.onnx"
DEFAULT_OUTPUT = "runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_slices.onnx"
DEFAULT_REFERENCE_DIR = "runs/palm_om/legacy_full_palm"


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def add_f32_initializer(model: Any, name: str, values: list[float]) -> str:
    from onnx import numpy_helper

    model.graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name))
    return name


def resize_attributes(node: Any) -> dict[str, Any]:
    import onnx

    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def is_target_resize(node: Any, shapes: dict[str, list[int]], arrays: dict[str, np.ndarray]) -> tuple[bool, dict[str, Any]]:
    if node.op_type != "Resize" or len(node.input) < 3 or not node.output:
        return False, {}

    attrs = resize_attributes(node)
    if attrs.get("mode") != b"linear" or attrs.get("coordinate_transformation_mode") != b"half_pixel":
        return False, {}

    input_name = node.input[0]
    output_name = node.output[0]
    if input_name not in shapes or output_name not in shapes:
        return False, {}

    scales_name = node.input[2]
    if scales_name not in arrays:
        return False, {}
    scales = np.asarray(arrays[scales_name], dtype=np.float32).reshape(-1)
    if scales.shape != (4,) or not np.allclose(scales, np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32)):
        return False, {}

    input_shape = shapes[input_name]
    output_shape = shapes[output_name]
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
        "scales_name": scales_name,
        "scales": scales.tolist(),
        "attrs": {key: (value.decode("utf-8") if isinstance(value, bytes) else value) for key, value in attrs.items()},
    }


def make_slice_node(model: Any, prefix: str, source: str, axis: int, start: int, end: int, output: str) -> Any:
    from onnx import helper

    starts = add_i64_initializer(model, f"{prefix}_starts", [start])
    ends = add_i64_initializer(model, f"{prefix}_ends", [end])
    axes = add_i64_initializer(model, f"{prefix}_axes", [axis])
    steps = add_i64_initializer(model, f"{prefix}_steps", [1])
    return helper.make_node("Slice", inputs=[source, starts, ends, axes, steps], outputs=[output], name=prefix)


def make_weighted_sum_nodes(
    model: Any,
    prefix: str,
    a: str,
    b: str,
    weight_a: float,
    weight_b: float,
    output: str,
) -> list[Any]:
    from onnx import helper

    weight_a_name = add_f32_initializer(model, f"{prefix}_weight_a", [weight_a])
    weight_b_name = add_f32_initializer(model, f"{prefix}_weight_b", [weight_b])
    a_scaled = f"{prefix}_a_scaled"
    b_scaled = f"{prefix}_b_scaled"
    return [
        helper.make_node("Mul", inputs=[a, weight_a_name], outputs=[a_scaled], name=f"{prefix}_mul_a"),
        helper.make_node("Mul", inputs=[b, weight_b_name], outputs=[b_scaled], name=f"{prefix}_mul_b"),
        helper.make_node("Add", inputs=[a_scaled, b_scaled], outputs=[output], name=f"{prefix}_add"),
    ]


def build_linear_half_pixel_2x_axis_nodes(
    model: Any,
    prefix: str,
    source: str,
    source_size: int,
    axis: int,
    output: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    from onnx import helper

    nodes: list[Any] = []
    terms: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    for out_index in range(source_size * 2):
        if out_index == 0:
            term = f"{prefix}_out_{out_index}"
            nodes.append(make_slice_node(model, f"{prefix}_slice_edge_first_{out_index}", source, axis, 0, 1, term))
            manifest_rows.append({"out_index": out_index, "kind": "edge_first", "src": [0], "weights": [1.0]})
        elif out_index == source_size * 2 - 1:
            term = f"{prefix}_out_{out_index}"
            nodes.append(
                make_slice_node(
                    model,
                    f"{prefix}_slice_edge_last_{out_index}",
                    source,
                    axis,
                    source_size - 1,
                    source_size,
                    term,
                )
            )
            manifest_rows.append(
                {"out_index": out_index, "kind": "edge_last", "src": [source_size - 1], "weights": [1.0]}
            )
        else:
            if out_index % 2 == 1:
                src0 = (out_index - 1) // 2
                src1 = src0 + 1
                weights = (0.75, 0.25)
            else:
                src0 = out_index // 2 - 1
                src1 = src0 + 1
                weights = (0.25, 0.75)
            first = f"{prefix}_out_{out_index}_src_{src0}"
            second = f"{prefix}_out_{out_index}_src_{src1}"
            term = f"{prefix}_out_{out_index}"
            nodes.append(make_slice_node(model, f"{prefix}_slice_{out_index}_a", source, axis, src0, src0 + 1, first))
            nodes.append(make_slice_node(model, f"{prefix}_slice_{out_index}_b", source, axis, src1, src1 + 1, second))
            nodes.extend(make_weighted_sum_nodes(model, f"{prefix}_blend_{out_index}", first, second, weights[0], weights[1], term))
            manifest_rows.append(
                {
                    "out_index": out_index,
                    "kind": "blend",
                    "src": [src0, src1],
                    "weights": [weights[0], weights[1]],
                }
            )
        terms.append(term)

    nodes.append(helper.make_node("Concat", inputs=terms, outputs=[output], name=f"{prefix}_concat", axis=axis))
    return nodes, manifest_rows


def build_resize_replacement(model: Any, rewrite_id: int, info: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    input_shape = info["input_shape"]
    height = int(input_shape[2])
    width = int(input_shape[3])
    prefix = f"resize_slices_{rewrite_id}"
    height_output = f"{prefix}_height_up"
    height_nodes, height_rows = build_linear_half_pixel_2x_axis_nodes(
        model,
        f"{prefix}_h",
        info["input_name"],
        height,
        2,
        height_output,
    )
    width_nodes, width_rows = build_linear_half_pixel_2x_axis_nodes(
        model,
        f"{prefix}_w",
        height_output,
        width,
        3,
        info["output_name"],
    )
    rewrite = {
        **info,
        "rewrite_prefix": prefix,
        "height_rule": height_rows,
        "width_rule": width_rows,
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
        "task": "rewrite_palm_bilinear_resize",
        "source_model": str(input_model),
        "output_model": str(output_model),
        "rewrites": rewrites,
        "rewritten_nodes": len(rewrites),
    }
    write_json(output_model.with_suffix(".manifest.json"), manifest)
    return manifest


def summarize_abs(diff: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(diff, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_p99": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
        f"{prefix}_max": float(np.max(arr)),
    }


def load_reference_items(reference_dir: Path, max_images: int) -> list[dict[str, Any]]:
    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    if max_images:
        items = items[:max_images]
    return items


def run_onnx_session(model_path: Path) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def verify_onnx_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    reference_dir = resolve_path(args.reference_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_model = resolve_path(args.input_model)
    rewritten_model = resolve_path(args.output_model)
    source_session = run_onnx_session(source_model)
    rewritten_session = run_onnx_session(rewritten_model)

    source_input = source_session.get_inputs()[0].name
    rewritten_input = rewritten_session.get_inputs()[0].name
    source_outputs = [item.name for item in source_session.get_outputs()]
    rewritten_outputs = [item.name for item in rewritten_session.get_outputs()]
    if source_outputs != rewritten_outputs:
        raise RuntimeError(f"Output names differ: {source_outputs} vs {rewritten_outputs}")

    output_abs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    items = load_reference_items(reference_dir, args.max_images)
    for item in items:
        ref = np.load(reference_dir / item["npz"])
        tensor = np.asarray(ref["input"], dtype=np.float32)
        source_values = [np.asarray(v, dtype=np.float32) for v in source_session.run(source_outputs, {source_input: tensor})]
        rewritten_values = [
            np.asarray(v, dtype=np.float32) for v in rewritten_session.run(rewritten_outputs, {rewritten_input: tensor})
        ]
        diffs = [np.abs(a - b) for a, b in zip(source_values, rewritten_values)]
        for diff in diffs:
            output_abs.append(diff.reshape(-1))
        rows.append(
            {
                "image": item["image"],
                "output0_mean_abs": float(np.mean(diffs[0])) if diffs else math.nan,
                "output0_max_abs": float(np.max(diffs[0])) if diffs else math.nan,
                "output1_mean_abs": float(np.mean(diffs[1])) if len(diffs) > 1 else math.nan,
                "output1_max_abs": float(np.max(diffs[1])) if len(diffs) > 1 else math.nan,
            }
        )

    all_diff = np.concatenate(output_abs) if output_abs else np.array([], dtype=np.float32)
    summary = {
        "task": "verify_resize_slices_equivalence",
        "source_model": str(source_model),
        "rewritten_model": str(rewritten_model),
        "reference_dir": str(reference_dir),
        "images": len(items),
        "output_names": source_outputs,
        **summarize_abs(all_diff, "all_outputs_abs"),
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "per_image.csv", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", default=DEFAULT_INPUT)
    parser.add_argument("--output-model", default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", default="runs/palm_om/legacy_full_palm/onnx_resize_slices_compare")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = rewrite_model(resolve_path(args.input_model), resolve_path(args.output_model))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.verify:
        verify_onnx_equivalence(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
