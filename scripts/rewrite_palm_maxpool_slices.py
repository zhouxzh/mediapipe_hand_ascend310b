#!/usr/bin/env python3
"""Rewrite fixed 2x2 stride-2 MaxPool nodes to Slice + Max.

This is an experimental rewrite for the legacy MediaPipe full palm detector on
Ascend 310B. CANN reports that MaxPoolV3 does not support FP32 input when
`--precision_mode=must_keep_origin_dtype` is used. Replacing the static
2x2/stride-2/pad-0 MaxPool pattern with elementwise Max over four strided
slices removes that MaxPoolV3 dependency while preserving ONNX numerics.
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


DEFAULT_INPUT = "runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_slices.onnx"
DEFAULT_OUTPUT = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx"
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


def add_i64_initializer(model: Any, name: str, values: list[int]) -> str:
    from onnx import numpy_helper

    model.graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    return name


def node_attributes(node: Any) -> dict[str, Any]:
    import onnx

    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def is_target_maxpool(node: Any, shapes: dict[str, list[int]]) -> tuple[bool, dict[str, Any]]:
    if node.op_type != "MaxPool" or len(node.input) != 1 or len(node.output) != 1:
        return False, {}
    attrs = node_attributes(node)
    if attrs.get("kernel_shape") != [2, 2] or attrs.get("strides") != [2, 2]:
        return False, {}
    if attrs.get("pads", [0, 0, 0, 0]) != [0, 0, 0, 0]:
        return False, {}
    input_name = node.input[0]
    output_name = node.output[0]
    if input_name not in shapes or output_name not in shapes:
        return False, {}
    input_shape = shapes[input_name]
    output_shape = shapes[output_name]
    if len(input_shape) != 4 or len(output_shape) != 4:
        return False, {}
    expected = [input_shape[0], input_shape[1], input_shape[2] // 2, input_shape[3] // 2]
    if input_shape[2] % 2 or input_shape[3] % 2 or output_shape != expected:
        raise ValueError(f"Unsupported MaxPool shape at {node.name}: {input_shape} -> {output_shape}")
    return True, {
        "input_name": input_name,
        "output_name": output_name,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def make_slice(model: Any, prefix: str, source: str, h_start: int, w_start: int, height: int, width: int, output: str) -> Any:
    from onnx import helper

    starts = add_i64_initializer(model, f"{prefix}_starts", [h_start, w_start])
    ends = add_i64_initializer(model, f"{prefix}_ends", [height, width])
    axes = add_i64_initializer(model, f"{prefix}_axes", [2, 3])
    steps = add_i64_initializer(model, f"{prefix}_steps", [2, 2])
    return helper.make_node("Slice", inputs=[source, starts, ends, axes, steps], outputs=[output], name=prefix)


def build_replacement(model: Any, rewrite_id: int, node: Any, info: dict[str, Any]) -> list[Any]:
    from onnx import helper

    height = int(info["input_shape"][2])
    width = int(info["input_shape"][3])
    prefix = f"maxpool_slices_{rewrite_id}"
    slice_names = [
        f"{prefix}_h0_w0",
        f"{prefix}_h0_w1",
        f"{prefix}_h1_w0",
        f"{prefix}_h1_w1",
    ]
    return [
        make_slice(model, f"{prefix}_slice_h0_w0", info["input_name"], 0, 0, height, width, slice_names[0]),
        make_slice(model, f"{prefix}_slice_h0_w1", info["input_name"], 0, 1, height, width, slice_names[1]),
        make_slice(model, f"{prefix}_slice_h1_w0", info["input_name"], 1, 0, height, width, slice_names[2]),
        make_slice(model, f"{prefix}_slice_h1_w1", info["input_name"], 1, 1, height, width, slice_names[3]),
        helper.make_node("Max", inputs=slice_names, outputs=list(node.output), name=f"{prefix}_max"),
    ]


def rewrite_model(input_model: Path, output_model: Path) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(input_model))
    shapes = collect_shapes(model)
    new_nodes = []
    rewrites: list[dict[str, Any]] = []
    rewrite_id = 0
    for index, node in enumerate(model.graph.node):
        matched, info = is_target_maxpool(node, shapes)
        if not matched:
            new_nodes.append(node)
            continue
        new_nodes.extend(build_replacement(model, rewrite_id, node, info))
        rewrites.append({"node_index": index, "node_name": node.name, **info})
        rewrite_id += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))

    manifest = {
        "task": "rewrite_palm_maxpool_slices",
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


def verify_onnx_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    import onnxruntime as ort

    reference_dir = resolve_path(args.reference_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_model = resolve_path(args.input_model)
    rewritten_model = resolve_path(args.output_model)
    source_session = ort.InferenceSession(str(source_model), providers=["CPUExecutionProvider"])
    rewritten_session = ort.InferenceSession(str(rewritten_model), providers=["CPUExecutionProvider"])

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
        "task": "verify_maxpool_slices_equivalence",
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
    parser.add_argument("--output-dir", default="runs/palm_om/legacy_full_palm/onnx_maxpool_slices_compare")
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
