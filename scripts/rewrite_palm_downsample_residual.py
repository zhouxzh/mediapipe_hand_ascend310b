#!/usr/bin/env python3
"""Rewrite legacy palm downsample residual Pad+Add blocks.

The legacy MediaPipe palm detector uses this residual pattern when channels
double after max pooling:

    Add(Pad(pool, channel_tail_zeros), conv_branch)

On Ascend 310B, the Pad tail can contain non-zero values in OM inference. This
script removes the dependency on padded zero channels by rewriting each block:

    first = Add(pool, Slice(conv_branch, channels 0:C))
    tail  = Slice(conv_branch, channels C:2C)
    out   = Concat(first, tail, axis=1)

For the original ONNX graph this is mathematically equivalent and preserves the
same downstream tensor names.
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


DEFAULT_INPUT = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx"
DEFAULT_OUTPUT = "runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_split.onnx"
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


def is_channel_tail_pad(node: Any, shapes: dict[str, list[int]], arrays: dict[str, np.ndarray]) -> tuple[bool, dict[str, Any]]:
    if node.op_type != "Pad" or len(node.input) < 2 or not node.output:
        return False, {}
    input_name = node.input[0]
    output_name = node.output[0]
    pads_name = node.input[1]
    if input_name not in shapes or output_name not in shapes or pads_name not in arrays:
        return False, {}

    input_shape = shapes[input_name]
    output_shape = shapes[output_name]
    pads = np.asarray(arrays[pads_name], dtype=np.int64).reshape(-1)
    rank = len(input_shape)
    if rank != 4 or len(output_shape) != 4 or pads.size != rank * 2:
        return False, {}

    before = pads[:rank]
    after = pads[rank:]
    expected = input_shape.copy()
    expected[1] += int(after[1])
    matched = (
        np.all(before == 0)
        and after[0] == 0
        and after[1] == input_shape[1]
        and after[2] == 0
        and after[3] == 0
        and expected == output_shape
    )
    if not matched:
        return False, {}

    return True, {
        "pad_input": input_name,
        "pad_output": output_name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "pads_name": pads_name,
        "pads": pads.tolist(),
        "channels": input_shape[1],
    }


def add_i64_initializer(model: Any, name: str, values: list[int]) -> str:
    from onnx import numpy_helper

    model.graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    return name


def build_rewrite_nodes(model: Any, prefix: str, add_node: Any, info: dict[str, Any], branch_name: str) -> list[Any]:
    from onnx import helper

    channels = int(info["channels"])
    output_channels = int(info["output_shape"][1])
    axes = add_i64_initializer(model, f"{prefix}_axes", [1])
    steps = add_i64_initializer(model, f"{prefix}_steps", [1])
    first_starts = add_i64_initializer(model, f"{prefix}_first_starts", [0])
    first_ends = add_i64_initializer(model, f"{prefix}_first_ends", [channels])
    tail_starts = add_i64_initializer(model, f"{prefix}_tail_starts", [channels])
    tail_ends = add_i64_initializer(model, f"{prefix}_tail_ends", [output_channels])

    first_slice = f"{prefix}_branch_first"
    tail_slice = f"{prefix}_branch_tail"
    first_sum = f"{prefix}_first_sum"
    return [
        helper.make_node(
            "Slice",
            inputs=[branch_name, first_starts, first_ends, axes, steps],
            outputs=[first_slice],
            name=f"{prefix}_slice_first",
        ),
        helper.make_node(
            "Add",
            inputs=[info["pad_input"], first_slice],
            outputs=[first_sum],
            name=f"{prefix}_add_first",
        ),
        helper.make_node(
            "Slice",
            inputs=[branch_name, tail_starts, tail_ends, axes, steps],
            outputs=[tail_slice],
            name=f"{prefix}_slice_tail",
        ),
        helper.make_node(
            "Concat",
            inputs=[first_sum, tail_slice],
            outputs=list(add_node.output),
            name=f"{prefix}_concat",
            axis=1,
        ),
    ]


def rewrite_model(input_model: Path, output_model: Path) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(input_model))
    shapes = collect_shapes(model)
    arrays = initializer_arrays(model)
    graph_outputs = {output.name for output in model.graph.output}

    pad_infos: dict[str, dict[str, Any]] = {}
    pad_node_indices: set[int] = set()
    consumer_indices: dict[str, list[int]] = {}
    for index, node in enumerate(model.graph.node):
        for input_name in node.input:
            consumer_indices.setdefault(input_name, []).append(index)
        matched, info = is_channel_tail_pad(node, shapes, arrays)
        if matched:
            if info["pad_output"] in graph_outputs:
                raise ValueError(f"Cannot rewrite Pad that is a graph output: {info['pad_output']}")
            pad_infos[info["pad_output"]] = info
            pad_node_indices.add(index)

    add_rewrites: dict[int, tuple[dict[str, Any], str]] = {}
    rewrites: list[dict[str, Any]] = []
    for pad_output, info in pad_infos.items():
        consumers = consumer_indices.get(pad_output, [])
        if len(consumers) != 1:
            raise ValueError(f"Expected one consumer for {pad_output}, got {consumers}")
        add_index = consumers[0]
        add_node = model.graph.node[add_index]
        if add_node.op_type != "Add" or len(add_node.input) != 2:
            raise ValueError(f"Expected Add consumer for {pad_output}, got {add_node.op_type}")
        branch_inputs = [name for name in add_node.input if name != pad_output]
        if len(branch_inputs) != 1:
            raise ValueError(f"Cannot identify branch input for {add_node.name}")
        branch_name = branch_inputs[0]
        branch_shape = shapes.get(branch_name, [])
        if branch_shape != info["output_shape"]:
            raise ValueError(f"Branch shape mismatch for {add_node.name}: {branch_shape} vs {info['output_shape']}")
        add_rewrites[add_index] = (info, branch_name)
        rewrites.append(
            {
                "pad_node_index": next(i for i, node in enumerate(model.graph.node) if node.output and node.output[0] == pad_output),
                "add_node_index": add_index,
                "pad_output": pad_output,
                "pad_input": info["pad_input"],
                "add_output": list(add_node.output),
                "branch_input": branch_name,
                "channels": info["channels"],
                "input_shape": info["input_shape"],
                "output_shape": info["output_shape"],
            }
        )

    new_nodes = []
    rewrite_id = 0
    for index, node in enumerate(model.graph.node):
        if index in pad_node_indices:
            continue
        if index in add_rewrites:
            info, branch_name = add_rewrites[index]
            prefix = f"downsample_split_{rewrite_id}"
            new_nodes.extend(build_rewrite_nodes(model, prefix, node, info, branch_name))
            rewrite_id += 1
            continue
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))

    manifest = {
        "task": "rewrite_palm_downsample_residual",
        "source_model": str(input_model),
        "output_model": str(output_model),
        "rewrites": rewrites,
        "rewritten_blocks": len(rewrites),
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
        rewritten_values = [np.asarray(v, dtype=np.float32) for v in rewritten_session.run(rewritten_outputs, {rewritten_input: tensor})]
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
        "task": "verify_downsample_split_equivalence",
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
    parser.add_argument("--output-dir", default="runs/palm_om/legacy_full_palm/onnx_downsample_split_compare")
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
