#!/usr/bin/env python3
"""Compare operator structure between two ONNX graphs.

This is a graph-level diagnostic tool. It does not run inference. It focuses on
operator counts, attribute signatures, Conv/channel structure, and the
Pad/Resize/MaxPool patterns that matter for the palm detector on Ascend 310B.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    import onnx

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return value.reshape(-1).tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, onnx.TensorProto):
        arr = onnx.numpy_helper.to_array(value)
        return json_safe(arr)
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def node_attrs(node: Any) -> dict[str, Any]:
    import onnx

    return {attr.name: json_safe(onnx.helper.get_attribute_value(attr)) for attr in node.attribute}


def attr_signature(node: Any) -> str:
    attrs = node_attrs(node)
    return json.dumps(attrs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tensor_shape(value_info: Any) -> list[int | str | None]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def collect_shapes(model: Any) -> dict[str, list[int | str | None]]:
    import onnx

    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shapes: dict[str, list[int | str | None]] = {}
    for value_info in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if value_info.type.HasField("tensor_type") and value_info.type.tensor_type.HasField("shape"):
            shapes[value_info.name] = tensor_shape(value_info)
    return shapes


def initializer_arrays(model: Any) -> dict[str, np.ndarray]:
    import onnx

    return {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}


def small_initializer(arrays: dict[str, np.ndarray], name: str) -> Any:
    if name not in arrays:
        return None
    arr = np.asarray(arrays[name])
    return json_safe(arr)


def is_int_shape(shape: list[Any] | None) -> bool:
    return bool(shape) and all(isinstance(item, int) for item in shape)


def analyze_conv(node: Any, arrays: dict[str, np.ndarray], shapes: dict[str, list[Any]]) -> dict[str, Any]:
    attrs = node_attrs(node)
    group = int(attrs.get("group", 1))
    weight_shape = list(arrays[node.input[1]].shape) if len(node.input) > 1 and node.input[1] in arrays else None
    input_shape = shapes.get(node.input[0]) if node.input else None
    output_shape = shapes.get(node.output[0]) if node.output else None
    in_channels = input_shape[1] if is_int_shape(input_shape) and len(input_shape) == 4 else None
    out_channels = output_shape[1] if is_int_shape(output_shape) and len(output_shape) == 4 else None
    if in_channels is None and weight_shape:
        in_channels = int(weight_shape[1]) * group
    if out_channels is None and weight_shape:
        out_channels = int(weight_shape[0])
    is_depthwise = bool(in_channels and group == in_channels and out_channels and int(out_channels) % int(in_channels) == 0)
    return {
        "name": node.name,
        "input": list(node.input),
        "output": list(node.output),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "weight_shape": weight_shape,
        "group": group,
        "is_depthwise": is_depthwise,
        "kernel_shape": attrs.get("kernel_shape"),
        "strides": attrs.get("strides", [1, 1]),
        "pads": attrs.get("pads", [0, 0, 0, 0]),
        "dilations": attrs.get("dilations", [1, 1]),
    }


def analyze_resize(node: Any, arrays: dict[str, np.ndarray], shapes: dict[str, list[Any]]) -> dict[str, Any]:
    attrs = node_attrs(node)
    return {
        "name": node.name,
        "input": list(node.input),
        "output": list(node.output),
        "input_shape": shapes.get(node.input[0]) if node.input else None,
        "output_shape": shapes.get(node.output[0]) if node.output else None,
        "attrs": attrs,
        "scales": small_initializer(arrays, node.input[2]) if len(node.input) > 2 else None,
        "sizes": small_initializer(arrays, node.input[3]) if len(node.input) > 3 else None,
    }


def analyze_pad(node: Any, arrays: dict[str, np.ndarray], shapes: dict[str, list[Any]]) -> dict[str, Any]:
    attrs = node_attrs(node)
    return {
        "name": node.name,
        "input": list(node.input),
        "output": list(node.output),
        "input_shape": shapes.get(node.input[0]) if node.input else None,
        "output_shape": shapes.get(node.output[0]) if node.output else None,
        "attrs": attrs,
        "pads": small_initializer(arrays, node.input[1]) if len(node.input) > 1 else attrs.get("pads"),
        "constant_value": small_initializer(arrays, node.input[2]) if len(node.input) > 2 else None,
    }


def analyze_pool(node: Any, shapes: dict[str, list[Any]]) -> dict[str, Any]:
    attrs = node_attrs(node)
    return {
        "name": node.name,
        "input": list(node.input),
        "output": list(node.output),
        "input_shape": shapes.get(node.input[0]) if node.input else None,
        "output_shape": shapes.get(node.output[0]) if node.output else None,
        "attrs": attrs,
    }


def analyze_slice(node: Any, arrays: dict[str, np.ndarray], shapes: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "name": node.name,
        "input": list(node.input),
        "output": list(node.output),
        "input_shape": shapes.get(node.input[0]) if node.input else None,
        "output_shape": shapes.get(node.output[0]) if node.output else None,
        "starts": small_initializer(arrays, node.input[1]) if len(node.input) > 1 else None,
        "ends": small_initializer(arrays, node.input[2]) if len(node.input) > 2 else None,
        "axes": small_initializer(arrays, node.input[3]) if len(node.input) > 3 else None,
        "steps": small_initializer(arrays, node.input[4]) if len(node.input) > 4 else None,
    }


def analyze_prelu(node: Any, arrays: dict[str, np.ndarray], shapes: dict[str, list[Any]]) -> dict[str, Any]:
    slope_shape = list(arrays[node.input[1]].shape) if len(node.input) > 1 and node.input[1] in arrays else None
    return {
        "name": node.name,
        "input": list(node.input),
        "output": list(node.output),
        "input_shape": shapes.get(node.input[0]) if node.input else None,
        "output_shape": shapes.get(node.output[0]) if node.output else None,
        "slope_shape": slope_shape,
    }


def analyze_node_sequence(nodes: list[Any], shapes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, node in enumerate(nodes):
        rows.append(
            {
                "index": index,
                "op_type": node.op_type,
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "input0_shape": shapes.get(node.input[0]) if node.input else None,
                "output0_shape": shapes.get(node.output[0]) if node.output else None,
            }
        )
    return rows


def find_tail_pad_residuals(nodes: list[Any], pads: list[dict[str, Any]], shapes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    producer = {}
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in nodes:
        for out in node.output:
            producer[out] = node
        for inp in node.input:
            consumers[inp].append(node)

    results = []
    for pad in pads:
        output = pad["output"][0] if pad["output"] else ""
        input_shape = pad.get("input_shape")
        output_shape = pad.get("output_shape")
        pad_values = pad.get("pads")
        if not (is_int_shape(input_shape) and is_int_shape(output_shape) and isinstance(pad_values, list)):
            continue
        if len(input_shape) != 4 or len(output_shape) != 4 or len(pad_values) != 8:
            continue
        tail_channel_pad = (
            pad_values[:4] == [0, 0, 0, 0]
            and pad_values[4] == 0
            and pad_values[5] == input_shape[1]
            and pad_values[6] == 0
            and pad_values[7] == 0
            and output_shape[1] == input_shape[1] * 2
        )
        add_consumers = [node for node in consumers.get(output, []) if node.op_type == "Add"]
        if tail_channel_pad and add_consumers:
            add_node = add_consumers[0]
            branch = [inp for inp in add_node.input if inp != output]
            results.append(
                {
                    "pad_name": pad["name"],
                    "pad_input": pad["input"][0],
                    "pad_output": output,
                    "pad_input_shape": input_shape,
                    "pad_output_shape": output_shape,
                    "add_name": add_node.name,
                    "add_output": list(add_node.output),
                    "branch_input": branch[0] if branch else None,
                    "branch_shape": shapes.get(branch[0]) if branch else None,
                    "pool_source": producer.get(pad["input"][0]).name if pad["input"][0] in producer else None,
                }
            )
    return results


def analyze_model(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    nodes = list(model.graph.node)
    shapes = collect_shapes(model)
    arrays = initializer_arrays(model)
    ops = Counter(node.op_type for node in nodes)
    signatures = Counter(f"{node.op_type} {attr_signature(node)}" for node in nodes)
    convs = [analyze_conv(node, arrays, shapes) for node in nodes if node.op_type == "Conv"]
    resize = [analyze_resize(node, arrays, shapes) for node in nodes if node.op_type == "Resize"]
    pads = [analyze_pad(node, arrays, shapes) for node in nodes if node.op_type == "Pad"]
    maxpools = [analyze_pool(node, shapes) for node in nodes if node.op_type == "MaxPool"]
    slices = [analyze_slice(node, arrays, shapes) for node in nodes if node.op_type == "Slice"]
    prelus = [analyze_prelu(node, arrays, shapes) for node in nodes if node.op_type == "PRelu"]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "ir_version": model.ir_version,
        "opset_import": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "inputs": [{"name": item.name, "shape": shapes.get(item.name)} for item in model.graph.input],
        "outputs": [{"name": item.name, "shape": shapes.get(item.name)} for item in model.graph.output],
        "node_count": len(nodes),
        "initializer_count": len(model.graph.initializer),
        "op_counts": dict(sorted(ops.items())),
        "op_signatures": dict(sorted(signatures.items())),
        "nodes": analyze_node_sequence(nodes, shapes),
        "conv": convs,
        "conv_summary": {
            "total": len(convs),
            "depthwise": sum(1 for item in convs if item["is_depthwise"]),
            "grouped_non_depthwise": sum(1 for item in convs if item["group"] != 1 and not item["is_depthwise"]),
            "pointwise_1x1": sum(1 for item in convs if item["kernel_shape"] == [1, 1]),
            "kernel_counts": dict(sorted(Counter(str(item["kernel_shape"]) for item in convs).items())),
            "stride_counts": dict(sorted(Counter(str(item["strides"]) for item in convs).items())),
        },
        "resize": resize,
        "pads": pads,
        "tail_pad_residuals": find_tail_pad_residuals(nodes, pads, shapes),
        "maxpool": maxpools,
        "slice": slices,
        "prelu": prelus,
        "special_op_counts": {
            "Resize": len(resize),
            "Pad": len(pads),
            "tail_pad_residuals": len(find_tail_pad_residuals(nodes, pads, shapes)),
            "MaxPool": len(maxpools),
            "Slice": len(slices),
            "PRelu": len(prelus),
        },
    }


def count_diff(left: dict[str, int], right: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(left) | set(right)):
        lval = int(left.get(key, 0))
        rval = int(right.get(key, 0))
        if lval != rval:
            rows.append({"key": key, "left": lval, "right": rval, "delta_right_minus_left": rval - lval})
    return rows


def compare_channel_pairs(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, (lconv, rconv) in enumerate(zip(left, right)):
        rows.append(
            {
                "index": index,
                "left_name": lconv["name"],
                "right_name": rconv["name"],
                "left_weight_shape": lconv["weight_shape"],
                "right_weight_shape": rconv["weight_shape"],
                "left_group": lconv["group"],
                "right_group": rconv["group"],
                "left_depthwise": lconv["is_depthwise"],
                "right_depthwise": rconv["is_depthwise"],
                "left_kernel": lconv["kernel_shape"],
                "right_kernel": rconv["kernel_shape"],
                "left_stride": lconv["strides"],
                "right_stride": rconv["strides"],
            }
        )
    return rows


def normalize_layer_name(name: str) -> str:
    # Keep semantic layer identity, but strip exporter suffixes that make the
    # same Keras layer appear as Conv2D1/Conv2D2 in different branches.
    result = name
    for suffix in ("_Transpose", ":0"):
        result = result.replace(suffix, "")
    result = result.replace("Conv2D1", "Conv2D")
    result = result.replace("Conv2D2", "Conv2D")
    result = result.replace("depthwise1", "depthwise")
    result = result.replace("depthwise2", "depthwise")
    return result


def layer_key(row: dict[str, Any]) -> str:
    return f"{row['op_type']}::{normalize_layer_name(str(row['name']))}"


def node_presence_diff(left_nodes: list[dict[str, Any]], right_nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    left_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    right_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in left_nodes:
        left_map[layer_key(row)].append(row)
    for row in right_nodes:
        right_map[layer_key(row)].append(row)

    only_left = []
    only_right = []
    for key in sorted(set(left_map) | set(right_map)):
        left_count = len(left_map.get(key, []))
        right_count = len(right_map.get(key, []))
        if left_count > right_count:
            only_left.extend(left_map[key][right_count:])
        elif right_count > left_count:
            only_right.extend(right_map[key][left_count:])
    return {"only_left": only_left, "only_right": only_right}


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
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def write_markdown(report: dict[str, Any], path: Path) -> None:
    left = report["left"]
    right = report["right"]
    lines = [
        "# ONNX Operator Compare",
        "",
        f"- left: `{left['path']}`",
        f"- right: `{right['path']}`",
        "",
        "## Summary",
        "",
        "| Metric | Left | Right |",
        "| --- | ---: | ---: |",
        f"| node_count | {left['node_count']} | {right['node_count']} |",
        f"| initializer_count | {left['initializer_count']} | {right['initializer_count']} |",
        f"| Conv total | {left['conv_summary']['total']} | {right['conv_summary']['total']} |",
        f"| Conv depthwise | {left['conv_summary']['depthwise']} | {right['conv_summary']['depthwise']} |",
        f"| Resize | {left['special_op_counts']['Resize']} | {right['special_op_counts']['Resize']} |",
        f"| Pad | {left['special_op_counts']['Pad']} | {right['special_op_counts']['Pad']} |",
        f"| tail Pad + Add residual | {left['special_op_counts']['tail_pad_residuals']} | {right['special_op_counts']['tail_pad_residuals']} |",
        f"| MaxPool | {left['special_op_counts']['MaxPool']} | {right['special_op_counts']['MaxPool']} |",
        f"| Slice | {left['special_op_counts']['Slice']} | {right['special_op_counts']['Slice']} |",
        f"| PRelu | {left['special_op_counts']['PRelu']} | {right['special_op_counts']['PRelu']} |",
        "",
        "## Operator Count Differences",
        "",
    ]
    if report["op_count_diff"]:
        lines.extend(["| op_type | Left | Right | Delta |", "| --- | ---: | ---: | ---: |"])
        for row in report["op_count_diff"]:
            lines.append(f"| `{row['key']}` | {row['left']} | {row['right']} | {row['delta_right_minus_left']} |")
    else:
        lines.append("No op_type count differences.")

    lines.extend(
        [
            "",
            "## Conv Channel Structure",
            "",
            "| Metric | Left | Right |",
            "| --- | --- | --- |",
            f"| kernel_counts | `{left['conv_summary']['kernel_counts']}` | `{right['conv_summary']['kernel_counts']}` |",
            f"| stride_counts | `{left['conv_summary']['stride_counts']}` | `{right['conv_summary']['stride_counts']}` |",
            "",
            "First paired Conv layers are written to `conv_pairs.csv`.",
            "",
            "## Special Patterns",
            "",
            "### Resize",
            "",
        ]
    )
    for side_name, side in [("left", left), ("right", right)]:
        lines.append(f"{side_name}:")
        for item in side["resize"]:
            lines.append(f"- `{item['name']}` `{item['input_shape']}` -> `{item['output_shape']}`, attrs=`{item['attrs']}`, scales=`{item['scales']}`")
    lines.extend(["", "### Tail Channel Pad + Add Residuals", ""])
    for side_name, side in [("left", left), ("right", right)]:
        lines.append(f"{side_name}:")
        for item in side["tail_pad_residuals"]:
            lines.append(
                f"- `{item['pad_name']}` -> `{item['add_name']}` "
                f"`{item['pad_input_shape']}` -> `{item['pad_output_shape']}`, branch=`{item['branch_shape']}`"
            )
    lines.extend(["", "### MaxPool", ""])
    for side_name, side in [("left", left), ("right", right)]:
        lines.append(f"{side_name}:")
        for item in side["maxpool"]:
            lines.append(f"- `{item['name']}` `{item['input_shape']}` -> `{item['output_shape']}`, attrs=`{item['attrs']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Left ONNX model, usually full palm.")
    parser.add_argument("--right", required=True, help="Right ONNX model, usually lite palm.")
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-dir", default="runs/onnx_op_compare/test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    left = analyze_model(resolve(args.left))
    right = analyze_model(resolve(args.right))
    report = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left": left,
        "right": right,
        "op_count_diff": count_diff(left["op_counts"], right["op_counts"]),
        "op_signature_diff": count_diff(left["op_signatures"], right["op_signatures"]),
        "conv_pairs": compare_channel_pairs(left["conv"], right["conv"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "conv_pairs.csv", report["conv_pairs"])
    write_csv(output_dir / "left_nodes.csv", left["nodes"])
    write_csv(output_dir / "right_nodes.csv", right["nodes"])
    presence = node_presence_diff(left["nodes"], right["nodes"])
    write_csv(output_dir / "nodes_only_left.csv", presence["only_left"])
    write_csv(output_dir / "nodes_only_right.csv", presence["only_right"])
    write_csv(output_dir / "op_count_diff.csv", report["op_count_diff"])
    write_csv(output_dir / "op_signature_diff.csv", report["op_signature_diff"])
    write_markdown(report, output_dir / "report.md")
    print(f"report={output_dir / 'report.md'}")
    print(json.dumps(
        {
            "left_node_count": left["node_count"],
            "right_node_count": right["node_count"],
            "op_count_diff": report["op_count_diff"],
            "left_special": left["special_op_counts"],
            "right_special": right["special_op_counts"],
            "left_conv_summary": left["conv_summary"],
            "right_conv_summary": right["conv_summary"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
