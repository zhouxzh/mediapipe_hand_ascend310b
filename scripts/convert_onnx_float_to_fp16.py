#!/usr/bin/env python3
"""Convert an ONNX model's float tensors to float16.

This is intentionally small and dependency-light so it can run on the Ascend
board with the installed ``onnx`` package. It rewrites FLOAT graph inputs,
outputs, value_info entries, initializers, and Constant tensors to FLOAT16.
Integer attributes such as Resize sizes, Slice indices, and shape constants are
left unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper


def convert_value_info(value_info: Any) -> bool:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type != TensorProto.FLOAT:
        return False
    tensor_type.elem_type = TensorProto.FLOAT16
    return True


def convert_tensor(tensor: TensorProto) -> bool:
    if tensor.data_type != TensorProto.FLOAT:
        return False
    array = numpy_helper.to_array(tensor).astype(np.float16)
    tensor.CopyFrom(numpy_helper.from_array(array, name=tensor.name))
    return True


def convert_constant_attribute(attribute: Any) -> bool:
    if attribute.type != onnx.AttributeProto.TENSOR:
        return False
    return convert_tensor(attribute.t)


def convert_model(input_path: Path, output_path: Path) -> dict[str, Any]:
    model = onnx.load(input_path)
    counts = {
        "graph_inputs": 0,
        "graph_outputs": 0,
        "value_info": 0,
        "initializers": 0,
        "constant_attributes": 0,
    }
    for item in model.graph.input:
        counts["graph_inputs"] += int(convert_value_info(item))
    for item in model.graph.output:
        counts["graph_outputs"] += int(convert_value_info(item))
    for item in model.graph.value_info:
        counts["value_info"] += int(convert_value_info(item))
    for initializer in model.graph.initializer:
        counts["initializers"] += int(convert_tensor(initializer))
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            counts["constant_attributes"] += int(convert_constant_attribute(attribute))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    try:
        onnx.checker.check_model(output_path)
        checker = "passed"
    except Exception as exc:  # pragma: no cover - checker messages are reported to users.
        checker = f"failed: {exc}"
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_size_bytes": input_path.stat().st_size,
        "output_size_bytes": output_path.stat().st_size,
        "counts": counts,
        "checker": checker,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = convert_model(Path(args.input), Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
