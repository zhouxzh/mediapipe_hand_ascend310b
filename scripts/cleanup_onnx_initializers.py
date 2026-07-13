#!/usr/bin/env python3
"""Remove unused ONNX initializers and refresh inferred shapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cleanup_model(input_model: Path, output_model: Path) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(input_model))
    used_inputs = {name for node in model.graph.node for name in node.input if name}
    kept = []
    removed = []
    for initializer in model.graph.initializer:
        if initializer.name in used_inputs:
            kept.append(initializer)
        else:
            removed.append(initializer.name)

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))
    manifest = {
        "task": "cleanup_onnx_initializers",
        "source_model": str(input_model),
        "output_model": str(output_model),
        "removed_initializers": removed,
        "removed_count": len(removed),
        "kept_count": len(kept),
        "nodes": len(model.graph.node),
    }
    output_model.with_suffix(".cleanup.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-model", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = cleanup_model(resolve_path(args.input_model), resolve_path(args.output_model))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
