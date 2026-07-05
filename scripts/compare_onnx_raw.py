#!/usr/bin/env python3
"""Compare raw outputs from two ONNX models on deterministic inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_pipeline.inference import OnnxModel  # noqa: E402


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def parse_shape(text: str) -> tuple[int, ...]:
    values = [int(item.strip()) for item in text.replace("x", ",").replace("X", ",").split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Invalid shape: {text}")
    return tuple(values)


def infer_shape(model: OnnxModel) -> tuple[int, ...]:
    shape = []
    for dim in model.inputs[0].shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(int(dim))
        else:
            raise ValueError(f"Cannot infer static ONNX input shape from {model.inputs[0].shape}; pass --shape.")
    return tuple(shape)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else float("nan"),
        "median": statistics.median(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# ONNX Raw Output Compare",
        "",
        f"- left: `{report['left']}`",
        f"- right: `{report['right']}`",
        f"- samples: `{report['samples']}`",
        f"- input_shape: `{report['input_shape']}`",
        "",
        "| Output | shape | max_abs | mean_abs | p95_abs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in report["outputs"]:
        lines.append(
            "| {index} | `{shape}` | {max_abs:.9g} | {mean_abs:.9g} | {p95_abs:.9g} |".format(
                index=item["index"],
                shape=item["shape"],
                max_abs=item["max_abs"]["max"],
                mean_abs=item["mean_abs"]["mean"],
                p95_abs=item["p95_abs"]["max"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--shape", default="")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--output-dir", default="runs/onnx_raw_compare/test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left_path = resolve(args.left)
    right_path = resolve(args.right)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left = OnnxModel(left_path)
    right = OnnxModel(right_path)
    input_shape = parse_shape(args.shape) if args.shape else infer_shape(left)
    rng = np.random.default_rng(args.seed)

    per_output: list[dict[str, Any]] = []
    for _sample_index in range(args.samples):
        x = rng.random(input_shape, dtype=np.float32)
        left_outputs = left(x)
        right_outputs = right(x)
        if len(left_outputs) != len(right_outputs):
            raise ValueError(f"Output count mismatch: left={len(left_outputs)}, right={len(right_outputs)}")
        if not per_output:
            per_output = [
                {
                    "shape": list(np.asarray(value).shape),
                    "max_abs": [],
                    "mean_abs": [],
                    "median_abs": [],
                    "p95_abs": [],
                    "p99_abs": [],
                }
                for value in left_outputs
            ]
        for index, (left_value, right_value) in enumerate(zip(left_outputs, right_outputs)):
            a = np.asarray(left_value, dtype=np.float32)
            b = np.asarray(right_value, dtype=np.float32)
            if a.shape != b.shape:
                if a.size != b.size:
                    raise ValueError(f"Output size mismatch for output {index}: left={a.shape}, right={b.shape}")
                b = b.reshape(a.shape)
            diff = np.abs(a - b)
            per_output[index]["max_abs"].append(float(np.max(diff)))
            per_output[index]["mean_abs"].append(float(np.mean(diff)))
            per_output[index]["median_abs"].append(float(np.median(diff)))
            per_output[index]["p95_abs"].append(float(np.percentile(diff, 95)))
            per_output[index]["p99_abs"].append(float(np.percentile(diff, 99)))

    outputs = []
    for index, item in enumerate(per_output):
        outputs.append(
            {
                "index": index,
                "shape": item["shape"],
                "max_abs": summarize(item["max_abs"]),
                "mean_abs": summarize(item["mean_abs"]),
                "median_abs": summarize(item["median_abs"]),
                "p95_abs": summarize(item["p95_abs"]),
                "p99_abs": summarize(item["p99_abs"]),
            }
        )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "left": str(left_path),
        "right": str(right_path),
        "samples": args.samples,
        "input_shape": list(input_shape),
        "outputs": outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, output_dir / "report.md")
    print(f"report={output_dir / 'report.md'}")
    for item in outputs:
        print(
            "output[{index}] shape={shape} max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} p95_abs={p95_abs:.9g}".format(
                index=item["index"],
                shape=item["shape"],
                max_abs=item["max_abs"]["max"],
                mean_abs=item["mean_abs"]["mean"],
                p95_abs=item["p95_abs"]["max"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
