#!/usr/bin/env python3
"""Compare raw outputs from one ONNX model and one Ascend OM model.

This is a model-level numerical check. It does not run MediaPipe decode, NMS,
ROI, or landmark postprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
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
from hand_pipeline.om_runtime import PersistentAclModel, PersistentAclRuntime  # noqa: E402


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def shape_from_onnx(model: OnnxModel) -> tuple[int, ...]:
    shape = []
    for dim in model.inputs[0].shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            raise ValueError(f"Cannot infer static ONNX input shape from {model.inputs[0].shape}; pass --shape.")
    return tuple(shape)


def parse_shape(text: str) -> tuple[int, ...]:
    normalized = text.replace("x", ",").replace("X", ",")
    values = [int(item.strip()) for item in normalized.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Invalid shape: {text}")
    return tuple(values)


def make_input(shape: tuple[int, ...], fill: str, seed: int) -> np.ndarray:
    if fill == "zeros":
        return np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else float("nan"),
        "median": statistics.median(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def compare_arrays(onnx_value: np.ndarray, om_value: np.ndarray, eps: float) -> dict[str, float | list[int]]:
    a = np.asarray(onnx_value, dtype=np.float32)
    b = np.asarray(om_value, dtype=np.float32)
    if a.shape != b.shape:
        if a.size != b.size:
            raise ValueError(f"Output size mismatch: ONNX shape={a.shape}, OM shape={b.shape}")
        b = b.reshape(a.shape)
    diff = np.abs(a - b)
    rel = diff / np.maximum(np.abs(a), eps)
    return {
        "shape": list(a.shape),
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "median_abs": float(np.median(diff)),
        "p95_abs": float(np.percentile(diff, 95)),
        "p99_abs": float(np.percentile(diff, 99)),
        "max_rel": float(np.max(rel)),
        "mean_rel": float(np.mean(rel)),
        "p95_rel": float(np.percentile(rel, 95)),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# ONNX vs OM Raw Output Compare",
        "",
        f"- ONNX: `{report['onnx']}`",
        f"- OM: `{report['om']}`",
        f"- samples: `{report['samples']}`",
        f"- input_shape: `{report['input_shape']}`",
        "",
        "## Summary",
        "",
        "| Output | shape | max_abs | mean_abs | p95_abs | max_rel | mean_rel |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["outputs"]:
        lines.append(
            "| {index} | `{shape}` | {max_abs:.9g} | {mean_abs:.9g} | {p95_abs:.9g} | {max_rel:.9g} | {mean_rel:.9g} |".format(
                index=item["index"],
                shape=item["shape"],
                max_abs=item["max_abs"]["max"],
                mean_abs=item["mean_abs"]["mean"],
                p95_abs=item["p95_abs"]["max"],
                max_rel=item["max_rel"]["max"],
                mean_rel=item["mean_rel"]["mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--om", required=True)
    parser.add_argument("--shape", default="", help="Input shape, for example 1,192,192,3.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--fill", choices=["random", "zeros"], default="random")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output-dir", default="runs/onnx_om_raw_compare/test")
    parser.add_argument(
        "--reload-om-each-sample",
        action="store_true",
        help="Create a fresh ACL model for each sample. Useful for detecting model-handle reuse drift.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    onnx_path = resolve(args.onnx)
    om_path = resolve(args.om)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = OnnxModel(onnx_path)
    input_shape = parse_shape(args.shape) if args.shape else shape_from_onnx(onnx_model)
    runtime = PersistentAclRuntime(device_id=args.device_id, finalize_on_release=True)
    om_model = None if args.reload_om_each_sample else PersistentAclModel(om_path, runtime=runtime)

    try:
        for index in range(args.warmup):
            x = make_input(input_shape, args.fill, args.seed + index)
            onnx_model(x)
            if args.reload_om_each_sample:
                sample_model = PersistentAclModel(om_path, runtime=runtime)
                try:
                    sample_model.infer(x)
                finally:
                    sample_model.release()
            else:
                assert om_model is not None
                om_model.infer(x)

        per_output: list[dict[str, list[float] | list[int]]] = []
        for sample_index in range(args.samples):
            x = make_input(input_shape, args.fill, args.seed + 1000 + sample_index)
            onnx_outputs = onnx_model(x)
            if args.reload_om_each_sample:
                sample_model = PersistentAclModel(om_path, runtime=runtime)
                try:
                    om_outputs = sample_model.infer(x)
                finally:
                    sample_model.release()
            else:
                assert om_model is not None
                om_outputs = om_model.infer(x)
            if len(onnx_outputs) != len(om_outputs):
                raise ValueError(f"Output count mismatch: ONNX={len(onnx_outputs)}, OM={len(om_outputs)}")
            if not per_output:
                per_output = [
                    {
                        "shape": list(np.asarray(onnx_outputs[index]).shape),
                        "max_abs": [],
                        "mean_abs": [],
                        "median_abs": [],
                        "p95_abs": [],
                        "p99_abs": [],
                        "max_rel": [],
                        "mean_rel": [],
                        "p95_rel": [],
                    }
                    for index in range(len(onnx_outputs))
                ]
            for index, (onnx_value, om_value) in enumerate(zip(onnx_outputs, om_outputs)):
                stats = compare_arrays(onnx_value, om_value, eps=args.eps)
                for key, value in stats.items():
                    if key == "shape":
                        per_output[index]["shape"] = value  # type: ignore[index]
                    else:
                        per_output[index][key].append(float(value))  # type: ignore[index, union-attr]

        outputs = []
        for index, item in enumerate(per_output):
            outputs.append(
                {
                    "index": index,
                    "shape": item["shape"],
                    "max_abs": summarize(item["max_abs"]),  # type: ignore[arg-type]
                    "mean_abs": summarize(item["mean_abs"]),  # type: ignore[arg-type]
                    "median_abs": summarize(item["median_abs"]),  # type: ignore[arg-type]
                    "p95_abs": summarize(item["p95_abs"]),  # type: ignore[arg-type]
                    "p99_abs": summarize(item["p99_abs"]),  # type: ignore[arg-type]
                    "max_rel": summarize(item["max_rel"]),  # type: ignore[arg-type]
                    "mean_rel": summarize(item["mean_rel"]),  # type: ignore[arg-type]
                    "p95_rel": summarize(item["p95_rel"]),  # type: ignore[arg-type]
                }
            )

        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "onnx": str(onnx_path),
            "om": str(om_path),
            "samples": args.samples,
            "warmup": args.warmup,
            "input_shape": list(input_shape),
            "fill": args.fill,
            "reload_om_each_sample": bool(args.reload_om_each_sample),
            "outputs": outputs,
        }
        json_path = output_dir / "summary.json"
        md_path = output_dir / "report.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report, md_path)
        print(f"report={md_path}")
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
    finally:
        if om_model is not None:
            om_model.release()
        runtime.release()


if __name__ == "__main__":
    raise SystemExit(main())
