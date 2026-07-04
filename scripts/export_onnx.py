#!/usr/bin/env python3
"""Export selected MediaPipe TFLite models to ONNX with tf2onnx."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.convert import available_export_groups
from hand_pipeline.convert import onnx_model_info
from hand_pipeline.convert import resolve_project_path
from hand_pipeline.convert import select_export_specs
from hand_pipeline.convert import strip_unused_onnx_opsets
from hand_pipeline.convert import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        choices=available_export_groups(),
        default=None,
        help="Model group to export. Defaults to legacy_full. Repeat to export more groups.",
    )
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--output-report", default="models/onnx/export_report.json")
    parser.add_argument("--no-clean-opsets", action="store_true")
    return parser.parse_args()


def run_tf2onnx(tflite_path: Path, onnx_path: Path, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--tflite",
        str(tflite_path),
        "--output",
        str(onnx_path),
        "--opset",
        str(opset),
    ]
    print(f"[tf2onnx] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    specs = select_export_specs(args.group)
    report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "opset": args.opset,
        "models": [],
    }

    for spec in specs:
        tflite_path = resolve_project_path(PROJECT_ROOT, spec.tflite)
        onnx_path = resolve_project_path(PROJECT_ROOT, spec.onnx)
        if not tflite_path.exists():
            raise FileNotFoundError(f"Cannot find TFLite model: {tflite_path}")

        run_tf2onnx(tflite_path, onnx_path, args.opset)
        cleanup = {"changed": False, "before": [], "after": []}
        if not args.no_clean_opsets:
            cleanup = strip_unused_onnx_opsets(onnx_path)

        item = {
            "key": spec.key,
            "group": spec.group,
            "role": spec.role,
            "tflite": str(tflite_path),
            "onnx_cleanup": cleanup,
            "onnx": onnx_model_info(onnx_path),
        }
        report["models"].append(item)
        print(
            f"[done] {spec.key}: {onnx_path.name} sha256={item['onnx']['sha256']}",
            flush=True,
        )

    report_path = resolve_project_path(PROJECT_ROOT, Path(args.output_report))
    write_json(report_path, report)
    print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
