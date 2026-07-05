#!/usr/bin/env python3
"""Build the optimized legacy full palm ONNX/OM artifacts.

This is the reproducible entry point for the Ascend 310B legacy full palm
detector fix discovered during OM alignment:

1. rewrite channel-padding residual Pad+Add blocks;
2. rewrite bilinear half-pixel Resize nodes to explicit arithmetic;
3. rewrite fixed 2x2 stride-2 MaxPool nodes to Slice + Max;
4. optionally verify ONNX equivalence against the original ONNX;
5. optionally compile the optimized ONNX to OM with single-threaded ATC.

Run the rewrite/verify step in the local `mediapipe_legacy` environment. Run
the `--compile-om` step on the Ascend 310B board.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import rewrite_palm_bilinear_resize
from scripts import rewrite_palm_downsample_residual
from scripts import rewrite_palm_maxpool_slices


DEFAULT_INPUT_ONNX = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx"
DEFAULT_DOWNSAMPLE_ONNX = "runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_split.onnx"
DEFAULT_RESIZE_ONNX = "runs/palm_om/build/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_slices.onnx"
DEFAULT_OPTIMIZED_ONNX = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx"
DEFAULT_REFERENCE_DIR = "runs/palm_om/legacy_full_palm"
DEFAULT_VERIFY_DIR = "runs/palm_om/legacy_full_palm/onnx_optimized_compare"
DEFAULT_OM_OUTPUT = "models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype"
DEFAULT_ATC_LOG = "runs/palm_om/atc_logs/downsample_resize_maxpool_slices_origin_dtype.log"


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_optimized_onnx(args: argparse.Namespace) -> dict[str, Any]:
    input_onnx = resolve_path(args.input_onnx)
    downsample_onnx = resolve_path(args.downsample_onnx)
    resize_onnx = resolve_path(args.resize_onnx)
    optimized_onnx = resolve_path(args.optimized_onnx)

    downsample_manifest = rewrite_palm_downsample_residual.rewrite_model(input_onnx, downsample_onnx)
    resize_manifest = rewrite_palm_bilinear_resize.rewrite_model(downsample_onnx, resize_onnx)
    if args.skip_maxpool_rewrite:
        if resize_onnx != optimized_onnx:
            optimized_onnx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resize_onnx, optimized_onnx)
        maxpool_manifest: dict[str, Any] = {"rewritten_nodes": 0, "skipped": True}
    else:
        maxpool_manifest = rewrite_palm_maxpool_slices.rewrite_model(resize_onnx, optimized_onnx)

    summary: dict[str, Any] = {
        "task": "build_optimized_legacy_full_palm_onnx",
        "input_onnx": str(input_onnx),
        "downsample_onnx": str(downsample_onnx),
        "resize_onnx": str(resize_onnx),
        "optimized_onnx": str(optimized_onnx),
        "downsample_rewrites": downsample_manifest.get("rewritten_blocks"),
        "resize_rewrites": resize_manifest.get("rewritten_nodes"),
        "maxpool_rewrites": maxpool_manifest.get("rewritten_nodes"),
        "verify": bool(args.verify_onnx),
    }
    write_json(optimized_onnx.with_suffix(".build.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.verify_onnx:
        verify_args = argparse.Namespace(
            input_model=str(input_onnx),
            output_model=str(optimized_onnx),
            reference_dir=args.reference_dir,
            output_dir=args.verify_dir,
            max_images=args.max_images,
        )
        summary["verify_summary"] = rewrite_palm_downsample_residual.verify_onnx_equivalence(verify_args)
    return summary


def atc_command(args: argparse.Namespace) -> list[str]:
    atc = shutil.which("atc")
    if not atc:
        raise RuntimeError("Cannot find `atc`. Run --compile-om on the Ascend 310B board after sourcing CANN env.")

    optimized_onnx = resolve_path(args.optimized_onnx)
    om_output = resolve_path(args.om_output)
    command = [
        atc,
        f"--model={optimized_onnx.as_posix()}",
        "--framework=5",
        f"--output={om_output.as_posix()}",
        "--input_format=ND",
        "--input_shape=input_1:1,192,192,3",
        f"--soc_version={args.soc_version}",
        f"--precision_mode={args.precision_mode}",
        "--op_compiler_cache_mode=force",
    ]
    if args.op_select_implmode:
        command.append(f"--op_select_implmode={args.op_select_implmode}")
    if args.output_type:
        command.append(f"--output_type={args.output_type}")
    if args.single_thread:
        taskset = shutil.which("taskset")
        nice = shutil.which("nice")
        if taskset and nice:
            command = [taskset, "-c", args.taskset_cpu, nice, "-n", str(args.nice), *command]
    return command


def compile_om(args: argparse.Namespace) -> dict[str, Any]:
    log_path = resolve_path(args.atc_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = atc_command(args)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)

    om_path = resolve_path(args.om_output).with_suffix(".om")
    summary = {
        "task": "compile_optimized_legacy_full_palm_om",
        "command": command,
        "returncode": process.returncode,
        "om": str(om_path),
        "atc_log": str(log_path),
        "exists": om_path.exists(),
        "size_bytes": om_path.stat().st_size if om_path.exists() else 0,
    }
    write_json(resolve_path(args.om_output).with_suffix(".compile.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if process.returncode != 0:
        raise SystemExit(process.returncode)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx", default=DEFAULT_INPUT_ONNX)
    parser.add_argument("--downsample-onnx", default=DEFAULT_DOWNSAMPLE_ONNX)
    parser.add_argument("--resize-onnx", default=DEFAULT_RESIZE_ONNX)
    parser.add_argument("--optimized-onnx", default=DEFAULT_OPTIMIZED_ONNX)
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--verify-dir", default=DEFAULT_VERIFY_DIR)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--skip-rewrite", action="store_true", help="Use an existing optimized ONNX.")
    parser.add_argument("--skip-maxpool-rewrite", action="store_true", help="Keep MaxPool nodes in the optimized ONNX.")
    parser.add_argument("--verify-onnx", action="store_true")
    parser.add_argument("--compile-om", action="store_true")
    parser.add_argument("--om-output", default=DEFAULT_OM_OUTPUT)
    parser.add_argument("--atc-log", default=DEFAULT_ATC_LOG)
    parser.add_argument("--soc-version", default="Ascend310B4")
    parser.add_argument("--precision-mode", default="must_keep_origin_dtype")
    parser.add_argument("--op-select-implmode", default="")
    parser.add_argument("--output-type", default="")
    parser.add_argument("--single-thread", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--taskset-cpu", default="0")
    parser.add_argument("--nice", type=int, default=19)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_rewrite:
        build_optimized_onnx(args)
    if args.compile_om:
        compile_om(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
