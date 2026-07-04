#!/usr/bin/env python3
"""Compile ONNX models to Ascend 310B OM models with single-threaded ATC."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.convert import available_export_groups
from hand_pipeline.convert import file_info
from hand_pipeline.convert import resolve_project_path
from hand_pipeline.convert import select_export_specs
from hand_pipeline.convert import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        choices=available_export_groups(),
        default=None,
        help="Model group to compile. Defaults to legacy_full. Repeat to compile more groups.",
    )
    parser.add_argument("--soc-version", default="Ascend310B4")
    parser.add_argument("--atc", default=os.environ.get("ATC", "atc"))
    parser.add_argument("--output-report", default="models/om/export_report.json")
    parser.add_argument("--log-dir", default="runs/atc_logs")
    parser.add_argument("--precision-mode", default="", help="Optional ATC precision_mode, for example force_fp32.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-taskset", action="store_true", help="Do not pin ATC to CPU core 0.")
    parser.add_argument("--no-nice", action="store_true", help="Do not lower ATC scheduling priority.")
    return parser.parse_args()


def build_command(args: argparse.Namespace, model_path: Path, output_stem: Path, input_shape: str, input_format: str) -> list[str]:
    command = [
        args.atc,
        f"--model={model_path}",
        "--framework=5",
        f"--output={output_stem}",
        f"--input_format={input_format}",
        f"--input_shape={input_shape}",
        f"--soc_version={args.soc_version}",
        "--enable_graph_parallel=0",
        "--ac_parallel_enable=0",
        "--op_compiler_cache_mode=enable",
    ]
    if args.precision_mode:
        command.append(f"--precision_mode={args.precision_mode}")
    if not args.no_nice:
        command = ["nice", "-n", "19", *command]
    if not args.no_taskset:
        command = ["taskset", "-c", "0", *command]
    return command


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TE_PARALLEL_COMPILER": "1",
            "TBE_PARALLEL_COMPILER": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def run_atc(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("[command] " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=single_thread_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - start
    return result.returncode, elapsed


def main() -> int:
    args = parse_args()
    specs = select_export_specs(args.group)
    report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "soc_version": args.soc_version,
        "single_thread": {
            "TE_PARALLEL_COMPILER": "1",
            "TBE_PARALLEL_COMPILER": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "taskset": "cpu0" if not args.no_taskset else "disabled",
            "nice": "19" if not args.no_nice else "disabled",
        },
        "models": [],
    }

    log_dir = resolve_project_path(PROJECT_ROOT, Path(args.log_dir))
    for spec in specs:
        model_path = resolve_project_path(PROJECT_ROOT, spec.onnx)
        output_stem = resolve_project_path(PROJECT_ROOT, spec.om_stem)
        output_om = output_stem.with_suffix(".om")
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot find ONNX model: {model_path}")
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{spec.key}_atc.log"
        if args.skip_existing and output_om.exists():
            item = {
                "key": spec.key,
                "group": spec.group,
                "role": spec.role,
                "onnx": str(model_path),
                "input_format": spec.input_format,
                "input_shape": spec.input_shape,
                "log": str(log_path),
                "skipped": True,
                "om": file_info(output_om),
            }
            report["models"].append(item)
            print(f"[skip] {spec.key}: {output_om}", flush=True)
            continue
        command = build_command(args, model_path, output_stem, spec.input_shape, spec.input_format)

        print(f"[atc] {spec.key}: {' '.join(command)}", flush=True)
        returncode, elapsed = run_atc(command, log_path)
        if returncode != 0:
            print(f"[failed] {spec.key}: returncode={returncode}, log={log_path}", file=sys.stderr)
            return returncode
        if not output_om.exists():
            raise FileNotFoundError(f"ATC returned 0 but OM file is missing: {output_om}")

        item = {
            "key": spec.key,
            "group": spec.group,
            "role": spec.role,
            "onnx": str(model_path),
            "input_format": spec.input_format,
            "input_shape": spec.input_shape,
            "log": str(log_path),
            "skipped": False,
            "elapsed_seconds": elapsed,
            "om": file_info(output_om),
        }
        report["models"].append(item)
        print(
            f"[done] {spec.key}: {output_om.name} elapsed={elapsed:.1f}s sha256={item['om']['sha256']}",
            flush=True,
        )

    report_path = resolve_project_path(PROJECT_ROOT, Path(args.output_report))
    write_json(report_path, report)
    print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
