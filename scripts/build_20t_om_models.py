#!/usr/bin/env python3
"""Compile deployed MediaPipe hand ONNX models for the current Ascend 310B board.

This script is intended to run on the board after sourcing the CANN environment.
It writes new OM files with a SoC suffix and does not overwrite the existing
deployed OM files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
    }


def detect_soc_version() -> str:
    try:
        import acl  # type: ignore[import-not-found]

        soc = str(acl.get_soc_name())
        if soc:
            return soc
    except Exception:
        pass
    return "Ascend310B1"


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    existing_path = env.get("PATH", "")
    if python_bin_dir and python_bin_dir not in existing_path.split(":"):
        existing_path = f"{python_bin_dir}:{existing_path}" if existing_path else python_bin_dir

    python_lib_dir = str(Path(sys.executable).resolve().parents[1] / "lib")
    existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if Path(python_lib_dir).exists() and python_lib_dir not in existing_ld_library_path.split(":"):
        existing_ld_library_path = (
            f"{python_lib_dir}:{existing_ld_library_path}" if existing_ld_library_path else python_lib_dir
        )

    py_paths: list[str] = []
    try:
        import site

        py_paths.extend(str(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = getattr(site, "getusersitepackages", lambda: "")()
        if user_site:
            py_paths.append(str(user_site))
    except Exception:
        pass
    existing_pythonpath = env.get("PYTHONPATH", "")
    for path in py_paths:
        if path and Path(path).exists() and path not in existing_pythonpath.split(":"):
            existing_pythonpath = f"{path}:{existing_pythonpath}" if existing_pythonpath else path
    env.update(
        {
            "TE_PARALLEL_COMPILER": "1",
            "TBE_PARALLEL_COMPILER": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PATH": existing_path,
            "LD_LIBRARY_PATH": existing_ld_library_path,
            "PYTHONPATH": existing_pythonpath,
        }
    )
    return env


def build_command(
    atc: str,
    model_path: Path,
    output_stem: Path,
    input_shape: str,
    soc_version: str,
    precision_mode: str,
    taskset_cpu: str,
    nice: int,
) -> list[str]:
    command = [
        atc,
        f"--model={model_path}",
        "--framework=5",
        f"--output={output_stem}",
        "--input_format=ND",
        f"--input_shape={input_shape}",
        f"--soc_version={soc_version}",

        "--op_compiler_cache_mode=force",
    ]
    if precision_mode:
        command.append(f"--precision_mode={precision_mode}")

    taskset = shutil.which("taskset")
    nice_bin = shutil.which("nice")
    if taskset and nice_bin:
        command = [taskset, "-c", taskset_cpu, nice_bin, "-n", str(nice), *command]
    return command


def run_atc(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("[command] " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=single_thread_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode, time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soc-version", default="auto", help="Use 'auto' to read acl.get_soc_name().")
    parser.add_argument("--atc", default=os.environ.get("ATC", "atc"))
    parser.add_argument("--suffix", default="", help="Output suffix. Defaults to lower-case soc version.")
    parser.add_argument(
        "--model-set",
        choices=["deployed_full", "legacy_lite", "all"],
        default="deployed_full",
        help="Which ONNX models to compile.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/om")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "runs/atc_20t/logs")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "runs/atc_20t")
    parser.add_argument("--taskset-cpu", default="0")
    parser.add_argument("--nice", type=int, default=19)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    atc = shutil.which(args.atc) or args.atc
    soc_version = detect_soc_version() if args.soc_version == "auto" else args.soc_version
    suffix = args.suffix or soc_version.lower()

    specs = [
        {
            "key": "optimized_palm",
            "model_set": "deployed_full",
            "onnx": ROOT / "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx",
            "output_stem": args.output_dir
            / f"mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype_{suffix}",
            "input_shape": "input_1:1,192,192,3",
            "precision_mode": "must_keep_origin_dtype",
        },
        {
            "key": "legacy_full_landmark",
            "model_set": "deployed_full",
            "onnx": ROOT / "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx",
            "output_stem": args.output_dir / f"mediapipe_legacy_0_10_14_hand_landmark_full_{suffix}",
            "input_shape": "input_1:1,224,224,3",
            "precision_mode": "",
        },
        {
            "key": "legacy_lite_palm",
            "model_set": "legacy_lite",
            "onnx": ROOT / "models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx",
            "output_stem": args.output_dir / f"mediapipe_legacy_0_10_14_palm_detection_lite_{suffix}",
            "input_shape": "input_1:1,192,192,3",
            "precision_mode": "",
        },
        {
            "key": "legacy_lite_landmark",
            "model_set": "legacy_lite",
            "onnx": ROOT / "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx",
            "output_stem": args.output_dir / f"mediapipe_legacy_0_10_14_hand_landmark_lite_{suffix}",
            "input_shape": "input_1:1,224,224,3",
            "precision_mode": "",
        },
    ]
    if args.model_set != "all":
        specs = [spec for spec in specs if spec["model_set"] == args.model_set]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "project_root": str(ROOT),
        "soc_version": soc_version,
        "suffix": suffix,
        "model_set": args.model_set,
        "atc": atc,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "models": [],
    }

    for spec in specs:
        onnx_path = Path(spec["onnx"])
        output_stem = Path(spec["output_stem"])
        output_om = output_stem.with_suffix(".om")
        log_path = args.log_dir / f"{spec['key']}_{suffix}.log"
        if not onnx_path.exists():
            raise FileNotFoundError(f"Cannot find ONNX: {onnx_path}")

        command = build_command(
            atc=atc,
            model_path=onnx_path,
            output_stem=output_stem,
            input_shape=str(spec["input_shape"]),
            soc_version=soc_version,
            precision_mode=str(spec["precision_mode"]),
            taskset_cpu=args.taskset_cpu,
            nice=args.nice,
        )

        item: dict[str, Any] = {
            "key": spec["key"],
            "onnx": str(onnx_path),
            "input_format": "ND",
            "input_shape": spec["input_shape"],
            "precision_mode": spec["precision_mode"],
            "output_stem": str(output_stem),
            "log": str(log_path),
            "command": command,
        }
        if args.skip_existing and output_om.exists():
            item.update({"skipped": True, "returncode": 0, "om": file_info(output_om)})
            report["models"].append(item)
            print(f"[skip] {spec['key']}: {output_om}", flush=True)
            continue

        print(f"[atc] {spec['key']}: {' '.join(command)}", flush=True)
        returncode, elapsed = run_atc(command, log_path)
        item.update(
            {
                "skipped": False,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "om": file_info(output_om),
            }
        )
        report["models"].append(item)
        if returncode != 0:
            print(f"[failed] {spec['key']}: returncode={returncode}, log={log_path}", file=sys.stderr)
            report_path = args.report_dir / f"compile_20t_om_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[report] {report_path}", flush=True)
            return returncode
        print(
            f"[done] {spec['key']}: {output_om.name} elapsed={elapsed:.1f}s sha256={item['om']['sha256']}",
            flush=True,
        )

    report_path = args.report_dir / f"compile_20t_om_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
