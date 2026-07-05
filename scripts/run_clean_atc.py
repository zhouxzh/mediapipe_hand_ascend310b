#!/usr/bin/env python3
"""Run ATC with a clean, single-threaded environment for one ONNX model."""

from __future__ import annotations

import argparse
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


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
    }


def add_python_runtime(env: dict[str, str]) -> None:
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    python_lib_dir = str(Path(sys.executable).resolve().parents[1] / "lib")
    path_parts = [part for part in env.get("PATH", "").split(":") if part]
    ld_parts = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
    if python_bin_dir not in path_parts:
        path_parts.insert(0, python_bin_dir)
    if Path(python_lib_dir).exists() and python_lib_dir not in ld_parts:
        ld_parts.insert(0, python_lib_dir)

    py_parts = [part for part in env.get("PYTHONPATH", "").split(":") if part]
    try:
        import site

        for path in site.getsitepackages():
            if path and Path(path).exists() and path not in py_parts:
                py_parts.insert(0, path)
        user_site = site.getusersitepackages()
        py_parts = [path for path in py_parts if Path(path) != Path(user_site)]
    except Exception:
        pass
    env["PATH"] = ":".join(path_parts)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    env["PYTHONPATH"] = ":".join(py_parts)


def build_env(mode: str, extra_pythonpath: str = "") -> dict[str, str]:
    env = os.environ.copy()
    if mode in ("current", "python_runtime"):
        if mode == "python_runtime":
            add_python_runtime(env)
        if extra_pythonpath:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{extra_pythonpath}:{existing}" if existing else extra_pythonpath
        env.update(
            {
                "TE_PARALLEL_COMPILER": "1",
                "TBE_PARALLEL_COMPILER": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        return env
    if mode != "clean":
        raise ValueError(f"Unsupported env mode: {mode}")

    # ATC/TBE should not pick up conda shared libraries such as libtinfo.so.
    ld_parts = [
        part
        for part in env.get("LD_LIBRARY_PATH", "").split(":")
        if part and "miniconda" not in part and "anaconda" not in part and "conda" not in part
    ]
    path_parts = [
        part
        for part in env.get("PATH", "").split(":")
        if part and "miniconda" not in part and "anaconda" not in part and "conda" not in part
    ]
    for required in ["/usr/local/Ascend/ascend-toolkit/latest/bin", "/usr/bin", "/bin"]:
        if required not in path_parts:
            path_parts.insert(0, required)

    py_parts = [
        part
        for part in env.get("PYTHONPATH", "").split(":")
        if part and "miniconda" not in part and "anaconda" not in part and "conda" not in part
    ]
    if extra_pythonpath:
        py_parts.insert(0, extra_pythonpath)

    env.update(
        {
            "PATH": ":".join(path_parts),
            "LD_LIBRARY_PATH": ":".join(ld_parts),
            "PYTHONPATH": ":".join(py_parts),
            "TE_PARALLEL_COMPILER": "1",
            "TBE_PARALLEL_COMPILER": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def build_command(args: argparse.Namespace, output_stem: Path) -> list[str]:
    atc = shutil.which(args.atc) or args.atc
    command = [
        atc,
        f"--model={resolve(args.model)}",
        "--framework=5",
        f"--output={output_stem}",
        f"--input_format={args.input_format}",
        f"--input_shape={args.input_shape}",
        f"--soc_version={args.soc_version}",
        f"--op_compiler_cache_mode={args.cache_mode}",
    ]
    if args.use_graph_parallel_options:
        command.extend(["--enable_graph_parallel=0", "--ac_parallel_enable=0"])
    if args.precision_mode:
        command.append(f"--precision_mode={args.precision_mode}")
    if args.extra_option:
        command.extend(args.extra_option)

    if not args.no_nice:
        nice_bin = shutil.which("nice")
        if nice_bin:
            command = [nice_bin, "-n", str(args.nice), *command]
    if not args.no_taskset:
        taskset = shutil.which("taskset")
        if taskset:
            command = [taskset, "-c", args.taskset_cpu, *command]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="Output stem, without .om.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--atc", default="atc")
    parser.add_argument("--input-format", default="ND")
    parser.add_argument("--input-shape", default="input_1:1,192,192,3")
    parser.add_argument("--soc-version", default="Ascend310B1")
    parser.add_argument("--precision-mode", default="")
    parser.add_argument("--cache-mode", choices=["enable", "force", "disable"], default="force")
    parser.add_argument(
        "--use-graph-parallel-options",
        action="store_true",
        help="Pass --enable_graph_parallel=0 and --ac_parallel_enable=0 for older ATC versions that accept them.",
    )
    parser.add_argument("--taskset-cpu", default="0")
    parser.add_argument("--nice", type=int, default=19)
    parser.add_argument("--no-taskset", action="store_true")
    parser.add_argument("--no-nice", action="store_true")
    parser.add_argument("--extra-pythonpath", default="")
    parser.add_argument(
        "--pycache-prefix",
        default=str(ROOT / "runs/atc_pycache"),
        help="Redirect Python bytecode cache lookup/writes during ATC/TBE imports.",
    )
    parser.add_argument("--env-mode", choices=["clean", "current", "python_runtime"], default="clean")
    parser.add_argument("--extra-option", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve(args.model)
    output_stem = resolve(args.output)
    output_om = output_stem.with_suffix(".om")
    log_path = resolve(args.log)
    report_path = resolve(args.report) if args.report else log_path.with_suffix(".json")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(args.env_mode, args.extra_pythonpath)
    if args.pycache_prefix:
        pycache_prefix = resolve(args.pycache_prefix)
        pycache_prefix.mkdir(parents=True, exist_ok=True)
        env["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)
    command = build_command(args, output_stem)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("[created_at] " + datetime.now().isoformat(timespec="seconds") + "\n")
        log.write("[command] " + " ".join(command) + "\n")
        log.write("[PATH] " + env.get("PATH", "") + "\n")
        log.write("[LD_LIBRARY_PATH] " + env.get("LD_LIBRARY_PATH", "") + "\n")
        log.write("[PYTHONPATH] " + env.get("PYTHONPATH", "") + "\n")
        log.write("[PYTHONPYCACHEPREFIX] " + env.get("PYTHONPYCACHEPREFIX", "") + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path),
        "output_stem": str(output_stem),
        "log": str(log_path),
        "command": command,
        "env_mode": args.env_mode,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "om": model_info(output_om),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
