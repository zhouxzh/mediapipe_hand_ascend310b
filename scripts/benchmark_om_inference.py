#!/usr/bin/env python3
"""Benchmark Ascend OM inference latency on the target board.

The primary metric is ``execute_ms``: a warmed-up synchronous
``acl.mdl.execute`` call using persistent device buffers. ``full_ms`` measures
the current Python runner path, including host/device copies and output
conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_pipeline.om_runtime import (  # noqa: E402
    ACL_MEMCPY_HOST_TO_DEVICE,
    PersistentAclModel,
    PersistentAclRuntime,
    _check_ret,
)


DEFAULT_MODELS = [
    ROOT / "models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om",
    ROOT / "models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(command: str, timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return f"unavailable: {exc}"
    return completed.stdout.strip()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def summarize_ms(samples: list[float]) -> dict[str, float]:
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples) if samples else float("nan"),
        "median": statistics.median(samples) if samples else float("nan"),
        "min": min(samples) if samples else float("nan"),
        "max": max(samples) if samples else float("nan"),
        "p90": percentile(samples, 0.90),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
    }


def dtype_name(dtype: np.dtype) -> str:
    return np.dtype(dtype).name


def infer_input_array(model: PersistentAclModel, fill: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if len(model.input_buffers) != 1:
        raise ValueError(f"Only one-input OM models are supported, got {len(model.input_buffers)} inputs")

    input_size = int(model.input_buffers[0]["size"])
    shape = model.input_shapes[0]
    if shape is not None:
        element_count = int(np.prod(shape))
        if element_count * np.dtype(np.float32).itemsize == input_size:
            dtype = np.dtype(np.float32)
        elif element_count * np.dtype(np.float16).itemsize == input_size:
            dtype = np.dtype(np.float16)
        elif element_count * np.dtype(np.uint8).itemsize == input_size:
            dtype = np.dtype(np.uint8)
        else:
            raise ValueError(f"Input shape {shape} does not match input size {input_size} bytes")
    elif input_size % np.dtype(np.float32).itemsize == 0:
        dtype = np.dtype(np.float32)
        shape = (input_size // np.dtype(np.float32).itemsize,)
    else:
        dtype = np.dtype(np.uint8)
        shape = (input_size,)

    if fill == "random":
        rng = np.random.default_rng(seed)
        if np.issubdtype(dtype, np.floating):
            array = rng.random(shape, dtype=np.float32).astype(dtype)
        else:
            array = rng.integers(0, 255, size=shape, dtype=dtype)
    else:
        array = np.zeros(shape, dtype=dtype)

    return np.ascontiguousarray(array), {
        "shape": list(shape),
        "dtype": dtype_name(dtype),
        "bytes": input_size,
        "fill": fill,
    }


def prepare_input_copy(model: PersistentAclModel, input_array: np.ndarray) -> tuple[bytes, object, int]:
    input_buffer = model.input_buffers[0]
    input_size = int(input_buffer["size"])
    input_bytes = model._prepare_input_bytes(input_array, input_size)
    host_input_ptr = model.acl.util.bytes_to_ptr(input_bytes)
    return input_bytes, host_input_ptr, input_size


def copy_input_to_device(model: PersistentAclModel, host_input_ptr: object, input_size: int) -> None:
    input_buffer = model.input_buffers[0]
    _check_ret(
        model.acl.rt.memcpy(
            input_buffer["ptr"],
            input_size,
            host_input_ptr,
            input_size,
            ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "acl.rt.memcpy host_to_device",
    )


def execute_model(model: PersistentAclModel) -> None:
    _check_ret(
        model.acl.mdl.execute(model.model_id, model.input_dataset, model.output_dataset),
        "acl.mdl.execute",
    )


def timed_loop(iterations: int, body) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        body()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def find_compile_soc_version(model_path: Path) -> str | None:
    sidecars = [
        model_path.with_suffix(".compile.remote.json"),
        Path(str(model_path) + ".compile.remote.json"),
    ]
    for sidecar in sidecars:
        if not sidecar.exists():
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            command = data.get("command") or []
            for item in command:
                text = str(item)
                if text.startswith("--soc_version="):
                    return text.split("=", 1)[1]
        except Exception:
            pass

    report = model_path.parent / "export_report_all.json"
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8-sig"))
            for item in data.get("models", []):
                om = item.get("om", {})
                if Path(str(om.get("path", ""))).name == model_path.name:
                    return str(data.get("soc_version") or "")
        except Exception:
            pass
    return None


def benchmark_model(
    model_path: Path,
    runtime: PersistentAclRuntime,
    warmup: int,
    iterations: int,
    fill: str,
    seed: int,
) -> dict[str, Any]:
    model = PersistentAclModel(model_path, runtime=runtime)
    try:
        input_array, input_info = infer_input_array(model, fill=fill, seed=seed)
        input_bytes, host_input_ptr, input_size = prepare_input_copy(model, input_array)
        output_info = [
            {
                "index": index,
                "shape": list(shape) if shape is not None else None,
                "bytes": int(buffer["size"]),
            }
            for index, (shape, buffer) in enumerate(zip(model.output_shapes, model.output_buffers))
        ]

        for _ in range(warmup):
            model.infer(input_array)

        copy_input_to_device(model, host_input_ptr, input_size)
        execute_only_ms = timed_loop(iterations, lambda: execute_model(model))

        def h2d_execute() -> None:
            copy_input_to_device(model, host_input_ptr, input_size)
            execute_model(model)

        h2d_execute_ms = timed_loop(iterations, h2d_execute)

        full_ms = timed_loop(iterations, lambda: model.infer(input_array))

        # Keep the prepared bytes alive until all copies are done.
        _ = input_bytes
        return {
            "model": str(model_path),
            "model_name": model_path.name,
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
            "compile_soc_version": find_compile_soc_version(model_path),
            "input": input_info,
            "outputs": output_info,
            "warmup": warmup,
            "iterations": iterations,
            "execute_ms": summarize_ms(execute_only_ms),
            "h2d_execute_ms": summarize_ms(h2d_execute_ms),
            "full_ms": summarize_ms(full_ms),
        }
    finally:
        model.release()


def child_command(args: argparse.Namespace, model_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--device-id",
        str(args.device_id),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--fill",
        args.fill,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model_path),
    ]


def newest_report(output_dir: Path) -> Path | None:
    reports = sorted(output_dir.glob("om_benchmark_*.json"), key=lambda path: path.stat().st_mtime)
    return reports[-1] if reports else None


def run_isolated_models(args: argparse.Namespace, model_paths: list[Path], report_path: Path) -> int:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    runtime_info: dict[str, Any] = {}
    child_root = args.output_dir / "_children" / report_path.stem

    for index, model_path in enumerate(model_paths):
        child_dir = child_root / f"{index:02d}_{model_path.stem}"
        child_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["OM_BENCHMARK_CHILD"] = "1"
        completed = subprocess.run(
            child_command(args, model_path, child_dir),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        print(completed.stdout, end="")

        child_report_path = newest_report(child_dir)
        if child_report_path is None:
            failures.append(
                {
                    "model": str(model_path),
                    "error": f"child benchmark exited {completed.returncode} without a report",
                }
            )
            continue

        child_report = json.loads(child_report_path.read_text(encoding="utf-8"))
        if not runtime_info:
            runtime_info = child_report.get("runtime", {})
        results.extend(child_report.get("results", []))
        failures.extend(child_report.get("failures", []))
        if completed.returncode != 0:
            warnings.append(
                {
                    "model": str(model_path),
                    "warning": f"child benchmark exited {completed.returncode} after writing {child_report_path}",
                }
            )

    report = {
        "runtime": runtime_info,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "fill": args.fill,
        "isolated_process_per_model": True,
        "results": results,
        "failures": failures,
        "warnings": warnings,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport={report_path}")
    return 2 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=Path, help="OM model path. Can be repeated.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--fill", choices=["zeros", "random"], default="zeros")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/om_inference_benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.model or DEFAULT_MODELS
    model_paths = [path if path.is_absolute() else ROOT / path for path in models]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"om_benchmark_{timestamp}.json"

    if len(model_paths) > 1 and os.environ.get("OM_BENCHMARK_CHILD") != "1":
        return run_isolated_models(args, model_paths, report_path)

    runtime = PersistentAclRuntime(device_id=args.device_id, finalize_on_release=True)
    try:
        acl = runtime.acl
        runtime_info = {
            "python": sys.executable,
            "acl": getattr(acl, "__file__", ""),
            "soc_name": acl.get_soc_name() if hasattr(acl, "get_soc_name") else "",
            "atc_path": run_text("command -v atc || true"),
            "npu_smi": run_text("npu-smi info 2>/dev/null | head -80"),
        }
        print("== Runtime ==")
        print(json.dumps(runtime_info, ensure_ascii=False, indent=2))

        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for model_path in model_paths:
            print(f"\n== Benchmark {model_path} ==")
            try:
                result = benchmark_model(
                    model_path=model_path,
                    runtime=runtime,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    fill=args.fill,
                    seed=args.seed,
                )
                results.append(result)
                for key in ("execute_ms", "h2d_execute_ms", "full_ms"):
                    summary = result[key]
                    print(
                        f"{key}: mean={summary['mean']:.3f} ms  "
                        f"p95={summary['p95']:.3f} ms  min={summary['min']:.3f} ms  max={summary['max']:.3f} ms"
                    )
            except Exception as exc:
                print(f"FAILED: {exc}")
                failures.append({"model": str(model_path), "error": str(exc)})

        report = {
            "runtime": runtime_info,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "fill": args.fill,
            "results": results,
            "failures": failures,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport={report_path}")
        return 2 if failures else 0
    finally:
        if os.environ.get("OM_BENCHMARK_CHILD") != "1":
            runtime.release()


if __name__ == "__main__":
    exit_code = main()
    if os.environ.get("OM_BENCHMARK_CHILD") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    raise SystemExit(exit_code)
