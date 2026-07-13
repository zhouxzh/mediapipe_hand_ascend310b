#!/usr/bin/env python3
"""Quantize the optimized legacy full palm detector with AMCT ONNX.

Run this on the Ascend 310B board in the base conda environment where
amct_onnx and its ONNX Runtime custom op are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl  # noqa: F401 - preload the intended OpenSSL runtime before pyarrow on Ascend images.
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.preprocess import image_to_tensor  # noqa: E402


DEFAULT_MODEL = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx"
DEFAULT_DATASET = "data/portable-hagridv2-mediapipe-hand/test-00000.parquet"
DEFAULT_OUTPUT_DIR = "runs/amct_palm_int8"
DEFAULT_SAVE_PREFIX = "mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_amct_int8"


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decode_image(image_value: dict[str, Any], image_name: str) -> np.ndarray:
    payload = image_value.get("bytes")
    if not payload:
        raise ValueError(f"Missing embedded JPEG bytes for {image_name}")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode embedded JPEG for {image_name}")
    return image


def selected_indices(total: int, count: int, mode: str) -> np.ndarray:
    if total <= 0:
        raise ValueError("Calibration dataset is empty")
    count = min(int(count), total)
    if count <= 0:
        raise ValueError("--calib-images must be positive")
    if mode == "head":
        return np.arange(count, dtype=np.int64)
    if mode == "even":
        return np.unique(np.linspace(0, total - 1, count, dtype=np.int64))
    raise ValueError(f"Unsupported selection mode: {mode}")


def load_calibration_tensors(
    dataset_path: Path,
    count: int,
    input_size: int,
    selection: str,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    parquet_file = pq.ParquetFile(dataset_path)
    table = parquet_file.read(columns=["image", "file_name", "width", "height"])
    rows = table.to_pylist()
    indices = selected_indices(len(rows), count, selection)
    tensors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for ordinal, row_index in enumerate(indices.tolist()):
        row = rows[row_index]
        image_name = str(row["file_name"])
        image = decode_image(row["image"], image_name)
        width = int(row["width"])
        height = int(row["height"])
        if image.shape[1] != width or image.shape[0] != height:
            raise ValueError(f"{image_name} shape {image.shape[1]}x{image.shape[0]} != metadata {width}x{height}")
        tensor, _ = image_to_tensor(image, input_size=input_size)
        tensors.append(np.ascontiguousarray(tensor.astype(np.float32, copy=False)))
        metadata.append(
            {
                "ordinal": ordinal,
                "dataset_index": int(row_index),
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )
    return tensors, metadata


def parse_skip_layers(text: str) -> list[str] | None:
    names = [item.strip() for item in text.split(",") if item.strip()]
    return names or None


def find_model_io(model_path: Path) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    return {
        "ir_version": model.ir_version,
        "opsets": {item.domain or "": item.version for item in model.opset_import},
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def run_calibration(modified_onnx: Path, tensors: list[np.ndarray]) -> dict[str, Any]:
    from amct_onnx.custom_op.amct_custom_op import AMCT_SO  # pylint: disable=import-error

    if AMCT_SO is None:
        raise RuntimeError("AMCT custom op library is not loaded. Build/install amct_onnx_op before calibration.")

    session = ort.InferenceSession(
        str(modified_onnx),
        sess_options=AMCT_SO,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    start = time.perf_counter()
    for index, tensor in enumerate(tensors, start=1):
        session.run(output_names, {input_name: tensor})
        if index == 1 or index == len(tensors) or index % 50 == 0:
            print(f"[calibration] {index}/{len(tensors)}", flush=True)
    elapsed = time.perf_counter() - start
    return {
        "input_name": input_name,
        "outputs": output_names,
        "images": len(tensors),
        "elapsed_seconds": elapsed,
        "images_per_second": len(tensors) / elapsed if elapsed > 0 else 0.0,
    }


def copy_if_requested(src: Path | None, dst_text: str) -> Path | None:
    if src is None or not dst_text:
        return None
    dst = resolve_path(dst_text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-prefix", default=DEFAULT_SAVE_PREFIX)
    parser.add_argument("--calib-images", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=192)
    parser.add_argument("--selection", choices=["even", "head"], default="even")
    parser.add_argument("--activation-offset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-layers", default="", help="Comma-separated AMCT layer names to leave unquantized.")
    parser.add_argument("--deploy-output", default="")
    parser.add_argument("--fakequant-output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import amct_onnx  # pylint: disable=import-error

    model_path = resolve_path(args.model)
    dataset_path = resolve_path(args.dataset)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "palm_int8_quant_config.json"
    modified_onnx = output_dir / "palm_int8_modified_for_calibration.onnx"
    record_file = output_dir / "palm_int8_calibration_record.txt"
    save_path = output_dir / args.save_prefix
    summary_path = output_dir / "summary.json"

    skip_layers = parse_skip_layers(args.skip_layers)
    print(f"[amct] model={model_path}", flush=True)
    print(f"[amct] dataset={dataset_path}", flush=True)
    print(f"[amct] output_dir={output_dir}", flush=True)

    tensors, calibration_metadata = load_calibration_tensors(
        dataset_path=dataset_path,
        count=args.calib_images,
        input_size=args.input_size,
        selection=args.selection,
    )
    print(f"[amct] loaded calibration tensors: {len(tensors)}", flush=True)

    amct_onnx.create_quant_config(
        str(config_file),
        str(model_path),
        skip_layers=skip_layers,
        batch_num=len(tensors),
        activation_offset=bool(args.activation_offset),
    )
    amct_onnx.quantize_model(
        str(config_file),
        str(model_path),
        str(modified_onnx),
        str(record_file),
    )
    calibration_info = run_calibration(modified_onnx, tensors)
    if not record_file.exists() or record_file.stat().st_size == 0:
        raise RuntimeError(f"Calibration record was not written: {record_file}")

    amct_onnx.save_model(str(modified_onnx), str(record_file), str(save_path))
    deploy_model = output_dir / f"{args.save_prefix}_deploy_model.onnx"
    fakequant_model = output_dir / f"{args.save_prefix}_fake_quant_model.onnx"
    if not deploy_model.exists():
        deploy_candidates = sorted(output_dir.glob("*deploy_model.onnx"))
        deploy_model = deploy_candidates[-1] if deploy_candidates else deploy_model
    if not fakequant_model.exists():
        fakequant_candidates = sorted(output_dir.glob("*fake_quant_model.onnx"))
        fakequant_model = fakequant_candidates[-1] if fakequant_candidates else fakequant_model

    stable_deploy = copy_if_requested(deploy_model if deploy_model.exists() else None, args.deploy_output)
    stable_fakequant = copy_if_requested(fakequant_model if fakequant_model.exists() else None, args.fakequant_output)

    summary = {
        "task": "quantize_palm_amct",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "versions": {
            "amct_onnx": getattr(amct_onnx, "__file__", ""),
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "model": file_info(model_path),
        "model_io": find_model_io(model_path),
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "calibration": {
            "requested_images": args.calib_images,
            "actual_images": len(tensors),
            "input_size": args.input_size,
            "selection": args.selection,
            "activation_offset": bool(args.activation_offset),
            "skip_layers": skip_layers or [],
            **calibration_info,
        },
        "calibration_images": calibration_metadata,
        "artifacts": {
            "config": file_info(config_file),
            "modified_onnx": file_info(modified_onnx),
            "record": file_info(record_file),
            "deploy_model": file_info(deploy_model),
            "fakequant_model": file_info(fakequant_model),
            "stable_deploy_model": file_info(stable_deploy) if stable_deploy else None,
            "stable_fakequant_model": file_info(stable_fakequant) if stable_fakequant else None,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
