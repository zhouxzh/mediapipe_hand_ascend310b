"""Model conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelExportSpec:
    """A TFLite model and its derived Ascend migration artifacts."""

    key: str
    group: str
    role: str
    tflite: Path
    onnx: Path
    om_stem: Path
    input_shape: str
    input_format: str = "ND"


MODEL_EXPORT_SPECS: tuple[ModelExportSpec, ...] = (
    ModelExportSpec(
        key="legacy_full_palm",
        group="legacy_full",
        role="palm_detector",
        tflite=Path("models/tflite/mediapipe_legacy_0_10_14_palm_detection_full.tflite"),
        onnx=Path("models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx"),
        om_stem=Path("models/om/mediapipe_legacy_0_10_14_palm_detection_full"),
        input_shape="input_1:1,192,192,3",
    ),
    ModelExportSpec(
        key="legacy_full_landmark",
        group="legacy_full",
        role="hand_landmark",
        tflite=Path("models/tflite/mediapipe_legacy_0_10_14_hand_landmark_full.tflite"),
        onnx=Path("models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx"),
        om_stem=Path("models/om/mediapipe_legacy_0_10_14_hand_landmark_full"),
        input_shape="input_1:1,224,224,3",
    ),
    ModelExportSpec(
        key="legacy_lite_palm",
        group="legacy_lite",
        role="palm_detector",
        tflite=Path("models/tflite/mediapipe_legacy_0_10_14_palm_detection_lite.tflite"),
        onnx=Path("models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite.onnx"),
        om_stem=Path("models/om/mediapipe_legacy_0_10_14_palm_detection_lite"),
        input_shape="input_1:1,192,192,3",
    ),
    ModelExportSpec(
        key="legacy_lite_landmark",
        group="legacy_lite",
        role="hand_landmark",
        tflite=Path("models/tflite/mediapipe_legacy_0_10_14_hand_landmark_lite.tflite"),
        onnx=Path("models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx"),
        om_stem=Path("models/om/mediapipe_legacy_0_10_14_hand_landmark_lite"),
        input_shape="input_1:1,224,224,3",
    ),
    ModelExportSpec(
        key="task_full_palm",
        group="task_full",
        role="palm_detector",
        tflite=Path("models/tflite/mediapipe_task_hand_detector_full.tflite"),
        onnx=Path("models/onnx/mediapipe_task_hand_detector_full.onnx"),
        om_stem=Path("models/om/mediapipe_task_hand_detector_full"),
        input_shape="input_1:1,192,192,3",
    ),
    ModelExportSpec(
        key="task_full_landmark",
        group="task_full",
        role="hand_landmark",
        tflite=Path("models/tflite/mediapipe_task_hand_landmark_full.tflite"),
        onnx=Path("models/onnx/mediapipe_task_hand_landmark_full.onnx"),
        om_stem=Path("models/om/mediapipe_task_hand_landmark_full"),
        input_shape="input_1:1,224,224,3",
    ),
)

DEFAULT_EXPORT_GROUPS = ("legacy_full",)


def available_export_groups() -> list[str]:
    return sorted({spec.group for spec in MODEL_EXPORT_SPECS} | {"all"})


def select_export_specs(groups: list[str] | tuple[str, ...] | None) -> list[ModelExportSpec]:
    selected_groups = set(groups or DEFAULT_EXPORT_GROUPS)
    if "all" in selected_groups:
        return list(MODEL_EXPORT_SPECS)
    specs = [spec for spec in MODEL_EXPORT_SPECS if spec.group in selected_groups]
    if not specs:
        raise ValueError(f"No model export specs selected for groups: {sorted(selected_groups)}")
    return specs


def resolve_project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _onnx_dim_to_value(dim: Any) -> int | str | None:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return str(dim.dim_param)
    return None


def _onnx_tensor_info(value_info: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    shape: list[int | str | None] = []
    elem_type: int | None = None
    if value_info.type.HasField("tensor_type"):
        elem_type = int(tensor_type.elem_type)
        if tensor_type.HasField("shape"):
            shape = [_onnx_dim_to_value(dim) for dim in tensor_type.shape.dim]
    return {
        "name": value_info.name,
        "elem_type": elem_type,
        "shape": shape,
    }


def onnx_model_info(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    info = file_info(path)
    info.update(
        {
            "opset_import": [
                {"domain": item.domain or "", "version": int(item.version)}
                for item in model.opset_import
            ],
            "node_count": len(model.graph.node),
            "inputs": [_onnx_tensor_info(item) for item in model.graph.input],
            "outputs": [_onnx_tensor_info(item) for item in model.graph.output],
        }
    )
    return info


def strip_unused_onnx_opsets(path: Path) -> dict[str, Any]:
    """Remove unused non-default opset imports added by converters."""

    import onnx

    model = onnx.load(str(path))
    before = [{"domain": item.domain or "", "version": int(item.version)} for item in model.opset_import]
    used_domains = {node.domain or "" for node in model.graph.node}
    kept = [item for item in model.opset_import if (item.domain or "") in used_domains or (item.domain or "") == ""]
    changed = len(kept) != len(model.opset_import)
    if changed:
        del model.opset_import[:]
        model.opset_import.extend(kept)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    after = [{"domain": item.domain or "", "version": int(item.version)} for item in kept]
    return {"changed": changed, "before": before, "after": after}
