"""Ascend 310B OM runtime adapter.

This module keeps AscendCL/ais_bench details behind the small project runtime
contract: ``__call__(tensor) -> list[np.ndarray]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


Shape = tuple[int, ...]


class AscendModel:
    """One-input OM model runner backed by ``ais_bench`` InferSession.

    The MediaPipe hand pipeline expects NHWC float32 tensors on host side.
    Preprocessing is already done in Python, so the converted OM should not use
    AIPP. Outputs are returned as host ``numpy.ndarray`` objects.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device_id: int = 0,
        mode: str = "static",
        expected_output_shapes: Sequence[Sequence[int]] | None = None,
        output_indices: Sequence[int] | None = None,
        acl_json_path: str | Path | None = None,
        debug: bool = False,
        loop: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.device_id = int(device_id)
        self.mode = mode
        self.expected_output_shapes = _normalize_shapes(expected_output_shapes)
        self.output_indices = None if output_indices is None else tuple(int(i) for i in output_indices)
        if not self.model_path.exists():
            raise FileNotFoundError(f"OM model does not exist: {self.model_path}")

        try:
            from ais_bench.infer.interface import InferSession
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import ais_bench. On the 310B board, install and source "
                "the CANN/AIT environment so 'ais_bench' and 'aclruntime' are available."
            ) from exc

        kwargs: dict[str, object] = {
            "device_id": self.device_id,
            "model_path": str(self.model_path),
            "debug": bool(debug),
            "loop": int(loop),
        }
        if acl_json_path is not None:
            kwargs["acl_json_path"] = str(acl_json_path)
        self.session = _make_infer_session(InferSession, kwargs)
        self.inputs = _safe_call(self.session, "get_inputs")
        self.outputs = _safe_call(self.session, "get_outputs")

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        input_tensor = np.ascontiguousarray(np.asarray(tensor, dtype=np.float32))
        outputs = _infer(self.session, [input_tensor], self.mode)
        arrays = _as_array_list(outputs)
        return self._normalize_outputs(arrays)

    def close(self) -> None:
        free_resource = getattr(self.session, "free_resource", None)
        if callable(free_resource):
            free_resource()

    def __enter__(self) -> "AscendModel":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _normalize_outputs(self, outputs: list[np.ndarray]) -> list[np.ndarray]:
        if self.output_indices is not None:
            try:
                outputs = [outputs[index] for index in self.output_indices]
            except IndexError as exc:
                raise ValueError(
                    f"Output index mapping {self.output_indices} is incompatible "
                    f"with {len(outputs)} OM outputs."
                ) from exc

        if self.expected_output_shapes is None:
            return [np.asarray(item) for item in outputs]
        return _reorder_by_shape(outputs, self.expected_output_shapes)


def _safe_call(obj: object, method_name: str) -> object | None:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _make_infer_session(infer_session_cls: object, kwargs: dict[str, object]) -> object:
    try:
        return infer_session_cls(**kwargs)  # type: ignore[misc]
    except TypeError:
        positional = [
            kwargs["device_id"],
            kwargs["model_path"],
        ]
        return infer_session_cls(*positional)  # type: ignore[operator]


def _infer(session: object, feeds: list[np.ndarray], mode: str) -> object:
    infer = getattr(session, "infer")
    try:
        return infer(feeds=feeds, mode=mode, out_array=True)
    except TypeError:
        try:
            return infer(feeds=feeds, mode=mode)
        except TypeError:
            return infer(feeds, mode)


def _as_array_list(outputs: object) -> list[np.ndarray]:
    if isinstance(outputs, np.ndarray):
        return [outputs]
    if isinstance(outputs, (list, tuple)):
        return [np.asarray(item) for item in outputs]
    return [np.asarray(outputs)]


def _normalize_shapes(shapes: Sequence[Sequence[int]] | None) -> tuple[Shape, ...] | None:
    if shapes is None:
        return None
    return tuple(tuple(int(dim) for dim in shape) for shape in shapes)


def _shape_matches(actual: Iterable[int], expected: Shape) -> bool:
    actual_tuple = tuple(int(dim) for dim in actual)
    return actual_tuple == expected


def _reorder_by_shape(outputs: list[np.ndarray], expected_shapes: tuple[Shape, ...]) -> list[np.ndarray]:
    if len(outputs) != len(expected_shapes):
        raise ValueError(
            f"Expected {len(expected_shapes)} OM outputs, got {len(outputs)}: "
            f"{[tuple(item.shape) for item in outputs]}"
        )
    if all(_shape_matches(output.shape, expected) for output, expected in zip(outputs, expected_shapes)):
        return [np.asarray(item) for item in outputs]

    unused = list(range(len(outputs)))
    ordered: list[np.ndarray] = []
    for expected in expected_shapes:
        match_pos = next((pos for pos in unused if _shape_matches(outputs[pos].shape, expected)), None)
        if match_pos is None:
            raise ValueError(
                f"Cannot map OM outputs {[tuple(item.shape) for item in outputs]} "
                f"to expected shapes {list(expected_shapes)}."
            )
        ordered.append(np.asarray(outputs[match_pos]))
        unused.remove(match_pos)
    return ordered
