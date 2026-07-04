"""Inference wrappers for TFLite, ONNX, and Ascend OM backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TfliteModel:
    """Small LiteRT wrapper for one-input MediaPipe TFLite models."""

    def __init__(self, model_path: str | Path, num_threads: int = 1) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.model_path = Path(model_path)
        self.interpreter = Interpreter(model_path=str(self.model_path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        input_index = self.inputs[0]["index"]
        self.interpreter.set_tensor(input_index, tensor)
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(item["index"]) for item in self.outputs]


class OnnxModel:
    """Small ONNX Runtime wrapper for one-input MediaPipe ONNX models."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

    def __call__(self, tensor: np.ndarray) -> list[np.ndarray]:
        input_name = self.inputs[0].name
        output_names = [item.name for item in self.outputs]
        return [np.asarray(item) for item in self.session.run(output_names, {input_name: tensor})]


class AclRuntime:
    """Process-level ACL runtime context."""

    def __init__(self, device_id: int = 0) -> None:
        import acl

        self.acl = acl
        self.device_id = device_id
        self.initialized = False
        self.context = None
        self.acl.init()
        self.initialized = True
        self._check("acl.rt.set_device", self.acl.rt.set_device(device_id))
        self.context, ret = self.acl.rt.create_context(device_id)
        self._check("acl.rt.create_context", ret)

    @staticmethod
    def _check(name: str, ret: int) -> None:
        if ret != 0:
            raise RuntimeError(f"{name} failed, ret={ret}")

    def close(self) -> None:
        if self.context is not None:
            self._check("acl.rt.destroy_context", self.acl.rt.destroy_context(self.context))
            self.context = None
        if self.initialized:
            self.acl.rt.reset_device(self.device_id)
            self.acl.finalize()
            self.initialized = False

    def __enter__(self) -> "AclRuntime":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class AclOmModel:
    """Minimal ACL OM model runner for static one-input float MediaPipe models."""

    ACL_MEM_MALLOC_NORMAL_ONLY = 2
    ACL_DEVICE = 0
    ACL_HOST = 1
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2
    ACL_MEMCPY_DEVICE_TO_DEVICE = 3
    ACL_FLOAT = 0
    ACL_FLOAT16 = 1
    ACL_INT8 = 2
    ACL_INT32 = 3
    ACL_UINT8 = 4
    ACL_INT16 = 6
    ACL_UINT16 = 7
    ACL_UINT32 = 8
    ACL_INT64 = 9
    ACL_UINT64 = 10
    ACL_DOUBLE = 11
    ACL_BOOL = 12

    DTYPE_MAP = {
        ACL_FLOAT: np.float32,
        ACL_FLOAT16: np.float16,
        ACL_INT8: np.int8,
        ACL_INT32: np.int32,
        ACL_UINT8: np.uint8,
        ACL_INT16: np.int16,
        ACL_UINT16: np.uint16,
        ACL_UINT32: np.uint32,
        ACL_INT64: np.int64,
        ACL_UINT64: np.uint64,
        ACL_DOUBLE: np.float64,
        ACL_BOOL: np.uint8,
    }

    def __init__(self, model_path: str | Path) -> None:
        import acl

        self.acl = acl
        self.model_path = Path(model_path)
        self.model_id: int | None = None
        self.model_desc = None
        self._output_shapes: list[tuple[int, ...]] | None = None
        self._output_dtypes: list[np.dtype] | None = None

        run_mode, ret = self.acl.rt.get_run_mode()
        self._check("acl.rt.get_run_mode", ret)
        self.run_mode = int(run_mode)
        # The alignment scripts pass normal numpy host buffers. Keep the copy
        # direction explicit instead of following AclLite's device-mode shortcut,
        # which is only valid when the input/output numpy pointer already lives
        # in device-visible memory.
        self.input_copy_policy = self.ACL_MEMCPY_HOST_TO_DEVICE
        self.output_copy_policy = self.ACL_MEMCPY_DEVICE_TO_HOST

        self.model_id, ret = self.acl.mdl.load_from_file(str(self.model_path))
        self._check("acl.mdl.load_from_file", ret)
        self.model_desc = self.acl.mdl.create_desc()
        ret = self.acl.mdl.get_desc(self.model_desc, self.model_id)
        self._check("acl.mdl.get_desc", ret)

    @staticmethod
    def _check(name: str, ret: int) -> None:
        if ret != 0:
            raise RuntimeError(f"{name} failed, ret={ret}")

    def _create_output_dataset(self) -> tuple[object, list[tuple[int, object, int]]]:
        dataset = self.acl.mdl.create_dataset()
        buffers: list[tuple[int, object, int]] = []
        output_count = self.acl.mdl.get_num_outputs(self.model_desc)
        for index in range(output_count):
            size = int(self.acl.mdl.get_output_size_by_index(self.model_desc, index))
            device_ptr, ret = self.acl.rt.malloc(size, self.ACL_MEM_MALLOC_NORMAL_ONLY)
            self._check("acl.rt.malloc(output)", ret)
            data_buffer = self.acl.create_data_buffer(device_ptr, size)
            _, ret = self.acl.mdl.add_dataset_buffer(dataset, data_buffer)
            self._check("acl.mdl.add_dataset_buffer(output)", ret)
            buffers.append((device_ptr, data_buffer, size))
        return dataset, buffers

    def _create_input_dataset(self, tensors: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> tuple[object, list[tuple[int, object]]]:
        if isinstance(tensors, np.ndarray):
            tensor_list = [tensors]
        else:
            tensor_list = list(tensors)
        input_count = int(self.acl.mdl.get_num_inputs(self.model_desc))
        if len(tensor_list) != input_count:
            raise ValueError(f"Input count mismatch: tensors={len(tensor_list)}, OM expects={input_count}")

        dataset = self.acl.mdl.create_dataset()
        buffers: list[tuple[int, object]] = []
        for index, tensor in enumerate(tensor_list):
            tensor = np.ascontiguousarray(tensor)
            size = int(tensor.nbytes)
            expected_size = int(self.acl.mdl.get_input_size_by_index(self.model_desc, index))
            if size != expected_size:
                raise ValueError(f"Input {index} size mismatch: tensor={size} bytes, OM expects={expected_size} bytes")
            host_ptr = self.acl.util.numpy_to_ptr(tensor)
            device_ptr, ret = self.acl.rt.malloc(size, self.ACL_MEM_MALLOC_NORMAL_ONLY)
            self._check("acl.rt.malloc(input)", ret)
            ret = self.acl.rt.memcpy(device_ptr, size, host_ptr, size, self.input_copy_policy)
            self._check("acl.rt.memcpy(input)", ret)
            data_buffer = self.acl.create_data_buffer(device_ptr, size)
            _, ret = self.acl.mdl.add_dataset_buffer(dataset, data_buffer)
            self._check("acl.mdl.add_dataset_buffer(input)", ret)
            buffers.append((device_ptr, data_buffer))
        return dataset, buffers

    def _destroy_input_dataset(self, dataset: object, buffers: list[tuple[int, object]]) -> None:
        for device_ptr, data_buffer in buffers:
            self.acl.destroy_data_buffer(data_buffer)
            self.acl.rt.free(device_ptr)
        self.acl.mdl.destroy_dataset(dataset)

    def output_shapes(self) -> list[tuple[int, ...]]:
        if self._output_shapes is not None:
            return self._output_shapes
        shapes: list[tuple[int, ...]] = []
        output_count = self.acl.mdl.get_num_outputs(self.model_desc)
        for index in range(output_count):
            dims, ret = self.acl.mdl.get_output_dims(self.model_desc, index)
            self._check("acl.mdl.get_output_dims", ret)
            shapes.append(tuple(int(item) for item in dims["dims"]))
        self._output_shapes = shapes
        return shapes

    def output_names(self) -> list[str]:
        names: list[str] = []
        output_count = self.acl.mdl.get_num_outputs(self.model_desc)
        for index in range(output_count):
            get_name = getattr(self.acl.mdl, "get_output_name_by_index", None)
            if get_name is None:
                names.append(str(index))
                continue
            name = get_name(self.model_desc, index)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            names.append(str(name))
        return names

    def output_dtypes(self) -> list[np.dtype]:
        if self._output_dtypes is not None:
            return self._output_dtypes
        dtypes: list[np.dtype] = []
        output_count = self.acl.mdl.get_num_outputs(self.model_desc)
        for index in range(output_count):
            dtype_code = self.acl.mdl.get_output_data_type(self.model_desc, index)
            if dtype_code not in self.DTYPE_MAP:
                raise TypeError(f"Unsupported ACL output dtype code: {dtype_code}")
            dtypes.append(np.dtype(self.DTYPE_MAP[dtype_code]))
        self._output_dtypes = dtypes
        return dtypes

    def __call__(self, tensor: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> list[np.ndarray]:
        input_dataset, input_buffers = self._create_input_dataset(tensor)
        output_dataset, output_buffers = self._create_output_dataset()
        try:
            ret = self.acl.mdl.execute(self.model_id, input_dataset, output_dataset)
            self._check("acl.mdl.execute", ret)
            return self._copy_outputs(output_buffers)
        finally:
            self._destroy_input_dataset(input_dataset, input_buffers)
            self._destroy_output_dataset(output_dataset, output_buffers)

    def _copy_outputs(self, output_buffers: list[tuple[int, object, int]]) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        for index, (device_ptr, _data_buffer, size) in enumerate(output_buffers):
            dtype = self.output_dtypes()[index]
            shape = self.output_shapes()[index]
            output = np.empty(size // dtype.itemsize, dtype=dtype)
            host_ptr = self.acl.util.numpy_to_ptr(output)
            ret = self.acl.rt.memcpy(host_ptr, size, device_ptr, size, self.output_copy_policy)
            self._check("acl.rt.memcpy(output)", ret)
            outputs.append(output.reshape(shape).copy())
        return outputs

    def _destroy_output_dataset(self, dataset: object, buffers: list[tuple[int, object, int]]) -> None:
        for device_ptr, data_buffer, _size in buffers:
            self.acl.destroy_data_buffer(data_buffer)
            self.acl.rt.free(device_ptr)
        self.acl.mdl.destroy_dataset(dataset)

    def close(self) -> None:
        if self.model_desc is not None:
            self.acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        if self.model_id is not None:
            self.acl.mdl.unload(self.model_id)
            self.model_id = None

    def __enter__(self) -> "AclOmModel":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
