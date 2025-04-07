import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import tensorrt as trt
from cuda.bindings import driver

__all__ = [
    "TensorRTModel"
]

from . import check_cuda_errors, int_address_to_ndarray, BACKENDS
from AITools.base.model_def import BaseModelHandler
from AITools.core.config import Config

_TENSORRT_MODEL_SUPPORT_SUFFIX = [".trt", ".engine", ".onnx"]
_TENSORRT_ONNX_SUPPORT_SUFFIX = [".onnx"]
_TENSORRT_ENGINE_DEFAULT_SUFFIX = ".trt"
_TENSORRT_ENGINE_DEFAULT_NAME = "model" + _TENSORRT_ENGINE_DEFAULT_SUFFIX
_TENSORRT_CONFIG_DEFAULT_USE_FP16 = True
_TENSORRT_CONFIG_DEFAULT_EXPLICIT_BATCH = True
_TENSORRT_CONFIG_DEFAULT_BATCH_CONFIG = (1, 4, 8)


@BACKENDS.register_component
class TensorRTModel(BaseModelHandler):
    """
    TensorRT adapter
    """

    _SUPPORTED_EXTENSIONS = _TENSORRT_MODEL_SUPPORT_SUFFIX

    def __init__(
            self,
            model_path: Union[str, Path],
            config: Union[Config, Dict[str, Any]],
            **kwargs
    ):
        super().__init__(model_path=model_path, config=config, **kwargs)
        self._model_file = self._resolve_model_file()
        self._need_build = self._model_file.suffix in _TENSORRT_ONNX_SUPPORT_SUFFIX
        self._model_info = OrderedDict()
        # I/O cache
        self._buffers = OrderedDict()

        # TensorRT
        self.trt_logger = self.config.get("logger", trt.Logger(trt.Logger.ERROR))
        self.trt_builder = None
        self.trt_network = None
        self.trt_config = None
        self.trt_profile = None
        self.trt_engine = None
        self.trt_context = None
        self.trt_runtime = None

    @property
    def info(self):
        return self._model_info

    def _resolve_model_file(self) -> Path:
        """Determine the model file path, preferentially selecting ".trt", ".engine" or ".onnx" files"""
        if self.model_path.is_dir():
            trt_files = [f for f in os.listdir(self.model_path)
                         if f.endswith(tuple(self._SUPPORTED_EXTENSIONS))]
            if trt_files:
                return self.model_path / trt_files[0]
        elif self.model_path.suffix in self._SUPPORTED_EXTENSIONS:
            return self.model_path
        else:
            raise FileNotFoundError(f"No valid model file found in {self.model_path}")

    def load(self, *args, **kwargs):
        """Load or build the TensorRT engine to initialize the execution context"""
        if self._need_build:
            self.build_engine()  # Build the engine from ONNX
            self.dump()  # Save the serialized engine
        else:
            self.load_engine()  # Load the serialized engine directly

        # Initializes the execution context
        self.trt_context = self.trt_engine.create_execution_context()

        # Initializes the memory cache
        self._setup_buffers()
        self._initialized = True

        self.run_hooks("on_model_load_finished", *args, **kwargs)

        return self

    def run(self, data, **kwargs):
        return self.infer(input_data=data, **kwargs)

    def destroy(self, *args, **kwargs):
        if self.trt_context:
            self.trt_context.__del__()
        if self.trt_engine:
            self.trt_engine.__del__()
        if self.trt_runtime:
            self.trt_runtime.__del__()

        # Free GPU memory
        for _, (host_buffer, device_buffer, _, _) in self._buffers.items():
            if isinstance(host_buffer, np.ndarray):
                check_cuda_errors(driver.cuMemHostUnregister(host_buffer.ctypes.data))
                del host_buffer
            else:
                check_cuda_errors(driver.cuMemFreeHost(host_buffer))
            check_cuda_errors(driver.cuMemFree(device_buffer))

        self._initialized = False
        self.run_hooks("on_model_destroy_finished", *args, **kwargs)

    def build_engine(self, batch_config: Union[list, tuple] = None):
        """Build the TensorRT engine from ONNX"""
        min_batch, opt_batch, max_batch = (batch_config
                                           or
                                           self.config.get("batch_config", _TENSORRT_CONFIG_DEFAULT_BATCH_CONFIG))
        self.trt_builder = trt.Builder(self.trt_logger)
        self.trt_network = self.trt_builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Analyze the ONNX model
        parser = trt.OnnxParser(self.trt_network, self.trt_logger)
        if not parser.parse_from_file(str(self._model_file)):
            raise RuntimeError(f"Failed to parse ONNX file: {[e.description for e in parser.errors]}")

        # Configure build parameters
        use_fp16 = self.config.get("use_fp16", _TENSORRT_CONFIG_DEFAULT_USE_FP16)
        self.trt_config = self.trt_builder.create_builder_config()
        if use_fp16 and self.trt_builder.platform_has_fast_fp16:
            self.trt_config.set_flag(trt.BuilderFlag.FP16)

        # Set the optimization profile
        self.trt_profile = self.trt_builder.create_optimization_profile()
        for i in range(self.trt_network.num_inputs):
            input_tensor = self.trt_network.get_input(i)
            input_name = input_tensor.name

            shape = list(input_tensor.shape)
            min_shape = list(shape)
            opt_shape = list(shape)
            max_shape = list(shape)

            # The explicit batch mode means that when the network is created,
            # the batch size of the input and output tensors of the network is
            # explicitly specified. This can improve the efficiency of the
            # network, but you need to manually set the batch size.
            if self.config.get("explicit_batch", _TENSORRT_CONFIG_DEFAULT_EXPLICIT_BATCH):
                if shape[0] == -1:
                    min_shape[0] = min_batch
                    opt_shape[0] = opt_batch
                    max_shape[0] = max_batch
                else:
                    min_shape[0] = shape[0]
                    opt_shape[0] = shape[0]
                    max_shape[0] = shape[0]
            else:
                raise ValueError("Explicit batch mode is required for building TensorRT engine building from onnx.")

            self.trt_profile.set_shape(input_name, min=tuple(min_shape), opt=tuple(opt_shape), max=tuple(max_shape))
        self.trt_config.add_optimization_profile(self.trt_profile)

        # Build a serialization engine
        self.trt_engine = self.trt_builder.build_engine(self.trt_network, self.trt_config)
        if self.trt_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

    def dump(self, path: Path = None):
        """Save the serialized engine file"""
        if path is None:
            path = self._model_file.with_suffix(_TENSORRT_ENGINE_DEFAULT_SUFFIX)
        if path.suffix not in _TENSORRT_MODEL_SUPPORT_SUFFIX:
            if path.is_dir():
                path = path / _TENSORRT_ENGINE_DEFAULT_NAME
            else:
                path = path.with_suffix(_TENSORRT_ENGINE_DEFAULT_SUFFIX)
        with open(path, "wb") as f:
            f.write(self.trt_engine.serialize())

    def load_engine(self, path: Path = None):
        """Load the pre-built TensorRT engine"""
        if path is None:
            path = self._model_file
        with open(path, "rb") as f:
            self.trt_runtime = self.trt_runtime if self.trt_runtime else trt.Runtime(self.trt_logger)
            self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())

    def _setup_buffers(self):
        """Allocate GPU/CPU memory for input and output"""
        for idx in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(idx)
            dtype = self.trt_engine.get_tensor_dtype(name)
            shape = self.trt_engine.get_tensor_shape(name)
            is_input = self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # Allocate CPU memory and GPU memory
            n_byte = trt.volume(shape) * dtype.itemsize
            host_buffer = check_cuda_errors(driver.cuMemAllocHost(n_byte))
            # host_buffer = np.empty(shape, dtype=self.nptype(dtype))
            # check_cuda_errors(driver.cuMemHostRegister(host_buffer.ctypes.data, n_byte, 0))
            device_buffer = check_cuda_errors(driver.cuMemAlloc(n_byte))

            if is_input:
                self._model_info.setdefault("inputs", []).append({
                    "name": name,
                    "dtype": self.nptype(dtype),
                    "shape": shape,
                    "host": host_buffer,
                    "device": device_buffer,
                    "size": n_byte
                })
            else:
                self._model_info.setdefault("outputs", []).append({
                    "name": name,
                    "dtype": self.nptype(dtype),
                    "shape": shape,
                    "host": host_buffer,
                    "device": device_buffer,
                    "size": n_byte
                })
            self._buffers[name] = [host_buffer, device_buffer, n_byte, is_input]
            self.trt_context.set_tensor_address(name, device_buffer)

    def infer(self, input_data: Dict[str, np.ndarray] = None, **kwargs) -> dict[Any, Any]:
        if not self.is_ready:
            raise RuntimeError("The TensorRT engine is not ready for inference")

        # Copy the input data to the GPU
        for input_info in self._model_info["inputs"]:
            if input_info["name"] not in input_data:
                raise ValueError(f"Missing input data for input tensor {input_info['name']}")
            name, dtype, shape, n_byte = input_info["name"], input_info["dtype"], \
                input_info["shape"], input_info["size"]
            host_buffer, device_buffer = input_info["host"], input_info["device"]
            if isinstance(host_buffer, np.ndarray):
                np.copyto(host_buffer, input_data[name])  # TODO move this memory copy operation to preprocess stage
                check_cuda_errors(driver.cuMemcpyHtoD(device_buffer, host_buffer.ctypes.data, n_byte))
            elif isinstance(host_buffer, int):
                np.copyto(int_address_to_ndarray(host_buffer, dtype, shape), input_data[name])
                check_cuda_errors(driver.cuMemcpyHtoD(device_buffer, host_buffer, n_byte))
            else:
                raise TypeError(f"Unsupported type for input tensor {name}")
            self.run_hooks("on_trt_input_memcpy_after", input_info=input_info, **kwargs)

        self.run_hooks("on_trt_infer_before", **kwargs)
        self.trt_context.execute_async_v3(0)
        self.run_hooks("on_trt_infer_after", **kwargs)

        outputs = {}
        for output_info in self._model_info["outputs"]:
            name, dtype, shape, n_byte = output_info["name"], output_info["dtype"], \
                output_info["shape"], output_info["size"]
            host_buffer, device_buffer = output_info["host"], output_info["device"]
            check_cuda_errors(driver.cuMemcpyDtoH(
                host_buffer.ctypes.data if isinstance(host_buffer, np.ndarray) else host_buffer,
                device_buffer,
                n_byte
            ))
            self.run_hooks("on_trt_output_memcpy_after", output_info=output_info, **kwargs)
            # TODO move this memory copy operation to postprocess stage
            outputs[name] = host_buffer.copy() if isinstance(host_buffer, np.ndarray) \
                else int_address_to_ndarray(host_buffer, dtype, shape).copy()
        return outputs

    @staticmethod
    def nptype(trt_type):

        """
        Returns the numpy-equivalent of a TensorRT :class:`DataType` .

        :arg trt_type: The TensorRT data type to convert.

        :returns: The equivalent numpy type.
        """
        return trt.nptype(trt_type)
