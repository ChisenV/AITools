import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import tensorrt as trt
from cuda.bindings import driver

from . import check_cuda_errors
from AITools.core.config import Config
from AITools.base.model_def import BaseModelHandler

_TENSORRT_MODEL_SUPPORT_SUFFIX = [".trt", ".engine", ".onnx"]
_TENSORRT_ONNX_SUPPORT_SUFFIX = [".onnx"]
_TENSORRT_ENGINE_DEFAULT_SUFFIX = ".trt"
_TENSORRT_ENGINE_DEFAULT_NAME = "model" + _TENSORRT_ENGINE_DEFAULT_SUFFIX
_TENSORRT_CONFIG_DEFAULT_USE_FP16 = True
_TENSORRT_CONFIG_DEFAULT_EXPLICIT_BATCH = True
_TENSORRT_CONFIG_DEFAULT_BATCH_CONFIG = (1, 4, 8)


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

        self.logger = self.config.get("logger", trt.Logger(trt.Logger.ERROR))
        self.use_fp16 = self.config.get("use_fp16", _TENSORRT_CONFIG_DEFAULT_USE_FP16)
        # The explicit batch mode means that when the network is created,
        # the batch size of the input and output tensors of the network is
        # explicitly specified. This can improve the efficiency of the
        # network, but you need to manually set the batch size.
        self.explicit_batch = self.config.get("explicit_batch", _TENSORRT_CONFIG_DEFAULT_EXPLICIT_BATCH)
        self.batch_config = self.config.get("batch_config", _TENSORRT_CONFIG_DEFAULT_BATCH_CONFIG)

        self.model_file = self._resolve_model_file()
        self.need_build = self.model_file.suffix in _TENSORRT_ONNX_SUPPORT_SUFFIX
        self.model_info = OrderedDict()
        # I/O cache
        self.buffers = OrderedDict()

        # TensorRT
        self.builder = None
        self.network = None
        self.config = None
        self.profile = None
        self.engine = None
        self.context = None
        self.runtime = None

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

    def load(self):
        """Load or build the TensorRT engine to initialize the execution context"""
        if self.need_build:
            self.build_engine()  # Build the engine from ONNX
            self.dump()  # Save the serialized engine
        else:
            self.load_engine()  # Load the serialized engine directly

        # Initializes the execution context
        self.context = self.engine.create_execution_context()

        # Initializes the memory cache
        self._setup_buffers()
        self._initialized = True

    def run(self, data):
        return self.infer(input_data=data)

    def destroy(self):
        if self.context:
            self.context.__del__()
        if self.engine:
            self.engine.__del__()
        if self.runtime:
            self.runtime.__del__()

        # Free GPU memory
        for _, (_, devicePtr, _, _) in self.buffers.items():
            check_cuda_errors(driver.cuMemFree(devicePtr))

        self._initialized = False

    def build_engine(self, batch_config: Union[list, tuple] = None):
        """Build the TensorRT engine from ONNX"""
        min_batch, opt_batch, max_batch = batch_config or self.batch_config
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Analyze the ONNX model
        parser = trt.OnnxParser(self.network, self.logger)
        if not parser.parse_from_file(self.model_file):
            raise RuntimeError(f"Failed to parse ONNX file: {[e.description for e in parser.errors]}")

        # Configure build parameters
        self.config = self.builder.create_builder_config()
        if self.use_fp16 and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)

        # Set the optimization profile
        self.profile = self.builder.create_optimization_profile()
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            input_name = input_tensor.name

            shape = list(input_tensor.shape)
            min_shape = list(shape)
            opt_shape = list(shape)
            max_shape = list(shape)

            if self.explicit_batch:
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

            self.profile.set_shape(input_name, min=tuple(min_shape), opt=tuple(opt_shape), max=tuple(max_shape))
        self.config.add_optimization_profile(self.profile)

        # 构建序列化引擎
        self.engine = self.builder.build_engine(self.network, self.config)
        if self.engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

    def dump(self, path: Path = None):
        """Save the serialized engine file"""
        if path is None:
            path = self.model_file.with_suffix(_TENSORRT_ENGINE_DEFAULT_SUFFIX)
        if path.suffix not in _TENSORRT_MODEL_SUPPORT_SUFFIX:
            if path.is_dir():
                path = path / _TENSORRT_ENGINE_DEFAULT_NAME
            else:
                path = path.with_suffix(_TENSORRT_ENGINE_DEFAULT_SUFFIX)
        with open(path, "wb") as f:
            f.write(self.engine.serialize())

    def load_engine(self, path: Path = None):
        """Load the pre-built TensorRT engine"""
        if path is None:
            path = self.model_file
        with open(path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def _setup_buffers(self):
        """Allocate GPU/CPU memory for input and output"""
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # Allocate GPU memory and CPU memory
            n_byte = trt.volume(shape) * dtype.itemsize
            cuda_err, device_buffer = driver.cuMemAlloc(n_byte)
            check_cuda_errors(cuda_err)
            host_buffer = np.empty(shape, dtype=self.nptype(dtype))

            if is_input:
                self.model_info.setdefault("inputs", []).append({
                    "name": name,
                    "dtype": self.nptype(dtype),
                    "shape": shape,
                    "host": host_buffer,
                    "device": device_buffer,
                    "size": n_byte
                })
            else:
                self.model_info.setdefault("outputs", []).append({
                    "name": name,
                    "dtype": self.nptype(dtype),
                    "shape": shape,
                    "host": host_buffer,
                    "device": device_buffer,
                    "size": n_byte
                })
            self.buffers[name] = [host_buffer.ctypes.data, device_buffer, n_byte, is_input]
            self.context.set_tensor_address(name, device_buffer)

    def infer(self, input_data: Dict[str, np.ndarray] = None) -> dict[Any, Any]:
        # Prepare work before inference
        if input_data is None:
            input_data = {}

        # Copy the input data to the GPU
        for input_info in self.model_info["inputs"]:
            if input_info["name"] not in input_data:
                raise ValueError(f"Missing input data for input tensor {input_info['name']}")
            np.copyto(input_info["host"].ctype.data, input_data[input_info["name"]].ravel())
            check_cuda_errors(driver.cuMemcpyHtoD(
                input_info["device"],
                input_info["host"].ctype.data,
                input_info["size"]
            ))

        self.context.execute_async_v3(0)

        outputs = {}
        for output_info in self.model_info["outputs"]:
            check_cuda_errors(driver.cuMemcpyDtoH(
                output_info["host"].ctype.data,
                output_info["device"],
                output_info["size"]
            ))
            outputs[output_info["name"]] = output_info["host"].copy()
        return outputs

    @staticmethod
    def nptype(trt_type):

        """
        Returns the numpy-equivalent of a TensorRT :class:`DataType` .

        :arg trt_type: The TensorRT data type to convert.

        :returns: The equivalent numpy type.
        """
        return trt.nptype(trt_type)
