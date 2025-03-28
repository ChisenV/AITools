import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
from cuda import cudart
from tensorrt import IBuilderConfig

from AITools.base.model_def import BaseModelHandler


class TensorRTModel(BaseModelHandler):
    """
    TensorRT adapter
    """

    def __init__(
            self,
            model_path: str,
            logger: trt.Logger = None,
            explicit_batch=True,
            use_fp16=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} does not exist.")
        self.model_path = model_path
        self.model_file = os.path.join(model_path, "model.trt")
        try:
            if not os.path.exists(self.model_file):
                raise FileNotFoundError(f"{self.model_file} does not exist.")
        except FileNotFoundError as e:
            if not os.path.exists(os.path.join(model_path, "model.onnx")):
                raise FileNotFoundError(f"{self.model_file} does not exist.")
            self.model_file = os.path.join(model_path, "model.onnx")

        self.need_build = True if self.model_file.endswith(".onnx") else False

        # Buildtime assets
        if logger is None:
            self.logger = trt.Logger(trt.Logger.Severity.ERROR)
        else:
            self.logger = logger

        self.use_fp16 = use_fp16

        self.builder = trt.Builder(self.logger)
        self.network = (self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                        if explicit_batch else self.builder.create_network())
        self.profile = self.builder.create_optimization_profile()
        self.config: IBuilderConfig = self.builder.create_builder_config()

        # Serialized model
        self.engine_bytes = None
        self.tensor_list = []
        self.tensor_name_list = []
        self.n_input = 0
        self.n_output = 0

        # Runtime assets
        self.runtime = None
        self.engine = None
        self.context = None

    def _build(self) -> None:
        parser = trt.OnnxParser(self.network, self.logger)

        # parse ONNX files
        success = parser.parse_from_file(str(self.model_file))  # parse from file

        if not success:
            for i in range(parser.num_errors):  # Get error information
                error = parser.get_error(i)
                print(error)  # Print error information
                print("error.code() = {}".format(error.code()))
                print("error.file() = {}".format(error.file()))
                print("error.func() = {}".format(error.func()))
                print("error.line() = {}".format(error.line()))
            parser.clear_errors()
            raise IOError(f"Fail parsing {self.model_file}")

        if self.use_fp16 and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 is supported on this platform, and will be used.")

        # self.engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        return self.builder.build_serialized_network(self.network, self.config)

    def load(self):
        if self.need_build:
            print("Building TensorRT engine...")
            self.engine_bytes = self._build()

            # serialize the engine to a file
            with open(os.path.join(self.model_path, "model.trt"), "wb") as f:
                f.write(self.engine_bytes)
            print(f"TensorRT engine saved to file: {os.path.join(self.model_path, 'model.trt')}")
        else:
            if not (self.model_file.endswith(".engine") or self.model_file.endswith(".trt")):
                raise IOError(f"{self.model_file} is not a TensorRT engine file.")

            print("Loading TensorRT engine...")
            with open(self.model_file, "rb") as f:
                self.engine_bytes = f.read()

        if self.runtime is None:  # Just in case we already have a runtime from outside
            self.runtime = trt.Runtime(self.logger)
        if self.engine is None:  # Just in case we already have an engine from outside
            self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
        if self.context is None:
            self.context = self.engine.create_execution_context()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            data_type = self.engine.get_tensor_dtype(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.n_input += is_input
            self.n_output += not is_input
            runtime_shape = self.context.get_tensor_shape(name)
            # if not is_input and not len(self.categories) == runtime_shape[1]:
            #     raise ValueError(f"Output '{name}' has wrong shape {runtime_shape}, "
            #                      f"not match to categories {self.categories}")
            n_byte = trt.volume(runtime_shape) * data_type.itemsize
            self.tensor_list.append((name, data_type, runtime_shape, is_input, n_byte))

        print(f"TensorRT engine has {len(self.tensor_list)} tensors: {self.tensor_list}")
        self._initialized = True
        return

    def serialize_engine(self, trt_file: Path):
        # Save engine bytes as TensorRT engine file
        with open(trt_file, "wb") as f:
            f.write(self.engine_bytes)
        return

    def infer(self, input_data=None, b_print_io: bool = False, b_get_timeline: bool = False) -> dict[Any, Any]:
        # Prepare work before inference
        if input_data is None:
            input_data = {}
        self.buffer = OrderedDict()
        for name, data_type, runtime_shape, is_input, n_byte in self.tensor_list:
            host_buffer = np.empty(runtime_shape, dtype=self.nptype(data_type))
            cuda_err, device_buffer = cudart.cudaMalloc(n_byte)
            self.buffer[name] = [host_buffer, device_buffer, n_byte]

        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data["data"])  # host_buffer

        for name, data_type, runtime_shape, is_input, n_byte in self.tensor_list:
            self.context.set_tensor_address(name, self.buffer[name][1])  # device_buffer
            if is_input:
                cudart.cudaMemcpy(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2],
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        self.context.execute_async_v3(0)

        outputs = {}
        for name, _, _, is_input, _ in self.tensor_list:
            if not is_input:
                cudart.cudaMemcpy(self.buffer[name][0].ctypes.data, self.buffer[name][1], self.buffer[name][2],
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
                outputs[name] = self.buffer[name][0]

        if b_print_io:
            for name, _, _, _, _ in self.tensor_list:
                print(name)
                print(self.buffer[name][0])

        for _, device_buffer, _ in self.buffer.values():
            cudart.cudaFree(device_buffer)

        return outputs

    @staticmethod
    def nptype(trt_type):

        """
        Returns the numpy-equivalent of a TensorRT :class:`DataType` .

        :arg trt_type: The TensorRT data type to convert.

        :returns: The equivalent numpy type.
        """
        import numpy as np

        limit_version = [1, 20, 0]
        np_ver = [int(i) for i in np.version.version.split(".")]

        if np_ver < limit_version:
            return trt.nptype(trt_type)

        mapping = {
            trt.float32: np.float32,
            trt.float16: np.float16,
            trt.int8: np.int8,
            trt.int32: np.int32,
            trt.bool: np.bool_,
            trt.uint8: np.uint8,
        }

        if trt_type in mapping:
            return mapping[trt_type]

        raise TypeError("Could not resolve TensorRT datatype to an equivalent numpy datatype.")

    def result(self, **kwargs):
        """
        Get the result of the inference.
        Args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError
