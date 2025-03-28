from abc import ABC, abstractmethod
from typing import Any, Dict
from enum import Enum


class ModelType(Enum):
    TRAINING = "training"
    DEPLOYMENT = "deployment"


class FrameworkType(Enum):
    PYTORCH = "pytorch"
    PADDLE = "paddle"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TENSORFLOW = "tensorflow"


# class BaseModel(abc.ABC):
#     """Base class for AI model inference with standardized processing pipeline"""
#
#     def __init__(
#             self,
#             model_type: ModelType,
#             framework: FrameworkType,
#             metadata: Optional[Dict] = None,
#             **kwargs
#     ):
#         """
#         Initialize base AI model
#         """
#         self.framework = framework
#         self._initialized = False
#
#     @abc.abstractmethod
#     def load_model(self) -> None:
#         """Load model and initialize inference engine"""
#         self._initialized = True
#
#     @abc.abstractmethod
#     def destroy(self) -> None:
#         """Release resources and clean up"""
#         self._initialized = False
#
#     @abc.abstractmethod
#     def preprocess(self, input_data: Any, **kwargs) -> Any:
#         """
#         Preprocess input data to model-ready format
#
#         :param input_data: Raw input data (e.g., image path, numpy array)
#         :return: Any
#         """
#         if not self._initialized:
#             raise RuntimeError("Model not initialized. Call load_model() first")
#
#     @abc.abstractmethod
#     def inference(self, inputs: Any, **kwargs) -> Any:
#         """
#         Execute model inference
#
#         :param inputs: Preprocessed input tensors
#         :return: Any
#         """
#         if not self._initialized:
#             raise RuntimeError("Model not initialized. Call load_model() first")
#
#     @abc.abstractmethod
#     def postprocess(self, outputs: Any, **kwargs) -> Any:
#         """
#         Convert raw model outputs to final results
#
#         :param outputs: Raw model outputs
#         :param kwargs: Additional postprocessing parameters
#         :return: Any
#         """
#         if not self._initialized:
#             raise RuntimeError("Model not initialized. Call load_model() first")
#
#     def __enter__(self):
#         self.load_model()
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.destroy()
#
#     def __call__(self, *args, **kwargs):
#         if not self._initialized:
#             raise RuntimeError("Model not initialized. Call load_model() first")
#
#
# class DeployModel(BaseModel, ABC):
#     def __init__(self, **kwargs):
#         """
#         需要的参数：模型、数据集、评价指标
#         :param kwargs:
#         """
#         super().__init__(**kwargs)
#         self.model = None


class BaseModelHandler(ABC):
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config  # 包含设备、精度等参数
        self.model_info = {}  # 存储输入/输出层名称、形状等元数据
        self._initialized = False

    @abstractmethod
    def load(self):
        """加载模型并初始化计算资源（如 GPU 绑定）"""
        pass

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """执行推理，返回原始输出（未后处理）"""
        pass

    @abstractmethod
    def destroy(self):
        """释放模型占用的资源（如显存、线程池）"""
        pass

    @property
    def is_ready(self) -> bool:
        return self._initialized

    def __call__(self, *args, **kwargs):
        if self.is_ready:
            return self.run(*args, **kwargs)
