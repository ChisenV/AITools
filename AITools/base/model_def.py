from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union, List, Callable
from enum import Enum

__all__ = [
    "ModelType",
    "FrameworkType",
    "BaseModelHandler",
]

from AITools.core.config import Config


class ModelType(Enum):
    TRAINING = "training"
    DEPLOYMENT = "deployment"


class FrameworkType(Enum):
    PYTORCH = "pytorch"
    PADDLE = "paddle"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TENSORFLOW = "tensorflow"


class BaseModelHandler(ABC):

    _SUPPORTED_EXTENSIONS = []

    def __init__(
            self,
            model_path: Union[str, Path],
            config: Union[Config, Dict[str, Any]],
            **kwargs
    ):
        self.model_path = Path(model_path)
        self.hooks: Dict[str, List[Callable]] = config.get("hooks", {})
        self.hooks.update(kwargs.pop("hooks", {}))
        self.config = config
        self.config.update(kwargs)
        self._initialized = False

    @abstractmethod
    def load(self, *args, **kwargs):
        """Load the model and initialize compute resources (such as GPU bindings)"""
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Perform inference, return original output (no post-processing)"""
        pass

    @abstractmethod
    def destroy(self, *args, **kwargs):
        """Free up resources occupied by the model (e.g. video memory, thread pool)"""
        pass

    @property
    def is_ready(self) -> bool:
        return self._initialized

    def __call__(self, *args, **kwargs):
        if self.is_ready:
            return self.run(*args, **kwargs)

    def __enter__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        self.destroy(*args, **kwargs)

    def run_hooks(self, names: Union[str, List[str]], *args, **kwargs):
        """Call hooks by name"""
        if isinstance(names, str):
            names = [names]
        for name in names:
            for hook in self.hooks.get(name, []):
                hook(*args, **kwargs)
