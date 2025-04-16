from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "BaseProcessor",
    "BasePreprocessor",
    "BasePostprocessor",
    "ComposeProcessor"
]


class BaseProcessor(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Processor the raw input"""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """ """
        pass


class BasePreprocessor(BaseProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BasePostprocessor(BaseProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ComposeProcessor(BaseProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, data: Any, *args, **kwargs) -> Any:
        pass

    def append(self, processor: BaseProcessor):
        pass