from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union, Dict

__all__ = [
    "BaseProcessor",
    "BasePreprocessor",
    "BasePostprocessor",
    "ComposeProcessor",
    "BaseEvaluator",
    "BaseSaver",
]

from . import Runnable


class BaseProcessor(Runnable):
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


class BaseEvaluator(BaseProcessor):
    """评估器基类"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def compute_one(self, *args, **kwargs) -> Any:
        """Core index calculation method"""
        pass

    @abstractmethod
    def compute_batch(self, *args, **kwargs) -> Any:
        """Core index calculation method"""
        pass

    def evaluate(self) -> Any:
        """完整评估流程"""
        predictions = []
        ground_truth = []

        return self.compute_batch(predictions, ground_truth)

    def run(self, *args, **kwargs) -> Any:
        """Processor the raw input"""
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """ """
        pass


class BaseSaver(BaseProcessor):
    """结果保存基类"""

    def __init__(self, output_dir: Union[str, Path], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save_batch(self, batch_data: Dict[str, Any], batch_id: int):
        """保存批次结果"""
        pass

    def finalize(self):
        """完成所有保存操作（如合并临时文件）"""
        pass
