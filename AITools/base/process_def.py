from abc import ABC, abstractmethod
from typing import Dict, Any, Union

from AITools import Config


class BaseProcessor(ABC):
    def __init__(self, config: Union[Config, Dict[str, Any]], *args, **kwargs):
        self.config = config

    @abstractmethod
    def run(self, data: Any, *args, **kwargs) -> Any:
        """Processor the raw input"""
        pass


class BasePreProcessor(BaseProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BasePostProcessor(BaseProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
