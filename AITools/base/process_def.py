from abc import ABC, abstractmethod
from typing import Dict, Any, Union

__all__ = [
    'BaseProcessor', 
    'BasePreProcessor', 
    'BasePostProcessor'
]


class BaseProcessor(ABC):
    def __init__(self, config: Union[Dict[str, Any]], *args, **kwargs):
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


class ComposeProcessor(BaseProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, data: Any, *args, **kwargs) -> Any:
        pass
