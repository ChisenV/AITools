from abc import abstractmethod
from typing import Dict, Any

from AITools.base import BaseEvaluator


class ClsEvaluator(BaseEvaluator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)


