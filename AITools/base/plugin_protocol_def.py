from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

__all__ = [
    'ParserPlugin',
    'ModelHandlerPlugin',
    'ProcessorPlugin',
]


# --------------------- Plugin protocol definition ---------------------
@runtime_checkable
class ParserPlugin(Protocol):
    """Configuration file parsing plugin protocol"""

    @classmethod
    def load(cls, path: Path, **kwargs) -> Dict[str, Any]:
        """Load configuration from file"""
        ...

    @classmethod
    def dump(cls, data: Dict[str, Any], path: Path, **kwargs) -> None:
        """Save the configuration to a file"""
        ...


@runtime_checkable
class ModelHandlerPlugin(Protocol):
    """Model handler plugin protocol"""

    def supported_framework_name(self) -> List[str]:
        """The name of the framework that the model handler supports"""
        ...

    def load(self, *args, **kwargs) -> Any:
        """Load the model and initialize the compute resources"""
        ...

    def run(self, *args, **kwargs) -> Any:
        """Perform the inference and return the original output"""
        ...

    def destroy(self, *args, **kwargs) -> Any:
        """Release the resources occupied by the model（如显存、线程池）"""
        ...

    def is_ready(self) -> bool:
        """Check if the model is ready to run"""
        ...


@runtime_checkable
class ProcessorPlugin(Protocol):
    def run(self, *args, **kwargs) -> Any:
        """Perform the operator and return output"""
        ...
