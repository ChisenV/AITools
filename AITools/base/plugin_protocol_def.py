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
class ModelHandlerPlugin(Protocol):  # TODO: maybe can inherit from `ParserPlugin` and `ProcessorPlugin`
    """Model handler plugin protocol"""

    # TODO: Remove this method after the framework is unified
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
        """Release the resources occupied by the model (e.g. GPU memory, thread pool)"""
        ...

    def is_ready(self) -> bool:  # TODO: Remove this method after the framework is unified
        """Check if the model is ready to run"""
        ...


# TODO: Refactor the processor plugin protocol, maybe rename it to 'Executable'
@runtime_checkable
class ProcessorPlugin(Protocol):
    def run(self, *args, **kwargs) -> Any:
        """Perform the operator and return output"""
        ...
