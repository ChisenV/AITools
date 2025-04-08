from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable

__all__ = [
    'ParserPlugin',
    'ModelHandlerPlugin',
    'ExecutablePlugin',
    'RunnablePlugin',
]


# --------------------- Plugin protocol definition ---------------------
@runtime_checkable
class ParserPlugin(Protocol):
    """Configuration file parsing plugin protocol"""

    @classmethod
    def load(cls, path: Path, **kwargs) -> Any:
        """Load configuration from file"""
        ...

    @classmethod
    def dump(cls, data: Dict[str, Any], path: Path, **kwargs) -> Any:
        """Save the configuration to a file"""
        ...


@runtime_checkable
class ExecutablePlugin(Protocol):
    def execute(self, *args, **kwargs) -> Any:
        """Perform the operator and return output"""
        ...


@runtime_checkable
class RunnablePlugin(Protocol):
    def run(self, *args, **kwargs) -> Any:
        """Perform the operator and return output"""
        ...


@runtime_checkable
class ModelHandlerPlugin(Protocol):
    """Model handler plugin protocol"""

    def create(self, *args, **kwargs) -> Any:
        """Create the model instance"""
        ...

    def destroy(self, *args, **kwargs) -> Any:
        """Release the resources occupied by the model (e.g. GPU memory, thread pool)"""
        ...

    def load(self, *args, **kwargs) -> Any:
        """Load configuration from file"""
        ...

    def dump(self, *args, **kwargs) -> Any:
        """Save the configuration to a file"""
        ...

    def run(self, *args, **kwargs) -> Any:
        """Perform the operator and return output"""
        ...
