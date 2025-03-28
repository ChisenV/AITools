from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

__all__ = ['ParserPlugin']


# --------------------- Plugin protocol definition ---------------------
@runtime_checkable
class ParserPlugin(Protocol):
    """Configuration file parsing plugin protocol"""

    @classmethod
    def parsable_file_extensions(cls) -> List[str]:
        """Enable the parser extension"""
        ...

    @classmethod
    def load(cls, path: Path, **kwargs) -> Dict[str, Any]:
        """Load configuration from file"""
        ...

    @classmethod
    def dump(cls, data: Dict[str, Any], path: Path, **kwargs) -> None:
        """Save the configuration to a file"""
        ...

