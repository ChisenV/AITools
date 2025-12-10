import sys
from enum import Enum

__all__ = []

if sys.version_info < (3, 11):
    def _generate_next_value_(name, start, count, last_values):
        # Automatically converts member names to lower case as values
        return name.lower()

    class StrEnum(str, Enum):
        _generate_next_value_ = _generate_next_value_

    __all__.append("StrEnum")
else:
    from enum import StrEnum
    __all__.append("StrEnum")
