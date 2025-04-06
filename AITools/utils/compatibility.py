import sys
from enum import Enum

__all__ = []

if sys.version_info < (3, 11):
    class StrEnum(str, Enum):
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()  # 自动将成员名转为小写作为值

    __all__.append("StrEnum")
