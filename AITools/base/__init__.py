class _FakeTorch:
    class Tensor:
        pass

    @staticmethod
    def from_numpy(*args, **kwargs):
        raise ImportError("PyTorch is not installed. Please install it to use this feature.")


try:
    import torch
except ImportError:
    torch = _FakeTorch


from .dataset_def import *
from .model_def import *
from .plugin_protocol_def import *
from .process_def import *
from .vision_def import *
