import ctypes

import numpy as np
from cuda.bindings import driver, nvrtc

__all__ = [
    "TensorRTModel",
    "check_cuda_errors",
]

from AITools.core.manager import ComponentManager

BACKENDS = ComponentManager("backends")


def _cuda_get_error_enum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


def check_cuda_errors(result):
    if isinstance(result, driver.CUresult):
        if result != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA error code={}({})".format(result.value, _cuda_get_error_enum(result)))
        else:
            return None
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cuda_get_error_enum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def int_address_to_ndarray(addr, dtype, shape) -> np.ndarray:
    """
    Encapsulate integer addresses as numpy arrays

    Args:
        addr (int): Memory address (such as the address obtained by id(obj) or C extension)
        dtype (np.dtype): Data type (e.g. np.float32, np.int64)
        shape (tuple): Array shape

    Return:
        np.ndarray: Encapsulated array (shared memory, no data copies)
    """
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    ptr_type = ctypes.POINTER(ctype * np.prod(shape))
    # Converts integer addresses to Pointers
    buffer = ctypes.cast(addr, ptr_type)
    # Convert to NumPy array (shared memory)
    arr = np.ctypeslib.as_array(buffer.contents).reshape(shape)

    return arr


from .trt import TensorRTModel

try:
    import onnxruntime
    from .onnx import *
except ImportError:
    pass
