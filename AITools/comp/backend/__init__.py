import ctypes

from cuda.bindings import driver, nvrtc, runtime as cudart
from cuda.bindings.driver import CUresult

__all__ = [
    'TensorRTModel',
]


def _cuda_get_error_enum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


def check_cuda_errors(result):
    if isinstance(result, CUresult):
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


def int_address_to_ndarray(addr, dtype, shape):
    """
    将整数地址封装为NumPy数组

    参数:
        addr (int): 内存地址（如通过`id(obj)`或C扩展获取的地址）
        dtype (np.dtype): 数据类型（如np.float32, np.int64）
        shape (tuple): 数组形状

    返回:
        np.ndarray: 封装后的数组（共享内存，无数据拷贝）
    """
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    ptr_type = ctypes.POINTER(ctype * np.prod(shape))
    # 将整数地址转换为指针
    buffer = ctypes.cast(addr, ptr_type)
    # 转换为NumPy数组（共享内存）
    arr = np.ctypeslib.as_array(buffer.contents).reshape(shape)

    return arr


from .trt import TensorRTModel

try:
    import onnxruntime
    from .onnx import *
except ImportError:
    pass

