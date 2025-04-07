import time

import numpy as np
import cuda.bindings.driver as cuda

from AITools.comp.backend import int_address_to_ndarray, check_cuda_result


def cuda_context():
    cuda.cuInit(0)
    device = check_cuda_result(cuda.cuDeviceGet(0))
    cuContext = check_cuda_result(cuda.cuCtxCreate(0, device))
    return cuContext


if __name__ == "__main__":
    cuContext = cuda_context()
    shape = (50, 2000, 2000, 3)
    cpu_data1 = np.empty(shape, dtype=np.float32)
    cpu_data2 = np.empty(shape, dtype=np.float32)
    # print(cpu_data)
    six = np.ones(cpu_data1.shape, dtype=cpu_data1.dtype) * 6
    # print(six)

    total_bytes = cpu_data1.nbytes
    print(total_bytes, type(total_bytes))

    np.copyto(cpu_data1, six)

    time1 = time.perf_counter()
    np.copyto(cpu_data2, cpu_data1)
    time2 = time.perf_counter()
    print(time2 - time1, "seconds")

    host_pinned = check_cuda_result(cuda.cuMemAllocHost(total_bytes))
    host_pinned_ndarray = int_address_to_ndarray(host_pinned, six.dtype, six.shape)

    time3 = time.perf_counter()
    np.copyto(host_pinned_ndarray, six)
    time4 = time.perf_counter()
    print(time4 - time3, "seconds")

    device = check_cuda_result(cuda.cuMemAlloc(total_bytes))

    for i in range(10):
        time5 = time.perf_counter()
        check_cuda_result(cuda.cuMemcpyHtoD(device, host_pinned, total_bytes))
        time6 = time.perf_counter()
        print(time6 - time5, "seconds")

    time.sleep(7)

    cuda.cuMemFreeHost(host_pinned)
    cuda.cuMemFree(device)

    cuda.cuCtxDestroy(cuContext)
