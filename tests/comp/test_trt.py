import time

import numpy as np
from cuda.bindings import driver
from PIL import Image

from AITools import TensorRTModel, Config
from AITools.comp.backend import int_address_to_ndarray

onnx_Path = (r"E:\python_project\ultralytics\project\det\XiDong\General_20241011_"
             r"ep1500_bs8_imgsz320_dataVer7_noPretrained_v8.3.0TOv8.2.0\model.onnx")

trt_Path = (r"E:\python_project\ultralytics\project\det\XiDong\General_20241011_"
             r"ep1500_bs8_imgsz320_dataVer7_noPretrained_v8.3.0TOv8.2.0\model.trt")


def hook1(*args, **kwargs):
    input_info = kwargs.get("input_info", {})
    name, dtype, shape, n_byte = input_info["name"], input_info["dtype"], \
        input_info["shape"], input_info["size"]
    _, device_buffer = input_info["host"], input_info["device"]
    _, hostPtr = driver.cuMemAllocHost(n_byte)
    driver.cuMemcpyDtoH(hostPtr, device_buffer, n_byte)
    dst = int_address_to_ndarray(hostPtr, dtype, shape)
    img = np.ascontiguousarray(dst.transpose((0, 2, 3, 1)), dtype=np.uint8)
    image = Image.fromarray(img[0])
    # 保存为文件
    image.save("./output.jpg")  # JPEG

    src = kwargs.get("in_img_to_hook", None)
    val = dst - src
    nonZ_idx = np.nonzero(val)
    print("nonZ_idx", nonZ_idx)
    # if len(nonZ_idx) > 0:
    #     print("diff", nonZ_idx)
    #     for i in nonZ_idx:
    #         print(i, dst[i])

    driver.cuMemFreeHost(hostPtr)


def test_trt_model():
    print(hook1)
    with TensorRTModel(
        trt_Path,
        config=Config(
            explicit_batch=True,
            use_fp16=True,
            hooks={
                "on_trt_input_memcpy_after": [
                    # hook1
                ]
            }
        ).dic
    ) as m:
        if m is None:
            print("load model failed")
            return
        # print(m.info)

        # 创建一张 320*320 的float32的图片
        # img = np.ones((1, 320, 320, 3), dtype=np.float32)
        total_iter = 10
        total_time = 0
        img = Image.open(r"E:\python_project\ultralytics\project\det\XiDong\Genera"
                         r"l_20241011_ep1500_bs8_imgsz320_dataVer7_noPretrained_v8"
                         r".3.0TOv8.2.0\image.jpg")
        img = np.array([img])
        print(img.shape)
        img = np.ascontiguousarray(img.transpose((0, 3, 1, 2)))
        for i in range(total_iter):
            # 随机创建
            # img = np.random.rand(1, 320, 320, 3).astype(np.float32)
            # reshape (1, 320, 320, 3) -> (1, 3, 320, 320)
            cur_time = time.perf_counter()
            out = m({
                "images": img
            }, in_img_to_hook=img)
            end_time = time.perf_counter()
            #print(f"trt time: {end_time - cur_time}")
            total_time += end_time - cur_time
        print(f"trt avg time: {total_time / total_iter}")
