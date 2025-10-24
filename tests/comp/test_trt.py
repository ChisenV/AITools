import os
import time

import numpy as np
from cuda.bindings import driver
from PIL import Image

from AITools import TensorRTModel, Config, IMG_FORMATS
from AITools.comp.backend import int_address_to_ndarray
from AITools.comp.processor import WarpAffineNorm2NCHW

onnx_Path = (r"E:\python_project\ultralytics\project\det\XiDong\General_20241011_"
             r"ep1500_bs8_imgsz320_dataVer7_noPretrained_v8.3.0TOv8.2.0\model.onnx")
onnx_Path = (r"G:\github\ultralyticshub\ultralytics-individual\OBB-EC\20250101-233045\weights\best.onnx")

trt_Path = (r"E:\python_project\ultralytics\project\det\XiDong\General_20241011_"
             r"ep1500_bs8_imgsz320_dataVer7_noPretrained_v8.3.0TOv8.2.0\model.trt")
trt_Path = (r"G:\github\ultralyticshub\ultralytics-individual\OBB-EC\20250101-233045\weights\best.trt")
trt_NMS_Path = (r"G:\github\ultralyticshub\ultralytics-individual\OBB-EC\20250101-233045\weights\best_nms.trt")


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


def infer_trt_model(path):
    print(hook1)
    with TensorRTModel(
        path,
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
        print(m.info)

        # 创建一张 320*320 的float32的图片
        # img = np.ones((1, 320, 320, 3), dtype=np.float32)
        total_iter = 100
        total_time = 0
        try:
            img = Image.open(r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.6\20250925\Fov\fod-test-20250909-AA_fov_17_0.jpg")
        except Exception as e:
            print(e)
            img = np.zeros((640,640,3), dtype=np.float32)
        print(m.info["inputs"][0].name, m.info["inputs"][0].dtype, m.info["inputs"][0].shape, m.info["inputs"][0].size)
        img = WarpAffineNorm2NCHW(model_input=m.info["inputs"][0]).run(np.array(img))
        print(img.shape)
        # img = np.ascontiguousarray(img.transpose((0, 3, 1, 2)))
        for i in range(total_iter):
            # 随机创建
            # img = np.random.rand(1, 320, 320, 3).astype(np.float32)
            # reshape (1, 320, 320, 3) -> (1, 3, 320, 320)
            cur_time = time.perf_counter()
            out = m({"images": img.data}, in_img_to_hook=img.data)
            end_time = time.perf_counter()
            #print(f"trt time: {end_time - cur_time}")
            total_time += end_time - cur_time
            # for k, v in out.items():
            #     print(f"{i} {k}: {v.shape}")
        print(f"trt avg time: {total_time / total_iter}")


def infer_trt_model_image_dir(path, image_dir):
    with TensorRTModel(
        path,
        config=Config(
            explicit_batch=True,
            use_fp16=True,
            hooks={
                "on_trt_infer_before": [
                    # hook1
                ],
                "on_trt_infer_after": [

                ]
            }
        ).dic
    ) as m:
        if m is None:
            print("load model failed")
            return
        print(m.info)

        # 创建一张 320*320 的float32的图片
        # img = np.ones((1, 320, 320, 3), dtype=np.float32)
        total_iter = 10
        total_time = 0
        image_paths = [os.path.join(image_dir, d) for d in os.listdir(image_dir) if d.endswith(tuple(IMG_FORMATS))]
        print(m.info["inputs"][0].name, m.info["inputs"][0].dtype, m.info["inputs"][0].shape,
              m.info["inputs"][0].size)
        # img = np.ascontiguousarray(img.transpose((0, 3, 1, 2)))
        for i, p in enumerate(image_paths):
            # 随机创建
            # img = np.random.rand(1, 320, 320, 3).astype(np.float32)
            # reshape (1, 320, 320, 3) -> (1, 3, 320, 320)
            try:
                img = Image.open(p)
                img = WarpAffineNorm2NCHW(model_input=m.info["inputs"][0]).run(np.array(img))
                cur_time = time.perf_counter()
                out = m({
                    "images": img.data
                }, in_img_to_hook=img.data)
                end_time = time.perf_counter()
                #print(f"trt time: {end_time - cur_time}")
            except Exception as e:
                print(f"error: {e}, {p}")
            total_time += end_time - cur_time
            # for k, v in out.items():
            #     print(f"{i} {k}: {v.shape}")
        print(f"trt avg time: {total_time / total_iter}s")


def test_trt_nms_model():
    TensorRTModel.trt_plugin()
    # infer_trt_model(trt_Path)
    # infer_trt_model(trt_NMS_Path)
    image_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.4\train_split\images\test"
    onnx_Path1 = (r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\model\best-v4.5.2-base.onnx")
    onnx_Path2 = (r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\model\best-v4.5.2-nms.onnx")
    onnx_Path3 = (r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\model\best-v4.5.2-post6.onnx")
    onnx_Path3 = (r"E:\python_project\ultralytics\project\BUBBLE\BUBBLE\20250609_083643\weights\export_plugin\best.onnx")
    onnx_Path3 = (r"E:\python_project\ultralytics\project\BUBBLE\BUBBLE\20250609_083643\weights\best.onnx")
    b = time.perf_counter()
    # infer_trt_model(onnx_Path1)
    # infer_trt_model(onnx_Path2)
    # infer_trt_model(onnx_Path3)
    e = time.perf_counter()
    total_time = e - b
    print(f"trt gen time: {total_time}")

    trt_Path1 = (r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\model\best-v4.5.2-base.trt")
    trt_Path2 = (r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\model\best-v4.5.2-nms.trt")
    trt_Path3 = (r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\model\best-v4.5.2-post6.trt")
    trt_Path3 = (r"E:\python_project\ultralytics\project\BUBBLE\BUBBLE\20250609_083643\weights\export_plugin\best.trt")
    trt_Path4 = (r"E:\python_project\ultralytics\project\BUBBLE\BUBBLE\20250609_083643\weights\best.trt")
    # infer_trt_model(trt_Path1)
    # infer_trt_model(trt_Path2)
    infer_trt_model(trt_Path3)
    infer_trt_model(trt_Path4)
    # infer_trt_model_image_dir(trt_Path1, image_dir)
    # infer_trt_model_image_dir(trt_Path3, image_dir)
    # infer_trt_model_image_dir(trt_Path3, image_dir)


if __name__ == '__main__':
    test_trt_nms_model()
