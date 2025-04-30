import time

import AITools
from AITools import ImageData
from AITools.comp.backend import TensorRTModel
from AITools.comp.processor import WarpAffineNorm2NCHW

if __name__ == '__main__':
    model = TensorRTModel(
        path=r"E:\python_project\AIToolsV2\example\3-deploy-a-model\model_ocr_cls.trt"
    )

    model.load()

    print(model.info)

    preprocessor = WarpAffineNorm2NCHW(model.info["inputs"][0], enable_normalization=True)

    im = AITools.imread(
        r"E:\python_ai_dataset\OCR\cls\Val500\(3D-OCR-SMT_IC)3D##AiDian##20231215102458##fov_2_0@[8]@[3262,1881]@[a=0]_0_1.bmp"
    )
    image = preprocessor.run(im, format="BGR")
    out = model.infer({
        image.input_name: image.to_numpy()
    })
    print(out)

    # total = 0
    # run_times = 1000
    # for i in range(run_times):
    #     s0 = time.perf_counter()
    #     image = preprocessor.run(im, format="BGR")
    #     # s1 = time.perf_counter()
    #     out = model.infer({
    #         image.input_name: image.to_numpy()
    #     })
    #     s2 = time.perf_counter()
    #     e0 = s2 - s0
    #     total += e0
    #     print(f"{i} {e0 = :.6f}")
    # print(f"total: {total}, avg {total / run_times:.6f}")

    # print(out)
    # for i in range(out['output0'].shape[2]):
    #     box_info = out['output0'][0, :, i].tolist()
    #     if box_info[4] > 0.5:
    #         print(box_info)
