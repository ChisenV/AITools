import copy
import os

import cv2
import numpy as np

from AITools import ImageData, ImageFormat, compute_affine_matrix, imread, imshow, YOLODataset
from AITools.comp.processor import VisualizeYOLODataset


def d():
    data = np.ones((1, 1, 3), dtype=np.uint8)
    im1 = ImageData(
        data=data,
        path="d",
        format=ImageFormat.RGB,
        shape=data.shape,
        id='0',
        input_name="input"
    )
    im2 = copy.deepcopy(im1)
    im2.id = '1'

    print(id(im1), im1)
    print(id(im2), im2)


def aff():
    img_dir = r"E:\python_ai_dataset\OCR\rec\gather\categories\Capacitor"
    imgs = os.listdir(img_dir)
    im = imread(os.path.join(img_dir, imgs[0]))
    ih, iw, ic = im.shape
    W, H = 320, 320
    src2dst, dst2src = compute_affine_matrix([iw, ih], [W, H])
    data = cv2.warpAffine(
        im,
        src2dst,
        [W, H],
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128)
    )
    imshow("aff1", data, 0)
    data = cv2.warpAffine(
        data,
        dst2src,
        [W, H],
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128)
    )
    imshow("aff2", data, 0)


if __name__ == "__main__":
    # d = YOLODataset(
    #     root=r"E:\python_ai_dataset\foreign-object-detect\2-FOVSlice\train\images-black-green-V3.1",
    #     with_label=True,
    #     task="seg",
    #     categories={
    #         0: "elem",
    #         1: "solder",
    #         2: "paster",
    #         3: "device"
    #     },
    #     read_image=False
    # )
    # assert len(d) == 440, "len(d) == {}".format(len(d))
    #
    # VisualizeYOLODataset(
    #     dataset=d,
    #     save_dir=r"E:\python_ai_dataset\foreign-object-detect\2-FOVSlice\train\images-black-green-V3.1-vis",
    #
    # )()

    m = np.zeros((100, 100), dtype=np.int8)

    cv2.fillPoly(m, [np.array([[-1, -1], [0, 50], [50, 50], [50, 0]])], 1)

    imshow("m", m, 0)

