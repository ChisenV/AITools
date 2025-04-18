import copy
import os

import cv2
import numpy as np

from AITools import ImageData, ImageFormat, compute_affine_matrix, imread, imshow


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
    aff()
