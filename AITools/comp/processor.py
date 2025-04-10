import cv2
import numpy as np

from AITools.base.process_def import BasePreProcessor
from AITools.base.vision_def import IOConfig, ImageData, ImageFormat
from AITools.comp import functions as F


class WarpAffineNorm2NCHW(BasePreProcessor):
    def __init__(self, model_input: IOConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input = model_input
        self.src2dst = None
        self.dst2src = None

    def run(self, im: ImageData, *args, **kwargs):
        ih, iw, ic = im.shape
        N, C, H, W = self.model_input.shape
        if ih != H or iw != W:
            self.src2dst, self.dst2src = F.compute_affine_matrix([iw, ih], [W, H])
            data = cv2.warpAffine(
                im.to_numpy(),
                self.src2dst,
                [W, H],
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(128, 128, 128)
            )
            data = data.astype(self.model_input.dtype) / 255.0
        else:
            data = im.to_numpy().astype(self.model_input.dtype) / 255.0
        if im.format == 'BGR':
            data = data[..., ::-1]
        if len(data.shape) == 3:
            data = np.ascontiguousarray(data.transpose((2, 0, 1)))[np.newaxis, :, :, :]

        return ImageData(
            data=data,
            path=im.path,
            format=ImageFormat.NCHW,
            shape=data.shape,
            id=im.id,
            input_name=im.input_name
        )

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)
