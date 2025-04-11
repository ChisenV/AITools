import cv2
import numpy as np

from AITools.base.process_def import BasePreProcessor
from AITools.base.vision_def import IOConfig, ImageData, ImageFormat
from AITools.comp import functions as F
from AITools.core.manager import ComponentManager

PROCESSORS = ComponentManager("Processors")


@PROCESSORS.register_component
class WarpAffineNorm2NCHW(BasePreProcessor):
    def __init__(self, model_input: IOConfig, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input = model_input
        self.src2dst = None
        self.dst2src = None
        self.last_src_size = None
        self.last_dst_size = None

        self.mean = self._val_norm_param(mean, "mean")
        self.std = self._val_norm_param(std, "std")

        self.enable_normalization = mean is not None and std is not None

    def _val_norm_param(self, param, param_name):
        """验证标准化参数的合法性"""
        if param is None:
            return None

        if isinstance(param, (int, float)):
            return [param] * self.model_input.shape[1]  # 按通道数扩展
        elif isinstance(param, (list, tuple)):
            if len(param) != self.model_input.shape[1]:
                raise ValueError(
                    f"{param_name} length ({len(param)}) must match "
                    f"input channels ({self.model_input.shape[1]})"
                )
            return list(param)
        else:
            raise TypeError(f"Unsupported {param_name} type: {type(param)}")

    def run(self, im: ImageData, *args, **kwargs):
        ih, iw, ic = im.shape
        N, C, H, W = self.model_input.shape
        if ic != C:
            raise ValueError(
                f"Input image channel mismatch. Expected {C} channels, got {ic} channels."
            )

        current_src_size = (iw, ih)
        current_dst_size = (W, H)
        if ih != H or iw != W:
            if current_src_size != self.last_src_size or current_dst_size != self.last_dst_size:
                self.src2dst, self.dst2src = F.compute_affine_matrix(current_src_size, current_dst_size)
                self.last_src_size = current_src_size
                self.last_dst_size = current_dst_size

            data = cv2.warpAffine(
                im.to_numpy().copy(),
                self.src2dst,
                [W, H],
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(128, 128, 128)
            )
            data = data.astype(np.float32) / 255.0
        else:
            data = im.to_numpy().copy().astype(np.float32) / 255.0
        if im.format == 'BGR':
            data = data[..., ::-1]

        if self.enable_normalization:
            if self.mean is not None:
                data -= np.array(self.mean, dtype=np.float32).reshape(1, 1, -1)
            if self.std is not None:
                data /= np.array(self.std, dtype=np.float32).reshape(1, 1, -1)

        data = data.astype(self.model_input.dtype)

        if len(data.shape) == 3:
            data = np.ascontiguousarray(data.transpose((2, 0, 1)))

        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)

        if data.shape != self.model_input.shape:
            raise RuntimeError(
                f"Processed data shape {data.shape} "
                f"doesn't match model input {self.model_input.shape}"
            )

        return ImageData(
            data=data,
            path=im.path,
            format=ImageFormat.NCHW,
            shape=data.shape,
            id=im.id,
            input_name=im.input_name
        )

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
