import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from AITools.base.process_def import BasePreprocessor, BaseProcessor
from AITools.base.vision_def import IOConfig, ImageData, ImageFormat, IMG_FORMATS
from AITools.comp import functions as F
from AITools.comp.dataset import OCRDatasetV2
from AITools.core.manager import ComponentManager

PROCESSORS = ComponentManager("Processors")


@PROCESSORS.register_component
class WarpAffineNorm2NCHW(BasePreprocessor):
    def __init__(self, model_input: IOConfig, mean=None, std=None, enable_normalization=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input = model_input
        self.src2dst = None
        self.dst2src = None
        self.last_src_size = None
        self.last_dst_size = None

        self.mean = self._val_norm_param(mean, "mean")
        self.std = self._val_norm_param(std, "std")

        self.enable_normalization = enable_normalization

    def _val_norm_param(self, param, param_name):
        """验证标准化参数的合法性"""
        if param is None:
            return {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }.get(param_name, None)

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

    def run(self, im: np.ndarray, *args, **kwargs):
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
                im.copy(),
                self.src2dst,
                [W, H],
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(128, 128, 128)
            )
            data = data.astype(np.float32) / 255.0
        else:
            data = im.copy().astype(np.float32) / 255.0
        if kwargs.get('format', 'RGB') == 'BGR':
            data = data[..., ::-1]

        if self.enable_normalization:
            data -= np.array(self.mean, dtype=np.float32).reshape(1, 1, -1)
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
            format=ImageFormat.NCHW,
            shape=data.shape,
            input_name=self.model_input.name
        )

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


@PROCESSORS.register_component
class VisualizeOCRDataset(BaseProcessor):
    def __init__(self, dataset=None, set_type='det', save_dir='', save_format='auto', label_file_name='Label.txt',
                 line_color=(128, 128, 128), box_enable=True, text_enable=True, text_color=(128, 128, 128), text_font=3,
                 text_box_enable=False, line_width=2, text_line_width_scale=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(dataset, OCRDatasetV2), \
            f"Unsupported dataset type: {dataset.__class__.__name__}, please use OCRDataset or its subclass."
        assert dataset.with_label, \
            f"Dataset is not in evaluation mode, please set dataset.is_eval=True."
        assert set_type in ['det', 'cls', 'rec'], \
            f"Unsupported dataset type: {set_type}, please use 'det', 'cls' or 'rec'."
        assert save_format in IMG_FORMATS or save_format == 'auto', \
            f"Unsupported save format: {save_format}, please use one of {IMG_FORMATS} or 'auto'."
        self.dataset = dataset
        self.set_type = set_type
        self.save_dir = Path(save_dir)
        self.save_format = save_format
        self.label_file_name = label_file_name
        self.box_enable = box_enable
        self.text_enable = text_enable
        self.line_color = line_color
        self.text_color = text_color
        self.text_font = text_font
        self.text_box_enable = text_box_enable
        self.line_width = line_width
        self.text_line_width_scale = text_line_width_scale

        self._visualize_func = getattr(self, "_visualize_{}".format(self.set_type))

    def run(self, *args, **kwargs):
        self.visualize(*args, **kwargs)

    def visualize(self, *args, **kwargs):
        if not os.path.exists(self.save_dir) and len(self.dataset) > 0:
            os.makedirs(self.save_dir)
        for data in tqdm(self.dataset, desc="Visualizing {}".format(self.set_type), ncols=128):
            img_path, contents = data
            img_name = os.path.basename(img_path) if self.save_format == 'auto' \
                else os.path.basename(img_path).rsplit('.', 1)[0] + '.{}'.format(self.save_format)
            img = F.imread(img_path)
            self._visualize_func(img, contents)
            F.imwrite(os.path.join(self.save_dir, img_name), img)

    def _visualize_det(self, img, contents):
        """
        :param img: np.ndarray
        :param contents: format in
                        [{"transcription": "text", "points": [[0, 0], [0, 1], [1, 1], [1, 0]], "difficult": 0}]
        :return:
        """
        text_box_offset_row = 0
        for tp in contents:
            F.plot_box_and_text_v2(img, box=tp["points"], text=tp["transcription"] if self.text_enable else "",
                                   lw=self.line_width, text_lw_scale=self.text_line_width_scale, box_color=self.line_color,
                                   text_color=self.text_color, font=self.text_font, text_box=self.text_box_enable,
                                   text_box_offset_y=text_box_offset_row)
            if len(tp["points"]) == 0:
                text_box_offset_row += 1
        return img

    def _visualize_cls(self, img, contents):
        F.plot_box_and_text_v2(img, box=[], text=contents, text_lw_scale=self.text_line_width_scale,
                               box_color=self.line_color, text_color=self.text_color, font=self.text_font,
                               text_box=self.text_box_enable)
        return img

    def _visualize_rec(self, img, contents):
        F.plot_box_and_text_v2(img, box=[], text=contents, text_lw_scale=self.text_line_width_scale,
                               box_color=self.line_color, text_color=self.text_color, font=self.text_font,
                               text_box=self.text_box_enable)
        return img

    def __call__(self, *args, **kwargs):
        return self.visualize(*args, **kwargs)


class Visualize(BaseProcessor):
    pass
