import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from AITools.base.process_def import BasePreprocessor, BaseProcessor
from AITools.base.vision_def import IOConfig, ImageData, ImageFormat, IMG_FORMATS
from AITools.comp import functions as F
from AITools.comp.dataset import OCRDatasetV2, YOLODataset
from AITools.core.manager import ComponentManager
from AITools.utils.plotting import Colors
from AITools.utils.property import threaded

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


@PROCESSORS.register_component
class VisualizeYOLODataset(BaseProcessor):
    def __init__(self, dataset=None, save_dir='', save_format='auto', box_enable=True, text_enable=True, text_font=3,
                 text_box_enable=False, line_width=2, text_line_width_scale=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(dataset, YOLODataset):
            raise ValueError("Unsupported dataset type: {}, please use YOLODataset.".format(dataset.__class__.__name__))
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.save_format = save_format
        self.box_enable = box_enable
        self.text_enable = text_enable
        self.text_font = text_font
        self.text_box_enable = text_box_enable
        self.line_width = line_width
        self.text_line_width_scale = text_line_width_scale

        self.colors = Colors()

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        self.visualize(*args, **kwargs)

    def visualize(self, *args, **kwargs):
        if not self.dataset.with_label:
            raise ValueError("The dataset is not labeled, please set `with_label=True` when initializing the dataset.")
        if self.dataset.is_read_image:
            raise ValueError(
                "The dataset is return image data, please set `is_read_image=False` when initializing the dataset.")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for data in tqdm(self.dataset, desc="Visualizing", ncols=128):
            img, lab = data
            self.visualize_one(img, lab)

    @threaded
    def visualize_one(self, image, label):
        im = F.imread(Path(image))
        h, w = im.shape[:2]
        if self.dataset.task == "cls":
            if not isinstance(label, str):
                label = str(label)

            F.plot_box_and_text_v2(
                im,
                text=label,
                lw=self.line_width,
                text_lw_scale=self.text_line_width_scale,
                text_color=self.colors(self.dataset.categories(label)),
                font=self.text_font,
                text_box=self.text_box_enable
            )
        else:
            with open(label, 'r', encoding='utf-8') as f:
                lines = [line for line in f.readlines()]

            for line_num, line in enumerate(lines, 1):
                parts = line.split()
                if not parts:
                    continue

                try:
                    class_id = int(parts[0])
                    class_str = self.dataset.categories(class_id)
                    values = np.array(list(map(float, parts[1:])))
                    if self.dataset.task == 'det':
                        # Expected format: class_id x_center y_center width height
                        if len(values) != 4:
                            raise ValueError(f"Detection label requires 4 values, got {len(values)}")
                        x1, y1 = values[0] - values[2] / 2, values[1] - values[3] / 2
                        x2, y2 = x1 + values[2], y1 + values[3]
                        points = [x1 * w, y1 * h, x2 * w, y2 * h]

                    elif self.dataset.task == 'obb':
                        # Expected format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                        if len(values) != 8:
                            raise ValueError(f"OBB label requires 8 values, got {len(values)}")
                        points = [values[i] * (w if i % 2 == 0 else h) for i in range(0, len(values))]

                    elif self.dataset.task == 'seg':
                        # Expected format: class_id x1 y1 x2 y2 ... (at least 3 points)
                        if len(values) < 6 or len(values) % 2 != 0:
                            raise ValueError(
                                f"Segmentation label requires even number of values (>=6), got {len(values)}")
                        points = [values[i] * (w if i % 2 == 0 else h) for i in range(0, len(values))]

                    elif self.dataset.task == 'pose':
                        # Expected format:
                        # class_id x_center y_center width height kp1_x kp1_y <p1-vis> kp2_x kp2_y <p2-vis>...
                        if len(values) < 4:
                            raise ValueError(f"Pose label requires at least 4 values, got {len(values)}")
                        bbox = values[:4]
                        x1, y1 = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2
                        x2, y2 = x1 + bbox[2], y1 + bbox[3]
                        points = [x1 * w, y1 * h, x2 * w, y2 * h]

                        # kps = values[4:]
                        # Auto-detect pose format (2D or 3D)
                        # keypoints = [
                        #     (kps[i], kps[i + 1], int(kps[i + 2]) if self.dataset.kpt_shape[-1] == 3 else 1)
                        #     for i in range(0, len(kps), self.dataset.kpt_shape[-1])
                        # ]

                    else:
                        raise ValueError(f"Unsupported task type: {self.dataset.task}")

                    color = self.colors(class_id)
                    F.plot_box_and_text_v2(
                        im,
                        box=points,
                        text=class_str,
                        lw=self.line_width,
                        box_color=color,
                        text_lw_scale=self.text_line_width_scale,
                        text_color=color,
                        font=self.text_font,
                        text_box=self.text_box_enable
                    )

                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid label format in file: '{label}'. Error: {str(e)}")

        basename = os.path.basename(image)
        F.imwrite(os.path.join(self.save_dir, basename), im)
