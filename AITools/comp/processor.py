import os
from pathlib import Path
from typing import Any, Union, List

import cv2
import numpy as np
from tqdm import tqdm
import math
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from AITools import Config
from AITools.base.process_def import BasePreprocessor, BaseProcessor
from AITools.base.vision_def import IOConfig, ImageData, ImageFormat, IMG_FORMATS
from AITools.comp import functions as F
from AITools.comp.dataset import OCRDatasetV2, YOLODataset
from AITools.core.manager import ComponentManager
from AITools.utils.plotting import Colors

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
        if ic > C:
            im = im[:, :, :3]
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
                        if len(parts[1:]) == 5:
                            values = np.array(list(map(float, parts[1:5])))
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


'''
def parse_args():
    """
    Parse arguments for cropping images.

    Returns:
        args: parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='Crop images')
    parser.add_argument('--input_dir', type=str, required=True, help='input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--width', type=int, required=True, help='width of the crop area')
    parser.add_argument('--height', type=int, required=True, help='height of the crop area')
    parser.add_argument('--suffix', type=str, default='"{}".format(basename)',
                        help='suffix of the output file names')
    parser.add_argument('--format', type=str, default='png', help='format of the output file names')
    return parser.parse_args()
'''


@PROCESSORS.register_component
class CropImages(BaseProcessor):
    def __init__(self, input_dir, output_dir, w: int, h: int, *args, suffix='"{}__{}___{}".format(basename, j, i)',
                 fmt='png', cope_with_label=False, yolo_task='det', dump_empty=True, label_selector: callable = None,
                 image_op: callable = None, **kwargs):
        """
        Crop images in a directory and save them to another directory.

        Args:
            input_dir: input directory containing images
            output_dir: output directory to save cropped images
            w: width of the crop area
            h: height of the crop area
            suffix: suffix of the output file names
            fmt: format of the output file names
            cope_with_label: whether to deal with labels
            yolo_task: yolo task type
            dump_empty: whether to dump empty label images
        """
        super().__init__(*args, **kwargs)
        self.input_dir = input_dir
        # self.output_dir = os.path.join(output_dir, "images") if deal_with_label else output_dir
        self.output_dir = output_dir
        self.w = w
        self.h = h
        self.suffix = suffix
        self.format = fmt
        self.cope_with_label = cope_with_label
        self.yolo_task = yolo_task
        self.dump_empty = dump_empty
        self.label_selector = label_selector
        if image_op is None:
            def image_op(im):
                return im
        self.image_op = image_op

    def run(self, *args, **kwargs):
        return self()

    def __call__(self, strict=True):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f'Crop images in \n\t{self.input_dir} \nto \n\t{self.output_dir} \n'
              f'with width {self.w} and height {self.h} ......')
        if self.w == 0 or self.h == 0:
            raise ValueError('Width and height must be greater than 0')
        pbar = tqdm(os.listdir(self.input_dir), desc='Cropping images')
        for filename in pbar:
            basename, ext = os.path.splitext(filename)
            if ext[1:] not in ['jpg', 'png', 'jpeg', 'bmp']:
                continue
            # Mutil-language friendly imread
            image_path = os.path.join(self.input_dir, filename)
            image = F.imread(image_path)
            _h, _w = self.h, self.w
            if image.shape[0] < self.h or image.shape[1] < self.w:
                if strict:
                    raise ValueError('Crop area is larger than the image size')
                else:
                    _h = image.shape[0] if image.shape[0] < _h else _h
                    _w = image.shape[1] if image.shape[1] < _w else _w
            n = [0, 0]
            s = [0, 0]
            if (image.shape[0] % _h) / _h > 0.1:
                n[0] = (image.shape[0] + _h - 1) // _h
                s[0] = (image.shape[0] - _h) // (n[0] - 1) if n[0] > 1 else _h
            else:
                n[0] = image.shape[0] // _h
                s[0] = _h
            if (image.shape[1] % _w) / _w > 0.1:
                n[1] = (image.shape[1] + _w - 1) // _w
                s[1] = (image.shape[1] - _w) // (n[1] - 1) if n[1] > 1 else _w
            else:
                n[1] = image.shape[1] // _w
                s[1] = _w
            for r in range(0, n[0]):
                for c in range(0, n[1]):
                    i, j = r * s[0], c * s[1]
                    # Mutil-language friendly imwrite
                    crop_img_path = os.path.join(self.output_dir, str(eval(self.suffix)) + f".{self.format}")
                    write = True
                    if self.cope_with_label:
                        write = self.deal_with_yolo_labels(image_path, crop_img_path, image.shape[:2],
                                                           (j, i, _w, _h), self.dump_empty)
                    if write or self.dump_empty:
                        F.imwrite(crop_img_path, self.image_op(image[i:i + _h, j:j + _w]))
                    pbar.set_postfix({'output': eval(self.suffix) + f".{self.format}"})

    def deal_with_yolo_labels(self, orig_img_path, crop_img_path, orig_size, crop_pos, dump_empty=True):
        """处理多种YOLO格式标签"""
        orig_label_path = F.img2label_path(orig_img_path)
        if not os.path.exists(orig_label_path):
            return

        with open(orig_label_path, 'r', encoding='utf-8') as f:
            orig_lines = [line.strip() for line in f.readlines() if line.strip()]

        crop_x, crop_y, crop_w, crop_h = crop_pos
        orig_h, orig_w = orig_size
        new_lines = []

        if self.yolo_task in {'det', 'detect', 'obb'}:
            for line in orig_lines:
                parts = line.split()
                class_id = parts[0]
                coords = list(map(float, parts[1:]))

                if self.yolo_task in {'det', 'detect'}:
                    processed = self._process_det(coords, orig_w, orig_h, crop_x, crop_y, crop_w, crop_h)
                elif self.yolo_task == 'obb':
                    processed = self._process_obb(coords, orig_w, orig_h, crop_x, crop_y, crop_w, crop_h)
                else:
                    processed = None

                if processed:
                    new_lines.append(f"{class_id} {' '.join(processed)}")
        elif self.yolo_task == 'seg':
            processed = self._process_seg_v2(orig_lines, orig_w, orig_h, crop_x, crop_y, crop_w, crop_h)
            new_lines = processed if processed else []

        new_label_path = F.img2label_path(crop_img_path)
        os.makedirs(os.path.dirname(new_label_path), exist_ok=True)

        if dump_empty:
            with open(new_label_path, 'w', encoding='utf-8') as f:
                if new_lines:
                    f.write("\n".join(new_lines))
                else:
                    f.write("")
        else:
            if new_lines:
                with open(new_label_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_lines))

        return not new_lines == []  # True: not empty, False: empty

    def _process_det(self, coords, orig_w, orig_h, x, y, w, h):
        """处理目标检测标签
        @params: coords: [cx, cy, bw, bh]
        @params: orig_w: 原图大小
        @params: orig_h: 原图大小
        @params: x: 裁剪框左上角x坐标
        @params: y: 裁剪框左上角y坐标
        @params: w: 裁剪框宽度
        @params: h: 裁剪框高度
        """
        cx, cy, bw, bh = coords
        # 转换到绝对坐标
        cx_abs = cx * orig_w
        cy_abs = cy * orig_h
        bw_abs = bw * orig_w
        bh_abs = bh * orig_h

        # 计算边界框
        xmin = cx_abs - bw_abs / 2
        ymin = cy_abs - bh_abs / 2
        xmax = cx_abs + bw_abs / 2
        ymax = cy_abs + bh_abs / 2

        # 转换到裁剪坐标系
        new_xmin = max(xmin - x, 0)
        new_ymin = max(ymin - y, 0)
        new_xmax = min(xmax - x, w)
        new_ymax = min(ymax - y, h)

        # 有效性检查
        if new_xmax <= 0 or new_ymax <= 0 or new_xmin >= w or new_ymin >= h:
            return None

        # 转换回YOLO格式
        new_cx = (new_xmin + new_xmax) / (2 * w)
        new_cy = (new_ymin + new_ymax) / (2 * h)
        new_bw = (new_xmax - new_xmin) / w
        new_bh = (new_ymax - new_ymin) / h

        return [f"{new_cx:.6f}", f"{new_cy:.6f}", f"{new_bw:.6f}", f"{new_bh:.6f}"]

    def _process_obb(self, coords, orig_w, orig_h, x, y, w, h):
        """处理旋转框标签"""
        # 转换所有顶点坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
        points = [(coords[i] * orig_w - x, coords[i + 1] * orig_h - y)
                  for i in range(0, 8, 2)]

        # 有效性检查：至少有一个顶点在裁剪区域内
        valid_points = [p for p in points
                        if 0 <= p[0] <= w and 0 <= p[1] <= h]
        if not valid_points:
            return None

        # 转换回相对坐标并保持顶点顺序
        normalized = [f"{(p[0] / w):.6f}" if i % 2 == 0 else f"{(p[1] / h):.6f}"
                      for p in points for i in (0, 1)]

        return normalized

    def _process_seg(self, orig_lines, orig_w, orig_h, crop_x, crop_y, crop_w, crop_h):
        """处理分割多边形标签, 将位于shape=[w，h]内的多边形轮廓提取出来"""
        new_lines = []
        mask_map = {}
        for line in orig_lines:
            parts = line.split()
            class_id = int(parts[0])
            class_mask = mask_map.setdefault(class_id, np.zeros((orig_h, orig_w), dtype=np.uint8))
            coords = list(map(float, parts[1:]))

            if len(coords) % 2 != 0:
                raise ValueError("The number of coordinates must be an even number.")

            abs_points = np.array(coords, dtype=np.float32).reshape(-1, 2) * [orig_w, orig_h]
            cv2.fillPoly(class_mask, [np.array(abs_points, dtype=np.int32)], color=255)

        # 创建裁剪后的mask
        for class_id, class_mask in mask_map.items():
            crop_mask = class_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            # 寻找新轮廓（使用最外层轮廓）
            contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            for contour in contours:
                if len(contour) >= 3:  # 有效多边形至少需要3个点
                    # 转换为相对坐标
                    rel_points = contour.squeeze().astype(np.float32) / [crop_w, crop_h]
                    normalized = [f"{p:.6f}" for point in rel_points.tolist() for p in point]
                    new_lines.append(f"{class_id} " + " ".join(normalized))
        return new_lines

    def _process_seg_v2(self, orig_lines, orig_w, orig_h, crop_x, crop_y, crop_w, crop_h):
        """处理分割多边形标签, 将位于shape=[w，h]内的多边形轮廓提取出来"""
        new_lines = []
        class_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for line in orig_lines:
            parts = line.split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) % 2 != 0:
                raise ValueError("The number of coordinates must be an even number.")

            abs_points = np.array(coords, dtype=np.float32).reshape(-1, 2) * [orig_w, orig_h]
            cv2.fillPoly(class_mask, [np.array(abs_points, dtype=np.int32)], color=255)
            crop_mask = class_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            crop_mask = self.image_op(crop_mask)
            cropped_h, cropped_w = crop_mask.shape[0], crop_mask.shape[1]
            # 寻找新轮廓（使用最外层轮廓）
            contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            for contour in contours:
                if self.label_selector is not None:
                    if not self.label_selector(contour=contour, crop_h=cropped_h, crop_w=cropped_w, class_id=class_id):
                        continue
                if len(contour) >= 3:  # 有效多边形至少需要3个点
                    # 转换为相对坐标
                    rel_points = contour.squeeze().astype(np.float32) / [cropped_w, cropped_h]
                    normalized = [f"{p:.6f}" for point in rel_points.tolist() for p in point]
                    new_lines.append(f"{class_id} " + " ".join(normalized))
            class_mask[:, :] = 0

        return new_lines


def str2numeric(string: str):
    if string.isdigit():
        return True, int(string)
    else:
        try:
            return True, float(string)
        except ValueError:
            return False, string


def get_start_point(img_file: str):
    img_name_split = img_file.rsplit('.', 1)[0].split('_')
    numeric_list = []
    for i in img_name_split:
        is_numeric, value = str2numeric(i)
        if is_numeric:
            numeric_list.append(value)
    return numeric_list[3:5]


def get_new_name(img_file: str):
    img_name_split = img_file.rsplit('.', 1)[0].split('_')
    for i in [9, 8, 7, 6, 5]:
        img_name_split.pop(i)
    return '_'.join(img_name_split)


class RecombineImage:
    def __init__(self, image_dir, save_dir, size: Union[int, list, tuple] = 2000, save_format='bmp'):
        assert save_format in IMG_FORMATS, f'save_format must be one of [{IMG_FORMATS}]'
        assert os.path.exists(image_dir), f'{image_dir} does not exist'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.size = [size, size] if not isinstance(size, (list, tuple)) else size
        self.save_format = save_format

        self._channels = 1

    def __call__(self, **kwargs):
        image_list = [img_file for img_file in os.listdir(self.image_dir)
                      if img_file.split('.')[-1] in IMG_FORMATS]
        combine_images = {}
        for img_file in image_list:
            start_point = kwargs.get("start_point", get_start_point)(img_file)
            new_name = kwargs.get("new_name", get_new_name)(img_file)
            if new_name in combine_images.keys():
                combine_images[new_name]['sub_images'].append({
                    'image_path': os.path.join(self.image_dir, img_file),
                    'start_point': start_point
                })
                combine_images[new_name]['max_x'] = max(combine_images[new_name]['max_x'],
                                                        start_point[0] + self.size[0])
                combine_images[new_name]['max_y'] = max(combine_images[new_name]['max_y'],
                                                        start_point[1] + self.size[1])
            else:
                combine_images[new_name] = {
                    'sub_images': [{
                        'image_path': os.path.join(self.image_dir, img_file),
                        'start_point': start_point,
                    }],
                    'max_x': start_point[0] + self.size[0],
                    'max_y': start_point[1] + self.size[1]
                }

        with tqdm(total=len(combine_images), desc='recombine images') as pbar:
            for image_name, value in combine_images.items():
                sub_images = value['sub_images']
                tmp = F.imread(sub_images[0]['image_path'], cv2.IMREAD_UNCHANGED)
                self._channels = 1 if len(tmp.shape) == 2 else tmp.shape[2]
                del tmp
                recombine_content = np.zeros((value['max_y'], value['max_x'], self._channels)
                                             if self._channels != 1 else (value['max_y'], value['max_x']),
                                             dtype=np.uint8)
                for each in sub_images:
                    replacement_content = F.imread(each['image_path'], cv2.IMREAD_UNCHANGED)
                    recombine_content[
                        each['start_point'][1]:(each['start_point'][1] + self.size[0]),
                        each['start_point'][0]:(each['start_point'][0] + self.size[1])
                    ] = replacement_content
                cv2.imwrite(os.path.join(self.save_dir, f'{image_name}.{self.save_format}'), recombine_content)
                pbar.update(1)


class PixelRuler:
    def __init__(self, step, length):
        self.step = step
        self.length = length
        self.ruler = np.arange(start=0, stop=length, step=step)

    def get_level(self, value):
        return int(round(value / self.step, 0))

    def get_ruler(self):
        return self.ruler

    def get_measure(self, value):
        index = self.get_level(value)
        if index > len(self.ruler) - 1:
            raise ValueError("Value is overflow")
        return self.ruler[index]

    def reset(self, step, length):
        self.step = step
        self.length = length
        self.ruler = np.arange(start=0, stop=length, step=step)


class MosaicImage(BaseProcessor):
    def __init__(self, image_dirs: Union[str, Path, List[Union[str, Path]]], **kwargs):
        super().__init__(**kwargs)
        self.image_dirs = image_dirs
        self.image_paths = F.get_img_files(image_dirs)
        self.ruler = PixelRuler(320, 50000)
        self.image_size = []

    def run(self, *args, **kwargs) -> Any:
        for img_path in self.image_paths:
            img = F.imread(img_path)
            if img is None:
                continue
            imh, imw = img.shape[:2]
            self.image_size.append([imw, imh])
            to_size = max(self.ruler.get_measure(imh), self.ruler.get_measure(imw))
            src2dst, dst2src = F.compute_affine_matrix((imw, imh), (to_size, to_size))
            img = cv2.warpAffine(img, src2dst, (to_size, to_size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

    def __call__(self, *args, **kwargs):
        pass


class CropConfig:
    def __init__(
            self,
            crop_width=1280,
            crop_height=1280,
            overlap_ratio_x=0.2,
            overlap_ratio_y=0.2,
            valid_ratio=0.8,
            random_offset=False,
            random_offset_ratio=0.1,
            output_format="jpg",
            jpg_quality=95,
            workers=None):

        self.crop_width = crop_width
        self.crop_height = crop_height

        self.overlap_ratio_x = overlap_ratio_x
        self.overlap_ratio_y = overlap_ratio_y

        self.valid_ratio = valid_ratio

        self.random_offset = random_offset
        self.random_offset_ratio = random_offset_ratio

        self.output_format = output_format.lower()

        self.jpg_quality = jpg_quality

        self.workers = workers or os.cpu_count()


class CropImagesV2(BaseProcessor):
    """
        cfg = CropConfig(
            crop_width=1280,
            crop_height=1280,
            overlap_ratio_x=0.25,
            overlap_ratio_y=0.25,
            valid_ratio=0.8,
            random_offset=True,
            random_offset_ratio=0.1,
            output_format="jpg",
            jpg_quality=95,
            workers=16
        )

        cropper = CropImage(cfg)

        # 单文件
        # cropper.process(
        #     "test.jpg",
        #     "output"
        # )

        # 文件夹
        cropper.process(
            r"D:\images",
            r"D:\patches"
        )

        # 文件列表
        # cropper.process(
        #     [
        #         "a.jpg",
        #         "b.jpg",
        #         "c.jpg"
        #     ],
        #     "output"
        # )
    """
    IMAGE_EXTS = IMG_FORMATS

    def __init__(self, config: Union[Config, CropConfig], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = config

    def __call__(self, *args, **kwargs):
        self.process(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        self.process(*args, **kwargs)

    def process(self, source, output_dir):
        files = self._collect_files(source)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(files)} images")
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures = []

            for file in files:
                futures.append(
                    pool.submit(self._process_one_image, file, output_dir)
                )

            for f in futures:
                f.result()

    def _collect_files(self, source):
        if isinstance(source, str):
            p = Path(source)

            if p.is_file():
                return [p]
            if p.is_dir():
                files = []
                for ext in self.IMAGE_EXTS:
                    files.extend(p.rglob(f"*{ext}"))
                    files.extend(p.rglob(f"*{ext.upper()}"))

                return sorted(files)

        elif isinstance(source, list):
            return [Path(x) for x in source]

        raise ValueError("Unsupported input")

    def _calc_axis_positions(
            self,
            image_size,
            crop_size,
            overlap_ratio,
            valid_ratio):

        if image_size <= crop_size:
            return [0]

        effective = image_size - crop_size * valid_ratio
        ideal_stride = crop_size * (1 - overlap_ratio)
        n = math.ceil(effective / ideal_stride) + 1

        if n <= 1:
            return [0]

        real_stride = (image_size - crop_size) / (n - 1)
        pos = []

        for i in range(n):
            p = round(i * real_stride)
            p = min(max(0, p), image_size - crop_size)
            pos.append(p)

        return sorted(list(set(pos)))

    def _process_one_image(self, image_path, output_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed: {image_path}")
            return

        h, w = image.shape[:2]

        xs = self._calc_axis_positions(
            w,
            self.cfg.crop_width,
            self.cfg.overlap_ratio_x,
            self.cfg.valid_ratio
        )
        ys = self._calc_axis_positions(
            h,
            self.cfg.crop_height,
            self.cfg.overlap_ratio_y,
            self.cfg.valid_ratio
        )

        basename = image_path.stem
        count = 0

        for y in ys:
            for x in xs:
                x0, y0 = x, y

                if self.cfg.random_offset:
                    stride_x = int(self.cfg.crop_width * (1 - self.cfg.overlap_ratio_x))
                    stride_y = int(self.cfg.crop_height * (1 - self.cfg.overlap_ratio_y))

                    dx = random.randint(
                        -int(stride_x * self.cfg.random_offset_ratio),
                        int(stride_x * self.cfg.random_offset_ratio)
                    )
                    dy = random.randint(
                        -int(stride_y * self.cfg.random_offset_ratio),
                        int(stride_y * self.cfg.random_offset_ratio)
                    )

                    x0 += dx
                    y0 += dy

                x0 = max(0, min(x0, w - self.cfg.crop_width))
                y0 = max(0, min(y0, h - self.cfg.crop_height))

                crop = image[
                    y0:y0+self.cfg.crop_height,
                    x0:x0+self.cfg.crop_width
                ]

                valid_ratio = (crop.shape[0] * crop.shape[1]) / (self.cfg.crop_width * self.cfg.crop_height)

                if valid_ratio < self.cfg.valid_ratio:
                    continue

                filename = (
                    f"{basename}"
                    f"_x{x0}"
                    f"_y{y0}"
                    f".{self.cfg.output_format}"
                )

                save_path = output_dir / filename

                self._save_image(crop, save_path)

                count += 1

        print(f"{image_path.name} -> {count} patches")

    def _save_image(self, image, save_path):

        ext = self.cfg.output_format

        if ext in ("jpg", "jpeg"):
            F.imwrite(
                str(save_path),
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    self.cfg.jpg_quality
                ]
            )

        elif ext == "png":
            F.imwrite(
                str(save_path),
                image,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    3
                ]
            )

        elif ext == "webp":
            F.imwrite(
                str(save_path),
                image,
                [
                    cv2.IMWRITE_WEBP_QUALITY,
                    95
                ]
            )

        else:
            F.imwrite(
                str(save_path),
                image
            )
