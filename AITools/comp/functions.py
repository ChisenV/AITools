import re
import hashlib
import glob
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Union
from datetime import datetime, timedelta

import cv2
import numpy as np
from tqdm import tqdm

__all__ = [
    "imread",
    "imwrite",
    "imshow",
    "parse_yolo_det_label",
    "parse_ppocr_label",
    "parse_voc_det_label",
    "yolo_to_absolute",
    "compute_affine_matrix",
    "invert_affine_transform",
    "plot_box_and_text_v2",
    "rotate_image",
    "rotate_bbox",
    "rotate_points",
    "rotate_image_min",
    "order_rectangle_points",
    "warp_affine_points",
    "img2label_path",
    "img2label_paths",
    "union_label",
    "union_labels",
    "convert_voc2yolo",
    "convert_coco2yolo",
    "segment2box",
    "get_img_files",
    "date_utils"
]

from .parser import XMLParser, JSONParser
from AITools.base.vision_def import (
    BoxFormat,
    BoundingBox,
    DetectionLabel,
    DataItem,
    ImageData,
    ImageFormat,
    OCRLabel,
    IMG_FORMATS,
)
from AITools.core.manager import ComponentManager

FUNCTIONS = ComponentManager("functions")

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors
_waitkey = cv2.waitKey


@FUNCTIONS.register_component
def imread(filename: Union[str, Path], flags: int = cv2.IMREAD_COLOR):
    """
    Read an image from a file.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.

    Returns:
        (np.ndarray): The read image.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


@FUNCTIONS.register_component
def imwrite(filename: Union[str, Path], img: np.ndarray, params=None):
    """
    Write an image to a file.

    Args:
        filename (str): Path to the file to write.
        img (np.ndarray): Image to write.
        params (list of ints, optional): Additional parameters. See OpenCV documentation.

    Returns:
        (bool): True if the file was written, False otherwise.
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


@FUNCTIONS.register_component
def imshow(winname: str, mat: np.ndarray, delay: int = None, scale: float = 1.0):
    """
    Displays an image in the specified window.

    Args:
        winname (str): Name of the window.
        mat (np.ndarray): Image to be shown.
        delay (int, optional): Delay in milliseconds. If 0, the window will stay open until the user closes it.
        If negative, the window will stay open indefinitely. Defaults to None.
        scale (float, optional): Scale factor for the image. Defaults to 1.0.
    """
    if mat is None:
        return

    if scale != 1.0:
        h, w = mat.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            raise ValueError("scale is too small, resulting image size is invalid")

        mat_show = cv2.resize(
            mat,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
        )
    else:
        mat_show = mat
    _imshow(winname.encode("unicode_escape").decode(), mat_show)
    if delay is not None:
        _waitkey(delay)


# Dataset label parser -------------------------------------------------------------------------------------------------
@FUNCTIONS.register_component
# async def parse_yolo_label(image_path, label_path):
def parse_yolo_det_label(image_path: str, label_path: str, normalized: bool = True):
    image = imread(image_path)
    labels = []
    with open(label_path) as f:
        for line in f:
            class_id, x_center, y_center, w, h = map(float, line.split())
            bbox = BoundingBox(
                coords=[x_center, y_center, w, h],
                format=BoxFormat.YOLO,
                normalized=True
            )
            if not normalized:
                bbox = yolo_to_absolute(bbox, image.shape[1], image.shape[0])
            labels.append(DetectionLabel(bbox=bbox, class_id=int(class_id)))
    return DataItem(
        image=ImageData(
            data=image,
            shape=image.shape,
            path=image_path,
            format=ImageFormat.BGR
        ),
        labels=labels
    )


@FUNCTIONS.register_component
# async def parse_ppocr_label(image_path, anno_label):
def parse_ppocr_label(image_path: str, anno_label: dict):
    image = imread(image_path)
    return DataItem(
        image=ImageData(
            data=image,
            shape=image.shape,
            path=image_path,
            format=ImageFormat.BGR
        ),
        labels=[
            OCRLabel(
                text=la["transcription"],
                text_box=BoundingBox(
                    coords=la["points"],
                    format=BoxFormat.POINTS,
                    normalized=False
                )
            )
            for la in anno_label
        ] if anno_label else None
    )


@FUNCTIONS.register_component
def parse_voc_det_label(image_path: str, label_path: str):
    image = imread(image_path)
    if Path(label_path).suffix == ".xml":
        anno_label = XMLParser.load(label_path)
    else:
        anno_label = None
    labels = None
    objects = anno_label['annotation'].get("object", None) if anno_label else []
    if isinstance(objects, dict):
        objects = [objects]
    if objects:
        labels = [
            DetectionLabel(
                bbox=BoundingBox(
                    coords=[
                        obj['bndbox']['xmin'],
                        obj['bndbox']['ymin'],
                        obj['bndbox']['xmax'],
                        obj['bndbox']['ymax']
                    ],
                    format=BoxFormat.XYXY,
                    normalized=False
                ),
                class_name=obj['name'],
                pose=obj['pose'],
                truncated=obj['truncated'],
                difficult=obj['difficult']
            )
            for obj in objects
        ]
    return DataItem(
        image=ImageData(
            data=image,
            shape=image.shape,
            path=image_path,
            format=ImageFormat.BGR
        ),
        labels=labels if labels else None
    )


@FUNCTIONS.register_component
def yolo_to_absolute(bbox: BoundingBox, width: int, height: int):
    if bbox.format != "yolo" or not bbox.normalized:
        return bbox
    x, y, w, h = bbox.coords
    return BoundingBox(
        coords=[
            (x - w / 2) * width,  # x_min
            (y - h / 2) * height,  # y_min
            (x + w / 2) * width,  # x_max
            (y + h / 2) * height  # y_max
        ],
        format=BoxFormat.XYXY,
        normalized=False
    )


# Image process functions ----------------------------------------------------------------------------------------------
@FUNCTIONS.register_component
def invert_affine_transform(im, om):
    i00 = im[0][0]
    i01 = im[0][1]
    i02 = im[0][2]
    i10 = im[1][0]
    i11 = im[1][1]
    i12 = im[1][2]

    D = i00 * i11 - i01 * i10
    D = 1.0 / D if D != 0 else 0

    A11 = i11 * D
    A12 = -i01 * D
    A21 = -i10 * D
    A22 = i00 * D

    om[0][0] = A11
    om[0][1] = A12
    om[0][2] = -A11 * i02 - A12 * i12
    om[1][0] = A21
    om[1][1] = A22
    om[1][2] = -A21 * i02 - A22 * i12


@FUNCTIONS.register_component
def compute_affine_matrix(from_size, to_size):
    scale_x = (float(to_size[0]) / (float(from_size[0])))
    scale_y = (float(to_size[1]) / (float(from_size[1])))
    scale = scale_x if scale_x < scale_y else scale_y

    src2dst = [[scale, 0, -scale * from_size[0] * 0.5 + to_size[0] * 0.5 + scale * 0.5 - 0.5],
               [0, scale, -scale * from_size[1] * 0.5 + to_size[1] * 0.5 + scale * 0.5 - 0.5]]
    dst2src = [[0, 0, 0], [0, 0, 0]]
    invert_affine_transform(src2dst, dst2src)

    return np.array(src2dst), np.array(dst2src)


@FUNCTIONS.register_component
def plot_box_and_text_v2(image, box, text: str = '', lw=None, text_lw_scale=0.5, box_color=(128, 128, 128),
                         text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX, text_box=False,
                         text_box_offset_x: int = 0, text_box_offset_y: int = 0):
    """

    :param image:
    :param lw: line width: max(round(sum(img_ori.shape) / 2 * 0.003), 2)
    :param box: [x1, y1, x2, y2] or [x1, y1, x2, y2, x3, y3, x4, y4]
    :param text: str
    :param box_color: (B, G, R)
    :param text_lw_scale: [0, 1]
    :param text_color: (B, G, R)
    :param font: cv2.FONT_HERSHEY_COMPLEX
    :param text_box: bool, whether to draw text box
    :param text_box_offset_x: int, >= 0
    :param text_box_offset_y: int, >= 0
    :return:
    """
    if lw is None:
        lw = max(round(sum(image.shape) / 2 * 0.003), 1)

    if not len(box) == 0:
        if not isinstance(box, np.ndarray):
            box = np.array(box, dtype=np.int32).reshape(-1, 2)
        elif len(box.shape) != 2:
            box = box.astype(np.int32).reshape(-1, 2)

        if box.shape[0] == 2:
            p1, p2 = box[0, :], box[1, :]
            cv2.rectangle(image, p1, p2, box_color, thickness=max(lw, 2), lineType=cv2.LINE_AA)
        elif box.shape[0] >= 3:
            p1 = box[0, :]
            cv2.polylines(image, [box], True, box_color, thickness=max(lw, 2), lineType=cv2.LINE_AA)
        else:
            raise ValueError("box shape is not correct.")
    else:
        p1 = [0, 0]

    if text:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(text, 0, fontScale=lw * text_lw_scale, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p1 = [p1[0] + w * text_box_offset_x, p1[1] + h * text_box_offset_y]
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        if text_box:
            cv2.rectangle(image, p1, p2, box_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, text, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw * text_lw_scale,
                    text_color, thickness=tf, lineType=cv2.LINE_AA)
    return image


@FUNCTIONS.register_component
def rotate_image(image, angle):
    """
    对图像进行任意角度旋转，旋转中心为图像中心

    参数:
        image: 输入图像 (numpy数组)
        angle: 旋转角度(度)，正值为逆时针旋转

    返回:
        旋转后的图像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


@FUNCTIONS.register_component
def rotate_image_min(image, angle):
    """
    计算旋转图像后的最小外接矩形

    参数:
        image: 输入图像
        angle: 旋转角度(度)，正值为逆时针旋转

    返回:
        (旋转后的图像, 最小外接矩形的宽高)
    """
    h, w = image.shape[:2]

    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    rotated_corners = rotate_points(corners, angle, (h, w))

    min_x = np.min(rotated_corners[:, 0])
    max_x = np.max(rotated_corners[:, 0])
    min_y = np.min(rotated_corners[:, 1])
    max_y = np.max(rotated_corners[:, 1])

    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_image = cv2.warpAffine(image, M, (new_w, new_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)

    return rotated_image, M


@FUNCTIONS.register_component
def order_rectangle_points(points):
    """
    将矩形框的四个顶点按左上、右上、右下、左下顺时针排序。

    参数：
        points (np.ndarray or list): 四个点的坐标，形状为(4,2)。

    返回：
        np.ndarray: 排序后的四个点，形状为(4,2)。
    """
    points = np.array(points)
    if points.shape != (4, 2):
        raise ValueError("输入必须是4个二维点，形状为(4, 2)。")

    # 1. 按y升序，x升序排序，确定左上点
    sorted_indices = np.lexsort((points[:, 0], points[:, 1]))
    sorted_points = points[sorted_indices]
    top_left = sorted_points[0]

    # 2. 计算剩余点相对于左上点的极角并排序
    remaining = sorted_points[1:]
    dx = remaining[:, 0] - top_left[0]
    dy = remaining[:, 1] - top_left[1]
    angles = np.arctan2(dy, dx)
    sorted_remaining = remaining[np.argsort(angles)]

    # 组合结果：左上、右上、右下、左下
    ordered_points = np.vstack([top_left.reshape(1, 2), sorted_remaining])
    return ordered_points


@FUNCTIONS.register_component
def warp_affine_points(points, M, round=None):
    points = np.asarray(points)
    points = np.column_stack((points, np.ones(len(points))))
    M = np.vstack((M, [0, 0, 1]))
    transformed = points @ M.T[:, :2]
    # if sort:
    #     sorted_indices = np.lexsort((transformed[:, 1], transformed[:, 0]))
    #     transformed = transformed[sorted_indices]
    if round is not None:
        transformed = np.round(transformed, round).astype(np.int32)
    return transformed


@FUNCTIONS.register_component
def rotate_bbox(bbox, angle, imgsz):
    """
    对图像中的矩形框进行旋转，旋转中心为图像中心

    参数:
        bbox: 原始矩形框 (x1, y1, x2, y2)
        angle: 旋转角度(度)，正值为逆时针旋转
        imgsz: 图像形状 (h, w)

    返回:
        旋转后的矩形框 (x1, y1, x2, y2)
    """
    h, w = imgsz
    x1, y1, x2, y2 = bbox

    center = (w / 2, h / 2)
    angle_rad = np.deg2rad(angle)

    points = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)

    points_centered = points - center
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    rotated_points = np.dot(points_centered, rotation_matrix.T)
    rotated_points += center

    min_x = np.min(rotated_points[:, 0])
    max_x = np.max(rotated_points[:, 0])
    min_y = np.min(rotated_points[:, 1])
    max_y = np.max(rotated_points[:, 1])

    return min_x, min_y, max_x, max_y


@FUNCTIONS.register_component
def rotate_points(points, angle, imgsz):
    """
    对图像中的一组点进行旋转，旋转中心为图像中心

    参数:
        points: 点集，格式为Nx2的numpy数组，每行表示一个点(x,y)
        angle: 旋转角度(度)，正值为逆时针旋转
        imgsz: 图像形状 (h, w)

    返回:
        旋转后的点集(Nx2 numpy数组)
    """
    h, w = imgsz

    center = np.array([w / 2, h / 2])
    angle_rad = np.deg2rad(angle)
    points_centered = points - center
    rotation_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ])

    rotated_points = np.dot(points_centered, rotation_matrix.T)
    rotated_points += center
    return rotated_points


@FUNCTIONS.register_component
def rotate_rectangle(cx, cy, w, h, radians=None, degrees=None, round=None, dtype=np.float64):
    """
    Rotate a rectangle by a given angle.

    Brief:
        clockwise:
            matrix1: [[np.sin(radians), -np.cos(radians)],[np.cos(radians), np.sin(radians)]]

            matrix2: [[math.cos(radians), math.sin(radians)],[-math.sin(radians), math.cos(radians)]]

        anticlockwise:
            matrix3: [[np.sin(radians), np.cos(radians)],[-np.cos(radians), np.sin(radians)]]

            matrix4: [[math.cos(radians), -math.sin(radians)],[math.sin(radians), math.cos(radians)]]

    Parameters:
        cx:
        cy:
        w:
        h:
        radians:
        degrees:
        round:
        dtype:

    Return:
        np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    """
    if radians is not None:
        assert degrees is None, "Either angle_radians or angle_degrees must be provided, not both"
    else:
        assert degrees is not None, "Either angle_radians or angle_degrees must be provided"
        radians = np.radians(degrees)

    point_matrix = np.array([
        [-w / 2, -h / 2],  # top left
        [w / 2, -h / 2],  # top right
        [w / 2, h / 2],  # bottom right
        [-w / 2, h / 2],  # bottom left
    ])

    rotation_matrix = [
        [np.cos(radians), np.sin(radians)],
        [-np.sin(radians), np.cos(radians)]
    ]

    rotated_point = point_matrix @ rotation_matrix + np.array([[cx, cy]] * 4)
    if round is not None:
        return rotated_point.round(round).astype(dtype)
    else:
        return rotated_point.astype(dtype)


def union_label(label_files, dst_file):
    dst_dirname = os.path.basename(os.path.dirname(dst_file))
    commonpath = os.path.commonpath(label_files)
    with open(dst_file, "w", encoding="utf-8") as f:
        for label_file in label_files:
            label_dir = os.path.dirname(label_file)
            subdir = str(label_dir).replace(commonpath + f"{os.sep}", "")
            with open(label_file, "r", encoding="utf-8") as f1:
                for line in f1:
                    path, label = line.strip().split("\t")
                    imgname = os.path.basename(path)
                    f.write(f"{dst_dirname}{os.sep}{subdir}{os.sep}{imgname}\t{label}\n")


def union_labels(dst_dir):
    label_files = [os.path.join(dst_dir, i, "Label.txt")
                   for i in os.listdir(dst_dir)
                   if os.path.isdir(os.path.join(dst_dir, i))]
    union_label(label_files, os.path.join(dst_dir, "Label.txt"))


@FUNCTIONS.register_component
def img2label_path(img_path, image_dirname="images", label_dirname="labels", postfix=".txt"):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}{image_dirname}{os.sep}", f"{os.sep}{label_dirname}{os.sep}"  # /images/, /labels/ substrings
    return sb.join(img_path.rsplit(sa, 1)).rsplit(".", 1)[0] + postfix


@FUNCTIONS.register_component
def img2label_paths(img_paths, image_dirname="images", label_dirname="labels", postfix=".txt"):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}{image_dirname}{os.sep}", f"{os.sep}{label_dirname}{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + postfix for x in img_paths]


@FUNCTIONS.register_component
def convert_voc2yolo(voc_dataset, save_dir, label_postfix=".txt", empty_label=True):
    """
    Args:
        voc_dataset:
        save_dir: the directory to save the converted labels
        label_postfix:
        empty_label:
    """
    if not voc_dataset.with_label:
        print("voc_dataset without label")
        return
    os.makedirs(save_dir, exist_ok=True)
    for i, item in enumerate(voc_dataset):
        img_path, lab_path = item
        xml_dump = False
        try:
            file_name = os.path.basename(lab_path).rsplit(".", 1)[0] + label_postfix
            if not os.path.exists(lab_path):
                if empty_label:
                    open(os.path.join(save_dir, file_name), "w").close()
                continue
            data = XMLParser.load(lab_path)
            objs = data['annotation'].get('object', [])
            if isinstance(objs, dict):
                objs = [objs]
            im_w, im_h = data['annotation']['size']['width'], data['annotation']['size']['height']
            if im_h <= 0 or im_w <= 0:
                h, w = imread(img_path).shape[0:2]
                im_w, im_h = w, h
                data['annotation']['size']['width'], data['annotation']['size']['height'] = w, h
                xml_dump = True
            with open(os.path.join(save_dir, file_name), "w") as f:
                for i, obj in enumerate(objs):
                    cla_id = voc_dataset.categories(obj['name'])
                    if cla_id is None:
                        continue
                    if voc_dataset.task == "det":
                        bbox = obj['bndbox']
                        x, y, w, h = bbox['xmin'], bbox['ymin'], bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin']
                        x, y, w, h = float(x + w / 2) / im_w, float(y + h / 2) / im_h, float(w) / im_w, float(h) / im_h
                        f.write(f"{cla_id} {x} {y} {w} {h}\n")
                    elif voc_dataset.task == "obb":
                        t = obj.get('type', '')
                        if t == 'robndbox':
                            if "segmentation" in obj:
                                try:
                                    seg = obj['segmentation']
                                    _seg_x1, _seg_y1, _seg_x2, _seg_y2 = seg['x1'], seg['y1'], seg['x2'], seg['y2']
                                    _seg_x3, _seg_y3, _seg_x4, _seg_y4 = seg['x3'], seg['y3'], seg['x4'], seg['y4']
                                    (seg_x1, seg_y1), (seg_x2, seg_y2), (seg_x3, seg_y3), (seg_x4, seg_y4) = cv2.RotatedRect(
                                        (_seg_x1, _seg_y1), (_seg_x2, _seg_y2), (_seg_x3, _seg_y3)
                                    ).points()
                                except Exception as e:
                                    seg_x1, seg_y1, seg_x2, seg_y2, seg_x3, seg_y3, seg_x4, seg_y4 = 0, 0, 0, 0, 0, 0, 0, 0
                            bbox = obj.get('robndbox', {})
                            cx, cy, w, h, angle = bbox['cx'], bbox['cy'], bbox['w'], bbox['h'], bbox['angle']
                            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv2.RotatedRect((cx, cy), (w, h), np.rad2deg(angle)).points()
                            if (seg_x1, seg_y1) != (x1, y1) or (seg_x2, seg_y2) != (x2, y2) or (seg_x3, seg_y3) != (x3, y3) or (seg_x4, seg_y4) != (x4, y4):
                                if isinstance(data['annotation']['object'], list):
                                    data['annotation']['object'][i]['segmentation'] = {
                                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                        'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4,
                                    }
                                elif isinstance(data['annotation']['object'], dict):
                                    data['annotation']['object']['segmentation'] = {
                                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                        'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4,
                                    }
                                xml_dump = True
                        elif t == 'bndbox':
                            bbox = obj.get('bndbox', {})
                            x1, y1, x3, y3 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                            x2, y2, x4, y4 = x3, y1, x1, y3
                            new_obj = {
                                'name': obj['name'], 'type': 'robndbox', 'pose': 'Unspecified',
                                'truncated': 1, 'difficult': 0,
                                'robndbox': {
                                    'cx': (x1 + x3) / 2, 'cy': (y1 + y3) / 2,
                                    'w': x3 - x1, 'h': y3 - y1, 'angle': 0
                                },
                                'segmentation': {
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4,
                                }
                            }
                            if isinstance(data['annotation']['object'], list):
                                data['annotation']['object'][i] = new_obj
                            elif isinstance(data['annotation']['object'], dict):
                                data['annotation']['object'] = new_obj
                            xml_dump = True
                        else:
                            continue
                        x1, y1, x2, y2 = float(x1) / im_w, float(y1) / im_h, float(x2) / im_w, float(y2) / im_h
                        x3, y3, x4, y4 = float(x3) / im_w, float(y3) / im_h, float(x4) / im_w, float(y4) / im_h
                        f.write(f"{cla_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
            if xml_dump:
                XMLParser.dump(data, lab_path, indent='')
                # print("New dump xml:", lab_path)
        except Exception as e:
            print(img_path, lab_path, str(e))
            raise e


def make_dirs(dir="new_dir/"):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / "labels", dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    """Converts COCO 91-class index (paper) to 80-class index (2014 challenge)."""
    return [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, None, 24, 25, None, None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60,
        None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, None, 73, 74, 75, 76, 77, 78, 79, None,
    ]


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


@FUNCTIONS.register_component
def convert_coco2yolo(
    json_file,
    save_dir,
    use_segments=False,
    cls91to80=False,
    cls_filter=None,
    label_exist_ok=False
):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    # save_dir = make_dirs()  # output directory
    coco80 = coco91_to_coco80_class()

    # Import json
    fn = Path(save_dir) / "labels"   # folder name
    if os.path.exists(fn) and not label_exist_ok:
        shutil.rmtree(fn)  # delete dir
    os.makedirs(fn, exist_ok=label_exist_ok)
    data = JSONParser().load(json_file)

    # Create image dict
    images = {"{:g}".format(x["id"]): x for x in data["images"]}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)

    # Write labels file
    for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
        img = images[f"{img_id:g}"]
        h, w, f = img["height"], img["width"], img["file_name"]

        bboxes = []
        segments = []
        for ann in anns:
            if ann["iscrowd"]:
                continue
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue
            box[box < 0] = 0.0
            box[box > 1] = 1.0  # if x > 1 and y > 1

            cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
            cls = cls_filter(cls) if cls_filter else cls
            if cls is None:
                continue
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
            # Segments
            if use_segments:
                if len(ann["segmentation"]) > 1:
                    s = merge_multi_segment(ann["segmentation"])
                    s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).astype(np.float64)
                else:
                    s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                    s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).astype(np.float64)
                s[s > 1] = 1.0
                s = [cls] + s.tolist()
                if s not in segments:
                    segments.append(s)

        # Write
        with open((fn / f).with_suffix(".txt"), "a") as file:
            for i in range(len(bboxes)):
                line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                file.write(("%g " * len(line)).rstrip() % line + "\n")

    categories = {}
    for cate in data["categories"]:
        categories[cate['id'] - 1] = cate['name']

    return categories


def convert_yolo2coco(dataset, output_json):
    """
    Converts YOLO format to COCO JSON format.

    Args:
        dataset: Yolo dataset
        output_json: Path to save COCO format JSON file
    """

    # Initialize COCO data structure
    coco_data = {
        "info": {
            "description": "COCO dataset converted from YOLO format",
            "version": "1.0",
            "year": int(datetime.now().strftime("%Y")),
            "contributor": "WQS",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, name in dataset.categories().items():
        coco_data["categories"].append({
            "id": i + 1,  # COCO uses 1-indexed categories
            "name": name,
            "supercategory": "none"
        })

    # Process images and annotations
    image_id = 1
    annotation_id = 1

    for img_path, label_path in tqdm(dataset, desc="Converting images"):
        img_path = Path(img_path)
        # Skip if label file doesn't exist
        if not Path(label_path).exists():
            continue

        # Read image to get dimensions
        try:
            img = imread(str(img_path))
            if img is None:
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        # Add image to COCO data
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "height": height,
            "width": width,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })

        # Read YOLO labels
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if dataset.task in ['det', 'detect']:
                if len(parts) < 5:
                    continue

                # YOLO format: class_id x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])

                # Convert from normalized YOLO format to COCO format
                # COCO: [x_min, y_min, width, height] in absolute pixels
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                bbox_w = bbox_width * width
                bbox_h = bbox_height * height

                # Ensure bbox is within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                bbox_w = min(bbox_w, width - x_min)
                bbox_h = min(bbox_h, height - y_min)

                # Check for valid bbox
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                # Get COCO category id (1-indexed)
                if class_id >= len(coco_data["categories"]):
                    print(f"Warning: class_id {class_id} exceeds available classes")
                    continue

                category_id = class_id + 1
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x_min), float(y_min), float(bbox_w), float(bbox_h)],
                    "area": float(bbox_w * bbox_h),
                    "segmentation": [],  # YOLO doesn't have segmentation
                    "iscrowd": 0
                }
            elif dataset.task in ['obb',]:
                # YOLO format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                annotation = {}
            elif dataset.task in ['seg', 'segmentation']:
                # YOLO format: class_id x1 y1 x2 y2 x3 y3 x4 y4 ...
                if len(parts) < 3:  # Need at least class_id and one point
                    continue

                class_id = int(parts[0])
                points = list(map(float, parts[1:]))

                # Check if number of points is even (x,y pairs)
                if len(points) % 2 != 0:
                    continue

                # Convert normalized points to absolute coordinates
                abs_points = []
                for i in range(0, len(points), 2):
                    x = points[i] * width
                    y = points[i + 1] * height
                    abs_points.extend([x, y])

                # Calculate bbox from segmentation
                x_coords = abs_points[0::2]
                y_coords = abs_points[1::2]

                x_min = max(0, min(x_coords))
                y_min = max(0, min(y_coords))
                x_max = min(width, max(x_coords))
                y_max = min(height, max(y_coords))

                bbox_w = x_max - x_min
                bbox_h = y_max - y_min

                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                # Get COCO category id
                if class_id >= len(coco_data["categories"]):
                    continue

                category_id = class_id + 1

                # Add annotation to COCO data
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x_min), float(y_min), float(bbox_w), float(bbox_h)],
                    "area": float(bbox_w * bbox_h),
                    "segmentation": [abs_points],  # COCO segmentation format
                    "iscrowd": 0
                }
            else:
                annotation = {}

            if annotation:
                coco_data["annotations"].append(annotation)
                annotation_id += 1
        image_id += 1

    # Save to JSON file
    JSONParser().dump(coco_data, output_json, indent=2)

    print(f"Conversion complete. Saved to {output_json}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total categories: {len(coco_data['categories'])}")

    return coco_data


def convert_labelme_json_to_ocr_txt(json_dir: Union[str, Path], output_txt: Union[str, Path]):
    json_dir = Path(json_dir)
    output_dir = Path(output_txt).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dirname = output_dir.name

    with open(output_txt, "w", encoding="utf-8") as out_file:
        for json_file in json_dir.glob("*.json"):
            data = JSONParser.load(json_file)
            image_path = f"{output_dirname}{os.sep}{os.path.basename(data['imagePath'])}"

            result_list = []

            for shape in data.get("shapes", []):
                transcription = shape.get("description", "")
                difficult = shape.get("difficult", False)

                points = [
                    [int(p[0]), int(p[1])]
                    for p in shape.get("points", [])
                ]

                item = {
                    "transcription": transcription,
                    "points": points,
                    "difficult": difficult
                }

                result_list.append(item)

            line = image_path + "\t" + JSONParser.dumps(result_list, ensure_ascii=False)
            out_file.write(line + "\n")


@FUNCTIONS.register_component
def segment2box(segment: np.ndarray, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (np.ndarray): The segment label.
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        (np.ndarray): The minimum and maximum x and y values of the segment.
    """
    x, y = segment.T  # segment xy
    # any 3 out of 4 sides are outside the image, clip coordinates first, https://github.com/ultralytics/ultralytics/pull/18294
    if np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum() >= 3:
        x = x.clip(0, width)
        y = y.clip(0, height)
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


@FUNCTIONS.register_component
def get_img_files(img_path, log_prefix=''):
    """
    Read image files from the specified path.

    Args:
        img_path (str | List[str]): Path or list of paths to image directories or files.
        log_prefix (str, optional): Prefix for logging errors. Defaults to ''.

    Returns:
        (List[str]): List of image file paths.

    Raises:
        FileNotFoundError: If no images are found or the path doesn't exist.
    """
    try:
        f = []  # image files
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                # F = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p, encoding="utf-8") as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                    # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise FileNotFoundError(f"{log_prefix}{p} does not exist")
        im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert im_files, f"{log_prefix}No images found in {img_path}. {IMG_FORMATS}"
    except Exception as e:
        raise FileNotFoundError(f"{log_prefix}Error loading data from {img_path}\n") from e
    return im_files


@FUNCTIONS.register_component
def rotate_image_around_point(image, center, angle_deg, imgsz):
    """
    以指定点为中心旋转图像，保持图像尺寸不变

    参数:
        image: 输入图像 (numpy数组)
        center: 旋转中心点坐标 (x, y)
        angle_deg: 旋转角度(度)，正值表示逆时针旋转

    返回:
        rotated_image: 旋转后的图像
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 计算旋转矩阵
    R = np.eye(3, dtype=np.float32)
    R[:2] = cv2.getRotationMatrix2D(center, angle_deg, 0.8)
    # 平移 缩放
    A = np.eye(3, dtype=np.float32)
    A[:2] = compute_affine_matrix((w, h), imgsz)[0]
    rotation_matrix = A @ R

    # 执行旋转（指定输出尺寸为原始尺寸）
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix[:2],
        imgsz,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))  # 填充黑边

    return rotated_image


def generate_yolo_empty_labels(images_dir, labels_dir, pbar: tqdm = None):
    """
    Generate YOLO empty labels.

    Args:
        images_dir (str, Path): Path to the image dataset directory.
        labels_dir (str, Path): Path to the output YOLO labels directory.
        pbar (tqdm): Progress bar.
    """

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    count = 0
    for basename in os.listdir(images_dir):
        img_path = os.path.join(images_dir, basename)
        if os.path.isfile(img_path) and basename.rsplit(
                '.', 1)[-1].lower() in IMG_FORMATS:
            lab_path = img2label_paths([img_path],
                                       os.path.basename(images_dir),
                                       os.path.basename(labels_dir))[0]
            if not os.path.exists(lab_path):
                with open(lab_path, 'w') as f:
                    f.write('')
                count += 1
                if pbar:
                    pbar.update()
                    pbar.set_postfix_str(f"Captured yolo empty labels: {os.path.basename(lab_path)}")
    return count


def date_utils(start_date: str, end_date: str = None, days: int = None) -> str:
    """
    通用日期计算函数
    :param start_date: 起始日期，字符串格式 "YYYY-MM-DD"
    :param end_date:   结束日期（可选），字符串格式 "YYYY-MM-DD"
    :param days:       天数（可选），整数，可以是正数或负数
    :return: 计算结果的字符串
    """
    # 转换为 datetime 对象
    start = datetime.strptime(start_date, "%Y-%m-%d")

    # 情况1：计算两个日期之间的相差天数
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
        delta_days = (end - start).days
        return f"{start_date} 到 {end_date} 相差 {delta_days} 天"

    # 情况2：计算经过 n 天后的日期
    if days is not None:
        new_date = start + timedelta(days=days)
        return f"{start_date} 经过 {days} 天后是 {new_date.strftime('%Y-%m-%d')}"

    return "请至少提供 end_date 或 days 参数"


def are_axis_aligned_rectangles_intersecting(rect1, rect2):
    """
    判断两个轴对齐矩形是否相交。

    参数:
    rect1 (tuple): 第一个矩形，格式为 (x1, y1, w1, h1)。
    rect2 (tuple): 第二个矩形，格式为 (x2, y2, w2, h2)。

    返回:
    bool: 如果相交返回 True，否则返回 False。
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 检查不相交的条件
    # 如果 rect1 在 rect2 的右侧
    if x1 > x2 + w2:
        return False
    # 如果 rect2 在 rect1 的右侧
    if x2 > x1 + w1:
        return False
    # 如果 rect1 在 rect2 的下方
    if y1 > y2 + h2:
        return False
    # 如果 rect2 在 rect1 的下方
    if y2 > y1 + h1:
        return False

    # 如果所有不相交的条件都不满足，则它们相交
    return True


@FUNCTIONS.register_component
def sanitize_filename(name: str, max_length: int = 50, replacement: str = "_") -> str:
    """
    生成跨平台安全文件名
    """

    if not name:
        return "EMPTY"

    # 替换非法字符
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', replacement, name)

    # 去除首尾空格和点
    name = name.strip(" .")

    # Windows保留名检查
    reserved = {
        "CON","PRN","AUX","NUL",
        "COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9",
        "LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9"
    }

    if name.upper() in reserved:
        name = f"_{name}"

    # 控制长度
    if len(name) > max_length:
        # 保留前半部分 + hash
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
        name = name[:max_length-9] + "_" + hash_suffix

    return name if name else "EMPTY"
