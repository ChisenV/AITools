import os
from pathlib import Path

import cv2
import numpy as np

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
    "warpAffine_points",
    "union_label",
]

from .parser import XMLParser
from AITools.base.vision_def import (
    BoxFormat,
    BoundingBox,
    DetectionLabel,
    DataItem,
    ImageData,
    ImageFormat,
    OCRLabel
)
from AITools.core.manager import ComponentManager


FUNCTIONS = ComponentManager("functions")

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors
_waitkey = cv2.waitKey


@FUNCTIONS.register_component
def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
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
def imwrite(filename: str, img: np.ndarray, params=None):
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
def imshow(winname: str, mat: np.ndarray, delay: int = None):
    """
    Displays an image in the specified window.

    Args:
        winname (str): Name of the window.
        mat (np.ndarray): Image to be shown.
        delay (int, optional): Delay in milliseconds. If 0, the window will stay open until the user closes it.
        If negative, the window will stay open indefinitely. Defaults to None.
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)
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
            (x - w/2) * width,   # x_min
            (y - h/2) * height,  # y_min
            (x + w/2) * width,   # x_max
            (y + h/2) * height   # y_max
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
        elif box.shape[0] >= 2:
            p1, p2 = box[0, :], box[2, :]
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
def warpAffine_points(points, M, round=None):
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
    with open(dst_file, "w", encoding="utf-8") as f:
        for label_file in label_files:
            with open(label_file, "r", encoding="utf-8") as f1:
                for line in f1:
                    f.write(f"{dst_dirname}/" + line)
