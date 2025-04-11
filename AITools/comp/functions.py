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
    "invert_affine_transform"
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
