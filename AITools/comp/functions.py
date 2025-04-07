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
    "yolo_to_absolute"
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
def imshow(winname: str, mat: np.ndarray):
    """
    Displays an image in the specified window.

    Args:
        winname (str): Name of the window.
        mat (np.ndarray): Image to be shown.
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


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
    objects = anno_label['annotation'].get("object", None)
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

