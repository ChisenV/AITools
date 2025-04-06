from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "imread",
    "imwrite",
    "imshow",
]

from AITools import BoundingBox, DetectionLabel, DataItem, ImageData, XMLParser
from AITools.base.vision_def import ImageFormat, OCRLabel, BoxFormat

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors


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


def imshow(winname: str, mat: np.ndarray):
    """
    Displays an image in the specified window.

    Args:
        winname (str): Name of the window.
        mat (np.ndarray): Image to be shown.
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


# YOLO label parser ----------------------------------------------------------------------------------------------------
# async def parse_yolo_label(image_path, label_path):
def parse_yolo_det_label(image_path, label_path):
    image = imread(image_path)
    labels = []
    with open(label_path) as f:
        for line in f:
            class_id, x_center, y_center, w, h = map(float, line.split())
            bbox = BoundingBox(
                coords=[x_center, y_center, w, h],
                format="yolo",
                normalized=True
            )
            labels.append(DetectionLabel(bbox=bbox, class_id=int(class_id)))
    return DataItem(
        data={
            "image": ImageData(
                data=image,
                shape=image.shape,
                path=image_path,
                format=ImageFormat.BGR
            )
        },
        labels=labels
    )


# async def parse_ppocr_label(image_path, anno_label):
def parse_ppocr_label(image_path, anno_label):
    image = imread(image_path)
    return DataItem(
        data={
            "image": ImageData(
                data=image,
                shape=image.shape,
                path=image_path,
                format=ImageFormat.BGR
            )
        },
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


def parse_voc_det(image_path, label_path):
    image = imread(image_path)
    if Path(label_path).suffix == ".xml":
        labels = XMLParser.load(label_path)
    else:
        labels = None
    return DataItem(
        data={
            "image": ImageData(data=image, shape=image.shape, path=image_path)
        },
        labels=[]
    )


def yolo_to_absolute(bbox: BoundingBox, img_width: int, img_height: int):
    if bbox.format != "yolo" or not bbox.normalized:
        return bbox
    x, y, w, h = bbox.coords
    return BoundingBox(
        coords=[
            (x - w/2) * img_width,   # x_min
            (y - h/2) * img_height,  # y_min
            (x + w/2) * img_width,   # x_max
            (y + h/2) * img_height   # y_max
        ],
        format="xyxy",
        normalized=False
    )

