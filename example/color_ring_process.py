import os
import shutil
import random
from pathlib import Path
from typing import Union, List
from datetime import datetime

os.environ["PATH"] = r"E:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

import cv2
import numpy as np
from tqdm import tqdm

from AITools import IMG_FORMATS, convert_coco2yolo, YOLODataset
from AITools.comp.functions import generate_yolo_empty_labels, convert_yolo2coco, img2label_path
from AITools.comp.processor import VisualizeYOLODataset


def rename(src_dir: Union[str, Path], offset: int = 0, include_yolo_label=False):
    src_dir = Path(src_dir)
    image_list = [i for i in os.listdir(src_dir) if i.rsplit('.', 1)[-1].lower() in IMG_FORMATS]
    idx = 0 + offset
    for i in tqdm(image_list):
        os.rename(src_dir / i, src_dir / f"UID{idx}.DIP_Resistor_ColorR.{i.rsplit('.', 1)[-1]}")
        if include_yolo_label:
            label_path = img2label_path(os.path.normpath(src_dir / i))
            label_dir = Path(label_path).parent
            os.rename(label_path, label_dir / f"UID{idx}.DIP_Resistor_ColorR.txt")
        idx += 1
    return idx


def convert_coco2yolo_cr_dataset(top_dir):
    annotation_file = Path(top_dir) / "Annotations" / "annotations.json"
    categories = convert_coco2yolo(annotation_file, top_dir, True)
    generate_yolo_empty_labels(Path(top_dir) / 'images', Path(top_dir) / 'labels')
    d = YOLODataset(top_dir, categories=categories, task='seg', fix_bad_data=True)
    VisualizeYOLODataset(d, Path(top_dir) / 'vis')()
    return categories


def collect_img(input_dir, output_dir):
    """ input_dir: ...\RunTime level """
    top_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir / "OK", exist_ok=True)
    os.makedirs(output_dir / "NG", exist_ok=True)
    list_dir = os.listdir(top_dir)
    for i in list_dir:
        ring_dir = top_dir / i / "Ring_错件"
        print(ring_dir)
        if ring_dir.exists():
            ring_ok_dir = ring_dir / "OK"
            ring_ng_dir = ring_dir / "NG"
            if ring_ok_dir.exists():
                print(ring_ok_dir)
                for j in os.listdir(ring_ok_dir):
                    if not j.rsplit(".", 1)[-1].lower() in IMG_FORMATS:
                        continue
                    shutil.move(ring_ok_dir / j, output_dir / "OK" / j)
            if ring_ng_dir.exists():
                print(ring_ng_dir)
                for j in os.listdir(ring_ng_dir):
                    if not j.rsplit(".", 1)[-1].lower() in IMG_FORMATS:
                        continue
                    shutil.move(ring_ng_dir / j, output_dir / "NG" / j)


if __name__ == '__main__':
    input_dir = r"E:\ds\ColorRing\anno\ColorRing_20260513"
    output_dir = r"E:\ds\ColorRing\anno\ColorRing_20260513-rename"
    # collect_img(input_dir, output_dir)

    # top_dir = Path(r"E:\python_ai_dataset\ColorRing\annoed")
    # input_dirs = [r"ColorRing_20260323", r"ColorRing_20260408", r"ColorRing_20260409",]
    id_offset = 594
    # for i in input_dirs:
    #     input_dir = Path(top_dir) / i
    #     convert_coco2yolo_cr_dataset(input_dir)

    cate = convert_coco2yolo_cr_dataset(input_dir)
    # rename(Path(input_dir) / "images", id_offset, include_yolo_label=True)

    d = YOLODataset(input_dir, categories=cate, task='seg', fix_bad_data=True)

    convert_yolo2coco(d, Path(input_dir) / "annotations.json")

