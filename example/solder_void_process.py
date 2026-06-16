import math
import os
from pathlib import Path
from PIL import Image
from collections import Counter
import json

import shutil
import random
from pathlib import Path
from typing import Union, List, Iterator
from datetime import datetime

os.environ["PATH"] = r"E:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

import cv2
import numpy as np
from tqdm import tqdm

from AITools.comp.dataset import dump_voc_dataset, YOLODataset
from AITools import IMG_FORMATS, T_co, JSONParser, Config, dump_yolo_dataset, VOCDataset, convert_voc2yolo
from AITools.comp.processor import CropImages, CropImagesV2, CropConfig, VisualizeYOLODataset


def get_all_dir(path):
    return [os.path.join(path, i, "JPEGImages") for i in os.listdir(path)
            if os.path.isdir(os.path.join(path, i))]

def get_dataset(path):
    all_dirs = get_all_dir(path)

    ds = VOCDataset(all_dirs, image_dirname="JPEGImages", read_image=True)
    print(len(all_dirs), len(ds), ds.categories())

    new_ds = VOCDataset(image_dirname="JPEGImages", categories=ds.categories())
    for imp, lap, _, la in tqdm(ds):
        if la is None:
            continue
        # print(imp, lap, la)
        is_OK = True
        if 'object' in la['annotation']:
            if isinstance(la['annotation']['object'], list):
                for obj in la['annotation']['object']:
                    if obj.get('name') in ['XiDong-Big', 'XiDong-ZhaXi']:
                        is_OK = False
                        break
            else:
                if la['annotation']['object'].get('name') in ['XiDong-Big', 'XiDong-ZhaXi']:
                    is_OK = False
        if not is_OK:
            new_ds.append(imp, lap)

    print(len(all_dirs), len(new_ds), new_ds.categories())

    dump_voc_dataset(new_ds, destination=r"E:\ds\SolderVoid\NG", label_dirname="voc")
    convert_voc2yolo(new_ds, save_dir=r"E:\ds\SolderVoid\NG\labels")


def vis_dataset():
    ds = YOLODataset(r"E:\ds\SolderVoid\NG", categories={0: 'Pin', 1: 'XiDong-ZhaXi', 2: 'XiDong-Big'})
    VisualizeYOLODataset(ds, r"E:\ds\SolderVoid\NG\vis")()


def split():
    dst_data = r"E:\ds\SolderVoid\non-supervision\SolderVoid\train-split"
    ds = YOLODataset(r"E:\ds\SolderVoid\non-supervision\SolderVoid\train", categories={0:"good"}, image_dirname="good", with_label=False)
    print(len(ds))
    subset = ds.split([0.6, 0.4])
    for i, (name, s) in enumerate(subset.items()):
        sub_dy = ds.subset(s)
        dump_yolo_dataset(sub_dy, destination=dst_data, sub_dirname=[name, "good"])


if __name__ == "__main__":
    # get_dataset(r"E:\ds\SolderVoid\anno")
    # vis_dataset()
    split()
