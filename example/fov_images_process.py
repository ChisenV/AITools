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

from build.lib.AITools import YOLODataset

os.environ["PATH"] = r"E:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

import cv2
import numpy as np
from tqdm import tqdm

from AITools import IMG_FORMATS, T_co, JSONParser, Config, dump_yolo_dataset
from AITools.comp.processor import CropImages, CropImagesV2, CropConfig


def obtain_images_info(image_dir):

    format_counter = Counter()
    resolution_counter = Counter()

    total_images = 0

    for file in tqdm(Path(image_dir).rglob("*")):
        if not file.suffix[1:] in IMG_FORMATS:
            # print(f"Skipping {file},", file.suffix)
            continue
        try:
            with Image.open(file) as img:
                total_images += 1

                format_counter[img.format] += 1

                resolution = f"{img.width}x{img.height}"
                resolution_counter[resolution] += 1

        except Exception:
            pass

    result = {
        "total_images": total_images,
        "formats": dict(sorted(format_counter.items())),
        "resolutions": dict(
            sorted(
                resolution_counter.items(),
                key=lambda item: (
                    int(item[0].split('x')[0]) *
                    int(item[0].split('x')[1])
                ),
                reverse=True
            )
        )
    }

    print(json.dumps(result, indent=4, ensure_ascii=False))
    """
    {
        "total_images": 12670,
        "formats": {
            "BMP": 904,
            "JPEG": 11766
        },
        "resolutions": {
            "5120x5120": 28,
            "5120x4096": 72,
            "5200x3500": 49,
            "4500x3600": 304,
            "4000x4000": 68,
            "4096x3072": 33,
            "4096x3000": 122,
            "4000x3020": 242,
            "4000x3000": 1021,
            "3792x3020": 1134,
            "3792x3000": 36,
            "3600x3000": 200,
            "3856x2800": 25,
            "3800x2800": 36,
            "3600x2800": 209,
            "3360x2960": 301,
            "3600x2600": 3734,
            "3160x2960": 70,
            "3000x2600": 114,
            "2992x2600": 288,
            "3200x2400": 198,
            "3300x2300": 105,
            "3200x2200": 263,
            "2448x2048": 190,
            "2400x2000": 236,
            "2000x2000": 1537,
            "1800x1800": 92,
            "1500x1400": 1963
        }
    }
    """


# def crop_images(image_dir):
#     images_list = [i for i in os.listdir(image_dir) if i.endswith(tuple(IMG_FORMATS))]
#
#     for img in tqdm(images_list):
#         img_path = Path(image_dir) / img
#         if not img_path.exists() or img_path.is_dir() or not img_path.suffix[1:] in IMG_FORMATS:
#             continue
#         with Image.open(img_path) as img:
#             img.format
#             img.width
#             img.height
#             c, r = 0, 0
#             if img.width < 2048:
#                 c = 1
#             elif 2048 <= img.width <= 4096:

class CLASSDataset:
    def __init__(self, roots, separate=False, image_dirname: str = "images", label_dirname: str = "labels"):
        self.roots = []
        self.separate = separate
        self.images = []
        self.labels = []
        self.root_map = {}
        self._index = 0
        self._begin = 0
        self.image_dirname = os.path.normpath(image_dirname)
        self.label_dirname = os.path.normpath(label_dirname)
        self.image_path_sep = f"{os.sep}{self.image_dirname}{os.sep}"
        self.label_path_sep = f"{os.sep}{self.label_dirname}{os.sep}"

        if isinstance(roots, str):
            self.images = [i for i in os.listdir(roots) if i.endswith(tuple(IMG_FORMATS))]
            self.roots = [roots]
        else:
            raise

    def __len__(self):
        return len(self.images)

    def __iter__(self) -> Iterator[T_co]:
        self._index = self._begin
        return self

    def __next__(self):
        if self._index < len(self):
            ret = self[self._index]
            self._index += 1
            return ret
        else:
            self._index = self._begin
            raise StopIteration(f"Iterator out of range, stop.")

    def __getitem__(self, index):
        image_path = Path(self.roots[0]) / self.images[index]
        label_path = self.img2label_path(image_path)
        la_json = JSONParser.load(label_path)
        if la_json is None:
            return image_path, None
        flags = la_json.get("flags", None)
        if flags is None:
            return image_path, None
        label = ""
        for flag_name, flag_v in flags.items():
            if flag_v:
                label = flag_name
        if not label:
            return image_path, None
        return image_path, label

    def img2label_path(self, image_path, label_postfix=".json"):
        image_path = str(image_path)
        if self.separate:
            return self.label_path_sep.join(image_path.rsplit(self.image_path_sep, 1)).rsplit(".", 1)[0] + label_postfix
        else:
            return image_path.rsplit(".", 1)[0] + label_postfix

# 配置
TILE_SIZE = 1280
OVERLAP = 256
STRIDE = TILE_SIZE - OVERLAP
SMALL_IMAGE_THRESHOLD = 1800

# =========================
# 随机裁剪
# =========================

def random_crop_positions(
        width,
        height,
        crop_num):

    max_x = width - TILE_SIZE
    max_y = height - TILE_SIZE

    positions = []

    for _ in range(crop_num):

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        positions.append((x, y))

    return positions


# =========================
# 保存Tile
# =========================

def save_tile(
        output_dir,
        img,
        x,
        y,
        tile_index,
        stem,
        suffix="png",
):

    tile = img.crop(
        (
            x,
            y,
            x + TILE_SIZE,
            y + TILE_SIZE
        )
    )

    out_file = (
        output_dir /
        f"{stem}_{tile_index}.{suffix}"
    )

    tile.save(
        out_file,
        quality=95
    )

def generate_positions(length, tile_size=TILE_SIZE):

    if length <= tile_size:
        return [0]

    num_tiles = math.ceil(length / tile_size)

    if num_tiles == 1:
        return [0]

    stride = (length - tile_size) / (num_tiles - 1)

    positions = []

    for i in range(num_tiles):

        pos = round(i * stride)

        if pos + tile_size > length:
            pos = length - tile_size

        positions.append(pos)

    return sorted(set(positions))


def process_image(image_path, output_dir, suffix="png"):

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"读取失败: {image_path}")
        print(e)
        return

    width, height = img.size

    stem = image_path.stem

    # =========================
    # 小图：Resize
    # =========================
    min_side = min(width, height)
    if min_side < SMALL_IMAGE_THRESHOLD:

        resized = img.resize(
            (TILE_SIZE, TILE_SIZE),
            Image.Resampling.LANCZOS
        )

        out_file = output_dir / f"{stem}_rs.{suffix}"

        resized.save(
            out_file,
            quality=95
        )

        print(
            f"{image_path.name} "
            f"-> resize"
        )

        return

    elif min_side < 3600:

        # if min_side < 2500:
        #     crop_num = 2
        # else:
        #     crop_num = 4
        crop_num = 2

        positions = random_crop_positions(
            width,
            height,
            crop_num
        )
        tile_index = 0
        for x, y in positions:
            save_tile(
                output_dir,
                img,
                x,
                y,
                tile_index,
                stem
            )

            tile_index += 1

        print(
            f"{image_path.name}"
            f" -> random crops:"
            f" {tile_index}"
        )
        return
    # =========================
    # 大图：滑窗裁剪
    # =========================

    xs = generate_positions(width)
    ys = generate_positions(height)

    tile_index = 0

    for y in ys:
        for x in xs:

            crop = img.crop(
                (
                    x,
                    y,
                    x + TILE_SIZE,
                    y + TILE_SIZE
                )
            )

            out_file = (
                output_dir
                / f"{stem}_{tile_index}.{suffix}"
            )

            crop.save(
                out_file,
                quality=95
            )

            tile_index += 1

    print(
        f"{image_path.name} "
        f"-> {tile_index} tiles"
    )


def crop_image_v2():
    cfg = CropConfig(
        crop_width=1280,
        crop_height=1280,
        overlap_ratio_x=0.1,
        overlap_ratio_y=0.1,
        valid_ratio=0.9,
        random_offset=True,
        random_offset_ratio=0.1,
        output_format="png",
        jpg_quality=100,
        workers=16
    )

    cropper = CropImagesV2(cfg)
    cropper(
        r"E:\ds\FOV\classified\Anomaly",
        r"E:\ds\FOV\classified\Anomaly-crop1280"
    )

def split():
    dst_data = r"E:\ds\FOV\classified\Normal-crop1280-split"
    ds = YOLODataset(r"E:\ds\FOV\classified", categories={0:"good"}, image_dirname="Normal-crop1280", with_label=False)
    print(len(ds))
    subset = ds.split([0.5, 0.25, 0.25])
    for i, (name, s) in enumerate(subset.items()):
        sub_dy = ds.subset(s)
        dump_yolo_dataset(sub_dy, destination=dst_data, sub_dirname=[name, "good"])


def main():
    image_dir = "E:\ds\FOV\Side"
    output_dir = Path(r"E:\ds\FOV\classified\Anomaly-crop1280")
    output_dir.mkdir(parents=True, exist_ok=True)
    cl = CLASSDataset(image_dir)
    print(len(cl))
    count = 0
    for image_path, l in tqdm(cl):
        if l is None:
            print(image_path)
        if l == "Normal":
            count += 1
            continue
        # process_image(image_path, output_dir)
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"读取失败: {image_path}", e)
            continue
        stem = image_path.stem
        width, height = img.size
        positions = random_crop_positions(
            width,
            height,
            1
        )
        tile_index = 0
        for x, y in positions:
            save_tile(
                output_dir,
                img,
                x,
                y,
                tile_index,
                stem,
                "jpg"
            )

    print(count)
    # crop_image_v2()

if __name__ == "__main__":
    # main()
    # split()

    CropImages(r"E:\ds\Anomaly\test\Fov",
        r"E:\ds\Anomaly\test\Fov-crop1280",
                       1280,1280)()
