import os
import shutil
import random
import time
from pathlib import Path
from typing import Union, List
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

os.environ["PATH"] = r"F:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

from AITools import IMG_FORMATS
from AITools.comp.dataset import OCRDatasetV2, dump_ocr_dataset, OCRRECDatasetV2, OCRCLSDatasetV2
from AITools.comp.functions import sanitize_filename, union_label, reverse_order_ocr_string, rotate_bbox_xyxyxyxy, \
    imread, rotate_image_around_point, imwrite, convert_ocr2yolo, rotate_image_auto, rotate_bbox_xyxyxyxy_auto, \
    order_rectangle_points
from AITools.comp.parser import JSONParser, YMLParser
from AITools.comp.processor import VisualizeOCRDataset

SEP_CHAR = "."
RENAME_LIST_FILE = "rename_list.txt"
date_num = int(datetime.now().strftime("%Y%m%d"))
unidirect_char = ['a', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'r', 's', 't', 'v', 'y',
                  'A', 'D', 'F', 'G', 'J', 'K', 'L', 'R', 'T', 'V', 'Y',
                  '2', '4', '5', '7', '!', '@', '#', '$', '%', '^', '&', '*']


def det_paths_level1(path):
    return [os.path.join(path, i) for i in os.listdir(path)
            if os.path.isdir(os.path.join(path, i))]


def dir_is_exist_image(d):
    return any(
        os.path.isfile(os.path.join(d, i))
        and
        i.rsplit('.', 1)[-1].lower() in IMG_FORMATS
        for i in os.listdir(d)
    )


def get_all_dir(top_dir):
    _all_dir = []
    for root, directory, files in os.walk(
            top_dir, topdown=False, onerror=None, followlinks=False):
        if dir_is_exist_image(root) and os.path.exists(
                os.path.join(root, "Label.txt")):
            _all_dir.append(root)
    return _all_dir


def box_random_expand(
    dataset: OCRDatasetV2,
    expand_ratio: Union[float, List[float]],
    expand_direction: Union[str, List[str]],
    seed: int = None
):
    if isinstance(expand_ratio, float):
        expand_ratio = [expand_ratio]
    if len(expand_ratio) > 4 or any([r < 0 for r in expand_ratio]):
        raise ValueError("Length of expand_ratio must be less 4 and it's elements must be greater than 0.")

    if isinstance(expand_direction, str):
        expand_direction = [expand_direction]
    if any([d not in ["left", "right", "top", "bottom"] for d in expand_direction]):
        raise ValueError("expand_direction must be one of 'left', 'right', 'top', 'bottom'")
    if len(expand_direction) < len(expand_ratio):
        raise ValueError("expand_direction must be at least as long as expand_ratio.")
    if seed is not None:
        random.seed(seed)
    if not getattr(dataset, "with_label", False):
        return

    for i, (pa, la) in enumerate(dataset):
        img = imread(pa)
        if img is None:
            continue
        h, w = img.shape[:2]

        for j, label in enumerate(la):
            points = np.array(label["points"], dtype=np.float32)

            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            for k, ratio in enumerate(expand_ratio):
                direction = expand_direction[k]

                box_w = x_max - x_min
                box_h = y_max - y_min
                ratio = random.uniform(0.0, ratio)
                if ratio <= 0:
                    continue
                if direction == "left":
                    delta = ratio * box_w
                    x_min = max(0, x_min - delta)
                elif direction == "right":
                    delta = ratio * box_w
                    x_max = min(w, x_max + delta)
                elif direction == "top":
                    delta = ratio * box_h
                    y_min = max(0, y_min - delta)
                elif direction == "bottom":
                    delta = ratio * box_h
                    y_max = min(h, y_max + delta)

                if x_min >= x_max or y_min >= y_max:
                    break

            new_points = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ], dtype=np.int32)

            la[j]["points"] = new_points.tolist()
        dataset[i] = (pa, la)

    return dataset


def rename_classified_OCRDatesetV2(primal_dir, classified_dir, final_dir, exclude_list=None):
    """
    exclude_list = [
        "4-Train996_reverse", "13-anno_20250225_AiDian_reverse", "25-OCR-opposite-20250927",
        "26-OCR-opposite-20250928"
    ]
    """
    src_list_all = det_paths_level1(primal_dir)
    src_list = []
    if exclude_list is not None:
        for src in src_list_all:
            if os.path.basename(src) in exclude_list:
                continue
            src_list.append(src)

    d_src = OCRDatasetV2(src_list, with_label=True)
    n2i = {v: k for k, v in d_src.image_map.items()}

    rename_list_file = open(os.path.join(classified_dir, RENAME_LIST_FILE), "w", encoding="utf-8")

    cls_list = det_paths_level1(classified_dir)
    for i in cls_list:
        cls_list += det_paths_level1(i)
    d_cls = OCRDatasetV2(cls_list)
    d_cls.with_label = True
    cnt = 0
    for i, n in d_cls.image_map.items():
        if n in n2i:
            cnt += 1
            im_p = d_cls[i][0]
            d_cls[i] = (im_p, d_src[n2i[n]][1])

    commonpath = os.path.commonpath(list(d_cls.roots_map.values()))

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        img_path = Path(_img_path)
        ext = img_path.suffix  # include .jpg
        parent = img_path.parent

        relative_parts = str(parent).replace(commonpath, "").split(os.sep)[1:]
        if not any("Type" in r for r in relative_parts):
            relative_parts = relative_parts + ["Type0"]
        name_parts = []

        if index is not None:
            name_parts.append(f"UID{index}")

        if relative_parts:
            name_parts.append(SEP_CHAR.join(relative_parts))

        if _label_data is not None and len(_label_data) > 0:
            for idx, _la in enumerate(_label_data):
                transcription = "{" + sanitize_filename(_la.get("transcription", "")) + "}"
                name_parts.append(transcription)
        else:
            name_parts.append("{}")

        basename = SEP_CHAR.join(name_parts) + ext
        shutil.copy(_img_path, os.path.join(_dst_dir, basename))
        rename_list_file.write(f"{index}, {os.path.basename(_img_path)}, {basename}\n")

        if _label_file is not None:
            dirname = os.path.basename(_dst_dir)
            if _label_data is not None:
                _label_str = _label_op(_label_data)
                _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
            else:
                _label_file.write(f"{dirname}/{basename}\t[]\n")

    dump_ocr_dataset(d_cls, final_dir, custom_image_label_op=image_label_op)
    rename_list_file.close()


def rename_classified_OCRDatasetV2_DIP(dip_dir: str, new_dir: str):
    dip_dataset = OCRDatasetV2(
        dip_dir,
        with_label=True,
    )
    subdir_lab_file_map = {}
    for img_path, lab_data in dip_dataset:
        img_basename = os.path.basename(img_path)
        subdir_name = img_basename.split(".")[1]
        if subdir_name in ["SMT_Crystal", "SMT_EleCapacitors"]:
            subdir_path = os.path.join(new_dir, subdir_name, "Type5")
            img_basename = img_basename.replace("Type0", "Type5")
        else:
            subdir_path = os.path.join(new_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        if subdir_path not in subdir_lab_file_map:
            subdir_lab_file_map[subdir_path] = open(os.path.join(subdir_path, "Label.txt"), "w", encoding='utf-8')
        subdir_lab_file_map[subdir_path].write(f"{os.path.basename(subdir_path)}/{img_basename}\t{dip_dataset.fmt_label_dumps(lab_data)}\n")
        shutil.copy(img_path, os.path.join(subdir_path, img_basename))

    lab_list = []
    for subdir_path, lab_file in subdir_lab_file_map.items():
        lab_file.close()
        lab_list.append(os.path.join(subdir_path, "Label.txt"))
    union_label(lab_list, os.path.join(new_dir, "Label.txt"), sep='/')


def reorder_OCRDatasetV2(classified_dir, final_dir, latest_dir, dip_dir=None):
    cls_dirs = det_paths_level1(classified_dir)
    for i in cls_dirs:
        cls_dirs += det_paths_level1(i)

    d_cls = OCRDatasetV2(cls_dirs, with_label=False)

    fnl_dirs = det_paths_level1(final_dir)
    for i in fnl_dirs:
        fnl_dirs += det_paths_level1(i)
    fnl_dirs = [i for i in fnl_dirs if os.path.exists(os.path.join(i, "Label.txt"))]
    d_fnl = OCRDatasetV2(fnl_dirs, with_label=True)
    d_new_map = {v: k for k, v in d_fnl.image_map.items()}

    d1, d2, d3 = {}, {}, {}
    new_dset = OCRDatasetV2(subject_to='image')
    new_dset.with_label = True
    with open(os.path.join(classified_dir, RENAME_LIST_FILE), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            idx, old_name, new_name = line.strip().rsplit(', ')
            d1[idx] = (old_name, new_name)
            d2[old_name] = new_name
            d3[new_name] = old_name

    for im, _ in d_cls:
        basename = os.path.basename(im)
        new_path = d2.get(basename, "")
        if new_path != "" and new_path in d_new_map:
            new_idx = d_new_map[new_path]
            new_dset.append(im, d_fnl[new_idx][1])

    commonpath = os.path.commonpath(list(d_cls.roots_map.values()))

    os.makedirs(latest_dir, exist_ok=True)
    replace_list_file = open(os.path.join(latest_dir, RENAME_LIST_FILE), "w", encoding="utf-8")
    offset = 0

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        img_path = Path(_img_path)
        ext = img_path.suffix
        parent = img_path.parent
        relative_parts = str(parent).replace(commonpath, "").split(os.sep)[1:]

        if not any("Type" in r for r in relative_parts):
            relative_parts = relative_parts + ["Type0"]
        name_parts = []

        if index is not None:
            name_parts.append(f"UID{index + offset}")

        if relative_parts:
            name_parts.append(SEP_CHAR.join(relative_parts))

        if _label_data is not None and len(_label_data) > 0:
            for idx, _la in enumerate(_label_data):
                transcription = "{" + sanitize_filename(_la.get("transcription", "")) + "}"
                name_parts.append(transcription)
        else:
            name_parts.append("{}")

        basename = SEP_CHAR.join(name_parts) + ext
        shutil.copy(_img_path, os.path.join(_dst_dir, basename))
        replace_list_file.write(f"{index + offset}, {os.path.basename(_img_path)}, {basename}\n")

        if _label_file is not None:
            dirname = os.path.basename(_dst_dir)
            if _label_data is not None:
                _label_str = _label_op(_label_data)
                _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
            else:
                _label_file.write(f"{dirname}/{basename}\t[]\n")

    dump_ocr_dataset(
        new_dset, latest_dir,
        custom_image_label_op=image_label_op,
        overwriting=True
    )

    offset = len(new_dset)
    dip_dirs = det_paths_level1(dip_dir)
    for i in dip_dirs:
        dip_dirs += det_paths_level1(i)
    dip_dirs = [i for i in dip_dirs if os.path.exists(os.path.join(i, "Label.txt"))]
    DIP_OCR_dataset = OCRDatasetV2(dip_dirs, with_label=True)
    commonpath = os.path.commonpath(list(DIP_OCR_dataset.roots_map.values()))

    dump_ocr_dataset(DIP_OCR_dataset, rf"{latest_dir}\DIP_OCR",
                     custom_image_label_op=image_label_op, )

    replace_list_file.close()

    latest_cV2_dir = det_paths_level1(latest_dir)
    for i in latest_cV2_dir:
        latest_cV2_dir += det_paths_level1(i)

    union_label([os.path.join(i, "Label.txt")
                 for i in latest_cV2_dir
                 if os.path.exists(os.path.join(i, "Label.txt"))],
                os.path.join(latest_dir, "Label.txt"), sep='/')


def union_label_all_OCRDatasetV2(latest_dir):
    for i in det_paths_level1(latest_dir):
        pro_label = os.path.join(i, "Label.txt")
        label_exs = os.path.exists(pro_label)
        sub_label = [os.path.join(i, typ, "Label.txt") for typ in os.listdir(i)
                     if os.path.isdir(os.path.join(i, typ))
                     and os.path.exists(os.path.join(i, typ, "Label.txt"))]

        if len(sub_label) > 0:
            pro_label_content = None
            if label_exs:
                with open(pro_label, "r", encoding="utf-8") as f1:
                    pro_label_content = f1.readlines()
            union_label(sub_label, pro_label)
            if pro_label_content is not None:
                with open(pro_label, "a", encoding="utf-8") as f1:
                    f1.write("".join(pro_label_content))


def reverse_OCRDatasetV2(top_dir, offset=None, angle=180, sample_rate=0.3):
    if offset is None:
        d_SMT = OCRDatasetV2(top_dir, with_label=True, subject_to='label')
        offset = len(d_SMT)

    reversible_list_path = os.path.join(top_dir, "reversible_list.txt")
    replace_list_file = os.path.join(top_dir, "replace_list.txt")

    with open(reversible_list_path, "r", encoding="utf-8") as f1:
        reversible_list_str = f1.readlines()
    reversible_list = [os.path.normpath(i.strip()) for i in reversible_list_str]

    ocr_dataset_map = {
        i: OCRDatasetV2(
            os.path.join(top_dir, i),
            with_label=True,
            subject_to='image'
        )
        for i in reversible_list
    }
    new_dataset = OCRDatasetV2()
    new_dataset.with_image = True
    new_dataset.with_label = True
    for name, dataset in ocr_dataset_map.items():
        # print(name, len(dataset))
        sample_index = dataset.sample(sample_rate, seed=date_num)
        sample_dataset = dataset.subset(sample_index)
        new_dataset += sample_dataset

    commonpath = os.path.commonpath(list(new_dataset.roots_map.values()))
    # print(len(new_dataset), commonpath)

    rm = {}
    with open(replace_list_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            rm[key] = value

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        img_path = Path(_img_path)
        ext = img_path.suffix  # include .jpg
        parent = img_path.parent
        relative_parts = str(parent).replace(commonpath, "").split(os.sep)[1:]
        if not any("Type" in r for r in relative_parts):
            relative_parts = relative_parts + ["Type0"]
        name_parts = []

        if index is not None:
            name_parts.append(f"UID{index + offset}")

        if relative_parts:
            name_parts.append(SEP_CHAR.join(relative_parts))

        im = imread(_img_path)

        dirname = os.path.basename(_dst_dir)
        if _label_data is not None and len(_label_data) > 0:
            _label_str = _label_op(_label_data)
            new_label = []
            label_data = JSONParser.loads(_label_str)
            for _la in label_data:
                _la["transcription"] = reverse_order_ocr_string(_la["transcription"], rm)
                _la["points"] = rotate_bbox_xyxyxyxy(
                    _la["points"], angle, im.shape[:2]).astype(np.int32).tolist()
                new_label.append(_la)
                transcription = "{" + sanitize_filename(_la.get("transcription", "")) + "}"
                name_parts.append(transcription)
            _label_str = JSONParser.dumps(new_label)
            basename = SEP_CHAR.join(name_parts) + ext
            if _label_file is not None:
                _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
        else:
            name_parts.append("{}")
            basename = SEP_CHAR.join(name_parts) + ext
            if _label_file is not None:
                _label_file.write(f"{dirname}/{basename}\t[]\n")

        im = rotate_image_around_point(im, (im.shape[1] // 2, im.shape[0] // 2), angle)
        imwrite(os.path.join(_dst_dir, basename), im)

    dump_ocr_dataset(
        new_dataset,
        rf"{top_dir}_reversible",
        custom_image_label_op=image_label_op,
    )

    latest_cV2_dir = det_paths_level1(rf"{top_dir}_reversible")
    for i in latest_cV2_dir:
        latest_cV2_dir += det_paths_level1(i)

    union_label([os.path.join(i, "Label.txt")
                 for i in latest_cV2_dir
                 if os.path.exists(os.path.join(i, "Label.txt"))],
                os.path.join(rf"{top_dir}_reversible", "Label.txt"), sep='/')
    # dset = OCRDatasetV2(rf"{top_dir}_reversible", with_label=True, subject_to='label')
    # print(len(dset))
    # VisualizeOCRDataset(dset, save_dir=rf"{top_dir}_reversible_vis")()


def multi_OCRDatasetV2_split(ocr_dataset_list, out_dir, dataset_class=OCRDatasetV2, split_ratio=None, seed=None, vis=False):
    if split_ratio is None:
        split_ratio = [0.7, 0.15, 0.15]
    dir_list = ocr_dataset_list

    all_dir = []
    for d in dir_list:
        all_dir += get_all_dir(d)

    ocr_dataset_map = {
        i: dataset_class(
            i,
            with_label=True,
            subject_to='label'
        )
        for i in tqdm(all_dir, desc="loading dataset")
    }
    split_map = {}
    for path, dataset in ocr_dataset_map.items():
        subsets = dataset.split(split_ratio, seed=seed)
        for name, sub_ids in subsets.items():
            subset = dataset.subset(sub_ids)
            if name not in split_map:
                split_map[name] = subset
            else:
                split_map[name] += subset
    total = 0
    for name, dataset in split_map.items():
        total += len(dataset)
        curr_dir = os.path.join(out_dir, f"{name}")
        os.makedirs(curr_dir, exist_ok=True)
        print(f"current dir: {curr_dir}")
        time.sleep(0.8)
        with open(os.path.join(curr_dir, "Label.txt"), "w", encoding='utf-8') as f:
            for img_path, lab_data in tqdm(dataset, desc=f"dump split dataset: {name}"):
                shutil.copy(img_path, os.path.join(curr_dir, os.path.basename(img_path)))
                f.write(f"{name}/{os.path.basename(img_path)}\t{dataset.fmt_label_dumps(lab_data)}\n")
        if vis:
            new_subset = OCRDatasetV2(
                curr_dir, with_label=True, subject_to='image')
            VisualizeOCRDataset(new_subset, save_dir=rf"{curr_dir}_vis")()

    print("total processed:", total)


def split_OCRDatasetV2(top_dir, split_ratio=None):
    if split_ratio is None:
        split_ratio = [0.7, 0.15, 0.15]
    top_dir2 = rf"{top_dir}_reversible"
    out_dir = rf"{top_dir}_split"
    dir_list = [top_dir, top_dir2]
    multi_OCRDatasetV2_split(dir_list, out_dir, split_ratio, vis=True)


def get_files_by_path(tree: dict, path_list: list):
    """
    根据多级路径返回文件列表

    :param tree: 多层级目录树
    :param path_list: 目录路径列表
    :return: 文件列表（若不存在返回空列表）
    """

    if not isinstance(path_list, list) or not path_list:
        return []

    node = tree

    for key in path_list:
        if not isinstance(node, dict):
            return []

        if key not in node:
            return []

        node = node[key]

    if isinstance(node, dict):
        return node.get("__files__", [])

    return []


def ocr_det_process():
    primal_dir = r"E:\python_ai_dataset\OCR\det\Label_new"
    classified_dir = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2"  # 需要手动从primal_dir分类到classified_dir
    final_dir = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260205"  # 如果标注有错误可以在这里用PPOCRLabel修改
    dip_old = r"E:\python_ai_dataset\OCR\det\gather\DIP_OCR_collect"
    dip_dir = r"E:\python_ai_dataset\OCR\det\gather\DIP_OCR"
    latest_dir = rf"E:\python_ai_dataset\OCR\det\gather\categoriesV2_{date_num}"  # 最终的数据集会包含dip_dir
    exclude_list = [
        "4-Train996_reverse", "13-anno_20250225_AiDian_reverse", "25-OCR-opposite-20250927",
        "26-OCR-opposite-20250928"
    ]
    sample_rate = 0.3
    split_ratio = [0.7, 0.15, 0.15]

    # generate_ocr_empty_label(r"F:\python_ai_dataset\OCR\det\Label_new\36-OCR-DIP-20260422")
    # 确保已经手动分类完成再执行rename_classified_OCRDatesetV2
    # rename_classified_OCRDatesetV2(latest_dir, classified_dir, final_dir, exclude_list)
    # rename_classified_OCRDatasetV2_DIP(dip_old, dip_dir)

    # 确保标注正确再执行以下
    reorder_OCRDatasetV2(classified_dir, final_dir, latest_dir, dip_dir)
    shutil.copy(r"E:\python_ai_dataset\OCR\det\gather\replace_list.txt", latest_dir)
    shutil.copy(r"E:\python_ai_dataset\OCR\det\gather\reversible_list.txt", latest_dir)
    union_label_all_OCRDatasetV2(rf"{latest_dir}\DIP_OCR")
    union_label_all_OCRDatasetV2(latest_dir)
    reverse_OCRDatasetV2(latest_dir, sample_rate=sample_rate)
    union_label_all_OCRDatasetV2(rf"{latest_dir}_reversible")
    split_OCRDatasetV2(latest_dir, split_ratio=split_ratio)


def ocr_det_process_v2():
    # 原始的数据集由标注员标注完成，数据是按时间批次标注，是一个长期累积的集合;
    # 其中的标注数据是有可能标注错误或者不规范的，且数据是杂乱的不经分类的
    # 以primal的标注数据为基础
    primal_data_dirs_top = Path(r"E:\ds\OCR\anno\Primal")
    primal_data_dir_list = [primal_data_dirs_top / i for i in os.listdir(primal_data_dirs_top)]
    # 手动地从primal_data_dir分类到classified_dir中，是不带标注数据的
    # 以Categorized的图像数据为基础
    classified_dir = Path(r"E:\ds\OCR\anno\Categorized\categories_20260224")
    # 一个stable带标注数据集，是按classified_dir的分类，并且文件被重命名了，如果标注有错误可以在这里用PPOCRLabel修改
    # 以stable的标注数据为标准，标准标注数据会覆盖基础数据
    stable_label_dir = r"E:\ds\OCR\anno\Stable\categorized_20260303"
    # 处理后的衍生最终版本
    derivative_dir = rf"E:\ds\OCR\anno\Derive\derived_{date_num}"
    derivative_rotate_dir = rf"E:\ds\OCR\anno\Derive\derived_{date_num}_r"
    # 分好训练验证测试集进行
    AITrain_dir = rf"E:\ds\OCR\anno\AITrain\OCRV6.6_{date_num}"
    split_ratio = [0.72, 0.16, 0.12]

    def step1():
        def get_exclude_list(path):
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                exclude_list = []
                for line in lines:
                    exclude_list.append(line.strip())
            return exclude_list

        # 0. 构建
        exclude_list = get_exclude_list(primal_data_dirs_top / "exclude_list.txt")
        src_list_all = det_paths_level1(primal_data_dirs_top)
        primal_list = [str(p) for p in src_list_all if os.path.basename(p) not in exclude_list]
        print("primal_list:", len(primal_list), primal_list)
        primal_ds = OCRDatasetV2(primal_list, with_label=True)
        print("primal_ds:", len(primal_ds), "set:", len(set(primal_ds.image_map.values())))

        classified_list = det_paths_level1(classified_dir)
        for c in classified_list:
            classified_list += det_paths_level1(c)
        print("classified_list:", len(classified_list), classified_list)
        categorized_ds = OCRDatasetV2(classified_list)
        print("categorized_ds:", len(categorized_ds), "set:", len(set(categorized_ds.image_map.values())))

        stable_list = det_paths_level1(stable_label_dir)
        for c in stable_list:
            stable_list += det_paths_level1(c)
        print("stable_list:", len(stable_list), stable_list)
        stable_ds = OCRDatasetV2(stable_list, with_label=True)
        print("stable_ds:", len(stable_ds), "set:", len(set(stable_ds.image_map.values())))

        def dataset_verification():
            # 1. 判断 primal 里的数据是否全部分类好，除了exclude_list里的
            large_ds = primal_ds if len(primal_ds) > len(categorized_ds) else categorized_ds
            little_ds = primal_ds if len(primal_ds) <= len(categorized_ds) else categorized_ds
            diff_count = 0
            rep_count = 0
            unikey = {}
            for idx, name in large_ds.image_map.items():
                if name not in unikey:
                    unikey[name] = idx
                else:
                    rep_count += 1
                    print(rep_count, "repect:", large_ds.image(unikey[name]), large_ds.image(idx))
                if name not in little_ds.image_map.values():
                    print("missing:", large_ds.image(idx))
                    diff_count += 1
            print(f"primal <=> categorized total diff: {diff_count}\n" + "="*65)

            # 2. 构建primal到stable的映射表，对比标注数据
            rename_path = os.path.join(stable_label_dir, "rename_list.txt")
            print(rename_path)
            rename_map_id2names = {}
            rename_map_old2id = {}
            rename_map_new2id = {}
            rename_map_old2new = {}
            rename_map_new2old = {}
            rename_map_missing = {}
            with open(rename_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    s = line.strip().split(", ")
                    s[0] = s[0].strip(" ").strip("\t").strip("\r").strip("\n")
                    s[1] = s[1].strip(" ").strip("\t").strip("\r").strip("\n")
                    s[2] = s[2].strip(" ").strip("\t").strip("\r").strip("\n")
                    if len(s[0]) > 5:
                        print(s[0], s[1], s[2])
                    rename_map_id2names[int(s[0])] = (s[1], s[2])
                    rename_map_old2id[s[1]] = int(s[0])
                    rename_map_new2id[s[2]] = int(s[0])
                    rename_map_old2new[s[1]] = s[2]
                    rename_map_new2old[s[2]] = s[1]
                    rename_map_missing[int(s[0])] = 1
            print("lines:", len(lines), len(rename_map_id2names), len(rename_map_old2id))

            missing_count = 0
            for idx, n in categorized_ds.image_map.items():
                if n not in rename_map_old2id.keys():
                    missing_count += 1
                    print("missing:", categorized_ds.image(idx, True))
                else:
                    rename_map_missing[rename_map_old2id[n]] = 0
            print("categorized set total missing:", missing_count)
            print("="*65)

            missing_count = 0
            for idx, n in stable_ds.image_map.items():
                if n not in rename_map_new2id.keys():
                    missing_count += 1
                    print("missing:", stable_ds.image(idx, True))
                else:
                    rename_map_missing[rename_map_new2id[n]] = 0
            print("stable      set total missing:", missing_count)
            print("="*65)

            for i, missing in rename_map_missing.items():
                if missing == 1:
                    print(i, rename_map_id2names[i])

            return rename_map_id2names, rename_map_old2id, rename_map_new2id, rename_map_old2new, rename_map_new2old

        rename_map_id2names, rename_map_old2id, rename_map_new2id, rename_map_old2new, rename_map_new2old = dataset_verification()
        rename_map_id2old, rename_map_id2new = {v: k for k, v in rename_map_old2id.items()}, {v: k for k, v in rename_map_new2id.items()}
        derivative_ds = OCRDatasetV2([], with_label=True)
        print("derivative_ds:", len(derivative_ds))

        primal_map = {}
        for p, l in primal_ds:
            basename = os.path.basename(p)
            if basename not in primal_map:
                primal_map[basename] = [l]
            else:
                primal_map[basename].append(l)
        primal_keys = primal_map.keys()
        stable_map = {}
        for p, l in stable_ds:
            basename = os.path.basename(p)
            if basename not in stable_map:
                stable_map[basename] = [l]
            else:
                stable_map[basename].append(l)
        stable_keys = stable_map.keys()

        for p, _ in categorized_ds:
            basename = os.path.basename(p)
            if basename not in primal_keys:
                print(basename)
            else:
                l = primal_map[basename][0]
                if basename in rename_map_old2new:
                    n = rename_map_old2new[basename]
                    if n in stable_keys:
                        l = stable_map[n][0]
                if l is None:
                    print(basename, "is None label.")
                else:
                    derivative_ds.append(p, l)
        print("derivative_ds:", len(derivative_ds))
        print("="*65)

        os.makedirs(derivative_dir, exist_ok=True)
        rename_list_file = open(os.path.join(derivative_dir, RENAME_LIST_FILE), "w", encoding="utf-8")
        commonpath = os.path.commonpath(list(derivative_ds.roots_map.values()))

        def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
            img_path = Path(_img_path)
            ext = img_path.suffix  # include .jpg
            parent = img_path.parent

            relative_parts = str(parent).replace(commonpath, "").split(os.sep)[1:]
            if not any("Type" in r for r in relative_parts):
                relative_parts = relative_parts + ["Type0"]
            name_parts = []

            if index is not None:
                name_parts.append(f"UID{index}")

            if relative_parts:
                name_parts.append(SEP_CHAR.join(relative_parts))

            if _label_data is not None and len(_label_data) > 0:
                for idx, _la in enumerate(_label_data):
                    transcription = "{" + sanitize_filename(_la.get("transcription", "")) + "}"
                    name_parts.append(transcription)
            else:
                name_parts.append("{}")

            basename = SEP_CHAR.join(name_parts) + ext
            try:
                shutil.copy(_img_path, os.path.join(_dst_dir, basename))
                rename_list_file.write(f"{index}, {os.path.normpath(_img_path)}, {os.path.join(_dst_dir, basename)}\n")
            except FileNotFoundError as e:
                print(e, _img_path, os.path.join(_dst_dir, basename))

            if _label_file is not None:
                dirname = os.path.basename(_dst_dir)
                if _label_data is not None:
                    _label_str = _label_op(_label_data)
                    _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
                else:
                    _label_file.write(f"{dirname}/{basename}\t[]\n")

        dump_ocr_dataset(derivative_ds, derivative_dir, custom_image_label_op=image_label_op, overwriting=True)
        rename_list_file.close()
        union_label_all_OCRDatasetV2(Path(derivative_dir) / "DIP_OCR")
        union_label_all_OCRDatasetV2(derivative_dir)
        print("="*65)

    def step2():
        derived_list = det_paths_level1(derivative_dir)
        for c in derived_list:
            derived_list += det_paths_level1(c)
        print("derived_list:", len(derived_list), derived_list)
        derived_ds = OCRDatasetV2(derived_list, with_label=True)
        print("derived_ds:", len(derived_ds), "set:", len(set(derived_ds.image_map.values())))
        # VisualizeOCRDataset(derived_ds, save_dir=rf"{derivative_dir}_vis")()
        print("="*65)

        def get_rotatable_list(path):
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                rotatable_list = []
                for line in lines:
                    rotatable_list.append(os.path.normpath(line.strip()))
            return rotatable_list

        rotatable_list = get_rotatable_list(classified_dir / r"rotatable_list.txt")

        def sample_condition(item):
            p, l = item
            p = os.path.normpath(p)
            for rot in rotatable_list:
                if rot in p:
                    return True
            return False

        sample = derived_ds.sample(0.5, condition=sample_condition, seed=date_num)
        rotate_ds = derived_ds.subset(sample)

        random.seed(date_num)
        offset = len(derived_ds)
        os.makedirs(derivative_rotate_dir, exist_ok=True)
        rotate_angle_file = open(os.path.join(derivative_rotate_dir, "rotate_angle.txt"), "w", encoding="utf-8")
        commonpath = os.path.commonpath(list(rotate_ds.roots_map.values()))
        def rotate_image_label(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
            img_path = Path(_img_path)
            ext = img_path.suffix  # include .jpg
            parent = img_path.parent
            img = imread(str(img_path))
            angle = random.choice([90, 180, 270])
            relative_parts = str(parent).replace(commonpath, "").split(os.sep)[1:]

            # rimg, M = rotate_image_auto(img, angle)

            if not any("Type" in r for r in relative_parts):
                relative_parts = relative_parts + ["Type0"]
            name_parts = []

            if index is not None:
                name_parts.append(f"UID{index + offset}")

            if relative_parts:
                name_parts.append(SEP_CHAR.join(relative_parts))

            if _label_data is not None and len(_label_data) > 0:
                for idx, _la in enumerate(_label_data):
                    transcription = "{" + sanitize_filename(_la.get("transcription", "")) + "}"
                    name_parts.append(transcription)
                    rpoints = rotate_bbox_xyxyxyxy_auto(np.array(_la.get("points")), angle, img.shape[:2]).astype(np.int32)
                    # rpoints = rotate_bbox_with_M(np.array(_la.get("points")), M).astype(np.int32)
                    ordered_points = order_rectangle_points(rpoints)
                    _la["points"] = ordered_points.tolist()
            else:
                name_parts.append("{}")

            basename = SEP_CHAR.join(name_parts) + ext
            try:
                rimg, _ = rotate_image_auto(img, angle)
                imwrite(os.path.join(_dst_dir, basename), rimg)
                rotate_angle_file.write(f"{index + offset}, {angle}, {os.path.normpath(_img_path)}, {os.path.join(_dst_dir, basename)}\n")
            except FileNotFoundError as e:
                print(e, _img_path, os.path.join(_dst_dir, basename))

            if _label_file is not None:
                dirname = os.path.basename(_dst_dir)
                if _label_data is not None:
                    _label_str = _label_op(_label_data)
                    _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
                else:
                    _label_file.write(f"{dirname}/{basename}\t[]\n")

        dump_ocr_dataset(rotate_ds, derivative_rotate_dir, custom_image_label_op=rotate_image_label, overwriting=True)
        rotate_angle_file.close()

        union_label_all_OCRDatasetV2(Path(derivative_rotate_dir) / "DIP_OCR")
        union_label_all_OCRDatasetV2(derivative_rotate_dir)
        derived_rotate_list = det_paths_level1(derivative_rotate_dir)
        for c in derived_rotate_list:
            derived_rotate_list += det_paths_level1(c)
        print("derived_rotate_list:", len(derived_rotate_list), derived_rotate_list)
        derived_rotate_ds = OCRDatasetV2(derived_rotate_list, with_label=True)
        print("derived_rotate_ds:", len(derived_rotate_ds), "set:", len(set(derived_rotate_ds.image_map.values())))
        time.sleep(0.5)
        # VisualizeOCRDataset(derived_rotate_ds, save_dir=rf"{derivative_rotate_dir}_vis")()
        print("="*65)

    def OCRDatesetV2_dump_for_AITrain(src_dirs, dst_dir, direct="to"):
        if src_dirs is None:
            return
        elif isinstance(src_dirs, str):
            src_dirs = [src_dirs]
        elif isinstance(src_dirs, list):
            pass

        os.makedirs(dst_dir, exist_ok=True)
        all_dirs = []
        for src_dir in src_dirs:
            sub_dirs = det_paths_level1(src_dir)
            for c in sub_dirs:
                sub_dirs += det_paths_level1(c)
            all_dirs += sub_dirs

        # ds = OCRDatasetV2(all_dirs, with_label=True)
        # print(f"{len(ds) = }", f"{len(all_dirs) = }")
        multi_OCRDatasetV2_split(all_dirs, dst_dir, split_ratio=split_ratio, vis=True)

        # def for_AITrain(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        #     img_path = Path(_img_path)
        #     basename = img_path.name
        #
        #     shutil.copy(_img_path, os.path.join(AITrain_dir, basename))
        #     if _label_file is not None:
        #         dirname = os.path.basename(AITrain_dir)
        #         if _label_data is not None:
        #             _label_str = _label_op(_label_data)
        #             _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
        #         else:
        #             _label_file.write(f"{dirname}/{basename}\t[]\n")
        #
        # dump_ocr_dataset(ds, dst_dir, custom_image_label_op=for_AITrain, overwriting=True)
        # del ds, all_dirs

    # step1()
    # step2()
    OCRDatesetV2_dump_for_AITrain([derivative_dir, derivative_rotate_dir], AITrain_dir)


def rename_dir_images():
    path = r"E:\datasets\OCR\annotated\2.Categorized\categories_20260224\DIP_OCR\SMT_EleCapacitors\Type5"
    images = [i for i in os.listdir(path) if i.rsplit('.', 1)[-1] in IMG_FORMATS]
    for i in images:
        print(i)
        os.rename(os.path.join(path, i),
                  os.path.join(path, i.replace("Type5", "Type0")))


def do_matting(det_dataset_map, out_dir, commonpath, offset=0):
    if all(len(det_dataset) == 0 for det_dataset in det_dataset_map.values()):
        print("no data in dataset")
        return

    mid = offset if offset is not None else 0
    for path, dataset in det_dataset_map.items():
        sub_path = Path(path.replace(commonpath, ""))
        new_path = path.replace(commonpath, out_dir)
        new_dirname = os.path.basename(new_path)
        os.makedirs(new_path, exist_ok=True)
        label_file = os.path.join(new_path, "Label.txt")
        sub_parts = list(Path(sub_path).parts[1:])
        if not any("Type" in p for p in sub_parts):
            sub_parts += ["Type0"]
        f = open(label_file, "w", encoding="utf-8")

        for img_path, lab_data in tqdm(dataset, desc=f"matting dataset: {sub_path}"):
            uid = os.path.basename(img_path).split(SEP_CHAR)[0]
            img = imread(img_path)
            for offset_id, la in enumerate(lab_data):
                save_img_name = SEP_CHAR.join(
                    [rf"MID{mid}", uid, *sub_parts, f"{{{sanitize_filename(la['transcription'])}}}", "png"]
                )
                rect = cv2.boundingRect(np.array(la["points"]))
                if len(img.shape) == 3:
                    roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]
                    imwrite(os.path.join(new_path, save_img_name), roi)
                    f.write(
                        f"{new_dirname}/{save_img_name}\t{OCRRECDatasetV2.fmt_label_dumps(la['transcription'])}\n")
                    # print(os.path.join(new_path, save_img_name),
                    # f"{new_dirname}/{save_img_name}\t{OCRRECDatasetV2.fmt_label_dumps(la['transcription'])}\n")
                    mid += 1
        f.close()
    return mid


def matting_ocr_dataset_to_rec(src_dir, out_dir, offset=None):
    all_dir = get_all_dir(src_dir)
    det_dataset_map = {
        i: OCRDatasetV2(i, with_label=True, subject_to='label')
        for i in all_dir
    }
    commonpath = os.path.commonpath(all_dir)
    # _offset = do_matting(det_dataset_map, out_dir, commonpath, offset)
    _offset = len(OCRRECDatasetV2(r"E:\python_ai_dataset\OCR\rec\gather\categoriesV2_20260303_done",
                              with_label=True, subject_to='label'))
    if offset is None:
        offset = _offset

    def expand_ratio(p):
        if "SMT_EleCapacitors" in p:
            return [0.3, 0.3, 0.25, 0.25]
        elif "SMT_IC" in p or "SMT_IC" in p:
            return [0.3, 0.3, 0.2, 0.2]
        elif "SMT_Inductor" in p or "SMT_QFN" in p:
            return [0.3, 0.3, 0.2, 0.2]
        elif "SMT_Mosfet" in p:
            return [0.4, 0.4, 0.45, 0.45]
        elif "Others" in p or "SMT_Diode" in p:
            return [0.4, 0.4, 0.35, 0.35]
        else:
            return [0.35, 0.35, 0.35, 0.35]

    expand_datasets = {
        path: box_random_expand(
            # dataset.subset(dataset.sample(0.3)),
            dataset,
            expand_ratio=expand_ratio(path),
            expand_direction=["left", "right", "top", "bottom"],
            seed=date_num
        )
        for path, dataset in tqdm(det_dataset_map.items(), desc="box random expand")
    }
    do_matting(expand_datasets, f"{out_dir}_expand_tmp", commonpath)

    expand_tmp_dirs = get_all_dir(f"{out_dir}_expand_tmp")
    rotate_dict = YMLParser.load(path=r"E:\python_ai_dataset\OCR\det\gather\rotate_list.yml")
    commonpath = os.path.commonpath(expand_tmp_dirs)

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        if not any(c in _label_data for c in unidirect_char):
            return
        img_path = Path(_img_path)
        ext = img_path.suffix
        parent = img_path.parent
        basename = img_path.name
        dirname = os.path.basename(parent)

        relative_dirs = str(parent).replace(commonpath, "").split(os.sep)[1:]

        rotate_file = get_files_by_path(rotate_dict, relative_dirs)
        if basename in rotate_file:
            return
        # basename = SEP_CHAR.join([f"MID{index + offset}"] + basename.split(f"{SEP_CHAR}")[1:])

        shutil.move(_img_path, os.path.join(_dst_dir, basename))
        _label_file.write(f"{dirname}/{basename}\t{_label_data}\n")

    dump_ocr_dataset(
        OCRRECDatasetV2(expand_tmp_dirs, with_label=True, subject_to='label'),
        f"{out_dir}_expand",
        custom_image_label_op=image_label_op
    )

    # union_label_all_OCRDatasetV2(f"{out_dir}/DIP_OCR")
    # union_label_all_OCRDatasetV2(out_dir)
    # union_label(
    #     [os.path.join(i, "Label.txt") for i in det_paths_level1(out_dir)],
    #     os.path.join(out_dir, "Label.txt")
    # )

    shutil.rmtree(f"{out_dir}_expand_tmp")

    expand_out_dir = f"{out_dir}_expand"
    out_offset = offset
    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        nonlocal out_offset
        dirname = os.path.basename(_dst_dir)
        basename = os.path.basename(_img_path)
        basename = SEP_CHAR.join([f"MID{index + offset}"] + basename.split(SEP_CHAR)[1:])
        os.rename(_img_path, os.path.join(_dst_dir, basename))
        _label_file.write(f"{dirname}/{basename}\t{_label_data}\n")
        out_offset = index + offset + 1

    dump_ocr_dataset(
        OCRRECDatasetV2(get_all_dir(expand_out_dir), with_label=True, subject_to='label'),
        expand_out_dir,
        custom_image_label_op=image_label_op,
        overwriting=True
    )

    union_label_all_OCRDatasetV2(f"{expand_out_dir}/DIP_OCR")
    union_label_all_OCRDatasetV2(expand_out_dir)
    union_label(
        [os.path.join(i, "Label.txt") for i in det_paths_level1(expand_out_dir)],
        os.path.join(expand_out_dir, "Label.txt")
    )
    return out_offset


def reverse_OCRRECDatasetV2(rec_dir, replace_list_file, offset=None, angle=180):
    d = OCRRECDatasetV2(get_all_dir(rec_dir), with_label=True, subject_to='label')
    if offset is None:
        offset = len(d)

    commonpath = os.path.commonpath(get_all_dir(rec_dir))

    rm = {}
    with open(replace_list_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            rm[key] = value

    _index = 0

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        nonlocal _index
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        if not any(c in _label_data for c in unidirect_char):
            return
        img_path = Path(_img_path)
        ext = img_path.suffix
        parent = img_path.parent
        uid = os.path.basename(img_path).split(SEP_CHAR)[1]
        relative_parts = [uid] + str(parent).replace(commonpath, "").split(os.sep)[1:]
        if not any("Type" in r for r in relative_parts):
            relative_parts = relative_parts + ["Type0"]
        name_parts = [f"MID{_index + offset}"]

        # if index is not None:
        _index += 1

        if relative_parts:
            name_parts.append(SEP_CHAR.join(relative_parts))

        im = imread(_img_path)

        dirname = os.path.basename(_dst_dir)
        _label_str = reverse_order_ocr_string(_label_data, rm)
        name_parts.append("{" + sanitize_filename(_label_str) + "}")
        basename = SEP_CHAR.join(name_parts) + ext

        if _label_file is not None:
            _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
        im = rotate_image_around_point(im, (im.shape[1] // 2, im.shape[0] // 2), angle)
        imwrite(os.path.join(_dst_dir, basename), im)

    dump_ocr_dataset(
        d,
        rf"{rec_dir}_reversible",
        custom_image_label_op=image_label_op,
    )
    union_label_all_OCRDatasetV2(f"{rec_dir}_reversible/DIP_OCR")
    union_label_all_OCRDatasetV2(rf"{rec_dir}_reversible")
    union_label(
        [os.path.join(i, "Label.txt") for i in det_paths_level1(rf"{rec_dir}_reversible")],
        os.path.join(rf"{rec_dir}_reversible", "Label.txt")
    )


def OCRRECDatesetV2_dump_for_AITrain(rec_dir, dst_dir, direct="to"):
    if direct == "to":
        d = OCRRECDatasetV2(
            get_all_dir(rec_dir),
            with_label=True,
            read_image=False,
            subject_to="label",
        )
    else:
        d = OCRDatasetV2(
            get_all_dir(rec_dir),
            with_label=True,
            read_image=False,
            subject_to="label",
        )

    os.makedirs(dst_dir, exist_ok=True)

    def op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        basename = os.path.basename(_img_path)
        im = imread(_img_path)
        h, w, _ = im.shape
        shutil.copy(_img_path, os.path.join(_dst_dir, basename))
        if _label_file is not None and _label_data is not None:
            dirname = os.path.basename(_dst_dir)
            if direct == "to":
                # _label_str = _label_op(_label_data)
                lab = [{"transcription": _label_data, "points": [[0, 0], [w, 0], [w, h], [0, h]], "difficult": 0}]
                _label_str = (str(lab).replace("'", '"')
                              .replace('"difficult": 0', '"difficult": false')
                              .replace('"difficult": 1', '"difficult": true'))
            else:
                _label_str = _label_data[0]["transcription"]
                # for la in _label_data:
                #     _label_str = la["transcription"]
            _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")

    dump_ocr_dataset(d, dst_dir, custom_image_label_op=op, overwriting=True)
    union_label_all_OCRDatasetV2(f"{dst_dir}/DIP_OCR")
    union_label_all_OCRDatasetV2(dst_dir)
    union_label(
        [os.path.join(i, "Label.txt") for i in det_paths_level1(dst_dir)],
        os.path.join(dst_dir, "Label.txt")
    )


def split_OCRRECDatasetV2(dir_list, out_dir, split_ratio=None, seed=None, vis=False):
    if split_ratio is None:
        split_ratio = [0.7, 0.15, 0.15]
    multi_OCRDatasetV2_split(dir_list, out_dir,
                             dataset_class=OCRRECDatasetV2,
                             split_ratio=split_ratio, seed=None, vis=vis)


def ocr_rec_process():
    _date_num = 20260303  # date_num
    det_dir = rf"E:\python_ai_dataset\OCR\det\gather\categoriesV2_{_date_num}"
    out_dir1 = r"E:\python_ai_dataset\OCR\rec\gather\categoriesV2_20260226"
    out_dir = rf"E:\python_ai_dataset\OCR\rec\gather\categoriesV2_{_date_num}"
    replace_file = r"E:\python_ai_dataset\OCR\det\gather\replace_list.txt"
    split_ratio = [0.6, 0.2, 0.2]

    # offset = matting_ocr_dataset_to_rec(det_dir, out_dir)
    # reverse_OCRRECDatasetV2(f"{out_dir}_done", replace_file, offset=offset)
    OCRRECDatesetV2_dump_for_AITrain(f"{out_dir}_done",
                                     f"{out_dir}_done_AITrain", direct="to")
    OCRRECDatesetV2_dump_for_AITrain(f"{out_dir}_expand",
                                     f"{out_dir}_expand_AITrain", direct="to")
    OCRRECDatesetV2_dump_for_AITrain(f"{out_dir}_done_reversible",
                                     f"{out_dir}_done_reversible_AITrain", direct="to")

    d1 = OCRRECDatasetV2(get_all_dir(f"{out_dir}_expand_AITrain"), with_label=True, subject_to="label")
    d2 = OCRRECDatasetV2(get_all_dir(f"{out_dir}_done_reversible_AITrain"), with_label=True, subject_to="label")
    d1_s = d1.subset(d1.sample(0.3, seed=_date_num))
    d2_s = d2.subset(d2.sample(0.3, seed=_date_num))
    dump_ocr_dataset(d1_s, f"{out_dir}_expand_AITrain_sample", image_file_op="move")
    dump_ocr_dataset(d2_s, f"{out_dir}_done_reversible_AITrain_sample", image_file_op="move")

    split_OCRRECDatasetV2([
        f"{out_dir}_done_AITrain",
        f"{out_dir}_expand_AITrain_sample",
        f"{out_dir}_done_reversible_AITrain_sample"
    ],
        f"{out_dir}_split",
        split_ratio=split_ratio,
        vis=False
    )


def OCRRECDatasetV2_to_cls(rec_dir_0, rec_dir_180, cls_dir):
    d0 = OCRRECDatasetV2(rec_dir_0, with_label=True, read_image=False, subject_to="label")
    d180 = OCRRECDatasetV2(rec_dir_180, with_label=True, read_image=False, subject_to="label")

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        img_path = Path(_img_path)
        shutil.copy(_img_path, os.path.join(_dst_dir, img_path.name))
        if _label_file is not None:
            _label_file.write(f"{ang}/{img_path.name}\t{ang}\n")

    ang = 0
    dump_ocr_dataset(d0, f"{cls_dir}/0", custom_image_label_op=image_label_op, overwriting=True)
    ang = 180
    dump_ocr_dataset(d180, f"{cls_dir}/180", custom_image_label_op=image_label_op, overwriting=True)


def ocr_cls_process():
    rec_dir_0 = rf"E:\python_ai_dataset\OCR\rec\gather\categoriesV2_20260303_done"
    rec_dir_180 = f"{rec_dir_0}_reversible"
    cls_dir = rf"E:\python_ai_dataset\OCR\cls\gather\categoriesV2_{date_num}"
    dst_dir = fr"{cls_dir}_split"

    OCRRECDatasetV2_to_cls(rec_dir_0, rec_dir_180, cls_dir)
    cls_dir_list = [rf"{cls_dir}/0", rf"{cls_dir}/180"]
    d = OCRCLSDatasetV2(cls_dir_list, with_label=True, read_image=False, subject_to="label",
                        categories={0: '0', 1: '180'})

    subsets = d.split([0.6, 0.2, 0.2], seed=date_num)
    for i, (name, subset_list) in enumerate(subsets.items()):
        dump_ocr_dataset(d.subset(subset_list), f"{dst_dir}/{name}", overwriting=True)


def parse_tree_diff(html_path, encoding="gb2312"):
    with open(html_path, "r", encoding=encoding, errors="ignore") as f:
        content = f.read()

    def create_soup(content):
        try:
            return BeautifulSoup(content, "lxml")
        except:
            return BeautifulSoup(content, "html.parser")

    soup = create_soup(content)

    tree = {}
    stack = []

    table = soup.find("table", class_="dc")

    for tr in table.find_all("tr"):
        td = tr.find("td", class_="AlignLeft")
        if not td:
            continue

        # 计算深度
        depth = 0
        for img in td.find_all("img"):
            alt = img.get("alt", "")
            if alt in ["|", "+", "\\", " "]:
                depth += 1

        # 判断是否目录
        dir_img = td.find("img", alt="<DIR>")
        text = td.get_text(strip=True)

        if not text:
            continue

        # 调整栈深度
        while len(stack) > depth:
            stack.pop()

        if dir_img:
            # 目录节点
            node = {}
            if not stack:
                tree[text] = node
            else:
                parent = stack[-1]
                parent[text] = node

            stack.append(node)

        else:
            # 文件节点
            if stack:
                current = stack[-1]
                current.setdefault("__files__", []).append(text)

    return tree


def ocr_rec_special_process():
    _date_num = 20260303
    commonpath = rf"E:\python_ai_dataset\OCR\rec\gather\categoriesV2_{_date_num}"

    d0 = OCRRECDatasetV2(get_all_dir(f"{commonpath}_done"), with_label=True, subject_to="label")
    d1 = OCRRECDatasetV2(get_all_dir(f"{commonpath}_expand"), with_label=True, subject_to="label")
    d2 = OCRRECDatasetV2(get_all_dir(f"{commonpath}_done_reversible"), with_label=True, subject_to="label")
    total_len = len(d0) + len(d1) + len(d2)
    print(f"{len(d0) = } + {len(d1) = } + {len(d2) = } => {total_len = }")
    d1_s = d1.subset(d1.sample(0.3, seed=_date_num))
    d2_s = d2.subset(d2.sample(0.3, seed=_date_num))
    dump_ocr_dataset(d1_s, f"{commonpath}_expand_sample", image_file_op="copy")
    dump_ocr_dataset(d2_s, f"{commonpath}_done_reversible_sample", image_file_op="copy")

    split_OCRRECDatasetV2([
        f"{commonpath}_done",
        f"{commonpath}_expand_sample",
        f"{commonpath}_done_reversible_sample"
    ],
        f"{commonpath}_rec_split",
        split_ratio=[0.6, 0.2, 0.2],
        seed=_date_num,
        vis=False
    )


def ocr_cls_special_process():
    commonpath = rf"E:\python_ai_dataset\OCR\cls\gather\categoriesV2_20260303_split"
    union_label_all_OCRDatasetV2(commonpath)


def ocr_rec_char_dict():

    chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-',
             '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':',
             ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
             'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a',
             'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{',
             '|', '}', '~']

    chars = sorted(chars)
    print(chars)

    # commonpath = rf"E:\python_ai_dataset\OCR\rec\gather\categoriesV2_20260303_split"
    # d = OCRDatasetV2(get_all_dir(commonpath), with_label=True, subject_to="label")
    #
    # char_dict = {}
    # for i, (img_path, label) in enumerate(d):
    #     f = False
    #     for l in label:
    #         for c in l['transcription']:
    #             char_dict[c] = char_dict.get(c, 0) + 1
    #             if c == ' ':
    #                 f = True
    #     if f:
    #         print(f"{i = } {img_path = } {label = }")
    # print(YMLParser.dumps(char_dict))
    #
    # char_dict_final = sorted(set(char_dict.keys()))
    with open(r"custom_en_dict.txt", 'w', encoding='utf-8') as f:
        for c in chars:
            f.write(c + '\n')
        # for c in chars:
        #     if c not in char_dict_final:
        #         print(f"{c = } not in char_dict_final")
    # return char_dict


def ocr_det_dataset2yolo():
    ocr_ds = OCRDatasetV2(get_all_dir(r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260303"),
                          with_label=True,
                          subject_to="label")
    # convert_ocr2yolo(ocr_ds,
    #                  r"E:\python_ai_dataset\OCR\det_yolo\categoriesV2_20260303\labels")
    subsets = ocr_ds.split([0.7,0.2,0.1], seed=date_num, grouped=True)
    for name, subset_list in subsets.items():
        subset = ocr_ds.subset(subset_list)
        dst_img_dir = rf"E:\python_ai_dataset\OCR\det_yolo\categoriesV2_{date_num}\images\{name}"
        dst_lab_dir = rf"E:\python_ai_dataset\OCR\det_yolo\categoriesV2_{date_num}\labels\{name}"
        dump_ocr_dataset(subset, dst_img_dir)
        ocr_subds = OCRDatasetV2(get_all_dir(dst_img_dir), with_label=True, subject_to="label")
        convert_ocr2yolo(ocr_subds, dst_lab_dir, task='det')


if __name__ == "__main__":
    # ocr_det_process()
    # ocr_rec_process()
    # ocr_rec_special_process()
    # ocr_cls_special_process()
    # ocr_cls_process()
    # d = parse_tree_diff(r"C:\Users\WQS\Desktop\h.txt")
    # print(YMLParser.dumps(d, allow_unicode=True, indent=2))

    # ocr_det_dataset2yolo()
    ocr_det_process_v2()
