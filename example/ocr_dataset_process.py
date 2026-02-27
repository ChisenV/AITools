import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

os.environ["PATH"] = r"E:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

from AITools import IMG_FORMATS
from AITools.comp.dataset import OCRDatasetV2, dump_ocr_dataset, OCRRECDatasetV2, OCRCLSDatasetV2
from AITools.comp.functions import sanitize_filename, union_label, reverse_order_ocr_string, rotate_bbox_xyxyxyxy, \
    imread, rotate_image_around_point, imwrite
from AITools.comp.parser import JSONParser
from AITools.comp.processor import VisualizeOCRDataset

SEP_CHAR = "."
RENAME_LIST_FILE = "rename_list.txt"


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
    from datetime import datetime
    date_num = int(datetime.now().strftime("%Y%m%d"))

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


def multi_OCRDatasetV2_split(ocr_dataset_list, out_dir, split_ratio=None, vis=False):
    if split_ratio is None:
        split_ratio = [0.7, 0.15, 0.15]
    dir_list = ocr_dataset_list

    all_dir = []
    for d in dir_list:
        all_dir += get_all_dir(d)

    ocr_dataset_map = {
        i: OCRDatasetV2(
            i,
            with_label=True,
            subject_to='label'
        )
        for i in tqdm(all_dir, desc="loading dataset")
    }
    split_map = {}
    for path, dataset in ocr_dataset_map.items():
        subsets = dataset.split(split_ratio)
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


def matting_ocr_dataset_to_rec(src_dir, out_dir):
    all_dir = get_all_dir(src_dir)
    det_dataset_map = {
        i: OCRDatasetV2(i, with_label=True, subject_to='label')
        for i in all_dir
    }
    if all(len(det_dataset) == 0 for det_dataset in det_dataset_map.values()):
        print("no data in dataset")
        return

    commonpath = os.path.commonpath(all_dir)

    mid = 0
    for path, dataset in det_dataset_map.items():
        sub_path = Path(path.replace(commonpath, ""))
        new_path = path.replace(commonpath, out_dir)
        new_dirname = os.path.basename(new_path)
        os.makedirs(new_path, exist_ok=True)
        label_file = os.path.join(new_path, "Label.txt")
        sub_parts = Path(sub_path).parts[1:]

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
                    f.write(f"{new_dirname}/{save_img_name}\t{OCRRECDatasetV2.fmt_label_dumps(la['transcription'])}\n")
                    # print(os.path.join(new_path, save_img_name),
                    # f"{new_dirname}/{save_img_name}\t{OCRRECDatasetV2.fmt_label_dumps(la['transcription'])}\n")
                    mid += 1
        f.close()


def ocr_det_process():
    primal_dir = r"E:\python_ai_dataset\OCR\det\Label_new"
    classified_dir = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2"  # 需要手动从primal_dir分类到classified_dir
    final_dir = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260205"  # 如果标注有错误可以在这里用PPOCRLabel修改
    dip_old = r"E:\python_ai_dataset\OCR\det\gather\DIP_OCR_collect"
    dip_dir = r"E:\python_ai_dataset\OCR\det\gather\DIP_OCR"
    latest_dir = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260226"  # 最终的数据集会包含dip_dir
    exclude_list = [
        "4-Train996_reverse", "13-anno_20250225_AiDian_reverse", "25-OCR-opposite-20250927",
        "26-OCR-opposite-20250928"
    ]
    sample_rate = 0.3
    split_ratio = [0.7, 0.15, 0.15]

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

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        img_path = Path(_img_path)
        ext = img_path.suffix
        parent = img_path.parent
        uid = os.path.basename(img_path).split(SEP_CHAR)[1]
        relative_parts = [uid] + str(parent).replace(commonpath, "").split(os.sep)[1:]
        if not any("Type" in r for r in relative_parts):
            relative_parts = relative_parts + ["Type0"]
        name_parts = []

        if index is not None:
            name_parts.append(f"MID{index + offset}")

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
            rec_dir,
            with_label=True,
            read_image=False,
            subject_to="label",
        )
    else:
        d = OCRDatasetV2(
            rec_dir,
            with_label=True,
            read_image=False,
            subject_to="label",
        )

    os.makedirs(dst_dir, exist_ok=True)
    label_file = open(os.path.join(dst_dir, "Label.txt"), "w", encoding="utf-8")

    def op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        basename = os.path.basename(_img_path)
        im = imread(_img_path)
        h, w, _ = im.shape
        shutil.copy(_img_path, os.path.join(dst_dir, basename))
        if _label_file is not None and _label_data is not None:
            dirname = os.path.basename(dst_dir)
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
            label_file.write(f"{dirname}/{basename}\t{_label_str}\n")

    dump_ocr_dataset(d, dst_dir, custom_image_label_op=op, overwriting=True)
    label_file.close()


def split_OCRRECDatasetV2(dir_list, out_dir, split_ratio=None, vis=False):
    if split_ratio is None:
        split_ratio = [0.7, 0.15, 0.15]
    multi_OCRDatasetV2_split(dir_list, out_dir, split_ratio, vis=vis)


def ocr_rec_process():
    det_dir = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260226"
    out_dir = r"E:\python_ai_dataset\OCR\rec\gather\categoriesV2_20260226"
    replace_file = r"E:\python_ai_dataset\OCR\det\gather\replace_list.txt"
    split_ratio = [0.6, 0.2, 0.2]
    # matting_ocr_dataset_to_rec(det_dir, out_dir)
    # union_label_all_OCRDatasetV2(f"{out_dir}/DIP_OCR")
    # union_label_all_OCRDatasetV2(out_dir)
    # union_label(
    #     [os.path.join(i, "Label.txt") for i in det_paths_level1(out_dir)],
    #     os.path.join(out_dir, "Label.txt")
    # )
    reverse_OCRRECDatasetV2(out_dir, replace_file)
    OCRRECDatesetV2_dump_for_AITrain(out_dir, f"{out_dir}_AITrain", direct="to")
    OCRRECDatesetV2_dump_for_AITrain(f"{out_dir}_reversible",
                                     f"{out_dir}_reversible_AITrain", direct="to")
    split_OCRRECDatasetV2([f"{out_dir}_AITrain", f"{out_dir}_reversible_AITrain"],
                          f"{out_dir}_split",
                          split_ratio=split_ratio, vis=True)


def OCRRECDatasetV2_to_cls():
    rec_dir_0 = r"E:\python_ai_dataset\OCR\rec\gather\categoriesV2_20260226"
    rec_dir_180 = f"{rec_dir_0}_reversible"

    cls_dir = r"E:\python_ai_dataset\OCR\cls\gather\categoriesV2_20260226"
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
    # OCRRECDatasetV2_to_cls()
    cls_dir = r"E:\python_ai_dataset\OCR\cls\gather\categoriesV2_20260226"
    dst_dir = fr"{cls_dir}_split"
    cls_dir_list = [rf"{cls_dir}/0", rf"{cls_dir}/180"]
    d = OCRCLSDatasetV2(cls_dir_list, with_label=True, read_image=False, subject_to="label",
                        categories={0: '0', 1: '180'})

    from datetime import datetime
    date_num = int(datetime.now().strftime("%Y%m%d"))

    subsets = d.split([0.6, 0.2, 0.2], seed=date_num)
    for i, (name, subset_list) in enumerate(subsets.items()):
        dump_ocr_dataset(d.subset(subset_list), f"{dst_dir}/{name}", overwriting=True)


if __name__ == "__main__":
    # ocr_det_process()
    # ocr_rec_process()
    ocr_cls_process()
