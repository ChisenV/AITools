import os

os.environ["PATH"] = r"E:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

import os.path
import random
import shutil
from pathlib import Path
import cv2
import numpy as np

from AITools import IMG_FORMATS
from AITools.comp.dataset import *
from AITools.comp.functions import *
from AITools.comp.functions import \
    convert_coco2yolo, \
    rotate_image_around_point, \
    generate_yolo_empty_labels, \
    date_utils, \
    convert_yolo2coco, \
    sanitize_filename, convert_labelme_json_to_ocr_txt
from AITools.comp.processor import VisualizeOCRDataset, VisualizeYOLODataset, CropImages
from AITools.utils.stitch_images import stitch_images

abs_file = r"E:\python_ai_dataset\train\Annotations\PinHole@201@201@1@pin0@ID27417(63.44)-NG.xml"
rel_file = r"train\Annotations\PinHole@201@201@1@pin0@ID27417(63.44)-NG.xml"


def path_test():
    print(Path(rel_file))

    print(os.path.normpath(rel_file).split(os.sep))
    print(os.path.join(*os.path.normpath(rel_file).split(os.sep)[1:]))
    print(os.path.dirname(abs_file))


def det_paths(path=r"E:\python_ai_dataset\OCR\det\gather\categories"):
    return [os.path.join(path, i) for i in os.listdir(path)
            if os.path.isdir(os.path.join(path, i))]


def rec_paths(top_dir=r"E:\python_ai_dataset\OCR\rec\gather\categories"):
    return [os.path.join(top_dir, i) for i in os.listdir(top_dir)
            if os.path.isdir(os.path.join(top_dir, i))]


def OCRDatesetV2_init_case1():
    d = OCRDatasetV2()
    assert len(d) == 0
    print(d.__dict__)


def OCRDatesetV2_init_case2():
    d = OCRDatasetV2(det_paths())
    assert len(d) != 0, "len(d) == {}".format(len(d))
    d = d + d
    assert len(d) != len(d) * 2, "len(d) == {}".format(len(d))


def OCRDatesetV2_init_case3():
    pass


def OCRDatesetV2_sample_case0():

    def condition(item):
        l = [
            "Audion", "Diode-type1", "IC-type3", "Mosfet",
            "Resistor-type7", "Resistor-type8", "RT"
        ]
        r = item[0] if isinstance(item, tuple) else item
        for i in l:
            if i in r:
                return False

        return True

    d = OCRDatasetV2(
        det_paths(r"E:\python_ai_dataset\OCR\det\gather\categories-copy"),
        with_label=True,
        read_image=False
    )
    subset = d.sample(0.45, seed=20250507, condition=condition)

    d_copy = d.copy()
    d_copy.wash(subset, mode='keep')
    dst_dir = r"E:\python_ai_dataset\OCR\det\gather\categories_20250507"

    def op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None):
        basename = os.path.basename(_img_path)
        dirname = os.path.basename(_dst_dir)
        im = imread(_img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        iterations = 1
        kernel = np.ones((3, 3), np.uint8)
        # 前景的区域大，先腐蚀后膨胀; 背景的区域大，先膨胀后腐蚀
        im = cv2.morphologyEx(
            im,
            cv2.MORPH_OPEN
            if np.count_nonzero(im) > im.shape[0] * im.shape[1] // 2
            else cv2.MORPH_CLOSE
            ,
            kernel,
            iterations=iterations
        )

        if _label_data is not None:
            new_label_data = []
            for i, _la in enumerate(_label_data):
                # _label_data[i]["points"]: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                rect = cv2.boundingRect(np.array(_label_data[i]["points"]))
                if len(im.shape) == 3:
                    roi = im[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]
                else:
                    roi = im[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                if roi.shape[0] * roi.shape[1] * 0.1 < np.count_nonzero(roi) < roi.shape[0] * roi.shape[1] * 0.9:
                    new_label_data.append(_label_data[i])
            if len(new_label_data) != 0:
                _label_str = _label_op(new_label_data) if _label_op is not None else str(new_label_data)
                _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
                imwrite(os.path.join(_dst_dir, basename), im)
        else:
            imwrite(os.path.join(_dst_dir, basename), im)

    dump_ocr_dataset(d_copy, dst_dir, custom_image_label_op=op, overwriting=True)
    label_files = [os.path.join(dst_dir, i, "Label.txt")
                   for i in os.listdir(dst_dir)
                   if os.path.isdir(os.path.join(dst_dir, i))]
    union_label(label_files, os.path.join(dst_dir, "Label.txt"))


def OCRDatesetV2_sample_case1():

    # dst_dir = r"E:\python_ai_dataset\OCR\det\from000\toAITrain\det"
    # d_train = OCRDatasetV2(
    #     r"E:\python_ai_dataset\OCR\det\from000\categories_20250320_v0.1.1.s",
    #     with_label=True,
    #     read_image=False,
    #     subject_to="label",
    #     label_file="Label_20250512_train.txt",
    # )
    # d_val = OCRDatasetV2(
    #     r"E:\python_ai_dataset\OCR\det\from000\categories_20250320_v0.1.1.s",
    #     with_label=True,
    #     read_image=False,
    #     subject_to="label",
    #     label_file="Label_20250512_val.txt",
    # )
    # d = d_train + d_val
    # print(len(d_train), len(d_val), len(d))

    dst_dir = r"E:\python_ai_dataset\OCR\det\from000\toAITrain\det_20250507_v0.1_test"
    d = OCRDatasetV2(
        # [
            r"E:\python_ai_dataset\OCR\det\from000\categories_20250320_v0.1.1.s\test",
            # r"E:\python_ai_dataset\OCR\det\from000\categories_20250320_v0.1.1.s\anno_20250512_val",
        # ],
        with_label=True,
        read_image=False,
        subject_to="label",
    )
    print(len(d))

    for i in d:
        print(i)
        break

    img_name_count = {}
    max_same_name_count = 0
    for idx, name in d.image_map.items():
        basename = os.path.basename(name)
        cur_cnt = img_name_count.get(basename, 0)
        img_name_count[basename] = cur_cnt + 1
        max_same_name_count = max(max_same_name_count, img_name_count[basename])
    print(f"{max_same_name_count = }, {len(img_name_count) = }")

    os.makedirs(dst_dir, exist_ok=True)
    label_file = open(os.path.join(dst_dir, "Label.txt"), "w", encoding="utf-8")

    def op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        basename = os.path.basename(_img_path)
        cur_cnt = img_name_count[basename]
        if cur_cnt > 1:
            img_name_count[basename] = cur_cnt - 1
            name, ext = basename.rsplit(".", 1)
            basename = name + f"_{cur_cnt}.{ext}"
        shutil.copy(_img_path, os.path.join(dst_dir, basename))
        if _label_file is not None and _label_data is not None:
            dirname = os.path.basename(dst_dir)
            _label_str = _label_op(_label_data)
            label_file.write(f"{dirname}/{basename}\t{_label_str}\n")

    dump_ocr_dataset(d, dst_dir, custom_image_label_op=op, overwriting=True)
    label_file.close()
    # label_files = [os.path.join(dst_dir, i, "Label.txt")
    #                for i in os.listdir(dst_dir)
    #                if os.path.isdir(os.path.join(dst_dir, i))]
    # union_label(label_files, os.path.join(dst_dir, "Label.txt"))


def OCRCLSDatesetV2_sample_case1():
    dst_dir = r"E:\python_ai_dataset\OCR\det\from000\toAITrain\cls_20250516_v0.1"
    d = OCRCLSDatasetV2(
        [
            r"E:\python_ai_dataset\OCR\det\from000\cls\Train1982_matting",
            r"E:\python_ai_dataset\OCR\det\from000\cls\Val500_matting",
        ],
        categories={
            0: "0",
            1: "180"
        },
        with_label=True,
        read_image=False,
        subject_to="label",
        label_file="cls_gt.txt"
    )
    print(len(d))

    for i in d:
        print(i)
        break

    img_name_count = {}
    max_same_name_count = 0
    for idx, name in d.image_map.items():
        basename = os.path.basename(name)
        cur_cnt = img_name_count.get(basename, 0)
        img_name_count[basename] = cur_cnt + 1
        max_same_name_count = max(max_same_name_count, img_name_count[basename])
    print(f"{max_same_name_count = }, {len(img_name_count) = }")

    os.makedirs(dst_dir, exist_ok=True)
    label_file = open(os.path.join(dst_dir, "Label.txt"), "w", encoding="utf-8")

    def op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        basename = os.path.basename(_img_path)
        cur_cnt = img_name_count[basename]
        if cur_cnt > 1:
            img_name_count[basename] = cur_cnt - 1
            name, ext = basename.rsplit(".", 1)
            basename = name + f"_{cur_cnt}.{ext}"

        shutil.copy(_img_path, os.path.join(dst_dir, basename))
        if _label_file is not None and _label_data is not None:
            dirname = os.path.basename(dst_dir)
            _label_str = _label_op(_label_data)
            label_file.write(f"{dirname}/{basename}\t{_label_str}\n")

    def op2(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        basename = os.path.basename(_img_path)
        cur_cnt = img_name_count[basename]
        if cur_cnt > 1:
            img_name_count[basename] = cur_cnt - 1
            name, ext = basename.rsplit(".", 1)
            basename = name + f"_{cur_cnt}.{ext}"
        os.makedirs(os.path.join(dst_dir, _label_op(_label_data)), exist_ok=True)
        shutil.copy(_img_path, os.path.join(dst_dir, _label_op(_label_data), basename))
        # if _label_file is not None and _label_data is not None:
        #     dirname = os.path.basename(dst_dir)
        #     _label_str = _label_op(_label_data)
        #     label_file.write(f"{dirname}/{basename}\t{_label_str}\n")

    dump_ocr_dataset(d, dst_dir, custom_image_label_op=op2, overwriting=True)
    label_file.close()
    # label_files = [os.path.join(dst_dir, i, "Label.txt")
    #                for i in os.listdir(dst_dir)
    #                if os.path.isdir(os.path.join(dst_dir, i))]
    # union_label(label_files, os.path.join(dst_dir, "Label.txt"))


def OCRRECDatesetV2_matting_for_AITrain():
    src_dir = r"E:\python_ai_dataset\OCR\det\Label_new\31-anno_20251121"
    matting_dir = r"E:\python_ai_dataset\OCR\rec\31-anno_20251121_matting"
    dst_dir = r"E:\python_ai_dataset\OCR\rec\rec_31_20251121_v0.1"
    od = OCRDatasetV2(
        src_dir,
        with_label=True,
        read_image=False,
        subject_to="label",
    )
    matting_ocr_dataset(od, matting_dir)

    d = OCRRECDatasetV2(
        matting_dir,
        with_label=True,
        read_image=False,
        subject_to="label",
        # label_file="v0.1.5_trainv0.1.4+binary.txt"
    )
    print(len(d))

    for i in d:
        print(i)
        break

    img_name_count = {}
    max_same_name_count = 0
    for idx, name in d.image_map.items():
        basename = os.path.basename(name)
        cur_cnt = img_name_count.get(basename, 0)
        img_name_count[basename] = cur_cnt + 1
        max_same_name_count = max(max_same_name_count, img_name_count[basename])
    print(f"{max_same_name_count = }, {len(img_name_count) = }")

    os.makedirs(dst_dir, exist_ok=True)
    label_file = open(os.path.join(dst_dir, "Label.txt"), "w", encoding="utf-8")

    def op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None):
        if not os.path.exists(_img_path):
            print(f"img not exists: {_img_path}")
            return
        basename = os.path.basename(_img_path)
        cur_cnt = img_name_count[basename]
        if cur_cnt > 1:
            img_name_count[basename] = cur_cnt - 1
            name, ext = basename.rsplit(".", 1)
            basename = name + f"_{cur_cnt}.{ext}"
        im = imread(_img_path)
        h, w, _ = im.shape
        shutil.copy(_img_path, os.path.join(dst_dir, basename))
        if _label_file is not None and _label_data is not None:
            dirname = os.path.basename(dst_dir)
            _label_str = _label_op(_label_data)
            lab = [{"transcription": _label_str, "points": [[0, 0], [w, 0], [w, h], [0, h]], "difficult": 0}]
            _label_str = (str(lab).replace("'", '"')
                          .replace('"difficult": 0', '"difficult": false')
                          .replace('"difficult": 1', '"difficult": true'))
            label_file.write(f"{dirname}/{basename}\t{_label_str}\n")

    dump_ocr_dataset(d, dst_dir, custom_image_label_op=op, overwriting=True)
    label_file.close()
    # label_files = [os.path.join(dst_dir, i, "Label.txt")
    #                for i in os.listdir(dst_dir)
    #                if os.path.isdir(os.path.join(dst_dir, i))]
    # union_label(label_files, os.path.join(dst_dir, "Label.txt"))


def OCRRECDatesetV2_sample_case0():

    imli = os.listdir(r"E:\python_ai_dataset\OCR\vis\gather\shunt")
    # imli = os.listdir(r"E:\python_ai_dataset\OCR\rec\gather_test\binary")

    def condition(item):
        r = item[0] if isinstance(item, tuple) else item

        bn = os.path.basename(r)
        if bn in imli:
            return True

        return False

    # d = OCRRECDatasetV2(
    d = OCRDatasetV2(
        det_paths(r"E:\python_ai_dataset\OCR\det\gather\categories"),
        # det_paths(r"E:\python_ai_dataset\OCR\rec\gather\categories_20250313_v0.1.0"),
        with_label=True,
        read_image=False
    )
    subset = d.sample(1., seed=20250507, condition=condition)

    d_copy = d.copy()
    d_copy.wash(subset, mode='keep')
    dst_dir = r"E:\python_ai_dataset\OCR\det\gather\categories_20250507_binary"
    # dst_dir = r"E:\python_ai_dataset\OCR\rec\gather\categories_20250507_v0.1.0_binary"

    def op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None):
        basename = os.path.basename(_img_path)
        dirname = os.path.basename(_dst_dir)
        im = imread(_img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        iterations = 1
        kernel = np.ones((3, 3), np.uint8)
        # 前景的区域大，先腐蚀后膨胀; 背景的区域大，先膨胀后腐蚀
        im = cv2.morphologyEx(
            im,
            cv2.MORPH_OPEN
            if np.count_nonzero(im) > im.shape[0] * im.shape[1] // 2
            else cv2.MORPH_CLOSE
            ,
            kernel,
            iterations=iterations
        )

        if _label_data is not None:
            if im.shape[0] * im.shape[1] * 0.1 < np.count_nonzero(im) < im.shape[0] * im.shape[1] * 0.9:
                _label_str = _label_op(_label_data)
                _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
                imwrite(os.path.join(_dst_dir, basename), im)
        else:
            imwrite(os.path.join(_dst_dir, basename), im)

    dump_ocr_dataset(d_copy, dst_dir, custom_image_label_op=op, overwriting=True)
    label_files = [os.path.join(dst_dir, i, "Label.txt")
                   for i in os.listdir(dst_dir)
                   if os.path.isdir(os.path.join(dst_dir, i))]
    union_label(label_files, os.path.join(dst_dir, "Label.txt"))


def OCRDatesetV2_sample_case2():
    d = OCRDatasetV2(
        det_paths(r"E:\python_ai_dataset\OCR\det\gather\categories-copy"),
        with_label=True,
        read_image=False
    )
    subset = d.sample(0.2, seed=20250414)
    # print(len(subset), subset)
    d_copy = d.copy()
    d_copy.wash(subset, mode='keep')
    dst_dir = r"E:\python_ai_dataset\OCR\det\gather\categories_20250415"

    def op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None):
        basename = os.path.basename(_img_path)
        dirname = os.path.basename(_dst_dir)
        im = imread(_img_path)
        angle = random.choice([-90, 90])
        im, M = rotate_image_min(im, angle)
        imwrite(os.path.join(_dst_dir, basename), im)
        if _label_data is not None:
            for i, _la in enumerate(_label_data):
                _label_data[i]["points"] = order_rectangle_points(
                    warp_affine_points(_la["points"], M, round=0)
                ).tolist()
            _label_str = _label_op(_label_data)
            _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")

    # dump(d_copy, dst_dir, custom_image_label_op=op, overwriting=True)
    #
    # label_files = [os.path.join(dst_dir, i, "Label.txt")
    #                for i in os.listdir(dst_dir)
    #                if os.path.isdir(os.path.join(dst_dir, i))]
    # union_label(label_files, os.path.join(dst_dir, "Label.txt"))
    #
    # d_copy2 = d.copy()
    # d_copy2.wash(subset, mode='drop')
    # val_test_set = d_copy2.split([0.4, 0.6], subset_name=["val", "test"], seed=20250414)
    # val_set = d_copy2.subset(val_test_set["val"])
    # testset = d_copy2.subset(val_test_set["test"])
    # dst_dir = r"E:\python_ai_dataset\OCR\det\gather\categories_20250415_val"
    # dump(val_set, dst_dir, custom_image_label_op=op, overwriting=True)
    # label_files = [os.path.join(dst_dir, i, "Label.txt")
    #                for i in os.listdir(dst_dir)
    #                if os.path.isdir(os.path.join(dst_dir, i))]
    # union_label(label_files, os.path.join(dst_dir, "Label.txt"))
    #
    # dst_dir = r"E:\python_ai_dataset\OCR\det\gather\categories_20250415_test"
    # dump(testset, dst_dir, custom_image_label_op=op, overwriting=True)
    # label_files = [os.path.join(dst_dir, i, "Label.txt")
    #                for i in os.listdir(dst_dir)
    #                if os.path.isdir(os.path.join(dst_dir, i))]
    # union_label(label_files, os.path.join(dst_dir, "Label.txt"))


def OCRDatesetV2_sample_case3():
    d = OCRDatasetV2(
        det_paths(r"E:\python_ai_dataset\OCR\det\Label_new\anno"),
        with_label=True,
        read_image=False,
        subject_to="label"
    )
    subsets = d.split([0.6, 0.2, 0.2], subset_name=["train", "val", "test"], seed=20250512)
    for subset_name, subset in subsets.items():
        print(subset_name, len(subset))
        d_copy = d.copy()
        d_copy.wash(subset, mode='keep')
        dst_dir = r"E:\python_ai_dataset\OCR\det\Label_new\anno_20250512_{}".format(subset_name)
        dump_ocr_dataset(d_copy, dst_dir, custom_image_label_op=None, overwriting=True)


def test_OCRDatesetV2_case4():
    print()
    src_list_all = det_paths(r"E:\python_ai_dataset\OCR\det\Label_new")
    print(src_list_all)
    src_list = []
    for src in src_list_all:
        if os.path.basename(src) in [
            "4-Train996_reverse", "13-anno_20250225_AiDian_reverse", "25-OCR-opposite-20250927",
            "26-OCR-opposite-20250928"
        ]:
            continue
        src_list.append(src)

    d_src = OCRDatasetV2(src_list, with_label=True,)
    # print(len(d_src), d_src[1])
    n2i = {v: k for k, v in d_src.image_map.items()}
    old_dir = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2"

    replace_list_file = open(os.path.join(old_dir, "replace_list.txt"), "w", encoding="utf-8")

    cV2_dir = det_paths(old_dir)
    for i in cV2_dir:
        cV2_dir += det_paths(i)
    for i in cV2_dir:
        print(i)
    d_cV2 = OCRDatasetV2(cV2_dir, )
    d_cV2.with_label = True
    cnt = 0
    for i, n in d_cV2.image_map.items():
        if n in n2i:
            cnt += 1
            im_p = d_cV2[i][0]
            d_cV2[i] = (im_p, d_src[n2i[n]][1])

    commonpath = os.path.commonpath(list(d_cV2.roots_map.values()))

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        img_path = Path(_img_path)
        ext = img_path.suffix  # 包含 .jpg
        parent = img_path.parent

        relative_parts = str(parent).replace(commonpath, "").split(os.sep)[1:]
        if not any("Type" in r for r in relative_parts):
            relative_parts = relative_parts + ["Type0"]
        name_parts = []

        if index is not None:
            name_parts.append(f"UID{index}")

        if relative_parts:
            name_parts.append(".".join(relative_parts))

        if _label_data is not None and len(_label_data) > 0:
            for idx, _la in enumerate(_label_data):
                transcription = "{" + sanitize_filename(_la.get("transcription", "")) + "}"
                name_parts.append(transcription)
        else:
            name_parts.append("{}")

        basename = ".".join(name_parts) + ext
        shutil.copy(_img_path, os.path.join(_dst_dir, basename))
        replace_list_file.write(f"{index}, {os.path.basename(_img_path)}, {basename}\n")

        if _label_file is not None:
            dirname = os.path.basename(_dst_dir)
            if _label_data is not None:
                _label_str = _label_op(_label_data)
                _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
            else:
                _label_file.write(f"{dirname}/{basename}\t[]\n")

    dump_ocr_dataset(
        d_cV2, r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260204",
        custom_image_label_op=image_label_op
    )
    print(len(d_cV2), cnt)
    replace_list_file.close()


def test_OCRDatesetV2_case5():
    print()
    dir_src = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2"
    cV2_dir = det_paths(dir_src)
    for i in cV2_dir:
        cV2_dir += det_paths(i)

    d_cV2 = OCRDatasetV2(cV2_dir, with_label=False)
    print(len(d_cV2))

    dir_new = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260205"
    cV2_new = det_paths(dir_new)
    for i in cV2_new:
        cV2_new += det_paths(i)
    cV2_new = [i for i in cV2_new if os.path.exists(os.path.join(i, "Label.txt"))]
    d_new = OCRDatasetV2(cV2_new, with_label=True)
    d_new_map = {v: k for k, v in d_new.image_map.items()}
    print(len(d_new))
    for i, (im, _) in enumerate(d_new[:10]):
        print(i, im)

    d1, d2, d3 = {}, {}, {}
    new_dset = OCRDatasetV2(subject_to='image')
    new_dset.with_label = True
    with open(os.path.join(dir_src, "replace_list.txt"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            idx, old_name, new_name = line.strip().rsplit(', ')
            d1[idx] = (old_name, new_name)
            d2[old_name] = new_name
            d3[new_name] = old_name

    for im, _ in d_cV2:
        basename = os.path.basename(im)
        new_path = d2.get(basename, "")
        if new_path is not "":
            new_idx = d_new_map[new_path]
            # print(im, d_new[new_idx])
            new_dset.append(im, d_new[new_idx][1])

    commonpath = os.path.commonpath(list(d_cV2.roots_map.values()))

    latest_dir = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260207"
    os.makedirs(latest_dir, exist_ok=True)
    replace_list_file = open(os.path.join(latest_dir, "replace_list.txt"), "w", encoding="utf-8")

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        img_path = Path(_img_path)
        ext = img_path.suffix  # 包含 .jpg
        parent = img_path.parent

        relative_parts = str(parent).replace(commonpath, "").split(os.sep)[1:]
        if not any("Type" in r for r in relative_parts):
            relative_parts = relative_parts + ["Type0"]
        name_parts = []

        if index is not None:
            name_parts.append(f"UID{index}")

        if relative_parts:
            name_parts.append(".".join(relative_parts))

        if _label_data is not None and len(_label_data) > 0:
            for idx, _la in enumerate(_label_data):
                transcription = "{" + sanitize_filename(_la.get("transcription", "")) + "}"
                name_parts.append(transcription)
        else:
            name_parts.append("{}")

        basename = ".".join(name_parts) + ext
        shutil.copy(_img_path, os.path.join(_dst_dir, basename))
        replace_list_file.write(f"{index}, {os.path.basename(_img_path)}, {basename}\n")

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
    print(len(new_dset))
    replace_list_file.close()

    save_d = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260207_vis"
    latest_cV2_dir = det_paths(latest_dir)
    for i in latest_cV2_dir:
        latest_cV2_dir += det_paths(i)

    union_label([os.path.join(i, "Label.txt")
                 for i in latest_cV2_dir
                 if os.path.exists(os.path.join(i, "Label.txt"))],
                os.path.join(latest_dir, "Label.txt"))
    # dset = OCRDatasetV2(latest_dir, with_label=True, subject_to='label')
    # print(len(dset))
    # VisualizeOCRDataset(dset, save_dir=save_d)()


def test_OCRDatesetV2_case6():
    txt_save_dir = r"E:\python_ai_dataset\OCR\det\annoed-copy\OCR-V6.4-待标注-dip-紫光V2"
    labelme_json_dir = rf"{txt_save_dir}\json"
    convert_labelme_json_to_ocr_txt(labelme_json_dir, os.path.join(txt_save_dir, "Label.txt"))


def test_OCRDatasetV2_case7():
    top_dir = r"E:\python_ai_dataset\OCR\det\annoed-copy"
    latest_dir = r"E:\python_ai_dataset\OCR\det\DIP_OCR"
    offset = 18504
    d = OCRDatasetV2(det_paths(top_dir), with_label=True, subject_to='image')
    print(len(d))

    def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, index=None):
        img_path = Path(_img_path)
        ext = img_path.suffix  # 包含 .jpg
        parent = img_path.parent
        dst_dir_parent = Path(_dst_dir).parent
        print(dst_dir_parent)

        relative_parts = [img_path.name.split('-')[0]]
        if not any("Type" in r for r in relative_parts):
            relative_parts = relative_parts + ["Type0"]
        name_parts = []

        if index is not None:
            name_parts.append(f"UID{index + offset}")

        if relative_parts:
            name_parts.append(".".join(relative_parts))

        if _label_data is not None and len(_label_data) > 0:
            for idx, _la in enumerate(_label_data):
                transcription = "{" + sanitize_filename(_la.get("transcription", "")) + "}"
                name_parts.append(transcription)
        else:
            name_parts.append("{}")

        basename = ".".join(name_parts) + ext
        print(basename)
        shutil.copy(_img_path, os.path.join(dst_dir_parent, basename))
        # replace_list_file.write(f"{index}, {os.path.basename(_img_path)}, {basename}\n")

        if _label_file is not None:
            dirname = os.path.basename(dst_dir_parent)
            if _label_data is not None:
                _label_str = _label_op(_label_data)
                _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")
            else:
                _label_file.write(f"{dirname}/{basename}\t[]\n")

    dump_ocr_dataset(
        d, latest_dir,
        custom_image_label_op=image_label_op,
        overwriting=True
    )
    union_label([os.path.join(i, "Label.txt")
                 for i in det_paths(latest_dir)
                 if os.path.exists(os.path.join(i, "Label.txt"))],
                os.path.join(latest_dir, "Label.txt"))


def test_OCRDatesetV2_init():
    print()
    # OCRDatesetV2_init_case1()
    # OCRDatesetV2_init_case2()
    # OCRDatesetV2_init_case3()
    # OCRDatesetV2_sample_case0()
    # OCRDatesetV2_sample_case1()
    # OCRDatesetV2_sample_case2()
    # OCRRECDatesetV2_sample_case0()
    # OCRCLSDatesetV2_sample_case1()
    # OCRRECDatesetV2_matting_for_AITrain()

    # union_labels(r"E:\python_ai_dataset\OCR\det\Label_new\anno")
    # union_labels(r"E:\python_ai_dataset\OCR\det\gather\anno_20250512_train")
    # union_labels(r"E:\python_ai_dataset\OCR\det\gather\anno_20250512_val")
    # union_labels(r"E:\python_ai_dataset\OCR\det\gather\anno_20250512_test")

    dir_top = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260205"
    save_d = r"E:\python_ai_dataset\OCR\det\gather\categoriesV2_20260205_vis"
    cV2_dir = det_paths(dir_top)
    for i in cV2_dir:
        cV2_dir += det_paths(i)

    union_label([os.path.join(i, "Label.txt")
                 for i in cV2_dir
                 if os.path.exists(os.path.join(i, "Label.txt"))],
                os.path.join(dir_top, "Label.txt"))
    dset = OCRDatasetV2(dir_top, with_label=True, subject_to='label')
    print(len(dset))
    VisualizeOCRDataset(dset, save_dir=save_d)()


def test_OCRDataset_vis():
    dir_path = r"E:\python_ai_dataset\OCR\det\DIP_OCR"
    vis_path = r"E:\python_ai_dataset\OCR\det\DIP_OCR_vis"

    dataset = OCRDatasetV2(
        dir_path, with_label=True, subject_to="image"
    )

    VisualizeOCRDataset(
        dataset, save_dir=vis_path, text_enable=True, line_color=(0, 180, 0), text_color=(0, 128, 0)
    )()


def test_YOLODataset_init():
    print()

    d = YOLODataset(
        root=r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\train_V4.5.2",
        with_label=True,
        image_dirname=f"images",
        label_dirname=f"labels",
        task="seg",
        categories={
            0: "elem",
            1: "solder",
            2: "paster",
            3: "device"
        },
        read_image=False
    )
    # assert len(d) == 440, "len(d) == {}".format(len(d))

    VisualizeYOLODataset(
        dataset=d,
        save_dir=r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\train_V4.5.2\vis",
    )()

    # def label_op(old_label_path, new_label_path):
    #     with open(old_label_path, "r") as f:
    #         lines = f.readlines()
    #     try:
    #         f = open(new_label_path, 'w', encoding='utf-8')
    #         for line_num, line in enumerate(lines, 1):
    #             parts = line.split()
    #             if not parts:
    #                 continue
    #
    #             class_id = int(parts[0])
    #             if class_id in [2, 3]:
    #                 continue
    #             values = list(map(float, parts[1:]))
    #             validate_normalized_coords(values, line_num, True)
    #             values_str = ' '.join(map(str, values))
    #             new_line = f"{class_id} {values_str}\n"
    #             f.write(new_line)
    #
    #     except ValueError as e:
    #         raise e
    #     finally:
    #         f.close()

    # dump_yolo_dataset(
    #     d,
    #     r"E:\python_ai_dataset\foreign-object-detect\2-FOVSlice\train\images-black-green-V3.1",
    #     lambda o, n: True,
    #     label_op
    # )


def test_res_img():
    dir_path = r"E:\python_ai_dataset\foreign-object-detect\2025\fod-20250527\fov-only-side\org"
    dst_path = r"E:\python_ai_dataset\foreign-object-detect\2025\fod-20250527\fov-only-side"
    img_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith(".jpg")]
    for i in img_path:
        im = imread(i)
        crop = im[0:2000, :]
        imwrite(os.path.join(dst_path, os.path.basename(i)), crop)


def test_VOCDataset():
    dir_data = r"E:\python_ai_dataset\foreign-object-detect\气泡\BUBBLE-1"
    d = VOCDataset(dir_data, with_label=True, categories={0:"BUBBLE"}, read_image=False)
    for i in d:
        print(i)


def test_convertVOC2yolo():
    dir_data = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.6\anno\fov-xz\annotations\annotations.json"
    dst_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.6\anno\fov-xz"
    crop_dir = r"E:\python_ai_dataset\foreign-object-detect\BUBBLE\UV_BUBBLE\images_crop_{}"
    # d = VOCDataset(dir_data, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    convert_coco2yolo(dir_data, dst_dir, use_segments=True)
    # assert len(os.listdir(dst_dir)) == len(d), "len(os.listdir(dst_dir)) == {}".format(len(os.listdir(dst_dir)))
    #
    # size = 1024
    # CropImages(os.path.join(dir_data, "images"), crop_dir.format(size),
    #            size, size, fmt="png", deal_with_label=True, yolo_task='det', dump_empty=False)()
    #
    # d_y = YOLODataset(crop_dir.format(size),
    #                   image_dirname="images",
    #                   label_dirname="labels",
    #                   with_label=True,
    #                   task="det",
    #                   categories={0: "BUBBLE"},
    #                   read_image=False)
    #
    # VisualizeYOLODataset(d_y, save_dir=os.path.join(crop_dir.format(size), "vis"))()

    vis_yolo = r"E:\opensource_project\ultralytics-individual\runs\detect\predict8"
    d_y = YOLODataset(dst_dir,
                      image_dirname="images",
                      label_dirname="labels",
                      with_label=True,
                      task="seg",
                      categories={0: "0", 1:"a", 2:"b", 3:"c", 4:"d", 5:"e"},
                      read_image=False)

    VisualizeYOLODataset(d_y, save_dir=os.path.join(dst_dir, "vis"))()


cate = {
    0: "entity",
    1: "solder",
    2: "paster",
    3: "device",
    4: "solderBall",
    5: "sticker",
    6: "footprint",
}


def test_coco2yolo():
    FOD_dir = ["异物-AIDIAN", "异物-BAINENG2D", "异物-BAINENG2D-V2", "异物-BAINENG3D"]
    tto_dir = ["fod_aidian", "fod_baineng2d_01", "fod_baineng2d_02", "fod_baineng3d"]
    dir_idx = 3
    src_dir  = rf"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\raw\异物4.5.2\{FOD_dir[dir_idx]}\pick"
    save_dir = rf"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\cooked\{tto_dir[dir_idx]}"
    image_key = tto_dir[dir_idx]
    coco_json = rf"{src_dir}\annotations\annotations.json"
    def cls_filter(cls):
        label_map = {
            0: 0,     # elem
            1: 1,     # solder
            2: None,  # paster
            3: None,  # device
            4: 1,     # solder ball, if 是 异物\异物-AIDIAN 等文件夹下用转为1
            5: None,  # sticker 贴纸
            6: 1,     # footprint 焊盘
        }
        return label_map[cls]

    convert_coco2yolo(
        coco_json,
        save_dir=save_dir,
        use_segments=True,
        cls_filter=cls_filter
    )

    if not os.path.exists(os.path.join(save_dir, "images")):
        shutil.copytree(rf"{src_dir}\images", os.path.join(save_dir, "images"))
    generate_yolo_empty_labels(Path(save_dir) / "images", Path(save_dir) / "labels")

    d_y = YOLODataset(save_dir,
                      image_dirname="images",
                      label_dirname="labels",
                      with_label=True,
                      task="seg",
                      categories=cate,
                      read_image=False)
    print(d_y.directories)
    if image_key is not None:
        for idx, i in enumerate(d_y):
            im_path, la_path = i
            im_dir_path = os.path.dirname(im_path)
            la_dir_path = os.path.dirname(la_path)
            new_im_dir = os.path.join(im_dir_path, f"{image_key}_" + os.path.basename(im_path))
            new_la_dir = os.path.join(la_dir_path, f"{image_key}_" + os.path.basename(la_path))
            # new_im_dir = os.path.join(im_dir_path, os.path.basename(im_path).replace(f"{image_key}_", ""))
            os.rename(im_path, new_im_dir)
            os.rename(la_path, new_la_dir)
            d_y[idx] = Path(new_im_dir), Path(new_la_dir)

    VisualizeYOLODataset(d_y, save_dir=os.path.join(save_dir, "vis"))()


def test_crop_image():
    size = 1280
    from_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\cooked\fod_xz_m\images"
    crop_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\crop\fod_xz_m"
    # crop_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.4\cooked\fod_baineng2d_02"
    CropImages(from_dir, crop_dir + "\images", 2048, 1500, fmt="png",
               cope_with_label=True, yolo_task='seg', dump_empty=False)()

    d_y = YOLODataset(crop_dir,
                      image_dirname="images",
                      label_dirname="labels",
                      with_label=True,
                      task="seg",
                      categories=cate,
                      read_image=False)

    VisualizeYOLODataset(d_y, save_dir=os.path.join(crop_dir, "vis"))()

def test_crop_image2():
    size = 1664
    from_dir = r"C:\Users\WQS\Documents\WXWork\1688856196451158\Cache\File\2025-09\FOV边缘\新建文件夹\622"
    crop_dir = r"C:\Users\WQS\Documents\WXWork\1688856196451158\Cache\File\2025-09\FOV边缘\新建文件夹\622-crop"
    # crop_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.4\cooked\fod_baineng2d_02"
    CropImages(from_dir, crop_dir + "\images", 2048, 1500, fmt="png",
               cope_with_label=False, yolo_task='seg', dump_empty=False)()

    # d_y = YOLODataset(crop_dir,
    #                   image_dirname="images",
    #                   label_dirname="labels",
    #                   with_label=True,
    #                   task="seg",
    #                   categories=cate,
    #                   read_image=False)
    #
    # VisualizeYOLODataset(d_y, save_dir=os.path.join(crop_dir, "vis"))()


def test_copy_image():
    src_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\crop"
    dst_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\train"
    dir_list = [i for i in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, i))]
    for d in dir_list:
        shutil.copytree(os.path.join(src_dir, d, "images"), os.path.join(dst_dir, "images"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(src_dir, d, "labels"), os.path.join(dst_dir, "labels"), dirs_exist_ok=True)
        print(os.path.join(src_dir, d, "images"))
        for i in os.listdir(os.path.join(src_dir, d, "images")):
            if not i.endswith(tuple(IMG_FORMATS)):
                continue
            if not os.path.exists(os.path.join(dst_dir, "images", i)):
                print(os.path.join(dst_dir, "images", i))


def test_union_labels():
    import json
    dst_dir = r"E:\python_ai_dataset\OCR\det\fromAITrain"
    # label_files = [os.path.join(dst_dir, i, "Label.txt")
    #                for i in os.listdir(dst_dir)
    #                if i != "rec_20250313_v0.1_val"]
    # union_label(label_files, os.path.join(dst_dir, "Label.txt"))

    new_label = []
    with open(os.path.join(dst_dir, "Label_val.txt"), "r", encoding="utf-8") as f1:
        for line in f1:
            path, label = line.split("\t")
            data = json.loads(label)
            lab_str = data[0]["transcription"]
            new_label.append(path+"\t"+lab_str)
    with open(os.path.join(dst_dir, "Label_val2.txt"), "w", encoding="utf-8") as f2:
        f2.write("\n".join(new_label))


def test_rename_yoloDataset():
    dir_data = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.3\cooked\alllabel\fod_real_small_roi_anno\fod_aidian\flying_canvas"
    dy = YOLODataset(dir_data,
                     with_label=True,
                     task="seg",
                     categories={0: "entity", 1: "solder"})

    for i in dy:
        im_path, la_path = i
        im_dir_path = os.path.dirname(im_path)
        la_dir_path = os.path.dirname(la_path)
        os.rename(im_path, os.path.join(im_dir_path, "fod_aidian_flying_" + os.path.basename(im_path)))
        os.rename(la_path, os.path.join(la_dir_path, "fod_aidian_flying_" + os.path.basename(la_path)))


def test_day():
    print()
    # 示例1：计算相差天数
    print(date_utils("2000", end_date="2025-09-03"))

    # 示例2：计算经过 n 天后的日期
    print(date_utils("2000", days=10000))


def test_filter_label_rename_yoloDataset():
    cate = {
        0: "entity",
        1: "solder",
        2: "paster",
        3: "device",
        4: "solderBall",
        5: "sticker"
    }
    def cls_filter(cls):
        label_map = {
            0: 0,     # elem
            1: 1,     # solder
            2: None,  # paster
            3: None,  # device
            4: None,  # solder ball
            5: None,  # sticker
        }
        return label_map[cls]
    patch = "fod_baineng3d_flying_canvas"
    dir_data = rf"E:\python_ai_dataset\foreign-object-detect\NEW\V4.3\cooked\alllabel\fod_real_small_roi_anno\fod_baineng3d\flying_canvas"
    dst_data = rf"E:\python_ai_dataset\foreign-object-detect\NEW\V4.3\cooked\alllabel\fod_real_small_roi_anno\fod_baineng3d\flying_canvas_2label"
    dy = YOLODataset(dir_data,
                     with_label=True,
                     task="seg",
                     categories=cate)

    def label_op(old_label_path, new_label_path):
        with open(old_label_path, "r") as f:
            lines = f.readlines()
        try:
            f = open(new_label_path, 'w', encoding='utf-8')
            for line_num, line in enumerate(lines, 1):
                parts = line.split()
                if not parts:
                    continue

                class_id = int(parts[0])
                class_id = cls_filter(class_id)
                if class_id is None:
                    continue
                values = list(map(float, parts[1:]))
                validate_normalized_coords(values, line_num, True)
                values_str = ' '.join(map(str, values))
                new_line = f"{class_id} {values_str}\n"
                f.write(new_line)

        except ValueError as e:
            raise e
        finally:
            f.close()

    dump_yolo_dataset(dy, dst_data, label_file_op=label_op)

    dy2 = YOLODataset(dst_data,
                      with_label=True,
                      task="seg",
                      categories={0: "entity", 1: "solder"})
    for i in dy2:
        im_path, la_path = i
        im_dir_path = os.path.dirname(im_path)
        la_dir_path = os.path.dirname(la_path)
        os.rename(im_path, os.path.join(im_dir_path, f"{patch}_" + os.path.basename(im_path)))
        os.rename(la_path, os.path.join(la_dir_path, f"{patch}_" + os.path.basename(la_path)))

    dy2 = YOLODataset(dst_data,
                      with_label=True,
                      task="seg",
                      categories={0: "entity", 1: "solder"})
    VisualizeYOLODataset(dy2, save_dir=os.path.join(dst_data, "vis"))()


def test_splitYOLODataset():
    dir_data = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\train_V4.5.2"
    dst_data = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\train_split_V4.5.3"
    dy = YOLODataset(dir_data,
                     with_label=True,
                     task="seg",
                     categories={0: "entity", 1: "solder"})
    subset = dy.split(ratio=[0.72, 0.18, 0.1], seed=20250902)
    for i, (name, s) in enumerate(subset.items()):
        sub_dy = dy.subset(s)
        dump_yolo_dataset(sub_dy, destination=dst_data, sub_dirname=name)


def test_voc2yolo():
    categories = {
        0: "DIP_IC",
        1: "SMT_BGA",
        2: "SMT_Capacitor",
        3: "SMT_Diode",
        4: "SMT_FOD",
        5: "SMT_IC",
        6: "SMT_Inductor",
        7: "SMT_LED",
        8: "SMT_Mosfet",
        9: "SMT_QFN",
        10: "SMT_Resistor",
        11: "SMT_Resistor_OCR",
        12: "SMT_TantalumCapacitors",
        13: "SMT_Transistor",
    }
    voc_dir =r"E:\python_ai_dataset\obb\AILocate-V2.8"
    dst_dir =r"E:\python_ai_dataset\obb\AILocate-V2.8-splits"
    voc_list = [os.path.join(voc_dir, d) for d in os.listdir(voc_dir)]
    for voc in voc_list:
        if not os.path.isdir(voc):
            continue
        # vd = VOCDataset(voc,
        #                 with_label=True,
        #                 image_dirname="JPEGImages",
        #                 label_dirname="Annotations",
        #                 categories=categories,
        #                 task='obb',)
        #
        # convertVOC2YOLO(vd, save_dir=os.path.join(voc, "labels"))

        dy = YOLODataset(voc,
                          image_dirname=r"JPEGImages",
                          label_dirname=r"labels",
                          with_label=True,
                          task="obb",
                          categories=categories,
                          read_image=False)

        subset = dy.split(ratio=[0.6, 0.2, 0.2], seed=20250613)
        cur_dirname = os.path.basename(voc)
        for i, (name, s) in enumerate(subset.items()):
            print(f"subset len {len(s) = }")
            if len(s) == 0:
                print(voc, len(dy))
                continue
            sub_dy = dy.subset(s)
            dump_yolo_dataset(sub_dy, destination=dst_dir, sub_dirname=name)

        VisualizeYOLODataset(dy, save_dir=os.path.join(voc_dir, "vis_all"))()
    # subset = ["train", "val", "test"]
    # subset_dir = [os.path.join(dst_dir, i) for i in subset]
    # subset_dirnames = os.listdir(subset_dir[0])
    # for i in subset_dirnames:
    #
    # subset = d_y.split(ratio=[0.7, 0.2, 0.1], seed=20250612)
    # for i, (name, s) in enumerate(subset.items()):
    #     sub_dy = d_y.subset(s)
    #     dump_yolo_dataset(sub_dy, destination=dst_dir, sub_dirname=name)

def test_voc2yolo2():
    dir_path = r"E:\python_ai_dataset\obb\AILocate-V2.8-splits\images"
    dir_path2 = r"E:\python_ai_dataset\obb\AILocate-V2.8.1-splits"
    categories = {
        0: "DIP_IC",
        1: "SMT_BGA",
        2: "SMT_Capacitor",
        3: "SMT_Diode",
        4: "SMT_FOD",
        5: "SMT_IC",
        6: "SMT_Inductor",
        7: "SMT_LED",
        8: "SMT_Mosfet",
        9: "SMT_QFN",
        10: "SMT_Resistor",
        11: "SMT_Resistor_OCR",
        12: "SMT_TantalumCapacitors",
        13: "SMT_Transistor",
    }
    categories2 = {
        0: "DIP_IC",
        1: "SMT_BGA",
        2: "SMT_Capacitor",
        3: "SMT_Diode",
        4: "SMT_IC",
        5: "SMT_Inductor",
        6: "SMT_LED",
        7: "SMT_Mosfet",
        8: "SMT_QFN",
        9: "SMT_Resistor",
        10: "SMT_Resistor_OCR",
        11: "SMT_TantalumCapacitors",
        12: "SMT_Transistor",
    }

    def label_op(old_label_path, new_label_path):
        with open(old_label_path, "r") as f:
            lines = f.readlines()
        try:
            f = open(new_label_path, 'w', encoding='utf-8')
            for line_num, line in enumerate(lines, 1):
                parts = line.split()
                if not parts:
                    continue

                class_id = int(parts[0])
                if class_id == 4:
                    continue
                if class_id > 4:
                    class_id -= 1
                values = list(map(float, parts[1:]))
                validate_normalized_coords(values, line_num, True)
                values_str = ' '.join(map(str, values))
                new_line = f"{class_id} {values_str}\n"
                f.write(new_line)

        except ValueError as e:
            raise e
        finally:
            f.close()

    for sub in ['train', 'val', 'test']:
        # path = os.path.join(dir_path, sub)
        # dirlist = [os.path.join(path, i) for i in os.listdir(path)]
        # dy = YOLODataset(
        #     dirlist,
        #     image_dirname=r"images",
        #     label_dirname=r"labels",
        #     with_label=True,
        #     task="obb",
        #     categories=categories,
        #     read_image=False
        # )
        # print(len(dy))
        # dump_yolo_dataset(dy, dir_path2, label_file_op=label_op)

        path = os.path.join(dir_path2, "images", sub)
        dirlist = [os.path.join(path, i) for i in os.listdir(path)]
        dy2 = YOLODataset(
            dirlist,
            image_dirname=r"images",
            label_dirname=r"labels",
            with_label=True,
            task="obb",
            categories=categories2,
            read_image=False
        )
        VisualizeYOLODataset(dy2, save_dir=rf"E:\python_ai_dataset\obb\AILocate-V2.8.1-splits-vis\{sub}")()


categories = {
    0: "SMT_IC",#"SMT_BGA",
    1: "SMT_Capacitor",
    2: "SMT_Diode",
    3: "SMT_IC",
    # 4: "SMT_QFN",
}

def test_vis_yolo():
    dir_path = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\raw\PCB-green-interface"
    cate_cooked = {
        0: "entity",
        1: "solder",
        2: "paster",
        3: "device",
        4: "ball", # solderBall
        5: "sticker",
        6: "footprint",
        7: "xizha"
    }
    convert_coco2yolo(
        Path(dir_path) / "annotations" / "annotations.json",
        dir_path,
        use_segments=True,
    )
    dy = YOLODataset(
        dir_path,
        image_dirname=r"images",
        label_dirname=r"labels",
        with_label=True,
        task="seg",
        categories=cate_cooked,
        read_image=False,
        subject_to="image"
    )
    VisualizeYOLODataset(dy, save_dir=os.path.join(dir_path, "vis"))()


def test_yolo_img_rotate():
    print()
    dy = YOLODataset(
        r"E:\python_ai_dataset\obb\AILocate-V2.8.2-small-patch-split\images\test",
        image_dirname=r"images",
        label_dirname=r"labels",
        with_label=True,
        task="obb",
        categories=categories,
        read_image=True
    )
    dst_dir = r"E:\python_ai_dataset\obb\AILocate-V2.8.2-small-patch-split\images\test_enh"
    os.makedirs(dst_dir, exist_ok=True)
    assert len(dy) != 0
    for i in dy:
        image_path, label_path, image, label = i
        if len(label) != 1:
            continue
        # print(image_path, label_path, label)
        h, w = image.shape[:2]
        points = np.array(label[0][1:], dtype=np.float32).reshape(-1, 2)
        points = points * np.array([w, h])
        rr = cv2.RotatedRect(points[0], points[1], points[2])
        center = rr.center
        angle = rr.angle
        filename, ext = os.path.basename(image_path).split('.', 1)

        print(os.path.basename(image_path), center, angle)
        # img = np.zeros((640,640,3), dtype=np.uint8)
        for deg in [0, 30, 45, 60, 75]:
            img = rotate_image_around_point(image, center, deg, (640, 640))
            # cv2.circle(img,(272, 276), radius=2, color=(0,0,255))
            # imshow("img", img, 0)
            imwrite(os.path.join(dst_dir, f"{filename}_{deg}.{ext}"), img)


def test_process_fod_dataset():
    dataset_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\raw"
    cooking_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\train_cooking"
    cooked_dir  = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\train_cooked"
    train_dir   = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\train"
    train_split_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\train_split_v4.7.0"
    dataset_txt = rf"{dataset_dir}/图像裁剪方案.txt"
    split_ratio = [0.83, 0.12, 0.05]
    from datetime import datetime
    date_num = int(datetime.now().strftime("%Y%m%d"))
    cate = {
        0: "entity",
        1: "solder",
        2: "paster",
        3: "device",
        4: "solderBall",
        5: "sticker",
        6: "footprint",
        7: "xizha"
    }
    cate_cooking = {
        0: 0,
        1: 1,
        2: None,
        3: None,
        4: 2,
        5: None,
        6: 1,
        7: None,
    }
    cate_cooked = {
        0: "entity",
        1: "solder",
        2: "solderBall",
        # 3: "footprint"
    }
    with open(dataset_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        line = line.strip().replace(" ", "").split(",")
        if len(line) != 3:
            continue
        image_dir, image_prefix, crop_scheme = line
        print(image_dir, image_prefix, crop_scheme)
        image_path = os.path.join(dataset_dir, Path(image_dir))
        label_path = os.path.join(Path(image_path).parent, "labels")
        annos_path = os.path.join(Path(image_path).parent, "annotations", "annotations.json")
        convert_coco2yolo(
            annos_path,
            Path(label_path).parent,
            use_segments=True,
        )

        generate_yolo_empty_labels(image_path, label_path)

        d_y = YOLODataset(str(Path(image_path).parent),
                          image_dirname="images",
                          label_dirname="labels",
                          with_label=True,
                          task="seg",
                          categories=cate,
                          read_image=False)

        # VisualizeYOLODataset(d_y, save_dir=os.path.join(Path(image_path).parent, "vis"))()

        def op(old_path, new_path):
            if not old_path.endswith(".txt"):
                shutil.copy(old_path, new_path)
            else:
                with open(old_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                with open(new_path, "w", encoding="utf-8") as f:
                    for line in lines:
                        parts = line.split()
                        class_id = int(parts[0])
                        if cate_cooking[class_id] is None:
                            continue
                        parts[0] = str(cate_cooking[class_id])
                        f.write(" ".join(parts) + "\n")

            basename = os.path.basename(new_path)
            if not basename.startswith(image_prefix):
                dir_path = os.path.dirname(new_path)
                ren_path = os.path.join(dir_path, f"{image_prefix}_" + basename)
                os.rename(new_path, ren_path)

        cooking_dataset_dir = os.path.join(cooking_dir, image_prefix)

        os.makedirs(cooking_dataset_dir, exist_ok=True)
        dump_yolo_dataset(d_y,
                          destination=cooking_dataset_dir,
                          image_file_op=op,
                          label_file_op=op
        )

        # d_y2 = YOLODataset(cooking_dataset_dir,
        #                    image_dirname="images",
        #                    label_dirname="labels",
        #                    with_label=True,
        #                    task="seg",
        #                    categories=cate,
        #                    read_image=False)

        # VisualizeYOLODataset(d_y2, save_dir=os.path.join(cooking_dataset_dir, "vis"))()

        dir_basename = os.path.basename(cooking_dataset_dir)
        cooked_dataset_dir = os.path.join(cooked_dir, dir_basename)

        if crop_scheme == "拼接":
            stitch_images(cooking_dataset_dir, cooked_dataset_dir, 1280, cate_cooked, visualize=False)
        elif crop_scheme == "待定或保留原尺寸":
            shutil.copytree(Path(cooking_dataset_dir) / "images", Path(cooked_dataset_dir) / "images")
            shutil.copytree(Path(cooking_dataset_dir) / "labels", Path(cooked_dataset_dir) / "labels")
            # d_y3 = YOLODataset(cooked_dataset_dir,
            #                    image_dirname="images",
            #                    label_dirname="labels",
            #                    with_label=True,
            #                    task="seg",
            #                    categories=cate_cooked,
            #                    read_image=False)
            # VisualizeYOLODataset(d_y3, save_dir=os.path.join(cooked_dataset_dir, "vis"))()
        else:
            w, h = crop_scheme.split("*")
            w, h = int(w), int(h)
            CropImages(Path(cooking_dataset_dir) / "images", Path(cooked_dataset_dir) / "images", w, h,
                       fmt="png", cope_with_label=True, yolo_task='seg', dump_empty=False)()
            # d_y3 = YOLODataset(cooked_dataset_dir,
            #                    image_dirname="images",
            #                    label_dirname="labels",
            #                    with_label=True,
            #                    task="seg",
            #                    categories=cate_cooked,
            #                    read_image=False)
            # VisualizeYOLODataset(d_y3, save_dir=os.path.join(cooked_dataset_dir, "vis"))()

        sub_images_path = os.path.join(cooked_dataset_dir, "images")
        sub_labels_path = os.path.join(cooked_dataset_dir, "labels")
        shutil.copytree(sub_images_path, os.path.join(train_dir, "images"), dirs_exist_ok=True)
        shutil.copytree(sub_labels_path, os.path.join(train_dir, "labels"), dirs_exist_ok=True)

    dy = YOLODataset(train_dir,
                     image_dirname="images",
                     label_dirname="labels",
                     with_label=True,
                     task="seg",
                     categories=cate_cooked,
                     read_image=False)
    VisualizeYOLODataset(dy, save_dir=os.path.join(train_dir, "vis"))()
    # subset = dy.split(ratio=split_ratio, seed=date_num)
    # for i, (name, s) in enumerate(subset.items()):
    #     sub_dy = dy.subset(s)
    #     dump_yolo_dataset(sub_dy, destination=train_split_dir, sub_dirname=name)

    shutil.rmtree(cooking_dir)
    shutil.rmtree(cooked_dir)
    # shutil.rmtree(train_dir)


def test_crop_img():
    CropImages(r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\20251112-test",
               r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\20251112-test2",
               1664, 1664,
               fmt="png", cope_with_label=False)()


def test_vis_coco2_ds():
    dir_name = "slice_image_train"
    json_file = r"train_2000_025"
    dataset_dir = Path(rf"E:\python_ai_dataset\COCOtoSLICE-S1-2000\{dir_name}")
    cooked_dataset_dir = Path(rf"E:\python_ai_dataset\COCOtoSLICE-S1-2000\{dir_name}_crop")
    categories = convert_coco2yolo(
        rf"E:\python_ai_dataset\COCOtoSLICE-S1-2000\{dir_name}\{json_file}.json",
        dataset_dir,
    )

    generate_yolo_empty_labels(dataset_dir / 'images', dataset_dir / 'labels')

    CropImages(
        Path(dataset_dir) / "images", Path(cooked_dataset_dir) / "images", 1000, 1000,
        fmt="png", cope_with_label=True, yolo_task='det', dump_empty=False,
    )()

    dy = YOLODataset(
        cooked_dataset_dir,
        image_dirname="images",
        label_dirname="labels",
        with_label=True,
        task="det",
        categories=categories,
    )
    convert_yolo2coco(dy, cooked_dataset_dir / f"{json_file}.json")
    VisualizeYOLODataset(dy, cooked_dataset_dir / 'vis')()


def test_vis_coco2_ds2():
    dir_name = "slice_image_train"
    json_file = r"train_2000_025"
    dataset_dir = Path(rf"E:\python_ai_dataset\COCOtoSLICE-S1-2000\{dir_name}")
    destination = rf"E:\python_ai_dataset\COCOtoSLICE-S1-2000\{dir_name}_new"
    categories = convert_coco2yolo(
        rf"E:\python_ai_dataset\COCOtoSLICE-S1-2000\{dir_name}\{json_file}.json",
        dataset_dir,
    )
    generate_yolo_empty_labels(dataset_dir / 'images', dataset_dir / 'labels')
    dy = YOLODataset(
        dataset_dir,
        image_dirname="images",
        label_dirname="labels",
        with_label=True,
        task="det",
        categories=categories,
    )
    print(dy.categories())
    print(dy.sample_info)
    new_cate_tmp = {
        dy.categories(name): name for name, num in dy.sample_info.items() if num > 500
    }
    print(new_cate_tmp)
    new_cate_n2i = {n: i for i, n in enumerate(sorted(new_cate_tmp.values()))}
    print(new_cate_n2i)
    old2new = {dy.categories(n): i_n for n, i_n in new_cate_n2i.items()}

    def la_op(old_path, new_path):
        with open(old_path, "r") as f:
            lines = f.readlines()
        try:
            f = open(new_path, 'w', encoding='utf-8')
            for line_num, line in enumerate(lines, 1):
                parts = line.split()
                if not parts:
                    continue

                class_id = int(parts[0])
                if class_id not in old2new.keys():
                    continue
                values = list(map(float, parts[1:]))
                validate_normalized_coords(values, line_num, True)
                values_str = ' '.join(map(str, values))
                new_line = f"{old2new[class_id]} {values_str}\n"
                f.write(new_line)
        except ValueError as e:
            raise e
        finally:
            f.close()

    dump_yolo_dataset(
        dy,
        destination=destination,
        label_file_op=la_op
    )
    generate_yolo_empty_labels(Path(destination) / 'images', Path(destination) / 'labels')
    ndy = YOLODataset(
        destination,
        categories=new_cate_n2i,
        task="det",
    )
    convert_yolo2coco(ndy, Path(destination) / f"{json_file}_less_cls.json")
    VisualizeYOLODataset(ndy, Path(destination) / 'vis')()
    # subset = dy.split(ratio=[0.5,0.3,0.2], seed=20251210)
    # for i, (name, s) in enumerate(subset.items()):
    #     sub_dy = dy.subset(s)
    #     dump_yolo_dataset(sub_dy,
    #                       destination=destination,
    #                       sub_dirname=name)
    #     convert_yolo2coco(sub_dy, Path(destination) / f"{json_file}_{name}.json")
    # VisualizeYOLODataset(dy, dataset_dir / 'vis')()


def test_convert_coco2yolo():
    dataset = r"E:\python_ai_dataset\COCOtoSLICE-S1-2000\VOC"
    d = VOCDataset(
        dataset,
        task='obb',
    )
    print(len(d), d.task, d.is_read_image)
    print(d.categories())
    print(d.sample_info)
    # convert_voc2yolo(d, Path(dataset) / 'labels')
    # dy = YOLODataset(
    #     dataset,
    #     categories=d.categories(),
    #     task='obb',
    #     fix_bad_data=True
    # )
    # VisualizeYOLODataset(dy, Path(dataset) / 'vis')()
    # convert_yolo2coco(dy, Path(dataset) / 'train.json')


def test_Server():
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 9000))
    server.listen(5)

    print("Server listening on 9000")

    while True:
        conn, addr = server.accept()
        print("Client:", addr)

        data = conn.recv(1024)
        print("Recv:", data.decode())

        conn.sendall(b"Hello Client")
        conn.close()


def test_Client():
    import socket

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 9000))

    client.sendall(b"Hello Server")
    resp = client.recv(1024)

    print(resp.decode())
    client.close()
