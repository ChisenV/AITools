import os.path
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from AITools import IMG_FORMATS
from AITools.comp.dataset import *
from AITools.comp.dataset import VOCDataset
from AITools.comp.functions import *
from AITools.comp.functions import convertCOCO2YOLO, rotate_image_around_point, generate_yolo_empty_labels
from AITools.comp.processor import VisualizeOCRDataset, VisualizeYOLODataset, CropImages

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

def OCRRECDatesetV2_sample_case1():
    dst_dir = r"E:\python_ai_dataset\OCR\det\from000\toAITrain\rec_20250516_v0.1_treated"
    d = OCRRECDatasetV2(
        r"E:\python_ai_dataset\OCR\det\from000\rec\trouble1-treated",
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
                    warpAffine_points(_la["points"], M, round=0)
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


def test_OCRDatesetV2_init():
    print()
    # OCRDatesetV2_init_case1()
    # OCRDatesetV2_init_case2()
    # OCRDatesetV2_init_case3()
    # OCRDatesetV2_sample_case0()
    OCRDatesetV2_sample_case1()
    # OCRDatesetV2_sample_case2()
    # OCRRECDatesetV2_sample_case0()
    # OCRCLSDatesetV2_sample_case1()
    # OCRRECDatesetV2_sample_case1()

    # union_labels(r"E:\python_ai_dataset\OCR\det\Label_new\anno")
    # union_labels(r"E:\python_ai_dataset\OCR\det\gather\anno_20250512_train")
    # union_labels(r"E:\python_ai_dataset\OCR\det\gather\anno_20250512_val")
    # union_labels(r"E:\python_ai_dataset\OCR\det\gather\anno_20250512_test")


def test_OCRDataset_vis():
    dir_path = r"E:\python_ai_dataset\OCR\det\gather\categories_20250507"
    vis_path = r"E:\python_ai_dataset\OCR\vis\gather\categories_20250507"

    dataset = OCRDatasetV2(
        det_paths(dir_path), with_label=True, subject_to="image")
    VisualizeOCRDataset(dataset, save_dir=vis_path, text_enable=True, line_color=(0, 180, 0), text_color=(0, 128, 0))()

    return


def test_YOLODataset_init():
    print()

    d = YOLODataset(
        root=r"E:\python_ai_dataset\foreign-object-detect\2-FOVSlice\train\images-black-green-V3.2\images\val",
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
        save_dir=r"E:\python_ai_dataset\foreign-object-detect\2-FOVSlice\train\images-black-green-V3.2-vis-val",
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
    dir_data = r"E:\python_ai_dataset\foreign-object-detect\BUBBLE\UV_BUBBLE"
    dst_dir = r"E:\python_ai_dataset\foreign-object-detect\BUBBLE\UV_BUBBLE\labels"
    crop_dir = r"E:\python_ai_dataset\foreign-object-detect\BUBBLE\UV_BUBBLE\images_crop_{}"
    # d = VOCDataset(dir_data, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    # convertVOC2YOLO(d, dst_dir)
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
    d_y = YOLODataset(vis_yolo,
                      image_dirname="images",
                      label_dirname="labels",
                      with_label=True,
                      task="det",
                      categories={0: "BUBBLE"},
                      read_image=False)

    VisualizeYOLODataset(d_y, save_dir=os.path.join(vis_yolo, "vis"))()


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
            5: None,  # sticker
            6: 1,     # footprint
        }
        return label_map[cls]

    convertCOCO2YOLO(
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
    dir_data = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\train"
    dst_data = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\train_split"
    dy = YOLODataset(dir_data,
                     with_label=True,
                     task="seg",
                     categories={0: "entity", 1: "solder"})
    subset = dy.split(ratio=[0.7, 0.2, 0.1], seed=20250829)
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
    # 1: "SMT_Capacitor",
    # 2: "SMT_Diode",
    # 3: "SMT_IC",
    # 4: "SMT_QFN",
}
def test_vis_yolo():
    dir_path = r"E:\python_ai_dataset\obb\3D-SMT_IC-std\3D-SMT_IC"
    dst_dir  = r"E:\python_ai_dataset\obb\3D-SMT_IC-std\3D-SMT_IC-split"
    path_list = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, i))]
    # for path in path_list:
    vd = VOCDataset(dir_path,
                    with_label=True,
                    image_dirname="JPEGImages",
                    label_dirname="Annotations",
                    categories=categories,
                    task='obb',)

    convertVOC2YOLO(vd, save_dir=os.path.join(dir_path, "labels"))

    dy = YOLODataset(
        dir_path,
        image_dirname=r"JPEGImages",
        label_dirname=r"labels",
        with_label=True,
        task="obb",
        categories=categories,
        read_image=False,
        subject_to="image"
    )
    # VisualizeYOLODataset(dy2, save_dir=os.path.join(path, "vis"))()
    subset = dy.split(ratio=[0.75, 0.25], seed=20250715)
    for i, (name, s) in enumerate(subset.items()):
        sub_dy = dy.subset(s)
        dump_yolo_dataset(sub_dy, destination=dst_dir, sub_dirname=name)


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