import os.path
import random
from pathlib import Path

import numpy as np

from AITools.comp.dataset import *
from AITools.comp.functions import *
from AITools.comp.processor import VisualizeOCRDataset, VisualizeYOLODataset

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


def OCRDatesetV2_sample_case1():
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


def test_OCRDatesetV2_init():
    print()
    # OCRDatesetV2_init_case1()
    # OCRDatesetV2_init_case2()
    OCRDatesetV2_init_case3()
    OCRDatesetV2_sample_case1()


def test_OCRDataset_vis():
    dir_path = r"E:\python_ai_dataset\OCR\det\gather\categories_20250415_val"
    vis_path = r"E:\python_ai_dataset\OCR\vis\gather\categories_20250415_val_vis"

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

