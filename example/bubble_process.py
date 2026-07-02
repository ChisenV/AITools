import os
import shutil
from datetime import datetime
from pathlib import Path

os.environ["PATH"] = r"E:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

from AITools import IMG_FORMATS, YOLODataset, VOCDataset, dump_yolo_dataset
from AITools.comp.functions import generate_yolo_empty_labels, img2label_path, convert_voc2yolo, convert_yolo2coco, \
    convert_yolo2voc
from AITools.comp.processor import VisualizeYOLODataset, CropImages

date_num = int(datetime.now().strftime("%Y%m%d"))


def convert_crop_VOCDataset(dir_data):
    # dir_data = r"E:\ds\Bubble\BUBBLE-AIDIAN"
    d = VOCDataset(dir_data, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    print(len(d))
    convert_voc2yolo(d, os.path.join(dir_data, "labels"))
    dy = YOLODataset(dir_data, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    VisualizeYOLODataset(dy, os.path.join(dir_data, "vis"))()
    CropImages(
        os.path.join(dir_data, "images"),
        os.path.join(dir_data, "images_crop_1024", "images"),
        1024,
        1024,
        cope_with_label=True,
        dump_empty=True,
    )()
    # dy = YOLODataset(
    #     os.path.join(dir_data, "images_crop_1024"),
    #     with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    # VisualizeYOLODataset(dy, os.path.join(dir_data, "vis"))()
    # convert_yolo2voc(dy, Path(dir_data) / "images_crop_1024" / "Annotations")

def collect_all_dataset(dirs, dst_dir=rf"E:\ds\Bubble\toTrain\Bubble_{date_num}", offset=0):
    # dirs = [
    #     r"E:\ds\Bubble\anno\BUBBLE-1\images_crop_1024\images",
    #     r"E:\ds\Bubble\anno\BUBBLE-2\images_crop_1024\images",
    #     r"E:\ds\Bubble\anno\BUBBLE-AIDIAN\images_crop_1024\images",
    #     r"E:\ds\Bubble\anno\UV_BUBBLE\images_crop_1024\images",
    # ]
    dy = YOLODataset(
        dirs, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False
    )
    total = len(dy)
    print(f"Total samples: {total}")

    zero_pad = len(str(total))

    def rename_copy_img(old_path, new_path, idx=None, image_name=None):
        ext = os.path.splitext(new_path)[1]
        new_name = f"UID{idx+offset:0{zero_pad}d}_Bubble{ext}"
        new_dir = os.path.dirname(new_path)
        new_dst = os.path.join(new_dir, new_name)
        shutil.copy(old_path, new_dst)

    def rename_copy_lab(old_path, new_path, idx=None, image_name=None):
        ext = os.path.splitext(new_path)[1]
        new_name = f"UID{idx+offset:0{zero_pad}d}_Bubble{ext}"
        new_dir = os.path.dirname(new_path)
        new_dst = os.path.join(new_dir, new_name)
        shutil.copy(old_path, new_dst)

    dump_yolo_dataset(
        dy,
        dst_dir,
        image_dirname="JPEGImages",
        image_file_op=rename_copy_img,
        label_file_op=rename_copy_lab,
    )
    dyn = YOLODataset(dst_dir, image_dirname="JPEGImages", with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    VisualizeYOLODataset(dyn, os.path.join(dst_dir, "vis"))()
    convert_yolo2voc(dyn, Path(dst_dir) / "Annotations")
    print(f"Done. Saved to {dst_dir}")
    return len(dyn)

def split_dataset(dir_data, dst_data):
    dy = YOLODataset(dir_data,image_dirname="JPEGImages",
                     with_label=True,
                     categories={0: "UV_BUBBLE"})
    subset = dy.split(ratio=[0.85, 0.15], seed=date_num)
    for i, (name, s) in enumerate(subset.items()):
        sub_dy = dy.subset(s)
        dump_yolo_dataset(sub_dy, destination=dst_data, sub_dirname=name)

def convert_dataset(dir_data, image_dirname="images"):
    dy = YOLODataset(dir_data,with_label=True,image_dirname=image_dirname,
                     categories={0: "UV_BUBBLE"})
    convert_yolo2voc(dy, Path(dir_data)/"Annotations")


def convert_dataset2(dir_data, image_dirname="JPEGImages"):
    dy = VOCDataset(dir_data,with_label=True,image_dirname=image_dirname,
                     categories={0: "UV_BUBBLE"})
    convert_voc2yolo(dy, Path(dir_data)/"labels-voc2yolo")

    dy = YOLODataset(dir_data,with_label=True,image_dirname=image_dirname,
                     label_dirname="labels-voc2yolo", categories={0: "UV_BUBBLE"})
    VisualizeYOLODataset(dy, os.path.join(dir_data, "vis-voc2yolo"))()

def crop_img(dir_data):
    CropImages(
        dir_data,
        os.path.join(dir_data, "images_crop_1024"),
        1024,
        1024,
        # cope_with_label=True,
        # dump_empty=True,
    )()

if __name__ == "__main__":
    # test_YOLODataset()
    # collect_all_dataset()
    # split_dataset(rf"E:\ds\Bubble\toTrain\Bubble_20260602",
    #               rf"E:\ds\Bubble\toTrain\Bubble_{date_num}_split")
    # convert_dataset(rf"E:\ds\Bubble\toTrain\Bubble_20260602")
    # top_dir = r"E:\ds\Bubble\anno"
    # dir_list = os.listdir(top_dir)
    #
    # for _dir in dir_list:
    #     crop_VOCDataset(os.path.join(top_dir, _dir))
    #     # convert_dataset2(os.path.join(top_dir, _dir), "images")
    # dst_dir = collect_all_dataset()
    # convert_dataset(dst_dir)

    # crop_img(r"E:\Jobs\bubble\1\fov")

    # 0. 某份标注好的数据集
    anno_dirs = [
        r"E:\ds\Bubble\anno\BUBBLE-AIDIAN-V2"
    ]
    # 1. 标注好的先转成yolo, 可视化检查标注，然后裁剪
    for anno_dir in anno_dirs:
        convert_crop_VOCDataset(anno_dir)
    # 2. 收集所有裁剪的数据集并重命名好
    dirs = [
        r"E:\ds\Bubble\anno\BUBBLE-1\images_crop_1024\images",
        r"E:\ds\Bubble\anno\BUBBLE-2\images_crop_1024\images",
        r"E:\ds\Bubble\anno\BUBBLE-AIDIAN\images_crop_1024\images",
        r"E:\ds\Bubble\anno\BUBBLE-AIDIAN-V2\images_crop_1024\images",
        r"E:\ds\Bubble\anno\UV_BUBBLE\images_crop_1024\images",
    ]
    next_idx = collect_all_dataset(dirs, offset=0)
    print(f"{next_idx = }")
