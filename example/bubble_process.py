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


def crop_VOCDataset(dir_data):
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

def test_YOLODataset():
    dir_data = fr"E:\ds\Bubble\BUBBLE-AIDIAN\images_crop_1024"
    dy = YOLODataset(dir_data, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    VisualizeYOLODataset(dy, os.path.join(dir_data, "vis"))()

def collect_all_dataset():
    dirs = [
        r"E:\ds\Bubble\anno\BUBBLE-1\images_crop_1024\images",
        r"E:\ds\Bubble\anno\BUBBLE-2\images_crop_1024\images",
        r"E:\ds\Bubble\anno\BUBBLE-AIDIAN\images_crop_1024\images",
        r"E:\ds\Bubble\anno\UV_BUBBLE\images_crop_1024\images",
    ]
    dy = YOLODataset(
        dirs, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False
    )
    total = len(dy)
    print(f"Total samples: {total}")

    dst_dir = rf"E:\ds\Bubble\toTrain\Bubble_{date_num}"
    zero_pad = len(str(total))

    def rename_copy_img(old_path, new_path, idx=None, image_name=None):
        ext = os.path.splitext(new_path)[1]
        new_name = f"UID{idx:0{zero_pad}d}_Bubble{ext}"
        new_dir = os.path.dirname(new_path)
        new_dst = os.path.join(new_dir, new_name)
        shutil.copy(old_path, new_dst)

    def rename_copy_lab(old_path, new_path, idx=None, image_name=None):
        ext = os.path.splitext(new_path)[1]
        new_name = f"UID{idx:0{zero_pad}d}_Bubble{ext}"
        new_dir = os.path.dirname(new_path)
        new_dst = os.path.join(new_dir, new_name)
        shutil.copy(old_path, new_dst)

    dump_yolo_dataset(
        dy,
        dst_dir,
        image_file_op=rename_copy_img,
        label_file_op=rename_copy_lab,
    )
    dyn = YOLODataset(dst_dir, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    VisualizeYOLODataset(dyn, os.path.join(dst_dir, "vis"))()
    print(f"Done. Saved to {dst_dir}")
    return dst_dir

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
    top_dir = r"E:\ds\Bubble\anno"
    # dir_list = os.listdir(top_dir)
    #
    # for _dir in dir_list:
    #     crop_VOCDataset(os.path.join(top_dir, _dir))
    #     # convert_dataset2(os.path.join(top_dir, _dir), "images")
    # dst_dir = collect_all_dataset()
    # convert_dataset(dst_dir)

    crop_img(r"E:\Jobs\bubble\1\fov")
