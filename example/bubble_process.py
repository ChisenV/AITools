import os
import shutil
from datetime import datetime

os.environ["PATH"] = r"E:\thirdparty\TensorRT\TensorRT-10.10.0.31\lib;" + os.environ["PATH"]

from AITools import IMG_FORMATS, YOLODataset, VOCDataset, dump_yolo_dataset
from AITools.comp.functions import generate_yolo_empty_labels, img2label_path, convert_voc2yolo
from AITools.comp.processor import VisualizeYOLODataset, CropImages

date_num = int(datetime.now().strftime("%Y%m%d"))


def test_VOCDataset():
    dir_data = r"E:\ds\Bubble\BUBBLE-AIDIAN"
    d = VOCDataset(dir_data, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    print(len(d))
    convert_voc2yolo(d, os.path.join(dir_data, "labels"))
    dy = YOLODataset(dir_data, with_label=True, categories={0: "UV_BUBBLE"}, read_image=False)
    VisualizeYOLODataset(dy, os.path.join(dir_data, "vis"))()
    CropImages(
        os.path.join(dir_data, "images"),
        os.path.join(dir_data, "images_crop_1024"),
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
    print(f"Done. Saved to {dst_dir}")

if __name__ == "__main__":
    # test_YOLODataset()
    collect_all_dataset()
