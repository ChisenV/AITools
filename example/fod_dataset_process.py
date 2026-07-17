import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from AITools import convert_coco2yolo, dump_yolo_dataset
from AITools.comp.dataset import YOLODataset
from AITools.comp.functions import generate_yolo_empty_labels, are_axis_aligned_rectangles_intersecting
from AITools.comp.processor import CropImages, VisualizeYOLODataset
from AITools.utils.stitch_images import stitch_images

date_num = int(datetime.now().strftime("%Y%m%d"))

cate_stable = {
    0: "entity",
    1: "solder",
    2: "paster",
    3: "device",
    4: "solderBall",
    5: "sticker",
    6: "footprint",
    7: "xizha"
}

def main(split_ratio=None, only_split=False):
    ver = r"v4.8"
    dataset_raw = r"E:\ds\Anomaly\anno"
    dataset_txt = rf"{dataset_raw}/图像裁剪方案.txt"
    cooking_dir = rf"E:\ds\Anomaly\processed\cooking\fod_{date_num}_{ver}"
    cooked_dir = rf"E:\ds\Anomaly\processed\cooked\fod_{date_num}_{ver}"
    stable_dir = rf"E:\ds\Anomaly\processed\stable\fod_{date_num}_{ver}"
    splits_dir = rf"E:\ds\Anomaly\processed\splits\fod_{date_num}_{ver}"
    if split_ratio is None:
        split_ratio = [0.80, 0.10, 0.10]

    cate_cooking = {
        0: 0,
        1: 1,
        2: None,
        3: None,
        4: 2,
        5: None,
        6: 1,
        7: 2,
    }
    cate_cooked = {
        0: "entity",
        1: "solder",
        2: "solderBall",
        # 3: "footprint"
    }

    def label_selector(**kwargs):
        contour = kwargs.get("contour")
        ch = kwargs.get("crop_h")
        cw = kwargs.get("crop_w")
        cls = kwargs.get("class_id")
        if cls == 0:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            x2 = 5 #cw * 0.01
            y2 = 5 #ch * 0.01
            w2 = cw - 10 #cw * 0.02
            h2 = ch - 10 #ch * 0.02
            return are_axis_aligned_rectangles_intersecting((x1, y1, w1, h1), (x2, y2, w2, h2))
        else:
            return True

    with open(dataset_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if only_split:
            break
        if i == 0:
            continue
        line = line.strip().replace(" ", "").split(",")
        if len(line) != 3:
            continue
        image_dir, image_prefix, crop_scheme = line
        print(image_dir, image_prefix, crop_scheme)
        image_path = os.path.join(dataset_raw, Path(image_dir))
        label_path = os.path.join(Path(image_path).parent, "labels")
        annos_path = os.path.join(Path(image_path).parent, "annotations", "annotations.json")
        if not os.path.exists(annos_path):
            print("[not exist]", annos_path)
            continue
        convert_coco2yolo(annos_path, Path(label_path).parent, use_segments=True,)

        generate_yolo_empty_labels(image_path, label_path)

        d_y = YOLODataset(str(Path(image_path).parent),
                          image_dirname="images",
                          label_dirname="labels",
                          with_label=True,
                          with_image=True,
                          task="seg",
                          categories=cate_stable,
                          read_image=False,
                          fix_bad_data=True)

        VisualizeYOLODataset(d_y, save_dir=os.path.join(Path(image_path).parent, "vis"))()

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

        d_y2 = YOLODataset(cooking_dataset_dir,
                           image_dirname="images",
                           label_dirname="labels",
                           with_label=True,
                           task="seg",
                           categories=cate_stable,
                           read_image=False,
                           fix_bad_data=True)

        VisualizeYOLODataset(d_y2, save_dir=os.path.join(cooking_dataset_dir, "vis"))()

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
            def iop(im):
                if im.shape[0] < 1280 and im.shape[1] < 1280:
                    im = cv2.resize(im, (1280, 1280), interpolation=cv2.INTER_LINEAR)
                return im
            w, h = crop_scheme.split("*")
            w, h = int(w), int(h)
            CropImages(Path(cooking_dataset_dir) / "images", Path(cooked_dataset_dir) / "images", w, h,
                       fmt="png", cope_with_label=True, yolo_task='seg', dump_empty=False,
                       label_selector=label_selector, image_op=iop)()
            # d_y3 = YOLODataset(cooked_dataset_dir,
            #                    image_dirname="images",
            #                    label_dirname="labels",
            #                    with_label=True,
            #                    task="seg",
            #                    categories=cate_cooked,
            #                    read_image=False,
            #                    fix_bad_data=True)
            # VisualizeYOLODataset(d_y3, save_dir=os.path.join(cooked_dataset_dir, "vis"))()

        sub_images_path = os.path.join(cooked_dataset_dir, "images")
        sub_labels_path = os.path.join(cooked_dataset_dir, "labels")
        shutil.copytree(sub_images_path, os.path.join(stable_dir, "images"), dirs_exist_ok=True)
        shutil.copytree(sub_labels_path, os.path.join(stable_dir, "labels"), dirs_exist_ok=True)

    dy = YOLODataset(stable_dir,
                     image_dirname="images",
                     label_dirname="labels",
                     with_label=True,
                     task="seg",
                     categories=cate_cooked,
                     read_image=False,
                     fix_bad_data=True)
    subset = dy.split(ratio=split_ratio, seed=date_num)
    for i, (name, s) in enumerate(subset.items()):
        if len(s) == 0:
            continue
        sub_dy = dy.subset(s)
        dump_yolo_dataset(sub_dy, destination=splits_dir, sub_dirname=name)
    VisualizeYOLODataset(dy, save_dir=os.path.join(stable_dir, "vis"))()

    print("remove cooking_dir and cooked_dir...")
    shutil.rmtree(cooking_dir)
    shutil.rmtree(cooked_dir)
    print("done!")
    # shutil.rmtree(stable_dir)


if __name__ == "__main__":
    main()