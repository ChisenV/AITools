import os
import shutil
from pathlib import Path

from AITools import convert_coco2yolo, dump_yolo_dataset
from AITools.comp.dataset import YOLODataset
from AITools.comp.functions import generate_yolo_empty_labels
from AITools.comp.processor import CropImages, VisualizeYOLODataset
from AITools.utils.stitch_images import stitch_images


def process_fod_dataset():
    dataset_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\raw"
    cooking_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\train_cooking"
    cooked_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\train_cooked"
    train_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.7\train"
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
