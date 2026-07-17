import json
import copy
import os
from collections import defaultdict
from pathlib import Path

from AITools import convert_coco2yolo, generate_yolo_empty_labels, YOLODataset
from AITools.comp.processor import VisualizeYOLODataset


def merge_coco_annotations(json1_path, json2_path, output_path):
    """
    合并两个COCO格式的标注JSON文件

    Args:
        json1_path: 第一个JSON文件路径
        json2_path: 第二个JSON文件路径
        output_path: 输出JSON文件路径
    """
    # 加载两个JSON文件
    with open(json1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open(json2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    # 创建合并后的数据结构
    merged = {
        "info": data1.get("info", ""),
        "licenses": data1.get("licenses", []),
        "categories": [],
        "images": [],
        "annotations": []
    }

    # 1. 合并categories（按名称去重）
    category_map = {}  # name -> category_id
    max_cat_id = 0

    for data in [data1, data2]:
        for cat in data.get("categories", []):
            cat_name = cat["name"]
            if cat_name not in category_map:
                # 分配新的ID
                max_cat_id += 1
                new_cat = copy.deepcopy(cat)
                new_cat["id"] = max_cat_id
                category_map[cat_name] = max_cat_id
                merged["categories"].append(new_cat)
            else:
                # 如果已存在，记录映射关系
                pass

    # 创建反向映射：原始ID -> 新ID（用于更新annotation中的category_id）
    id_mapping_cat = {}  # (dataset_index, old_cat_id) -> new_cat_id

    # 重新构建映射关系
    for idx, data in enumerate([data1, data2]):
        for cat in data.get("categories", []):
            old_id = cat["id"]
            new_id = category_map[cat["name"]]
            id_mapping_cat[(idx, old_id)] = new_id

    # 2. 合并images（按file_name去重）
    image_map = {}  # file_name -> new_image_id
    max_img_id = 0
    image_id_mapping = {}  # (dataset_index, old_image_id) -> new_image_id

    for idx, data in enumerate([data1, data2]):
        for img in data.get("images", []):
            file_name = img["file_name"]
            if file_name not in image_map:
                max_img_id += 1
                new_img = copy.deepcopy(img)
                new_img["id"] = max_img_id
                image_map[file_name] = max_img_id
                merged["images"].append(new_img)
                image_id_mapping[(idx, img["id"])] = max_img_id
            else:
                # 如果文件名重复，使用已有的ID
                image_id_mapping[(idx, img["id"])] = image_map[file_name]

    # 3. 合并annotations，更新image_id和category_id
    max_anno_id = 0
    anno_id_mapping = {}  # 用于记录旧的annotation ID映射（可选）

    # 先找到最大的annotation ID
    for data in [data1, data2]:
        for anno in data.get("annotations", []):
            if anno["id"] > max_anno_id:
                max_anno_id = anno["id"]

    current_anno_id = max_anno_id + 1

    for idx, data in enumerate([data1, data2]):
        for anno in data.get("annotations", []):
            new_anno = copy.deepcopy(anno)

            # 更新image_id
            old_img_id = anno["image_id"]
            new_img_id = image_id_mapping.get((idx, old_img_id))
            if new_img_id is None:
                # 如果找不到映射，跳过该标注
                print(f"Warning: Image ID {old_img_id} not found in dataset {idx}")
                continue
            new_anno["image_id"] = new_img_id

            # 更新category_id
            old_cat_id = anno["category_id"]
            new_cat_id = id_mapping_cat.get((idx, old_cat_id))
            if new_cat_id is None:
                print(f"Warning: Category ID {old_cat_id} not found in dataset {idx}")
                continue
            new_anno["category_id"] = new_cat_id

            # 分配新的annotation ID
            new_anno["id"] = current_anno_id
            current_anno_id += 1

            merged["annotations"].append(new_anno)

    # 4. 保存合并后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"合并完成！")
    print(f"  - Categories: {len(merged['categories'])}")
    print(f"  - Images: {len(merged['images'])}")
    print(f"  - Annotations: {len(merged['annotations'])}")

    return merged


def merge_coco_annotations_simple(json1_path, json2_path, output_path):
    """
    简化版本：直接合并两个JSON文件，不做去重处理

    适用于两个数据集类别和图像ID不冲突的情况
    """
    with open(json1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open(json2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    # 计算偏移量
    max_img_id = max([img["id"] for img in data1.get("images", [])] + [0])
    max_cat_id = max([cat["id"] for cat in data1.get("categories", [])] + [0])
    max_anno_id = max([anno["id"] for anno in data1.get("annotations", [])] + [0])

    merged = copy.deepcopy(data1)

    # 添加第二个数据集的categories（ID偏移）
    for cat in data2.get("categories", []):
        new_cat = copy.deepcopy(cat)
        new_cat["id"] = cat["id"] + max_cat_id
        merged["categories"].append(new_cat)

    # 添加第二个数据集的images（ID偏移）
    for img in data2.get("images", []):
        new_img = copy.deepcopy(img)
        new_img["id"] = img["id"] + max_img_id
        merged["images"].append(new_img)

    # 添加第二个数据集的annotations（ID偏移）
    for anno in data2.get("annotations", []):
        new_anno = copy.deepcopy(anno)
        new_anno["id"] = anno["id"] + max_anno_id
        new_anno["image_id"] = anno["image_id"] + max_img_id
        new_anno["category_id"] = anno["category_id"] + max_cat_id
        merged["annotations"].append(new_anno)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"合并完成（简单模式）！")
    print(f"  - Categories: {len(merged['categories'])}")
    print(f"  - Images: {len(merged['images'])}")
    print(f"  - Annotations: {len(merged['annotations'])}")

    return merged


# 使用示例
if __name__ == "__main__":
    # 方法1：智能合并（按文件名和类别名去重）
    # merge_coco_annotations(
    #     r"E:\ds\Anomaly\anno\5-FlyEle\annotations\annotations.json",
    #     r"E:\ds\Anomaly\anno\3-FlyEle_red\annotations\annotations.json",
    #     r"E:\ds\Anomaly\test\annotations.json")

    # 方法2：简单合并（直接拼接，ID偏移）
    # merge_coco_annotations_simple("annotations1.json", "annotations2.json", "merged_annotations.json")

    # 方法3：从文件列表合并多个JSON
    def merge_multiple_coco(json_files, output_path):
        """合并多个COCO JSON文件"""
        if not json_files:
            return

        merged_data = None
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if merged_data is None:
                merged_data = data
            else:
                # 使用智能合并
                # 先保存为临时文件，然后合并
                temp_path = "temp_merge.json"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_data, f, ensure_ascii=False)
                merge_coco_annotations(temp_path, json_file, output_path)
                with open(output_path, 'r', encoding='utf-8') as f:
                    merged_data = json.load(f)

        # 删除临时文件
        import os
        if os.path.exists("temp_merge.json"):
            os.remove("temp_merge.json")

        print(f"合并 {len(json_files)} 个文件完成！")
        return merged_data

    # 示例：合并多个文件
    # merge_multiple_coco(["ann1.json", "ann2.json", "ann3.json"], "final_merged.json")

    # dir_path = r"E:\ds\Anomaly\test\5-FlyEle"
    # annos_path = os.path.join(dir_path, "annotations", "annotations.json")
    # convert_coco2yolo(annos_path, Path(dir_path), use_segments=True, )
    # generate_yolo_empty_labels(Path(dir_path) / "images", Path(dir_path) / "labels")
    #
    # d_y = YOLODataset(str(dir_path),
    #                   image_dirname="images",
    #                   label_dirname="labels",
    #                   with_label=True,
    #                   with_image=True,
    #                   task="seg",
    #                   # categories=cate_stable,
    #                   read_image=False,
    #                   fix_bad_data=True)
    # VisualizeYOLODataset(d_y, save_dir=os.path.join(dir_path, "vis"))()
