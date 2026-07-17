import json
import os

def convert_to_coco_format(json_files):
    coco_format = {
        # ---少锡---
        #"categories": [{"id": 1, "name": "Xi", "color": [255, 0, 255], "supercategory": ""}, {"id": 2, "name": "ShaoXi", "color": [255, 0, 0], "supercategory": ""}, {"id": 3, "name": "LouTong", "color": [255, 85, 0], "supercategory": ""}],
        
        # ---异物---
        "categories": [{"id": 1, "name": "YuanJian", "color": [244, 108, 59], "supercategory": ""}, {"id": 2, "name": "KongHanPan", "color": [170, 0, 0], "supercategory": ""}, {"id": 3, "name": "BiaoQian", "color": [31, 216, 109], "supercategory": ""}, {"id": 4, "name": "SheBei", "color": [85, 255, 255], "supercategory": ""}, {"id": 5, "name": "XiZhu", "color": [255, 255, 0], "supercategory": ""}, {"id": 6, "name": "HongBaiBiaoQian", "color": [255, 0, 255], "supercategory": ""}, {"id": 7, "name": "HanDian", "color": [0, 0, 255], "supercategory": ""}, {"id": 8, "name": "XiZha", "color": [255, 170, 255], "supercategory": ""}, {"id": 9, "name": "BaiDian", "color": [0, 153, 153], "supercategory": ""}, {"id": 10, "name": "SuiXie", "color": [0, 153, 0], "supercategory": ""}],

        # ---色环---
        # "categories": [{"id": 1, "name": "body", "color": [155, 120, 155], "supercategory": ""}, {"id": 2, "name": "black", "color": [4, 4, 4], "supercategory": ""}, {"id": 3, "name": "brown", "color": [134, 67, 0], "supercategory": ""}, {"id": 4, "name": "red", "color": [255, 0, 4], "supercategory": ""}, {"id": 5, "name": "orange", "color": [255, 85, 0], "supercategory": ""}, {"id": 6, "name": "yellow", "color": [255, 255, 0], "supercategory": ""}, {"id": 7, "name": "green", "color": [0, 255, 0], "supercategory": ""}, {"id": 8, "name": "blue", "color": [0, 71, 212], "supercategory": ""}, {"id": 9, "name": "purple", "color": [170, 0, 255], "supercategory": ""}, {"id": 10, "name": "gray", "color": [118, 118, 118], "supercategory": ""}, {"id": 11, "name": "white", "color": [255, 255, 255], "supercategory": ""}, {"id": 12, "name": "gold", "color": [243, 243, 0], "supercategory": ""}, {"id": 13, "name": "silver", "color": [192, 192, 192], "supercategory": ""}],

        "images": [],
        "annotations": [],
        "info": "",
        "licenses": []
    }

    # 预定义类别及其对应的 ID
    category_mapping = {
        #--少锡--
        # "Xi": 1,
        # "ShaoXi": 2,
        # "LouTong": 3,

        #--异物--
        "YuanJian": 1,
        "KongHanPan": 2,
        "BiaoQian": 3,
        "SheBei": 4,
        "XiZhu": 5,
        "HongBaiBiaoQian": 6,
        "HanDian": 7,
        "XiZha": 8,
        "BaiDian":9

        #--色环--
        # "body": 1,
        # "black": 2,
        # "brown": 3,
        # "red": 4,
        # "orange": 5,
        # "yellow": 6,
        # "green": 7,
        # "blue": 8,
        # "purple": 9,
        # "gray": 10,
        # "white": 11,
        # "gold": 12,
        # "silver": 13
    }

    annotation_id = 1

    for json_file in json_files:
        print(f"正在处理文件: {json_file}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"从文件 {json_file} 解码 JSON 时发生错误: {e}")
            continue
        except Exception as e:
            print(f"处理文件 {json_file} 时发生错误: {e}")
            continue

        # 检查 imagePath 是否存在
        if 'imagePath' not in data:
            print(f"警告: 在 {json_file} 中未找到 'imagePath'。跳过此文件。")
            continue
        
        image_file_name = os.path.basename(data['imagePath'])  # 获取最后的图片名称
        
        # 检查图片的宽度和高度是否存在
        if 'imageWidth' not in data or 'imageHeight' not in data:
            print(f"警告: 在 {json_file} 中未找到 'imageWidth' 或 'imageHeight'。跳过此文件。")
            continue

        image_info = {
            "id": len(coco_format['images']) + 1,
            "width": data['imageWidth'],
            "height": data['imageHeight'],
            "file_name": image_file_name,
            "license": "",
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_format['images'].append(image_info)
        print(f"已添加图像信息: {image_info}")

        # 检查 shapes 是否存在
        if 'shapes' not in data:
            print(f"警告: 在 {json_file} 中未找到 'shapes'。跳过此文件。")
            continue

        # 处理每一个 shape
        for shape in data['shapes']:
            # 获取标签
            label = shape['label']
            category_id = category_mapping.get(label)  # 获取预定义的类别 ID

            if category_id is None:
                print(f"警告: 类别 '{label}' 未被识别。跳过此形状。")
                continue  # 如果标签未在映射中定义，则跳过该形状

            # 获取点的信息
            points = shape['points']
            if len(points) < 3:
                print(f"警告: 在形状中点的数量不足，来自 {json_file} 的形状将被跳过。")
                continue
            
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            x_min = min(x_coords)
            y_min = min(y_coords)
            width = max(x_coords) - x_min
            height = max(y_coords) - y_min

            # 添加注释
            annotation_info = {
                "id": annotation_id,
                "iscrowd": 0,
                "image_id": image_info["id"],  # 关联到当前图像
                "category_id": category_id,  # 使用预定义的 id
                "segmentation": [[coord for point in points for coord in point]],  # 格式化为嵌套列表
                "area": width * height,  # 计算面积
                "bbox": [x_min, y_min, width, height]
            }
            coco_format['annotations'].append(annotation_info)
            print(f"已添加注释信息: {annotation_info}")
            annotation_id += 1

    print("所有文件处理完毕。")
    return coco_format

def main(input_dir, output_file):
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个 JSON 文件。")
    coco_data = convert_to_coco_format(json_files)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False)  # 不格式化的 JSON 文件
    print(f"已将 COCO 格式数据写入文件: {output_file}")

if __name__ == "__main__":
    input_directory = r"E:\ds\Anomaly\anno\FOD_20260630_TestPoints\jsons"  # 输入JSON文件夹路径
    output_json_file = r"E:\ds\Anomaly\anno\FOD_20260630_TestPoints\annotations\annotations.json"  # 输出COCO格式文件名
    main(input_directory, output_json_file)
