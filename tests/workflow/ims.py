import os.path
import shutil
from pathlib import Path

import cv2
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Any

from AITools import imread, imwrite, imshow, get_img_files, compute_affine_matrix, YOLODataset
from AITools.comp.processor import VisualizeYOLODataset


def analyze_image_dimensions(file_list):
    """
    统计图像列表的宽高分布情况

    参数:
        file_list: 图像文件路径列表

    返回:
        包含统计信息的字典
    """
    # 初始化存储容器
    widths = []
    heights = []
    aspect_ratios = []
    errors = []

    # 遍历所有文件
    for img_path in file_list:
        try:
            # 读取图像
            img = imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")

            # 获取尺寸
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

        except Exception as e:
            errors.append(str(e))
            continue

    # 转换为NumPy数组
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # 计算基本统计量
    stats = {
        'total_images': len(file_list),
        'valid_images': len(widths),
        'error_files': len(errors),
        'width_stats': {
            'min': np.min(widths),
            'max': np.max(widths),
            'mean': np.mean(widths),
            'median': np.median(widths),
            'std': np.std(widths),
            'percentile_25': np.percentile(widths, 25),
            'percentile_75': np.percentile(widths, 75)
        },
        'height_stats': {
            'min': np.min(heights),
            'max': np.max(heights),
            'mean': np.mean(heights),
            'median': np.median(heights),
            'std': np.std(heights),
            'percentile_25': np.percentile(heights, 25),
            'percentile_75': np.percentile(heights, 75)
        },
        'aspect_ratio_stats': {
            'min': np.min(aspect_ratios),
            'max': np.max(aspect_ratios),
            'mean': np.mean(aspect_ratios),
            'median': np.median(aspect_ratios),
            'std': np.std(aspect_ratios)
        },
        'dimension_distribution': defaultdict(int),
        'errors': errors
    }

    # 统计尺寸分布（精确尺寸）
    for w, h in zip(widths, heights):
        stats['dimension_distribution'][(w, h)] += 1

    # 添加最常见的尺寸
    if stats['dimension_distribution']:
        most_common = max(stats['dimension_distribution'].items(), key=lambda x: x[1])
        stats['most_common_dimension'] = {
            'size': most_common[0],
            'count': most_common[1],
            'percentage': (most_common[1] / len(widths)) * 100
        }

    return stats


def print_stats(stats):
    # 打印统计结果
    print(f"总图像数: {stats['total_images']}")
    print(f"有效图像数: {stats['valid_images']}")
    print(f"错误文件数: {stats['error_files']}")

    print("\n宽度统计:")
    for k, v in stats['width_stats'].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n高度统计:")
    for k, v in stats['height_stats'].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n宽高比统计:")
    for k, v in stats['aspect_ratio_stats'].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n最常见尺寸:")
    common = stats['most_common_dimension']
    print(f"  尺寸: {common['size'][0]}x{common['size'][1]}")
    print(f"  数量: {common['count']}")
    print(f"  占比: {common['percentage']:.2f}%")

    # 显示错误文件（如果有）
    if stats['errors']:
        print("\n错误文件:")
        for err in stats['errors'][:5]:  # 只显示前5个错误
            print(f"  - {err}")
        if len(stats['errors']) > 5:
            print(f"  ... 还有 {len(stats['errors']) - 5} 个错误")


def get_resize_transform(original_size, target_sizes=None):
    """
    计算将图像约束到目标尺寸的仿射变换矩阵

    参数:
        original_size: 原始图像尺寸 (width, height)
        target_sizes: 目标尺寸列表，默认为 [320, 640, 960, 1280, 1600]

    返回:
        target_size: 选择的目标尺寸
        transform_matrix: 2x3 仿射变换矩阵
    """
    if target_sizes is None:
        target_sizes = [320, 640, 960, 1280, 1600, 1920, 2240]

    orig_w, orig_h = original_size
    max_side = max(orig_w, orig_h)

    # 选择合适的目标尺寸：大于等于原始最大边且最接近的尺寸
    if max_side <= target_sizes[0] - 160:
        target_size = target_sizes[0]
    else:
        target_size = min([size for size in target_sizes if size - 160 < max_side <= size + 160], default=max(target_sizes))

    transform_matrix, _ = compute_affine_matrix((orig_w, orig_h), (target_size, target_size))
    return target_size, transform_matrix


def stitch_images_to_canvas(
        image_list: List[str],
        canvas_size: int = 1280
) -> dict:
    """
    优化后的图像拼接函数，在放置大图像后立即尝试填充小图像

    参数:
        image_data_list: 包含(原始图像, 目标尺寸, 变换矩阵)的列表
        canvas_size: 画布尺寸，默认为1280

    返回:
        包含所有拼接后画布的列表
    """
    # 定义网格单位大小（最小尺寸320）
    grid_unit = 320
    grid_size = canvas_size // grid_unit  # 1280/320=4，所以是4×4网格

    image_data_in_rang_list = []
    image_data_out_rang_list = []
    for img_path in image_list:
        image = imread(img_path)
        size = image.shape[:2]
        orig_h, orig_w = size
        target_size, transform = get_resize_transform((orig_w, orig_h))
        if target_size < canvas_size:
            image_data_in_rang_list.append((img_path, target_size, transform))
        else:
            image_data_out_rang_list.append((img_path, target_size, transform))

    # 按目标尺寸从大到小排序（优先放置大尺寸图像）
    sorted_images = sorted(image_data_in_rang_list, key=lambda x: x[1], reverse=True)

    # 初始化画布列表和剩余图像
    canvases = {}
    remaining_images = sorted_images.copy()
    cnt = 0
    canvas_cnt = 0
    while remaining_images:
        # 创建新画布
        # current_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        canvas_info = []
        current_grid = np.zeros((grid_size, grid_size), dtype=int)
        # 尝试在当前画布上放置尽可能多的图像
        while True:
            placed_any = False

            # 首先尝试放置最大的剩余图像
            for idx in range(len(remaining_images)):
                img_path, target_size, transform = remaining_images[idx]

                # 计算图像占用的网格单位数
                grid_units = target_size // grid_unit  # 320->1, 640->2, 960->3

                # 在当前网格中查找空闲位置
                placement = find_best_placement(current_grid, grid_units, grid_size)

                if placement is not None:
                    # 放置图像
                    row, col = placement
                    cnt += 1
                    canvas_info.append((img_path, target_size, transform, canvas_cnt, row * grid_unit, col * grid_unit))

                    # 标记网格为已占用
                    for i in range(grid_units):
                        for j in range(grid_units):
                            current_grid[row + i][col + j] = 1

                    # 从剩余图像中移除已放置的图像
                    remaining_images.pop(idx)
                    placed_any = True
                    break  # 重新开始放置过程

            # 如果没有放置任何图像，尝试放置更小的图像
            if not placed_any:
                # 按从大到小顺序尝试放置所有剩余图像
                for idx in range(len(remaining_images)):
                    img_path, target_size, transform = remaining_images[idx]

                    # 计算图像占用的网格单位数
                    grid_units = target_size // grid_unit  # 320->1, 640->2, 960->3

                    # 在当前网格中查找空闲位置
                    placement = find_best_placement(current_grid, grid_units, grid_size)

                    if placement is not None:
                        # 放置图像
                        row, col = placement
                        cnt += 1
                        canvas_info.append((img_path, target_size, transform, canvas_cnt, row * grid_unit, col * grid_unit))
                        # 标记网格为已占用
                        for i in range(grid_units):
                            for j in range(grid_units):
                                current_grid[row + i][col + j] = 1

                        # 从剩余图像中移除已放置的图像
                        remaining_images.pop(idx)
                        placed_any = True
                        break  # 重新开始放置过程

            # 如果本轮没有放置任何图像，结束当前画布的填充
            if not placed_any:
                break

        # 添加当前画布到结果列表
        canvases[canvas_cnt] = canvas_info
        canvas_cnt += 1

    for i, (img_path, target_size, transform) in enumerate(image_data_out_rang_list):
        canvases[canvas_cnt] = [(img_path, target_size, transform, canvas_cnt, 0, 0)]
        cnt += 1
        canvas_cnt += 1
    print(f"all image:{cnt}")
    return canvases


def find_best_placement(
        grid: np.ndarray,
        grid_units: int,
        grid_size: int
) -> Optional[Tuple[int, int]]:
    """
    在网格中查找最佳放置位置

    参数:
        grid: 当前网格状态
        grid_units: 占用的网格单位数
        grid_size: 网格总大小

    返回:
        最佳放置位置(row, col)或None（如果没有合适位置）
    """
    best_placement = None
    best_score = -1

    # 遍历所有可能的放置位置
    for row in range(grid_size - grid_units + 1):
        for col in range(grid_size - grid_units + 1):
            # 检查该区域是否空闲
            if all(
                    grid[row + i][col + j] == 0
                    for i in range(grid_units)
                    for j in range(grid_units)
            ):
                # 计算放置位置评分
                score = evaluate_placement(grid, row, col, grid_units, grid_size)

                # 更新最佳位置
                if score > best_score:
                    best_score = score
                    best_placement = (row, col)

    return best_placement


def evaluate_placement(
        grid: np.ndarray,
        row: int,
        col: int,
        grid_units: int,
        grid_size: int
) -> float:
    """
    评估放置位置的质量，返回评分（越高越好）

    参数:
        grid: 当前网格状态
        row, col: 放置位置的起始坐标
        grid_units: 占用的网格单位数
        grid_size: 网格总大小

    返回:
        放置位置的质量评分
    """
    score = 0.0

    # 1. 优先放置在左上角（减少碎片）
    score += (grid_size - row) * 0.1 + (grid_size - col) * 0.1

    # 2. 优先放置在已有图像旁边（减少分散）
    if row > 0 and grid[row - 1][col] == 1:
        score += 0.5
    if col > 0 and grid[row][col - 1] == 1:
        score += 0.5

    # 3. 惩罚创建孤立区域
    isolated_penalty = 0
    for i in range(grid_units):
        for j in range(grid_units):
            # 检查周围是否有其他占用区域
            has_neighbor = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = row + i + di, col + j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    if grid[ni][nj] == 1:
                        has_neighbor = True
                        break
            if not has_neighbor:
                isolated_penalty += 0.1

    score -= isolated_penalty

    # 4. 优先放置大图像（已在排序中处理）

    return score


def place_image_on_canvas(
        canvas: np.ndarray,
        image: np.ndarray,
        x: int,
        y: int,
        name: str = None
) -> None:
    """
    将图像放置在画布的指定位置

    参数:
        canvas: 目标画布
        image: 要放置的图像
        x, y: 放置位置的左上角坐标
    """
    h, w = image.shape[:2]
    canvas[y:y + h, x:x + w] = image
    if name is not None:
        cv2.putText(canvas, name, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def do(data_dir, save_dir):
    basename = os.path.basename(save_dir)
    save_image_dir = os.path.join(save_dir, "images")
    save_label_dir = os.path.join(save_dir, "labels")
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    canvas_size = 1280
    cate = {
        0: "entity",
        1: "solder",
        2: "paster",
        3: "device",
        4: "solderBall",
        5: "sticker",
        6: "footprint",
    }
    dsy = YOLODataset(data_dir,
                      image_dirname="images",
                      label_dirname="labels",
                      with_label=True,
                      task="seg",
                      categories=cate,
                      read_image=False)
    canvases = stitch_images_to_canvas(dsy.images, canvas_size)

    for canvas_id, canvas in canvases.items():
        current_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        current_canvas_label_lines = []
        is_out_canvas = False
        for info in canvas:
            image_path, target_size, transform, canvas_cnt, canvas_x, canvas_y = info
            if target_size >= canvas_size:
                shutil.copy(image_path, os.path.join(save_image_dir, f"canvas_{canvas_id}.{os.path.basename(image_path).rsplit('.', 1)[-1]}"))
                shutil.copy(dsy.img2label_path(image_path), os.path.join(save_label_dir, f"canvas_{canvas_id}.txt"))
                is_out_canvas = True
                break
            image = imread(image_path)
            orig_h, orig_w = image.shape[:2]
            zeros = np.zeros((orig_h, orig_w), dtype=np.uint8)
            trans = cv2.warpAffine(
                image,
                transform,
                (target_size, target_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            place_image_on_canvas(current_canvas, trans, canvas_x, canvas_y)
            label_path = dsy.img2label_path(image_path)
            if not os.path.exists(label_path):
                continue
            with open(label_path, 'r', encoding='utf-8') as f:
                orig_lines = [line.strip() for line in f.readlines() if line.strip()]
            for line in orig_lines:
                parts = line.split()
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                if len(coords) % 2 != 0:
                    raise ValueError("The number of coordinates must be an even number.")

                abs_points = np.array(coords, dtype=np.float32).reshape(-1, 2) * [orig_w, orig_h]
                np.round(abs_points, out=abs_points)
                cv2.fillPoly(zeros, [np.array([abs_points], dtype=np.int32)], color=255)
                trans_zeros = cv2.warpAffine(
                    zeros,
                    transform,
                    (target_size, target_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )
                contours, _ = cv2.findContours(trans_zeros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                assert len(contours) == 1
                for contour in contours:
                    if len(contour) >= 3:  # 有效多边形至少需要3个点
                        # 转换为相对坐标
                        rel_points = (contour.squeeze().astype(np.float32) + [canvas_x, canvas_y]) / [canvas_size, canvas_size]
                        normalized = [f"{p:.6f}" for point in rel_points.tolist() for p in point]
                        current_canvas_label_lines.append(f"{class_id} " + " ".join(normalized))
                zeros[:] = 0
        if is_out_canvas:
            continue
        imwrite(f"{save_image_dir}/canvas_{canvas_id}.png", current_canvas)
        open(f"{save_label_dir}/canvas_{canvas_id}.txt", 'w', encoding='utf-8').write("\n".join(current_canvas_label_lines))

    ds2 = YOLODataset(save_dir,
                      image_dirname="images",
                      label_dirname="labels",
                      with_label=True,
                      task="seg",
                      categories=cate,
                      read_image=False)
    for idx, i in enumerate(ds2):
        im_path, la_path = i
        im_dir_path = os.path.dirname(im_path)
        la_dir_path = os.path.dirname(la_path)
        new_im_dir = os.path.join(im_dir_path, f"{basename}_" + os.path.basename(im_path))
        new_la_dir = os.path.join(la_dir_path, f"{basename}_" + os.path.basename(la_path))
        os.rename(im_path, new_im_dir)
        os.rename(la_path, new_la_dir)
        ds2[idx] = Path(new_im_dir), Path(new_la_dir)

    VisualizeYOLODataset(ds2, save_dir=os.path.join(save_dir, "vis"))()


if __name__ == "__main__":
    src_top_dir = r"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\cooked"
    dirlist = [
        # "fod_aidian", "fod_baineng2d_01", "fod_baineng2d_02",
        "fod_baineng3d"]
    for d in dirlist:
        data_dir = rf"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\cooked\{d}"
        save_dir = rf"E:\python_ai_dataset\foreign-object-detect\NEW\V4.5\crop\{d}"
        do(data_dir, save_dir)
