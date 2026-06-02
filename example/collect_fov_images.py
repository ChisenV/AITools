import os
import shutil
import re
import argparse
import logging
import time
from logging.handlers import RotatingFileHandler

# 定义目标文件夹名称
DAT_DIR = "dat"          # 存放 .dat 文件
LEFT_DIR = "Side"        # 存放 _0.* 文件（左图）
RIGHT_DIR = "Top"        # 存放 _1.* 文件（右图）

# 支持的图片扩展名（用于 side_thumb 查找）
SUPPORTED_IMG_EXTS = {
    '.jpg',
    # '.jpeg',
    '.bmp',
    # '.png',
    # '.tiff',
    # '.tif',
    # '.webp'
}

def setup_dirs(dst_dir):
    """创建三个目标文件夹（如果不存在）"""
    for dir_name in [DAT_DIR, LEFT_DIR, RIGHT_DIR]:
        dir_path = os.path.join(dst_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)

def extract_base_name(filename):
    """
    提取文件名的基础名（去掉扩展名及末尾 _0 / _1）
    例如：fov_10_0.dat -> fov_10
          fov_10_0.jpg -> fov_10
          fov_10_1.jpg -> fov_10
    """
    base = os.path.splitext(filename)[0]
    base = re.sub(r'_[01]$', '', base)
    return base

def find_side_thumb_file(job_dir_path):
    """
    在 job_dir 下查找 side_thumb 图片文件（任意支持的扩展名）
    返回 (文件完整路径, 扩展名) 或 (None, None)
    """
    if not os.path.isdir(job_dir_path):
        return None, None
    for fname in os.listdir(job_dir_path):
        # 检查文件名是否以 side_thumb 开头（忽略大小写）
        if fname.lower().startswith('side_thumb.'):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_IMG_EXTS:
                return os.path.join(job_dir_path, fname), ext
    return None, None

def classify_and_move(src_dir, dst_dir, jobId=0, operation="copy", start_idx=0):
    """
    将同一个基础名的三个文件分配相同的 UID，然后拷贝/移动
    返回：下一个可用的 idx（已使用 start_idx 后的最大值+1）
    """
    if not os.path.exists(src_dir):
        logging.info(f"src_dir:{src_dir} not exist")
        return start_idx

    pattern_dat = re.compile(r"\.dat$", re.IGNORECASE)
    pattern_left = re.compile(r"_0\.[a-zA-Z0-9]+$", re.IGNORECASE)
    pattern_right = re.compile(r"_1\.[a-zA-Z0-9]+$", re.IGNORECASE)

    # 第一步：收集所有需要处理的文件及其类型
    files_info = []  # 每个元素: (full_path, filename, base_name, category)
    base_set = set()

    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)
        if not os.path.isfile(file_path):
            continue
        if "FOD" in filename:
            continue

        if pattern_dat.search(filename):
            category = DAT_DIR
        elif pattern_left.search(filename):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_IMG_EXTS:
                continue   # 不支持的图片格式，跳过
            category = LEFT_DIR
        elif pattern_right.search(filename):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_IMG_EXTS:
                continue
            category = RIGHT_DIR
        else:
            continue

        base = extract_base_name(filename)
        files_info.append((file_path, filename, base, category))
        base_set.add(base)

    # 第二步：为基础名分配全局唯一的 idx（从 start_idx 开始，按任意顺序）
    base_list = sorted(base_set)  # 排序使结果可预测
    base_to_idx = {base: start_idx + i for i, base in enumerate(base_list)}
    next_idx = start_idx + len(base_list)

    # 第三步：执行拷贝/移动
    for file_path, filename, base, category in files_info:
        idx = base_to_idx[base]
        output_dir = os.path.join(dst_dir, category)
        dest_path = os.path.join(output_dir, f"UID{idx}_JID{jobId}_{filename}")
        try:
            if operation == "copy":
                shutil.copy(file_path, dest_path)
            elif operation == "move":
                shutil.move(file_path, dest_path)
            logging.info(f"{operation}: {file_path} -> {os.path.basename(dest_path)}")
        except Exception as e:
            logging.error(f"{operation} fail {filename}: {e}")

    return next_idx

def collect_fov_images(base_dirs, dst_dir, deal_type="fov", start_idx=0, start_job_idx=0):
    """
    遍历多个源根目录，处理每个根目录下所有子目录中的 fov 文件夹。
    base_dirs : list of str, 每个元素为一个根目录路径
    """
    setup_dirs(dst_dir)
    deal_type = deal_type.lower()

    # 配置日志：同时输出到控制台和 dst_dir/process.log
    log_file = os.path.join(dst_dir, "process.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件轮转 handler：每个文件最大 2MB，保留 99 个历史文件
    file_handler = RotatingFileHandler(
        log_file, maxBytes=2*1024*1024, backupCount=99, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    idx = start_idx
    global_job_id = start_job_idx  # 全局 JID 计数器

    start_time = time.time()
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            logging.info(f"base_dir 不存在，跳过: {base_dir}")
            continue
        # 获取该根目录下的所有子文件夹
        try:
            job_dir_list = os.listdir(base_dir)
        except NotADirectoryError:
            logging.info(f"跳过非目录: {base_dir}")
            continue

        for job_dir in job_dir_list:
            fov_copied = False
            fov_dir = os.path.join(base_dir, job_dir, "fov")
            is_exist = os.path.exists(fov_dir)
            fov_len = len(os.listdir(fov_dir)) if is_exist else 0

            if deal_type == "fov" or deal_type == "both":
                if not is_exist or fov_len == 0:
                    continue
                src_path = fov_dir
                idx = classify_and_move(src_path, dst_dir, jobId=global_job_id, operation="copy", start_idx=idx)
                fov_copied = True

            if deal_type == "thumb" or deal_type == "both":
                # 动态查找 side_thumb 图片（支持多种格式）
                job_dir_path = os.path.join(base_dir, job_dir)
                thumb_path, thumb_ext = find_side_thumb_file(job_dir_path)
                # 条件判断：只有当 fov 存在且已处理时才处理
                if (not is_exist or fov_len == 0) and not fov_copied:
                    # 没有 fov 数据，不处理 thumb
                    continue
                if thumb_path and os.path.exists(thumb_path):
                    # 目标文件名保留原始扩展名
                    dest_thumb = os.path.join(dst_dir, f"JID{global_job_id}_thumb{thumb_ext}")
                    shutil.copy(thumb_path, dest_thumb)
                    logging.info(f"copy thumb: {thumb_path} -> {os.path.basename(dest_thumb)}")
                elif not fov_copied and thumb_path is None:
                    # 既没有 fov 也没有 thumb，跳过
                    continue

            logging.info(f"处理完成: job_dir={job_dir} (JID={global_job_id}, fov len={fov_len})")
            global_job_id += 1

    # 记录结束时间并计算耗时
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60.0

    logging.info(f"最后使用的 UID 索引: {idx}")
    logging.info(f"总共处理了 {global_job_id - start_job_idx} 个任务（JID 范围 {start_job_idx}~{global_job_id-1}）")
    logging.info(f"总耗时: {elapsed_minutes:.2f} 分钟 ({elapsed_seconds:.2f} 秒)")


if __name__ == "__main__":
    """
    # 处理两个源目录，目标目录为默认，起始 UID 从 100 开始，JID 从 0 开始
    python collect_fov_images.py --base_dirs E:\Jobs1 E:\Jobs2 --dst_dir D:\output --start_idx 100
    
    # 使用短选项
    python collect_fov_images.py -b E:\Jobs1 E:\Jobs2 -d D:\output -s 100
    
    # 只处理一个源目录（仍需以列表形式给出）
    python collect_fov_images.py -b E:\Jobs
    """
    parser = argparse.ArgumentParser(description="将多个源目录下的 FOV 数据分类拷贝/移动到目标目录")
    parser.add_argument("--base_dirs", "-b", nargs='+', required=True,
                        help="一个或多个原始数据根目录（每个目录下应包含多个子任务文件夹，每个子任务文件夹下有 fov 子目录）")
    parser.add_argument("--dst_dir", "-d", type=str, default=r"E:\ds\FOV",
                        help="目标根目录（将在其下创建 dat / left / right 三个子文件夹）")
    parser.add_argument("--start_idx", "-s", type=int, default=0,
                        help="输出文件 UID 的起始编号")
    parser.add_argument("--start_job_idx", "-j", type=int, default=0,
                        help="输出文件 JID 的起始编号")
    parser.add_argument("--type", "-t", type=str, default="fov", choices=["fov", "thumb", "both"],
                        help="处理 fov / thumb")
    args = parser.parse_args()

    collect_fov_images(args.base_dirs, args.dst_dir, args.type, args.start_idx, args.start_job_idx)
