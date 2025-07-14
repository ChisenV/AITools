import cv2
import os
import argparse
import natsort

from AITools import IMG_FORMATS


def create_video(image_folder, output_video, fps=30, img_size=None):
    """
    将图像文件夹中的图片合成为视频

    参数:
        image_folder: 存放图像的文件夹路径
        output_video: 输出视频文件路径（需包含扩展名，如 .mp4）
        fps: 视频帧率（默认30）
        img_size: 视频尺寸（宽度,高度），为None则使用第一张图像的尺寸
    """
    # 获取所有图像文件并自然排序
    images = [img for img in os.listdir(image_folder)
              if img.endswith(tuple(IMG_FORMATS))]
    images = natsort.natsorted(images)  # 自然排序（1,2,10而不是1,10,2）

    if not images:
        raise FileNotFoundError(f"No images found in {image_folder}")

    # 读取第一张图像确定尺寸
    first_img_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_img_path)
    if frame is None:
        raise IOError(f"Failed to read image: {first_img_path}")

    # 设置视频尺寸
    height, width = frame.shape[:2]
    if img_size:
        width, height = img_size

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 处理每张图像
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipped corrupted image {img_name}")
            continue

        # 调整图像尺寸
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))

        video.write(img)
        print(f"Processed: {img_name}")

    video.release()
    print(f"Video saved to {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to video')
    parser.add_argument('--input', type=str, required=True, help='Input image folder path')
    parser.add_argument('--output', type=str, required=True, help='Output video file path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--width', type=int, default=None, help='Output video width')
    parser.add_argument('--height', type=int, default=None, help='Output video height')

    args = parser.parse_args()
    img_size = (args.width, args.height) if args.width and args.height else None

    create_video(
        image_folder=args.input,
        output_video=args.output,
        fps=args.fps,
        img_size=img_size
    )
