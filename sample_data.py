import os
import shutil
import random
import argparse
from pathlib import Path

# Supported image formats.
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

def sample_images(src_dir, dst_dir, sample_ratio=0.3, seed=42):
    """
    Sample images from each src_dir subdirectory into dst_dir.

    :param src_dir: Source directory with class subdirectories.
    :param dst_dir: Output directory.
    :param sample_ratio: Sampling ratio in (0.0, 1.0].
    :param seed: Random seed.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"源目录不存在: {src_path}")

    # Set the random seed.
    random.seed(seed)

    # Iterate over source subdirectories.
    for item in src_path.iterdir():
        if item.is_dir():
            # Read the subdirectory name.
            class_name = item.name
            if class_name == "good":
                continue
            class_dst = dst_path / class_name  # Destination subdirectory.

            # Create the destination subdirectory.
            class_dst.mkdir(parents=True, exist_ok=True)

            # Find supported image files.
            image_files = [f for f in item.iterdir() 
                         if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

            if not image_files:
                print(f"警告: 子文件夹 '{class_name}' 中没有找到图片文件。")
                continue

            # Compute the sample count.
            num_samples = max(1, int(len(image_files) * sample_ratio))  # Sample at least one image.
            sampled_files = random.sample(image_files, num_samples)

            # Copy images.
            for img_file in sampled_files:
                shutil.copy2(img_file, class_dst / img_file.name)

            print(f"已从 '{class_name}' 采样 {num_samples}/{len(image_files)} 张图片到 '{class_dst}'")

    print(f"\n✅ 所有图片采样完成，输出路径: {dst_path}")

def main():
    parser = argparse.ArgumentParser(description="从 test 文件夹的子文件夹中按比例采样图片并复制到指定路径")
    parser.add_argument('--src', type=str, default='/data/luoqianhao/datasets/anomaly_datesets/mvtec_2d/gangguan_temp_cls/test',
                        help='源目录路径 (默认: ./test)')
    parser.add_argument('--dst', type=str, default="/data/luoqianhao/datasets/anomaly_datesets/mvtec_2d/gangguan_temp_cls/train",
                        help='目标输出路径 (必需)')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='采样比例 (0.0 ~ 1.0, 默认: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError("采样比例必须在 (0, 1] 范围内")

    sample_images(args.src, args.dst, args.ratio, args.seed)

if __name__ == '__main__':
    main()
