import os
import argparse
from pathlib import Path


def check_correspondence(image_root, gt_root, extensions=('png', 'jpg', 'jpeg', 'bmp', 'tif')):
    """
    检查两个文件夹中的文件是否基于文件名前缀一一对应

    参数:
        image_root: 图像文件夹路径
        gt_root: GT 文件夹路径
        extensions: 要检查的文件扩展名列表
    """
    # 获取图像文件夹中所有符合条件的文件（不含扩展名）
    image_files = set()
    for ext in extensions:
        image_files.update(
            f.stem for f in Path(image_root).glob(f'*.{ext.lower()}') if f.is_file()
        )
        image_files.update(
            f.stem for f in Path(image_root).glob(f'*.{ext.upper()}') if f.is_file()
        )

    # 获取 GT 文件夹中所有符合条件的文件（不含扩展名）
    gt_files = set()
    for ext in extensions:
        gt_files.update(
            f.stem for f in Path(gt_root).glob(f'*.{ext.lower()}') if f.is_file()
        )
        gt_files.update(
            f.stem for f in Path(gt_root).glob(f'*.{ext.upper()}') if f.is_file()
        )

    # 计算差异
    only_in_images = image_files - gt_files
    only_in_gt = gt_files - image_files

    # 输出结果
    print(f"图像文件夹: {image_root}")
    print(f"GT 文件夹: {gt_root}")
    print(f"图像文件数量: {len(image_files)}")
    print(f"GT 文件数量: {len(gt_files)}")

    if only_in_images or only_in_gt:
        print("\n❌ 发现不匹配的文件:")
        if only_in_images:
            print(f"  仅存在于图像文件夹中的文件 ({len(only_in_images)}):")
            for f in sorted(only_in_images):
                print(f"    {f}")
        if only_in_gt:
            print(f"  仅存在于 GT 文件夹中的文件 ({len(only_in_gt)}):")
            for f in sorted(only_in_gt):
                print(f"    {f}")
        return False
    else:
        print("\n✅ 所有文件完全对应")
        return True


def main():
    parser = argparse.ArgumentParser(description='检查两个文件夹中的文件是否一一对应')
    parser.add_argument('--image_root', required=True, help='图像文件夹路径')
    parser.add_argument('--gt_root', required=True, help='GT 文件夹路径')
    parser.add_argument('--extensions', nargs='+', default=['png', 'jpg', 'jpeg', 'bmp', 'tif'],
                        help='要检查的文件扩展名列表，用空格分隔')

    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.image_root):
        print(f"错误: 图像文件夹不存在: {args.image_root}")
        return

    if not os.path.exists(args.gt_root):
        print(f"错误: GT 文件夹不存在: {args.gt_root}")
        return

    # 执行检查
    check_correspondence(args.image_root, args.gt_root, args.extensions)


if __name__ == "__main__":
    main()