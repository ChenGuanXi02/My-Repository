import os
from PIL import Image


def align_gt_to_img(img_dir, gt_dir):
    # 常见的图像扩展名
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # 获取所有有效文件
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(valid_exts)]

    # 构建 GT 的无后缀映射字典，防止原图是 .jpg 而 GT 是 .png 导致匹配失败
    gt_dict = {os.path.splitext(f)[0]: f for f in gt_files}

    # 兼容不同版本的 Pillow 库
    try:
        resample_method = Image.Resampling.NEAREST
    except AttributeError:
        resample_method = Image.NEAREST

    mismatched_count = 0
    processed_count = 0
    missing_gt_count = 0

    print("=" * 50)
    print(f"开始检查并修复数据对齐问题...")
    print(f"原图目录: {img_dir}")
    print(f"标签目录: {gt_dir}")
    print("=" * 50)

    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]

        if base_name in gt_dict:
            gt_name = gt_dict[base_name]
            img_path = os.path.join(img_dir, img_name)
            gt_path = os.path.join(gt_dir, gt_name)

            try:
                # 仅读取头部信息，获取原图尺寸 (width, height)
                with Image.open(img_path) as img:
                    img_size = img.size

                    # 打开 GT 图片进行对比
                with Image.open(gt_path) as gt:
                    gt_size = gt.size

                    # 如果尺寸不一致，触发修复机制
                    if img_size != gt_size:
                        mismatched_count += 1

                        # 使用最近邻插值强制 Resize 为原图尺寸
                        fixed_gt = gt.resize(img_size, resample=resample_method)

                        # 直接覆盖保存原 GT 文件
                        fixed_gt.save(gt_path)
                        # 如果不需要看刷屏日志，可以把下面这行注释掉
                        print(f"已修复: {gt_name} (从 {gt_size} 调整为 {img_size})")

            except Exception as e:
                print(f"处理文件 [{img_name}] 时发生错误: {e}")
        else:
            missing_gt_count += 1

        processed_count += 1
        if processed_count % 5000 == 0:
            print(f"进度: 已扫描 {processed_count} 张图片...")

    print("=" * 50)
    print("处理完成！数据汇总：")
    print(f"总计扫描原图: {len(img_files)} 张")
    print(f"尺寸不匹配并成功修复: {mismatched_count} 张")
    if missing_gt_count > 0:
        print(f"⚠️ 警告: 发现 {missing_gt_count} 张原图没有对应的 GT 标签")
    print("=" * 50)


if __name__ == '__main__':
    # 替换为你实际的文件夹路径
    IMGS_DIR = "/home/chenzeyu/TrainAndTest/Inpainting32K/CNN-Based/Imgs_test"
    GTS_DIR = "/home/chenzeyu/TrainAndTest/Inpainting32K/CNN-Based/GT_test"

    # 友情提示：因为是直接覆盖保存，建议运行前在终端执行一下 cp -r 命令备份 GT 文件夹
    align_gt_to_img(IMGS_DIR, GTS_DIR)