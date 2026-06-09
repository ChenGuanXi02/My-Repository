import os
from PIL import Image


def check_image_sizes(img_dir, gt_dir, output_txt):
    """
    检查两个文件夹中同名（忽略后缀）图片的尺寸是否对应。
    """
    mismatched_files = []

    # 允许的图片后缀
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    # 获取并过滤两个文件夹中的文件
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(valid_exts)]

    # 为了防止后缀不同导致匹配不上，我们建立一个 {文件名(无后缀): 完整文件名} 的字典
    gt_dict = {os.path.splitext(f)[0]: f for f in gt_files}

    print(f"正在扫描 {img_dir} 和 {gt_dir} ...")

    # 开始比对
    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]

        # 如果在标签文件夹中找到了同名的文件
        if base_name in gt_dict:
            gt_name = gt_dict[base_name]

            img_path = os.path.join(img_dir, img_name)
            gt_path = os.path.join(gt_dir, gt_name)

            try:
                # 打开图片获取尺寸 (Image.open 是懒加载，仅读取头部信息，速度很快)
                with Image.open(img_path) as img, Image.open(gt_path) as gt:
                    if img.size != gt.size:
                        # 记录不匹配的信息：原图名, 原图尺寸, 标签图名, 标签图尺寸
                        info = f"{img_name} {img.size}  !=  {gt_name} {gt.size}"
                        mismatched_files.append(info)
            except Exception as e:
                print(f"读取文件时出错 [{img_name} 或 {gt_name}]: {e}")
        else:
            # 如果你想知道哪些原图完全没有对应的标签，也可以在这里打印
            # print(f"警告: 找不到 {img_name} 对应的标签文件")
            pass

    # 将结果写入 txt 文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"总计不匹配数量: {len(mismatched_files)}\n")
        f.write("-" * 50 + "\n")
        for line in mismatched_files:
            f.write(line + '\n')

    # 在终端输出最终结果
    print("=" * 40)
    print(f"检查完毕！")
    print(f"尺寸不匹配的图片数量: {len(mismatched_files)}")
    print(f"详细名单已保存至: {os.path.abspath(output_txt)}")
    print("=" * 40)


if __name__ == '__main__':
    # ------------------ 请修改这里的路径 ------------------
    # 替换为你实际的原图文件夹路径和标签文件夹路径
    IMAGE_FOLDER = "/home/chenzeyu/TrainAndTest/Inpainting32K/Imgs_train"
    GT_FOLDER = "/home/chenzeyu/TrainAndTest/Inpainting32K/GT_train"
    OUTPUT_FILE = "mismatched_sizes.txt"
    # -----------------------------------------------------

    check_image_sizes(IMAGE_FOLDER, GT_FOLDER, OUTPUT_FILE)