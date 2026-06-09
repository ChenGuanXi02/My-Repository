import os
from pathlib import Path

# ================= 配置区域 =================
# 输入你的路径
image_root = '/home/chenzeyu/TrainAndTest/Inpainting32K/Imgs_train/'
gt_root = '/home/chenzeyu/TrainAndTest/Inpainting32K/GT_train/'

# 定义需要扫描的扩展名（不区分大小写）
img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
gt_extensions = {'.png', '.jpg', '.tif', '.tiff'}  # 你的GT主要是png，但也包含以此防万一


# ===========================================

def get_file_stems(directory, extensions):
    """获取目录下所有符合后缀的文件的文件名（不含后缀）"""
    stems = set()
    files_map = {}  # 记录 stem -> full_filename 的映射，方便后续打印

    if not os.path.exists(directory):
        print(f"❌ 错误: 找不到目录 {directory}")
        return stems, files_map

    print(f"正在扫描: {directory} ...")
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            # 获取小写后缀
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                # 获取不含后缀的文件名 (stem)
                stem = os.path.splitext(file)[0]
                stems.add(stem)
                files_map[stem] = file
                count += 1

    print(f"  -> 找到 {count} 个文件")
    return stems, files_map


def main():
    print("=" * 50)
    print("开始检查数据集匹配情况...")
    print("=" * 50)

    # 1. 获取文件名列表
    img_stems, img_map = get_file_stems(image_root, img_extensions)
    gt_stems, gt_map = get_file_stems(gt_root, gt_extensions)

    print("-" * 30)

    # 2. 计算交集和差集
    common = img_stems & gt_stems  # 交集：两者都有
    only_in_img = img_stems - gt_stems  # 只有图片没有GT
    only_in_gt = gt_stems - img_stems  # 只有GT没有图片

    # 3. 输出结果
    print(f"✅ 匹配成功 (Image和GT都有): {len(common)} 对")

    if len(only_in_img) > 0:
        print(f"⚠️  Image中有，但GT中没有: {len(only_in_img)} 个")
        print("   (可能导致训练时找不到标签报错，或者被代码过滤掉)")
        print("   示例文件: ", list(only_in_img)[:5])
    else:
        print("OK: Image中的文件都在GT里找到了。")

    if len(only_in_gt) > 0:
        print(f"⚠️  GT中有，但Image中没有: {len(only_in_gt)} 个")
        print("   (这意味着有些标签是多余的)")
        print("   示例文件: ", list(only_in_gt)[:5])
    else:
        print("OK: GT中的文件都在Image里找到了。")

    print("=" * 50)

    # 4. 深度分析：为什么还是 669 batch？
    # 669 * 32 = 21408
    print("【分析建议】")
    if len(common) > 21500:
        print(f"当前物理匹配数量 ({len(common)}) 远大于 训练加载数量 (约21408)。")
        print("可能原因：")
        print("1. 代码中的 glob 读取方式没有生效（例如 .tif 后缀大小写不匹配）。")
        print("2. 代码中存在 assert len(images) == len(gts)，如果数量不一致，可能程序默默退出了或者只加载了部分。")
        print("3. DataLoader 的 drop_last=True 丢弃了最后不足一个 batch 的数据。")
    else:
        print(f"当前物理匹配数量 ({len(common)}) 与 训练加载数量 ({21408}) 相近。")
        print("说明确实只有这么多文件能匹配上，可能 .tif 文件确实没有对应的 .png 标签。")


if __name__ == "__main__":
    main()