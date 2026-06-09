import os
import shutil


def gather_images():
    # 定义所有源文件夹路径
    source_dirs = [
        "/home/chenzeyu/TrainAndTest/Inpainting32K/GT_train",
        "/home/chenzeyu/TrainAndTest/Inpainting32K/TM-Based/GT_test",
        "/home/chenzeyu/TrainAndTest/Inpainting32K/CNN-Based/GT_test",
        "/home/chenzeyu/TrainAndTest/Inpainting32K/GAN-Based/GT_test",
        "/home/chenzeyu/TrainAndTest/Inpainting32K/DM-Based/GT_test"
    ]

    # 定义目标文件夹路径
    target_dir = "/home/chenzeyu/TrainAndTest/Inpainting32K/GT"

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"📁 创建了目标文件夹: {target_dir}")

    total_processed = 0
    duplicate_count = 0

    print("🚀 开始汇总图片...\n" + "=" * 50)

    for src_dir in source_dirs:
        # 检查源文件夹是否存在
        if not os.path.exists(src_dir):
            print(f"⚠️ 警告: 找不到文件夹 {src_dir}，已跳过。")
            continue

        print(f"📂 正在处理: {src_dir}")

        # 获取该文件夹下的所有文件
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        count_in_dir = 0

        for file_name in files:
            src_file_path = os.path.join(src_dir, file_name)
            tgt_file_path = os.path.join(target_dir, file_name)

            # 检查目标文件夹是否已经存在同名文件（防止覆盖）
            if os.path.exists(tgt_file_path):
                # 如果你想直接覆盖，可以把下面两行注释掉，并取消 duplicate_count 的累加
                # print(f"  跳过: {file_name} 已存在。")
                duplicate_count += 1
                continue

            try:
                # =======================================================
                # 【默认使用复制】安全起见，保留原文件，复制到新文件夹
                shutil.copy2(src_file_path, tgt_file_path)

                # 【如果你想剪切/移动（省空间）】请注释掉上面那行，取消注释下面这行：
                # shutil.move(src_file_path, tgt_file_path)
                # =======================================================

                count_in_dir += 1
                total_processed += 1

                # 每处理 5000 张打印一次进度，防止终端刷屏卡顿
                if total_processed % 5000 == 0:
                    print(f"  ...已汇总 {total_processed} 张图片...")

            except Exception as e:
                print(f"❌ 处理文件 {file_name} 时发生错误: {e}")

        print(f"  ✅ 完成! 从该文件夹成功转移了 {count_in_dir} 张图片。\n")

    print("=" * 50)
    print("🎉 全部处理完毕！")
    print(f"总计成功汇总图片: {total_processed} 张")
    if duplicate_count > 0:
        print(f"发现并跳过了 {duplicate_count} 个重名文件。")
    print(f"所有图片现已存放在: {target_dir}")


if __name__ == '__main__':
    gather_images()