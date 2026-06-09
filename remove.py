import os

# ================= 配置区域 =================
# 目标文件夹路径
target_dir = '/home/chenzeyu/TrainAndTest/260226DiffForensics/FRFakeAndCasiav2Fake/GT'

# 文件名前缀匹配
prefix = 'Tp'

# 【安全开关】True = 只打印不删除；False = 真的删除
dry_run = False


# ===========================================

def main():
    if not os.path.exists(target_dir):
        print(f"❌ 错误: 找不到目录 {target_dir}")
        return

    print(f"正在扫描目录: {target_dir}")
    print(f"筛选条件: 文件名以 '{prefix}' 开头")
    if dry_run:
        print("⚠️  当前为模拟模式 (dry_run=True)，不会真正删除文件。")
    else:
        print("🚨 当前为执行模式，文件将被永久删除！")

    print("-" * 40)

    count = 0
    deleted_count = 0

    # 遍历目录
    for filename in os.listdir(target_dir):
        # 检查前缀
        if filename.startswith(prefix):
            count += 1
            file_path = os.path.join(target_dir, filename)

            if dry_run:
                # 模拟模式：只打印
                print(f"[模拟删除] {filename}")
            else:
                # 执行模式：真删除
                try:
                    os.remove(file_path)
                    print(f"✅ 已删除: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 删除失败 {filename}: {e}")

    print("-" * 40)
    if dry_run:
        print(f"扫描结束。共发现 {count} 个以 '{prefix}' 开头的文件。")
        print("💡 确认无误后，请将代码中的 dry_run = True 改为 False 再运行一次。")
    else:
        print(f"操作完成。共删除了 {deleted_count} 个文件。")


if __name__ == "__main__":
    main()