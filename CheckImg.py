from PIL import Image
import os

def check_images(folder):
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(('.jpg', '.png', '.jpeg', '.tif')):
                path = os.path.join(root, name)
                try:
                    with Image.open(path) as img:
                        img.verify()  # 检查完整性
                except Exception as e:
                    print(f"损坏图像: {path}, 原因: {e}")

# 用法
check_images("/home/chenzeyu/pycharmprojects/CamoDiffusion/dataset/CASIAv2/TestDataset/NIST16/Imgs/")