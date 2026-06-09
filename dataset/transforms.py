import torch
import numpy as np
from PIL import Image
import io
from torchvision import transforms
import random


class GaussianNoise:
    """添加高斯噪声（带随机种子控制）"""

    def __init__(self, sigma=0.1, seed=None):
        self.sigma = sigma  # 噪声标准差
        self.seed = seed  # 随机种子，确保可复现

    def __call__(self, img):
        """
        img: PIL图像
        返回: 带噪声的PIL图像
        """
        # 设置随机种子
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # 转为Tensor进行噪声处理
        img_tensor = transforms.ToTensor()(img)
        noise = torch.randn_like(img_tensor) * self.sigma
        img_tensor = img_tensor + noise

        # 裁剪边界
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # 转回PIL图像
        return transforms.ToPILImage()(img_tensor)


class JPEGCompression:
    """模拟JPEG压缩（通过调整质量参数引入失真）"""

    def __init__(self, quality=50):
        self.quality = quality  # JPEG质量（1-100，越低压缩越强）

    def __call__(self, img):
        """
        img: PIL图像
        """
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)  # 转为PIL图像

        # 保存为JPEG再读取，模拟压缩
        # 添加subsampling=0参数确保跨平台一致性
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality, subsampling=0)
        buffer.seek(0)
        img = Image.open(buffer).convert('RGB')  # 若为灰度图则用'L'
        return img


# 组合变换示例
def get_distortion_transform(distortion_type, **kwargs):
    """根据类型获取对应变换"""
    if distortion_type == 'gaussian_blur':
        return transforms.GaussianBlur(
            kernel_size=kwargs.get('kernel_size', 5),
            sigma=kwargs.get('sigma', (0.1, 2.0))
        )
    elif distortion_type == 'gaussian_noise':
        return GaussianNoise(sigma=kwargs.get('sigma', 0.1))
    elif distortion_type == 'jpeg_compression':
        return JPEGCompression(quality=kwargs.get('quality', 50))
    else:
        raise ValueError(f"不支持的干扰类型: {distortion_type}")