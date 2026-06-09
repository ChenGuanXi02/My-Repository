import torch
import torch.nn as nn
import torch.nn.functional as F
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Conv_Block2(nn.Module):
    def __init__(self, channels):
        super(Conv_Block2, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels*4, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels*2, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(channels*2)
        # 修改输出通道数
        self.conv3 = DepthwiseSeparableConv(channels*2, channels*2, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(channels*2)

    def forward(self, x1, x2, x3, x4):
        fuse = torch.cat((x1, x2, x3, x4), dim=1)
        fuse = F.relu(self.bn1(self.conv1(fuse)))
        fuse = F.relu(self.bn2(self.conv2(fuse)))
        fuse = self.bn3(self.conv3(fuse))
        return fuse
class Decoder(nn.Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.num_classes = class_num
        # 用1×1卷积将通道统一为128（可根据需求调整）,目前用于融合频域噪声分支
        self.conv_c1 = nn.Conv2d(64, 128, kernel_size=1)  # input1: 64→128
        self.conv_c2 = nn.Conv2d(128, 128, kernel_size=1)  # input2: 128→128
        self.conv_c3 = nn.Conv2d(320, 128, kernel_size=1)  # input3: 320→128
        self.conv_c4 = nn.Conv2d(512, 128, kernel_size=1)  # input4: 512→128
        self.conv_block = Conv_Block2(channels=128)

    def forward(self, inputs):
        a1, a2, a3, a4 = inputs
        # c1,c2,c3,c4 是pvt输出的f1,f2,f3,f4
        # a1,a2,a3,a4 是是频率分支的四个输出
        # stage2:频率噪声分支
        # 上采样到88×88（假设输入特征为input1~input4）
        a2_upsampled = F.interpolate(a2, size=(88, 88), mode='bilinear', align_corners=False)
        a3_upsampled = F.interpolate(a3, size=(88, 88), mode='bilinear', align_corners=False)
        a4_upsampled = F.interpolate(a4, size=(88, 88), mode='bilinear', align_corners=False)
        # 应用卷积
        a1_aligned = self.conv_c1(a1)  # 假设input1已上采样到88×88
        a2_aligned = self.conv_c2(a2_upsampled)
        a3_aligned = self.conv_c3(a3_upsampled)
        a4_aligned = self.conv_c4(a4_upsampled)
        # 融合（拼接后通道数为128×4=512）
        xf = self.conv_block(a1_aligned, a2_aligned, a3_aligned, a4_aligned)
        # print('xf是图像分支的融合特征结果')
        # print(xf.shape)
        return xf

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # (B,C,H,W)
    c1 = torch.randn(1, 64, 88, 88).to(device)
    c2 = torch.randn(1, 128, 44, 44).to(device)
    c3 = torch.randn(1, 320, 22, 22).to(device)
    c4 = torch.randn(1, 512, 11, 11).to(device)

    print('c1234')
    print(c1.shape)
    print(c2.shape)
    print(c3.shape)
    print(c4.shape)

    Model = Decoder(dims=[64, 128, 320, 512], dim=256).to(device)
    out1= Model([c1, c2, c3, c4])
    print('xf')
    print(out1.shape)