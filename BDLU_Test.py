import torch
import torch.nn as nn
import torch.nn.functional as F
# 实验，边界引导分支
# 子模块：用于局部查找相似特征
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    # print(H,W)
    xgrid, ygrid = coords.split([1, 1], dim=-1)

    # print('xgrid',xgrid[0,0], xgrid[-1,-1])
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    # print('xgrid', xgrid[0,0], xgrid[-1,-1])
    grid = torch.cat([xgrid, ygrid], dim=-1)
    # print('grid', grid.shape) #[b*h*w, 9, 9, 2]
    # print('img', img.shape) #[b*h*w, 1, h, w]
    # exit()
    img = F.grid_sample(img.contiguous(), grid.contiguous(), align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    # print('img', img.shape)
    return img

# 功能：构建多尺度相关性金字塔，并根据输入坐标在金字塔各层上采样邻域相关性特征。
# 核心作用：用于捕捉特征图中像素间的长程依赖或局部相似性（如光流估计、特征匹配）。
class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            # print('corr', corr.shape)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            # print('coords', coords.shape)
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            # print('delta', delta)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl, mode='bilinear', mask=False)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

# 基础卷积块，包含卷积、批量归一化和 ReLU 激活。
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 融合多尺度特征（四个输入），通过多层卷积逐步增强特征表达。
class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*4, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels*2, 5, 1, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)
        self.conv3 = nn.Conv2d(channels*2, channels*4, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels*4)

    def forward(self, x1, x2, x3, x4):
        fuse = torch.cat((x1, x2, x3, x4), dim=1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

class BDLU(nn.Module):
    def __init__(self, in_channels=(512, 320, 128, 64), mid_channel=128, edge_channel=16,
                 patch_size=8, image_size=288):
        super(BDLU, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size  # e.g., 288
        self.edge_channel = edge_channel

        # 侧卷积：从 backbone 不同阶段提取中间特征（降维到 mid_channel）
        self.side_conv1 = nn.Conv2d(in_channels[0], mid_channel, 3, 1, 1)  # 输入512→128
        self.side_conv2 = nn.Conv2d(in_channels[1], mid_channel, 3, 1, 1)  # 输入320→128
        self.side_conv3 = nn.Conv2d(in_channels[2], mid_channel, 3, 1, 1)  # 输入128→128
        self.side_conv4 = nn.Conv2d(in_channels[3], mid_channel, 3, 1, 1)  # 输入64→128
        # 特征融合块（调用前面的 Conv_Block）
        self.conv_block = Conv_Block(mid_channel)  # 输入4*128→输出4*128

        self.decoder_embed_edge = nn.Linear(mid_channel * 4, patch_size**2 * edge_channel, bias=True)
        self.conv_edge = nn.Sequential(
            nn.Conv2d(edge_channel, edge_channel, kernel_size=1),
            nn.BatchNorm2d(edge_channel),
            nn.GELU(),
            nn.Conv2d(edge_channel, edge_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(edge_channel),
            nn.GELU(),
        )
        self.decoder_pred_edge = nn.Sequential(
            nn.Conv2d(edge_channel, edge_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(edge_channel),
            nn.GELU(),
            nn.Conv2d(edge_channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.edge_lookup_downsample = nn.Sequential(
            nn.Conv2d(edge_channel, edge_channel * 4, 3, 2, 1),
            nn.BatchNorm2d(edge_channel * 4),
            nn.GELU(),
            nn.Conv2d(edge_channel * 4, edge_channel * 4, 3, 2, 1),
            nn.BatchNorm2d(edge_channel * 4),
            nn.GELU(),
            nn.Conv2d(edge_channel * 4, edge_channel * 8, 3, 2, 1),
            nn.BatchNorm2d(edge_channel * 8),
            nn.GELU(),
            nn.Conv2d(edge_channel * 8, edge_channel * 8, 1),
            nn.BatchNorm2d(edge_channel * 8),
            nn.GELU(),
        )

        # self.final_lookup_downsample = nn.Sequential(
        #     nn.Conv2d(4 * 5 * 5, 128, 3, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.GELU(),
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.GELU(),
        #     nn.Conv2d(256, 512, 3, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.GELU(),
        #     nn.Conv2d(512, 512, 1)
        # )
        self.final_lookup_upsample = nn.Sequential(
            # 从 [1, 100, 22, 22] 开始上采样
            nn.Conv2d(100, 128, 3, 1, 1),  # 保持尺寸 22x22
            nn.BatchNorm2d(128),
            nn.GELU(),

            # 第一次上采样: 22x22 → 44x44
            nn.ConvTranspose2d(128, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),

            # 第二次上采样: 44x44 → 88x88
            nn.ConvTranspose2d(192, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # 最终调整通道数
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 1)  # 输出 [1, 256, 88, 88]
        )

    def unpatchify(self, x, d_dim):
        p = self.patch_size
        b, n, _ = x.shape
        h = w = int(n ** 0.5)
        x = x.view(b, h, w, p, p, d_dim)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, d_dim, h * p, w * p)

    def coords_grid(self, batch, ht, wd, device):
        coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def forward(self, x1, x2, x3, x4):
        b = x1.shape[0]

        # 侧卷积：从 backbone 不同阶段提取特征（降维到 mid_channel）
        E4 = self.side_conv1(x1)  # x1 是最高层特征（如512通道）→ [B, 128, H4, W4]
        E3 = self.side_conv2(x2)  # x2 → [B, 128, H3, W3]
        E2 = self.side_conv3(x3)  # x3 → [B, 128, H2, W2]
        E1 = self.side_conv4(x4)  # x4 → [B, 128, H1, W1]

        # 插值到相同尺寸（以 E2 的尺寸为基准）
        target_size = E2.size()[2:]

        # # #
        E4 = F.interpolate(E4, size=target_size, mode='bilinear')
        E3 = F.interpolate(E3, size=target_size, mode='bilinear')
        E1 = F.interpolate(E1, size=target_size, mode='bilinear')

        E5 = self.conv_block(E4, E3, E2, E1)  # shape: [b, c, h, w]

        # linear projection to patch tokens
        B, C, H, W = E5.shape
        x = E5.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        x_trans = self.decoder_embed_edge(x)  # [B, N, p^2 * d]
        print("x_trans的shape:", x_trans.shape)
        x_edge_conv = self.unpatchify(x_trans, self.edge_channel)  # [B, d, H', W']
        print("x_edge_conv的shape:", x_edge_conv.shape)
        edge_feature = self.conv_edge(x_edge_conv)  # [B, d, H', W']
        # 预测边界概率（二值分割）
        pred_edge = self.decoder_pred_edge(edge_feature)  # [B, 1, H', W']（概率图）
        edge_enhanced = edge_feature * pred_edge

        # downsample edge-enhanced features
        edge_down = self.edge_lookup_downsample(edge_enhanced)
        h, w = edge_down.shape[2], edge_down.shape[3]  # 获取edge_down的实际高和宽

        # correlation lookup
        corr_block = CorrBlock(edge_down, edge_down, num_levels=4, radius=2)
        coords = self.coords_grid(b, h, w, edge_down.device)  # 使用h和w生成coords
        lookup_ori = corr_block(coords)
        print("corr_block输出的shape:", lookup_ori.shape)
        # 最终下采样和特征增强
        x_l = self.final_lookup_upsample(lookup_ori)  # [B, 512, H_final, W_final]

        # ❗️在全部处理完后再上采样 pred_edge
        # print("F.interpolate前的pred_edge shape:", pred_edge.shape)
        pred_edge = F.interpolate(pred_edge, size=(352, 352), mode='bilinear', align_corners=False)
        # print("F.interpolate后的pred_edge shape:", pred_edge.shape)

        return x_l, pred_edge
    # 加了一个pred_edge
# 引导收！

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

    Model = BDLU(
            in_channels=[64, 128, 320, 512],  # [64,128,320,512]
            mid_channel=128,
            edge_channel=16,
            patch_size=8,
            image_size=288
        ).to(device)
    out1, out2 = Model(c1, c2, c3, c4)
    print('xl')
    print(out1.shape)
    print('pred_edge')
    print(out2.shape)