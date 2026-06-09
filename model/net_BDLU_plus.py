import os
import warnings
from functools import partial

import math
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_

from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention


# 陈泽实验！灵感来源medsegdiff v1,频率分支应用
import torch.fft
class FFParser(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2) * 0.02)
        self.h = h
        self.w = w

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape

        # 校验频域权重维度是否匹配
        assert self.h == H, f"FFParser权重高度维度不匹配：输入H={H}, 权重H={self.h}"
        assert self.w == (W // 2 + 1), f"FFParser权重宽度维度不匹配：输入W//2+1={W // 2 + 1}, 权重W={self.w}"

        if spatial_size is None:
            a = H
            b = W
        else:
            a, b = spatial_size

        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')  # 2D FFT
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight  # Apply learnable spectral filter
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')  # Inverse FFT

        return x

class FeatureAlignAndFuse(nn.Module):
    def __init__(self, x_channels, f_channels, target_h, target_w):
        super().__init__()
        # 1x1卷积对齐通道数
        self.conv_align = nn.Conv2d(x_channels, f_channels, kernel_size=1)
        # DynamicConditionFusion负责融合
        self.dynamic_fusion = DynamicConditionFusion(f_channels, target_h, target_w)

    def forward(self, x, f):
        # Step 1: 先对齐通道数
        x_aligned = self.conv_align(x)

        # Step 2: 再对齐空间分辨率
        if x_aligned.shape[2:] != f.shape[2:]:
            x_aligned = F.interpolate(x_aligned, size=f.shape[2:], mode='bilinear', align_corners=False)

        # Step 3: DynamicConditionFusion融合
        fusion = self.dynamic_fusion(x_aligned, f)

        return fusion

class DynamicConditionFusion(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()

        # 初始化FFParser
        # self.ffparser = FFParser(dim=channels, h=height//2 + 1, w=width)

        # 1deepseek
        # 修改此处：h 应为原高度，w 应为 width//2 + 1
        # self.ffparser = FFParser(dim=channels, h=height, w=width // 2 + 1)  # ✅ 修复维度

        # 2deepseek微调失败后给出的修改
        h_fft = height
        w_fft = width // 2 + 1
        self.ffparser = FFParser(dim=channels, h=h_fft, w=w_fft)

        # LayerNorm: 注意是对每个通道进行的，所以normalized_shape=(C, H, W)
        self.layer_norm_f1 = nn.LayerNorm([channels, height, width])
        self.layer_norm_x1 = nn.LayerNorm([channels, height, width])

    def forward(self, x1, f1):
        """
        :param x1: 噪声特征 (B, C, H, W)
        :param f1: 图像特征 (B, C, H, W)
        :return: 融合后的特征 fusion1 (B, C, H, W)
        """
        # Step 1: FFParser滤波
        x1_filtered = self.ffparser(x1)

        # Step 2: LayerNorm归一化
        x1_norm = self.layer_norm_x1(x1_filtered)
        f1_norm = self.layer_norm_f1(f1)

        # Step 3: 注意力亲和力Affinity
        affinity = x1_norm * f1_norm  # element-wise乘法

        # Step 4: 融合
        fusion = affinity * f1  # Affinity map乘原图特征

        return fusion
# 频率收!

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

        # 轻量化特征增强模块（通过 nn.Sequential 实现）
        self.edge_enhancer = nn.Sequential(
            nn.Conv2d(edge_channel, edge_channel, kernel_size=3, padding=1, groups=edge_channel, bias=False),
            nn.BatchNorm2d(edge_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_channel, edge_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(edge_channel),
            nn.Sigmoid()  # 注意：这里我们使用 Sigmoid 来生成注意力权重
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
        E4 = F.interpolate(E4, size=target_size, mode='bilinear')
        E3 = F.interpolate(E3, size=target_size, mode='bilinear')
        E1 = F.interpolate(E1, size=target_size, mode='bilinear')

        E5 = self.conv_block(E4, E3, E2, E1)  # shape: [b, c, h, w]

        # linear projection to patch tokens
        B, C, H, W = E5.shape
        x = E5.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        x_trans = self.decoder_embed_edge(x)  # [B, N, p^2 * d]
        x_edge_conv = self.unpatchify(x_trans, self.edge_channel)  # [B, d, H', W']
        edge_feature = self.conv_edge(x_edge_conv)  # [B, d, H', W']
        # 1. 生成边缘概率图（不变）
        pred_edge = self.decoder_pred_edge(edge_feature)  # [B,1,H',W']

        # LEG plus
        # 2. 生成注意力权重（不替换原始特征）
        edge_attention = self.edge_enhancer(edge_feature)  # [B,d,H',W']（权重）
        # 3. 先用注意力增强原始特征，再用概率图增强
        edge_feature_weighted = edge_feature * edge_attention  # 保留特征信息+通道-空间筛选

        edge_enhanced = edge_feature_weighted * pred_edge  # 进一步空间筛选高置信度边缘

        # downsample edge-enhanced features
        edge_down = self.edge_lookup_downsample(edge_enhanced)
        h, w = edge_down.shape[2], edge_down.shape[3]  # 获取edge_down的实际高和宽

        # correlation lookup
        corr_block = CorrBlock(edge_down, edge_down, num_levels=4, radius=2)
        coords = self.coords_grid(b, h, w, edge_down.device)  # 使用h和w生成coords
        lookup_ori = corr_block(coords)
        # 最终下采样和特征增强
        x_l = self.final_lookup_upsample(lookup_ori)  # [B, 512, H_final, W_final]

        # ❗️在全部处理完后再上采样 pred_edge
        # print("F.interpolate前的pred_edge shape:", pred_edge.shape)
        pred_edge = F.interpolate(pred_edge, size=(352, 352), mode='bilinear', align_corners=False)
        # print("F.interpolate后的pred_edge shape:", pred_edge.shape)

        return x_l, pred_edge
    # 加了一个pred_edge
# 引导收！

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)  # Fixme: Check Here
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # 零嵌入层
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # set mask_proj weight to 0
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)
            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = mask_chans

        # time_embed

        self.time_embed = nn.ModuleList()
        for i in range(0, len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))

        # patch_embed
        # self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
        #                                       embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        # 删除了mask_chans，去掉零嵌入层,stride=4

        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # 陈泽实验尝试
        # 加到 PyramidVisionTransformerImpr.__init__() 中
        # 微调384维度不对，deepseek修改
        self.fuse1 = FeatureAlignAndFuse(x_channels=256, f_channels=64, target_h=88, target_w=88)
        self.fuse2 = FeatureAlignAndFuse(x_channels=256, f_channels=128, target_h=44, target_w=44)
        self.fuse3 = FeatureAlignAndFuse(x_channels=256, f_channels=320, target_h=22, target_w=22)
        self.fuse4 = FeatureAlignAndFuse(x_channels=256, f_channels=512, target_h=11, target_w=11)

# 加个noise_feats
    def forward_features(self, x, timesteps, cond_img, noise_feats=None):

        # 陈泽实验
        if noise_feats is not None:
            x1, x2, x3, x4 = noise_feats
        else:
            x1 = x2 = x3 = x4 = None  # 兼容无融合情况
            print('#######################################noise_feats is None')
        # 收

        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0]))
        time_token = time_token.unsqueeze(dim=1)

        B = x.shape[0]
        outs = []

        # stage 1
        # x, H, W = self.patch_embed1(cond_img, x) # 融合了噪声图和真实图,删除了x
        x, H, W = self.patch_embed1(cond_img)

        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 陈泽实验Medseg

        f1 = x
        if x1 is not None:
            a1 = self.fuse1(x1, f1)
        # a1属于ff分支，x还是x，纯图像特征，输入到下一层的PVT中，目前的x是f1
        # #陈泽收尾

        outs.append(x) # F1

        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 2
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 陈泽实验Medseg

        f2 = x
        if x2 is not None:
            a2 = self.fuse2(x2, f2)
        # #陈泽收尾

        outs.append(x) # F2

        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 3
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 陈泽实验Medseg

        f3 = x
        if x3 is not None:
            a3 = self.fuse3(x3, f3)
        # #陈泽收尾

        outs.append(x) # F3

        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))
        time_token = time_token.unsqueeze(dim=1)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 陈泽实验Medseg
        f4 = x
        if x4 is not None:
            a4 = self.fuse4(x4, f4)
        # #陈泽收尾
        outs.append(x) # F4

        outs.append(a1)
        outs.append(a2)
        outs.append(a3)
        outs.append(a4)
        # a1,a2,a3,a4为频率分支的特征输入
        # outs = [f1,f2,f3,f4,a1,a2,a3,a4]
        return outs

    def forward(self, x, timesteps, cond_img, noise_feats=None):
        x = self.forward_features(x, timesteps, cond_img, noise_feats=noise_feats)

        #        x = self.head(x[3])

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)  # Fixme: Check Here
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


from timm.models.layers import DropPath
import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def Downsample(
        dim,
        dim_out=None,
        factor=2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
        nn.Conv2d(dim * (factor ** 2), dim if dim_out is None else dim_out, 1)
    )


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)
        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim) # encoder加一层
        )

    def forward(self, x, timesteps):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        _xfeature = []
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
            _xfeature.append(x)

        return x, _xfeature  # x is the last output, _xfeature is the list of intermediate features

# TFF灵感来源STNet，用于融合三个特征
def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class Tri_TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tri_TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 3, in_channel)
        self.catconvC = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 3, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.convC = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB, xC):
        x_ABdiff = xA - xB  # 通过相减获得粗略的变化表示: (B,C,H,W)
        x_BCdiff = xB - xC

        x_diffA = self.catconvA(torch.cat([x_ABdiff, xA], dim=1)) #将变化特征与xA拼接,通过DWConv提取特征: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W);  (B,2C,H,W)-catconvA-->(B,C,H,W)
        x_diffB = self.catconvB(torch.cat([x_ABdiff, x_BCdiff, xB], dim=1)) #将变化特征与xB拼接,通过DWConv提取特征: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W);  (B,2C,H,W)-catconvB-->(B,C,H,W)
        x_diffC = self.catconvC(torch.cat([x_BCdiff, xC], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA)) # 通过卷积映射到1个通道,生成空间描述符,然后通过sigmoid生成权重: (B,C,H,W)-convA->(B,1,H,W)
        B_weight = self.sigmoid(self.convB(x_diffB)) # 通过卷积映射到1个通道,生成空间描述符,然后通过sigmoid生成权重: (B,C,H,W)-convB->(B,1,H,W)
        C_weight = self.sigmoid(self.convC(x_diffC)) # 通过卷积映射到1个通道,生成空间描述符,然后通过sigmoid生成权重: (B,C,H,W)-convB->(B,1,H,W)

        xA = A_weight * xA # 使用生成的权重A_weight调整对应输入xA: (B,1,H,W) * (B,C,H,W) == (B,C,H,W)
        xB = B_weight * xB # 使用生成的权重B_weight调整对应输入xB: (B,1,H,W) * (B,C,H,W) == (B,C,H,W)
        xC = C_weight * xC

        x = self.catconv(torch.cat([xA, xB, xC], dim=1)) # 两个特征拼接,然后恢复与输入相同的shape: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W); (B,2C,H,W)--catconv->(B,C,H,W)

        return x

# 融合频率分支的四个特征：
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
# #########
class Decoder(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        # c分支的融合
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        # a分支的融合
        # 第二套卷积层用于处理 a1, a2, a3, a4
        # 用1×1卷积将通道统一为128（可根据需求调整）,目前用于融合频域噪声分支
        self.conv_c1 = nn.Conv2d(64, 128, kernel_size=1)  # input1: 64→128
        self.conv_c2 = nn.Conv2d(128, 128, kernel_size=1)  # input2: 128→128
        self.conv_c3 = nn.Conv2d(320, 128, kernel_size=1)  # input3: 320→128
        self.conv_c4 = nn.Conv2d(512, 128, kernel_size=1)  # input4: 512→128
        self.conv_block = Conv_Block2(channels=128)
        # #########################

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(
            # ConvModule(in_channels=embedding_dim//8+1, out_channels=embedding_dim//8, kernel_size=1,
            #            norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )

        self.encoder = Encoder(embedding_dim)

        # 新增BDLU模块
        self.bdlu = BDLU(
            in_channels=dims,  # [64,128,320,512]
            mid_channel=128,
            edge_channel=16,
            patch_size=8,
            image_size=288
        )
        self.Tri_TFF = Tri_TFF(in_channel=256, out_channel=256)

    def forward(self, inputs, timesteps, x):

        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        c1, c2, c3, c4, a1, a2, a3, a4 = inputs
        # c1,c2,c3,c4 是pvt输出的f1,f2,f3,f4
        # a1,a2,a3,a4 是是频率分支的四个输出

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        # stage0:Encoder处理的最后一层噪声x
        # print('噪声特征x')
        # print(x.shape)

        # stage1:图像分支四个特征的融合
        _c = self._process_and_fuse_features_c(c1, c2, c3, c4)
        # print('_c是图像分支的融合特征结果')
        # print(_c.shape)

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
        # 融合（拼接后通道数为256）
        xf = self.conv_block(a1_aligned, a2_aligned, a3_aligned, a4_aligned)
        # print('xf是噪声频率分支的融合特征结果')
        # print(xf.shape)

        # stage3:边缘引导分支
        xl, pred_edge = self.bdlu(c1, c2, c3, c4)
        # print('边缘引导分支特征xl', xl.shape)
        # print('边缘引导分支特征pred_edge', pred_edge.shape)

        # 融合三分支特征
        TriXFusion = self.Tri_TFF(xl, _c, xf)

        # 融合后的特征与噪声特征x结合
        x = torch.cat([TriXFusion, x], dim=1)

        # diffusion decoder
        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        # x = self.pred(torch.cat([x, _x.pop(-1)], dim=1))
        x = self.pred(x)
        # print('最终送入预测头的特征', x.shape)
        return x, c1, c2, c3, c4, pred_edge

    def _process_and_fuse_features_c(self, c1, c2, c3, c4):
        n = c4.shape[0]
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        L34 = self.linear_fuse34(torch.cat([_c4, _c3], dim=1))
        L2 = self.linear_fuse2(torch.cat([L34, _c2], dim=1))
        _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))
        return _c

    def _process_and_fuse_features_a(self, a1, a2, a3, a4):
        n = a4.shape[0]
        _a4 = self.linear_a4(a4).permute(0, 2, 1).reshape(n, -1, a4.shape[2], a4.shape[3])
        _a4 = resize(_a4, size=a1.size()[2:], mode='bilinear', align_corners=False)
        _a3 = self.linear_a3(a3).permute(0, 2, 1).reshape(n, -1, a3.shape[2], a3.shape[3])
        _a3 = resize(_a3, size=a1.size()[2:], mode='bilinear', align_corners=False)
        _a2 = self.linear_a2(a2).permute(0, 2, 1).reshape(n, -1, a2.shape[2], a2.shape[3])
        _a2 = resize(_a2, size=a1.size()[2:], mode='bilinear', align_corners=False)
        _a1 = self.linear_a1(a1).permute(0, 2, 1).reshape(n, -1, a1.shape[2], a1.shape[3])

        L34 = self.linear_fuse34_a(torch.cat([_a4, _a3], dim=1))
        L2 = self.linear_fuse2_a(torch.cat([L34, _a2], dim=1))
        _a = self.linear_fuse1_a(torch.cat([L2, _a1], dim=1))
        return _a

class net(nn.Module):
    def __init__(self, class_num=2, mask_chans=0, **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        self.backbone = pvt_v2_b4_m(in_chans=3, mask_chans=mask_chans)
        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=256, class_num=class_num, mask_chans=mask_chans)
        self._init_weights()  # load pretrain
        self.encoder = Encoder(embedding_dim=256)

    def forward(self, x, timesteps, cond_img):
        # ####nosie_feats
        noise_final, noise_feats = self.encoder(x, timesteps)
        features = self.backbone(x, timesteps, cond_img, noise_feats=noise_feats)
        # 上面的这个features包含[f1,f2,f3,f4,a1,a2,a3,a4]
        features, layer1, layer2, layer3, layer4, pred_edge = self.decode_head(features, timesteps, noise_final)
        # print('net return的features.shape:', features.shape)
        # print('net return的pred_edge.shape:', pred_edge.shape)
        # print(f"net return的pred_edge type: {type(pred_edge)}, shape: {pred_edge.shape}")
        return features, pred_edge
    # 原来只返回features，现在返回features和pred_edge

    # def _download_weights(self, model_name):
    #     _available_weights = [
    #         'pvt_v2_b0',
    #         'pvt_v2_b1',
    #         'pvt_v2_b2',
    #         'pvt_v2_b3',
    #         'pvt_v2_b4',
    #         'pvt_v2_b4_m',
    #         'pvt_v2_b5',
    #     ]
    #     assert model_name in _available_weights, f'{model_name} is not available now!'
    #     from huggingface_hub import hf_hub_download
    #     return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')
    from huggingface_hub import hf_hub_download

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0',
            'pvt_v2_b1',
            'pvt_v2_b2',
            'pvt_v2_b3',
            'pvt_v2_b4',
            'pvt_v2_b4_m',
            'pvt_v2_b5',
        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'

        # 指定本地文件路径
        local_path = os.path.join('./pretrained_weights', f'{model_name}.pth')
        from huggingface_hub import hf_hub_download
        if os.path.exists(local_path):
            print(f"Loading pretrained weights from local path: {local_path}")
            return local_path
        else:
            print(f"Local pretrained weights not found, downloading from Hugging Face Hub")
            return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        pretrained_dict = torch.load(self._download_weights('pvt_v2_b4_m')) #for save mem
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=False)

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img):
        return self.forward(x, timesteps, cond_img)

    def extract_features(self, cond_img):
        # do nothing
        return cond_img


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass