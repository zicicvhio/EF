import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = conv_block(in_features=in_features,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        self.c1 = conv_block(in_features=in_features,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

        self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x




import torch
import torch.nn as nn

#Github地址：https://github.com/zcablii/Large-Selective-Kernel-Network
#论文地址：https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf
# 微信公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
https://github.com/AIFengheshu/Plug-play-modules
"""

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 定义各个卷积层
        # 深度可分离卷积，保持输入和输出通道数一致，卷积核大小为5
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 空间卷积，卷积核大小为7，膨胀率为3，增加感受野
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 1x1卷积，用于降维
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 结合平均和最大注意力的卷积
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 最后的1x1卷积，将通道数恢复到原始维度
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # 对输入进行两种不同的卷积操作以生成注意力特征
        attn1 = self.conv0(x)  # 第一个卷积特征
        attn2 = self.conv_spatial(attn1)  # 空间卷积特征

        # 对卷积特征进行1x1卷积以降维
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # 将两个特征在通道维度上拼接
        attn = torch.cat([attn1, attn2], dim=1)
        # 计算平均注意力特征
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # 计算最大注意力特征
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 拼接平均和最大注意力特征
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 通过卷积生成注意力权重，并应用sigmoid激活函数
        sig = self.conv_squeeze(agg).sigmoid()
        # 根据注意力权重调整特征
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1)
        # 最终卷积恢复到原始通道数
        attn = self.conv(attn)
        # 通过注意力特征加权原输入
        return x * attn

######################

import torch
from torch import nn
import einops
from typing import Union

# 论文题目：Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation
# 中文题目:  拓扑几何约束管状结构分割的动态蛇卷积
# 英文论文链接：https://arxiv.org/pdf/2307.08388
# 中文论文链接：
# https://yaoleiqi.github.io/publication/2023_ICCV/DSCNet_Chinese.pdf
# 官方github：https://github.com/YaoleiQi/DSCNet
# 所属机构：东南大学人工智能新一代技术及其跨学科应用教育部重点实验室，江苏省医学信息处理国际联合实验室，中法生物医学信息研究中心
# 关键词：先验知识融合，动态蛇形卷积，多视角特征融合，持续同调，管状结构分割

"""Dynamic Snake Convolution Module"""


class DSConv_pro(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 9,
            extend_scope: float = 1.0,
            morph: int = 0,
            if_offset: bool = True,
            # device: str | torch.device = "cuda",
            device: Union[str, torch.device] = "cuda",
    ):
        """
        动态蛇形卷积模块 (Dynamic Snake Convolution) 的实现。
        Args:
            in_channels: 输入通道数，默认为1。
            out_channels: 输出通道数，默认为1。
            kernel_size: 卷积核大小，默认为9。
            extend_scope: 卷积核扩展范围，用于控制卷积操作的影响范围。默认为1。
            morph: 卷积核形态类型，沿x轴（0）或y轴（1）。参见论文了解更多细节。
            if_offset: 是否进行形变操作，若为 False，则为标准卷积。默认为 True。
        """
        super().__init__()
        if morph not in (0, 1):
            raise ValueError("morph 应该是 0 或 1。")

        # 保存输入参数
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # 偏移量归一化层，使用分组归一化处理偏移特征
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        # 输出特征归一化层，使用分组归一化处理输出特征
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        # 偏移卷积层，生成特征偏移量，用于动态卷积
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        # 动态蛇形卷积沿x轴的卷积核，卷积核大小为 (kernel_size, 1)
        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        # 动态蛇形卷积沿y轴的卷积核，卷积核大小为 (1, kernel_size)
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # 生成偏移量图，数值范围为 [-1, 1]
        offset = self.offset_conv(input)
        # 对偏移量进行分组归一化处理
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # 获取变形卷积的偏移坐标映射
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        # 根据偏移坐标获取插值后的特征
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        # 根据卷积形态选择合适的卷积操作
        if self.morph == 0:
            # 沿 x 轴的动态蛇形卷积
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            # 沿 y 轴的动态蛇形卷积
            output = self.dsc_conv_y(deformed_feature)

        # 使用分组归一化和 ReLU 激活函数处理卷积结果
        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
        offset: torch.Tensor,
        morph: int,
        extend_scope: float = 1.0,
        device: Union[str, torch.device] = "cuda",
):
    """
    计算动态蛇形卷积的2D坐标映射。
    Args:
        offset: 网络预测的偏移量，形状为 [B, 2*K, W, H]，其中 K 表示卷积核大小。
        morph: 卷积核形态类型，沿x轴（0）或y轴（1）。
        extend_scope: 扩展范围，控制卷积的偏移范围，默认为 1。
        device: 数据所在设备，默认为 'cuda'。
    Return:
        y_coordinate_map: y轴的坐标映射，形状为 [B, K_H * H, K_W * W]
        x_coordinate_map: x轴的坐标映射，形状为 [B, K_H * H, K_W * W]
    """
    # 检查 morph 参数是否为有效值
    if morph not in (0, 1):
        raise ValueError("morph 应该是 0 或 1。")

    # 获取批大小、宽度和高度
    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2  # 计算卷积核大小
    center = kernel_size // 2  # 中心位置
    device = torch.device(device)  # 确定设备

    # 将偏移量拆分为 x 和 y 的偏移
    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    # 生成 y 轴的中心坐标，扩展到每个卷积核位置和高度
    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    # 生成 x 轴的中心坐标，扩展到每个卷积核位置和宽度
    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    # 根据 morph 值来处理不同形态的卷积
    if morph == 0:
        """
        初始化卷积核，并展开卷积核：
            y：只需要0
            x：范围为 -num_points//2 到 num_points//2（由卷积核大小决定）
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        # 将 y 和 x 的扩展分布到对应的宽和高上
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        # 计算新的 y 和 x 坐标
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        # 重复 y 和 x 坐标，以适应批次维度
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        # 调整 y 偏移并初始化偏移量
        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # 中心位置保持不变，其他位置开始摇摆
        # 偏移量是一个迭代过程
        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                    y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                    y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        # 调整 y 偏移后，将偏移量应用到坐标
        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")
        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        # 生成 y 和 x 坐标图
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        初始化卷积核，并展开卷积核：
            y：范围为 -num_points//2 到 num_points//2（由卷积核大小决定）
            x：只需要0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        # 将 y 和 x 的扩展分布到对应的宽和高上
        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        # 计算新的 y 和 x 坐标
        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        # 重复 y 和 x 坐标，以适应批次维度
        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        # 调整 x 偏移并初始化偏移量
        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # 中心位置保持不变，其他位置开始摇摆
        # 偏移量是一个迭代过程
        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                    x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                    x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        # 调整 x 偏移后，将偏移量应用到坐标
        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        # 生成 y 和 x 坐标图
        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
        input_feature: torch.Tensor,
        y_coordinate_map: torch.Tensor,
        x_coordinate_map: torch.Tensor,
        interpolate_mode: str = "bilinear",
):
    """根据坐标图插值DSCNet的特征

    Args:
        input_feature: 待插值的特征图，形状为 [B, C, H, W]
        y_coordinate_map: 沿y轴的坐标图，形状为 [B, K_H * H, K_W * W]
        x_coordinate_map: 沿x轴的坐标图，形状为 [B, K_H * H, K_W * W]
        interpolate_mode: nn.functional.grid_sample的插值模式，可以为 'bilinear' 或 'bicubic' ，默认是 'bilinear'。

    Return:
        interpolated_feature: 插值后的特征图，形状为 [B, C, K_H * H, K_W * W]
    """

    # 检查插值模式是否正确
    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode 应为 'bilinear' 或 'bicubic'。")

    # 获取 y 和 x 的最大值
    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    # 缩放 y 坐标图到指定范围
    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    # 缩放 x 坐标图到指定范围
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    # 增加一维，使得坐标图的形状适配 grid_sample 的输入
    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # 合并 x 和 y 坐标图，生成 grid，形状为 [B, H, W, 2]，其中 [:, :, :, 2] 表示 [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    # 使用 grid_sample 进行插值
    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
        coordinate_map: torch.Tensor,
        origin: list,
        target: list = [-1, 1],
):
    """将坐标图的值从 origin=[min, max] 映射到 target=[a,b]，用于 DSCNet

    Args:
        coordinate_map: 需要缩放的坐标图
        origin: 坐标图的原始值范围，例如 [coordinate_map.min(), coordinate_map.max()]
        target: 坐标图的目标值范围，默认是 [-1, 1]

    Return:
        coordinate_map_scaled: 缩放后的坐标图
    """
    min, max = origin
    a, b = target

    # 将坐标图限制在 [min, max] 范围内
    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    # 计算缩放因子并应用到坐标图上
    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block =DSConv_pro(512, 512).to(device)

    # 若input形状为B C H W，先用下面代码变换张量形状
    input = torch.rand(8, 512, 12, 15).to(device)  # 输入 B C H W

    output = block(input)


    print(output.size())  # 输出的形状
    # block = PPA(in_features=64, filters=64)  # 输入通道数，输出通道数
    # input = torch.rand(6, 512, 12, 15)  # 输入 B C H W
    # output = block(input)
    # print(input.size())
    # print(output.size())