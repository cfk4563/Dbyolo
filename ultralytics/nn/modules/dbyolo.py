import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv

#######################################
#######################################
######## Concat and conv ##############

class CaC(nn.Module):
    def __init__(self, in_channels):
        super(CaC, self).__init__()
        # 1x1 卷积将拼接后的 2*in_channels 降维到 in_channels
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)  # 添加 BN 提升稳定性

    def forward(self, x):
        x_ccd = x[0]  # [B, C, H, W]
        x_dem = x[1]  # [B, C, H, W]
        # 沿通道维度拼接
        out = torch.cat([x_ccd, x_dem], dim=1)  # [B, 2*C, H, W]
        # 1x1 卷积降维
        out = self.conv(out)  # [B, C, H, W]
        out = self.bn(out)  # [B, C, H, W]
        return out


################ Attention feature fusion ###########

class Cat(nn.Module):
    def __init__(self, in_channels):
        super(Cat, self).__init__()
        # 降维以计算注意力
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 可学习的融合权重
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        输入：
            x_ccd: CCD 特征图 [batch_size, in_channels, height, width]
            x_dem: DEM 特征图 [batch_size, in_channels, height, width]
        输出：
            融合后的特征图 [batch_size, in_channels, height, width]
        """
        x_ccd = x[0]
        x_dem = x[1]
        batch_size, C, height, width = x_ccd.size()

        # 计算注意力
        proj_query = self.query_conv(x_ccd).view(batch_size, -1, width * height).permute(0, 2, 1)  # [B, H*W, C//8]
        proj_key = self.key_conv(x_dem).view(batch_size, -1, width * height)  # [B, C//8, H*W]
        energy = torch.bmm(proj_query, proj_key)  # [B, H*W, H*W]
        attention = F.softmax(energy, dim=-1)  # 注意力权重

        # 应用注意力到 DEM 特征
        proj_value = self.value_conv(x_dem).view(batch_size, -1, width * height)  # [B, C, H*W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(batch_size, C, height, width)  # [B, C, H, W]

        # 残差连接融合
        out = self.gamma * out + x_ccd
        return out

#############################################
#############################################
#############################################
#############################################
################## TFAM #####################
#############################################
#############################################

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class TFAM(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(Cat, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in // 2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


#############################################
#############################################
#############################################
################### DFM #####################

import math
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k

class DFM(nn.Module):
    """Fuse two feature into one feature."""

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, x, log=None, module_name=None,
                img_name=None):
        # channel part
        t1 = x[0]
        t2 = x[1]
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        return fuse

#############################################
#############################################
#############################################
################## DWRConv ##################

class DWRConv(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()

        self.conv_3x3 = Conv(in_c, out_c // 4, 3, s=2)

        self.conv_3x3_d1 = Conv(out_c // 4, out_c // 4, 3, d=1)
        self.conv_3x3_d2 = Conv(out_c // 4, out_c // 4, 3, d=2)
        self.conv_3x3_d3 = Conv(out_c // 4, out_c // 4, 3, d=3)
        self.conv_3x3_d4 = Conv(out_c // 4, out_c // 4, 3, d=4)

        # self.conv_1x1 = Conv(out_c , out_c , k=3)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1 = self.conv_3x3_d1(conv_3x3) + conv_3x3
        x2 = self.conv_3x3_d2(x1) + conv_3x3
        x3 = self.conv_3x3_d3(x2) + conv_3x3
        x4 = self.conv_3x3_d4(x3) + conv_3x3
        x_out = torch.cat([x1, x2, x3, x4], dim=1)
        x_out = x_out
        return x_out


