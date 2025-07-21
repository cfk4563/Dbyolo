import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

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

class AaN(nn.Module):
    def __init__(self, in_channels):
        super(AaN, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x_ccd = x[0]
        x_dem = x[1]
        out = self.bn(x_ccd + x_dem)
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

################### 消融实验  ################
class Cat_noAttention(nn.Module):
    def __init__(self, in_channels):
        super(Cat_noAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        x_ccd = x[0]
        x_dem = x[1]

        # 残差连接融合
        out = self.gamma * x_dem + x_ccd
        return out

###################################################
###################################################

class Cat_noGamma(nn.Module):
    def __init__(self, in_channels):
        super(Cat_noGamma, self).__init__()
        # 降维以计算注意力
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)


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
        out = out + x_ccd
        return out

###################################################
###################################################

class Cat_onlyResnet(nn.Module):
    def __init__(self, in_channels):
        super(Cat_onlyResnet, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        x_ccd = x[0]
        x_dem = x[1]

        # 残差连接融合
        out = x_dem + x_ccd
        return out

###################################################
###################################################

class Cat_onlyAttention(nn.Module):
    def __init__(self, in_channels):
        super(Cat_onlyAttention, self).__init__()
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

###################################################
###################################################
###################################################


import pywt
import ultralytics.nn.modules.lowlevel as lowlevel

class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)

        return x

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)

        return x

#################################################
##################################################
####################################################
#####################################################

class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()

        # 卷积1x1，保持输入通道数不变，激活函数使用ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

        # 卷积1x1 + 3x3卷积，扩大特征图的通道数，经过中间层后再降回原通道数
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

        # 卷积1x1 + 5x5卷积，再经过1x1卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

        # 卷积1x1 + 7x7卷积，再经过1x1卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

        # 将多尺度的特征图（四个卷积的输出）拼接在一起，并进行融合卷积
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 四个不同尺度的卷积操作
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # 将不同尺度的特征图拼接
        x_f = torch.cat([x1, x2, x3, x4], dim=1)

        # 通过融合卷积得到输出特征图
        out = self.convmix(x_f)

        return out

class MDFM(nn.Module):
    def __init__(self, in_d):
        super(MDFM, self).__init__()
        self.in_d = in_d
        self.out_d = in_d
        # 定义多尺度特征融合（MSFF）模块
        self.MPFL = MSFF(inchannel=in_d, mid_channel=64)  ##64

        # 差异增强卷积：对输入特征进行增强
        self.conv_diff_enh = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

        # 差异增强后的卷积操作，得到输出特征图
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

        # 差异计算卷积：计算输入特征图之间的差异
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

        # 融合卷积：将差异特征图进行融合
        self.convmix = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 3, groups=self.in_d, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

        # 卷积上采样：对输入特征图进行上采样
        # self.conv_up = Conv(int(in_d * 0.5), in_d, 1, act=nn.ReLU())

    def forward(self, x):
        # 输入x是一个包含两个特征图的元组，x[0]和x[1]
        x1, x2 = x[0], x[1]
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]

        # 对x2进行上采样
        # x2 = self.conv_up(x2)

        # 计算特征图之间的差异，并使用卷积模块进行处理
        x_sub = torch.abs(x1 - x2)
        x_att = torch.sigmoid(self.conv_sub(x_sub))

        # 对x1和x2进行差异增强
        x1 = (x1 * x_att) + self.MPFL(self.conv_diff_enh(x1))
        x2 = (x2 * x_att) + self.MPFL(self.conv_diff_enh(x2))

        # 特征图融合：将x1和x2堆叠到一起
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)

        # 根据注意力机制（x_att）对融合后的特征图进行加权
        x_f = x_f * x_att

        # 最终卷积操作，得到输出特征图
        out = self.conv_dr(x_f)

        return out

###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################


class BiFPN(nn.Module):
    def __init__(self, c1):
        super(BiFPN, self).__init__()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))
