
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        # self.bn = nn.BatchNorm2d(attention_channel)
        self.bn = nn.LayerNorm(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.bn(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class dif_atten(nn.Module):
    def __init__(self, in_planes, aim_kernel):
        super(dif_atten, self).__init__()
        self.in_planes = in_planes
        self.aim_kernel = aim_kernel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp0 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=max(32, self.in_planes // 16), kernel_size=1, stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=max(32, self.in_planes // 16), out_channels=self.aim_kernel * self.aim_kernel,
                      kernel_size=1, stride=1, padding=0)
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=max(32, self.in_planes // 16), kernel_size=1, stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=max(32, self.in_planes // 16), out_channels=self.aim_kernel * self.aim_kernel,
                      kernel_size=1, stride=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1):
        fd = x0 - x1

        fd_avg = self.avg_pool(fd)
        fd_max = self.max_pool(fd)

        z1 = self.mlp0(fd_avg)
        z2 = self.mlp0(fd_max)

        Mfd = self.sigmoid((z1 + z2).view(-1, 1, 1, 1, self.aim_kernel, self.aim_kernel))

        return Mfd

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt_multi_scale(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_multi_scale, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_upper_h = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h_u = nn.Conv2d(inp // 2, inp, kernel_size=(2, 1), stride=2, padding=0)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_upper_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_w_u = nn.Conv2d(inp // 2, inp, kernel_size=(1, 2), stride=2, padding=0)
        self.conv_upper = nn.Conv2d(inp // 2, inp, kernel_size=2, stride=2, padding=0)

        self.inp = inp

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.bn_x = nn.BatchNorm2d(1)
        self.bn_y = nn.BatchNorm2d(1)
        self.conv_x = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
        self.conv_y = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.conv_x_1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.conv_y_1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.conv_final = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding =0)

        self.down = nn.Conv2d(inp // 2, inp, kernel_size=2, stride=2)



    def enhance(self, x, f_upper):

        identity = x  # (b,c,h,w)
        b, c, h, w = x.size()
        x_h = self.pool_h(x)                                        # (b,c,h,1)
        x_h = torch.squeeze(x_h,dim=3)                              # (b,c,h)
        x_h_h = torch.matmul(torch.transpose(x_h, 1, 2), x_h)       # (b,h,h)
        x_h_h = torch.sigmoid(x_h_h)                                # (b,h,h)
        x_h_c = torch.bmm(x_h_h, torch.transpose(x_h, 1, 2))        # (b,h,c)
        x_h_out = torch.unsqueeze(x_h_c.permute(0,2,1),3)   # (b,c,h,1)

        p = self.down(f_upper)                                      # (b,c,h,w)
        p_h = self.pool_upper_h(p)                                  # (b,c,h,1)
        p_h = torch.squeeze(p_h, dim=3)                             # (b,c,h)
        p_h_h = torch.matmul(torch.transpose(p_h, 1, 2), x_h)       # (b,h,h)
        p_h_h = torch.sigmoid(p_h_h)                                # (b,h,h)
        p_h_c = torch.bmm(p_h_h, torch.transpose(p_h, 1, 2))        # (b,h,c)
        p_h_out = torch.unsqueeze(p_h_c.permute(0,2,1), 3)          # (b,c,h,1)

        h_out = torch.cat([x_h_out, p_h_out], dim=3).permute(0,3,2,1)             # (b,2,h,c)
        h_out = self.conv_x(h_out)                                  # (b,1,h,c)
        h_out = h_out.permute(0,3,2,1)                              # (b,c,h,1)

        x_w = self.pool_w(x)                                        # (b,c,1,w)
        x_w = torch.squeeze(x_w,dim=2)                              # (b,c,w)
        x_w_w = torch.matmul(torch.transpose(x_w, 1, 2), x_w)       # (b,w,w)
        x_w_w = torch.sigmoid(x_w_w)                                # (b,w,w)
        x_w_c = torch.bmm(x_w_w, torch.transpose(x_w, 1, 2))        # (b,w,c)
        x_w_out = torch.unsqueeze(x_w_c.permute(0,2,1),2)           # (b,c,1,w)

        p_w = self.pool_upper_w(p)                                  # (b,c,1,w)
        p_w = torch.squeeze(p_w, dim=2)                             # (b,c,w)
        p_w_w = torch.matmul(torch.transpose(p_w, 1, 2), p_w)       # (b,w,w)
        p_w_w = torch.sigmoid(p_w_w)                                # (b,w,w)
        p_w_c = torch.bmm(p_w_w, torch.transpose(p_w, 1, 2))        # (b,w,c)
        p_w_out = torch.unsqueeze(p_w_c.permute(0, 2, 1), 2)        # (b,c,1,w)

        w_out = torch.cat([x_w_out, p_w_out], dim=2).permute(0,2,1,3)           # (b,2,c,w)
        w_out = self.conv_y(w_out).permute(0,2,3,1)                 # (b,c,w,1)

        y = torch.cat([h_out, w_out], dim=2)                        # (b,c,h+w,1)
        y = self.act(self.bn1(self.conv1(y)))                       # (b,mip,h+w,1)
        res_h, res_w = torch.split(y, [h, w], dim=2)                # (b,mip,h,1), (b,mip,w,1)
        res_w = res_w.permute(0, 1, 3, 2)                           # (b,mip,1,w)
        res_h = self.conv_h(res_h).sigmoid()                        # (b,c,h,1)
        res_w = self.conv_w(res_w).sigmoid()                        # (b,c,1,w)
        out = (identity + p) * res_w * res_h + identity
        return out

    def forward(self, x_rgb, x_t, f_upper):
        rgb_out = self.enhance(x_rgb, f_upper)
        t_out = self.enhance(x_t, f_upper)
        return rgb_out, t_out

    def forward_old(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class ODConv2d_fusion(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d_fusion, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()
        self.conv = nn.Conv2d(2 * out_planes, out_planes, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.dif_mask = dif_atten(in_planes=in_planes, aim_kernel=kernel_size)
        self.gelu = nn.GELU()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common
        self.coordAtten = CoordAtt_multi_scale(in_planes, in_planes)

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x, x_atten):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x_atten)
        batch_size, in_planes, height, width = x.size()
        dif_mask = self.dif_mask(x, x_atten).half()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * dif_mask * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention

        channel_attention_compute = (channel_attention.unsqueeze(dim=1)).unsqueeze(dim=1)
        filter_attention_compute = (filter_attention.unsqueeze(dim=2)).unsqueeze(dim=1)
        spatial_attention_compute = spatial_attention
        kernel_attention_compute = kernel_attention
        dif_atten_compute = dif_mask
        compute_k_n = channel_attention_compute * filter_attention_compute * spatial_attention_compute * kernel_attention_compute * dif_atten_compute
        compute_k = torch.sum(compute_k_n, dim=1)
        return output, compute_k


    def _forward_impl_pw1x(self, x, x_atten, kernel_f=None):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x_atten)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output, self.weight.squeeze(dim=0)

    def forward(self, x_rgb, x_t, feature_upper=None):
        if feature_upper is not None:
            x_rgb, x_t = self.coordAtten(x_rgb, x_t, feature_upper)

        rgb_od, t_kernel = self._forward_impl(x_rgb, x_t)
        t_od, rgb_kernel = self._forward_impl(x_t, x_rgb)

        f_out = torch.cat((rgb_od, t_od), dim=1)
        f_out = self.softmax(f_out)
        out_r = f_out[:, :self.out_planes, :, :]
        out_t = f_out[:, self.out_planes:, :, :]
        out = x_rgb * out_r + x_t * out_t + x_rgb + x_t
        return out, rgb_kernel, t_kernel

class CDC(nn.Module):
    def __init__(self, arg, kernel_size, padding):
        super().__init__()
        """
        kernel_size [7, 5, 3]
        padding     [3, 2, 1]
        """
        self.c = arg
        self.od_kernel_size = kernel_size
        self.od_padding = padding

        self.ODConv2d_fusion = ODConv2d_fusion(self.c, self.c, kernel_size=self.od_kernel_size, padding=self.od_padding)

    def forward(self, x):
        x_rgb = x[0]
        x_t = x[1]
        # x_t = x[0]  # thermal b, c, h, w
        # x_rgb = x[1]  # rgb b, c, h, w
        if len(x) > 2:
            f_upper = x[2]
        else:
            f_upper = None

        res, rgb_kernel, t_kernel = self.ODConv2d_fusion(x_rgb, x_t, f_upper)

        return res

if __name__ == '__main__':
    x = torch.randn(1, 64, 80, 80)
    model = CDC(arg=64, kernel_size=7, padding=3)
    y = model([x,x,x])
    print(y.shape)