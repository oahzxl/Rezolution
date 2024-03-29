import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import math
from mmseg.ops import resize
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import constant_init, kaiming_init
from einops import rearrange
from mmcv.ops.carafe import CARAFEPack


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x


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


class RevolutionNaive(nn.Module):
    def __init__(self,
                 channels,
                 mid_channels,
                 kernel_size,
                 stride,
                 ratio,
                 norm_cfg,
                 align_corners):
        super(RevolutionNaive, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.align_corners = align_corners
        self.ratio = ratio
        self.mid = channels // 2 if channels // 2 % 2 == 0 else channels // 2 + 1
        self.query_conv = nn.Conv1d(in_channels=self.mid, out_channels=self.mid, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=self.mid, out_channels=self.mid, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=self.mid, out_channels=self.mid, kernel_size=1)

        # self.query_self = nn.Conv1d(in_channels=self.mid, out_channels=self.mid // 2, kernel_size=1)
        # self.key_self = nn.Conv1d(in_channels=self.mid, out_channels=self.mid // 2, kernel_size=1)
        # self.value_self = nn.Conv1d(in_channels=self.mid, out_channels=self.mid, kernel_size=1)

        self.pe = torch.zeros(500, self.mid)
        position = torch.arange(0, 500).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.mid, 2) *
                             -(math.log(10000.0) / self.mid))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).permute(0, 2, 1).cuda()

        self.conv_expand_h = nn.Conv1d(
            in_channels=self.mid,
            out_channels=self.mid * self.ratio,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1)
        self.conv_expand_w = nn.Conv1d(
            in_channels=self.mid,
            out_channels=self.mid * self.ratio,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1)
        self.conv_compress = nn.Conv1d(
            in_channels=channels,
            out_channels=self.mid,
            kernel_size=1,
            padding=0,
            stride=1)

        self.act = nn.ReLU()

        self.conv_back_w = nn.Conv1d(self.mid, channels, kernel_size=1, stride=1, padding=0)
        self.conv_back_h = nn.Conv1d(self.mid, channels, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):

        out = resize(
            x,
            size=(x.shape[-2] * self.ratio, x.shape[-1] * self.ratio),
            mode='bilinear',
            align_corners=self.align_corners)

        n, c, h, w = x.size()

        x_h = self.pool_h(x).squeeze(-1)
        x_w = self.pool_w(x).squeeze(-2)
        x = torch.cat([x_h,x_w], dim=2)
        x = self.act(self.conv_compress(x))
        x_h, x_w = torch.split(x, [h, w], dim=2)

        x_h = self.act(self.conv_expand_h(x_h))
        x_w = self.act(self.conv_expand_w(x_w))
        if self.ratio != 1:
            x_h = rearrange(x_h, 'b (c n) x -> b c (x n)', n=self.ratio)
            x_w = rearrange(x_w, 'b (c n) x -> b c (x n)', n=self.ratio)

        x_h = x_h + self.pe[:, :, :x_h.size(-1)]

        # x_h_query = self.query_self(x_h)
        # x_h_key = self.key_self(x_h)
        # x_h_value = self.value_self(x_h)
        # x_h = torch.bmm(x_h_query.permute(0, 2, 1), x_h_key)
        # x_h = torch.bmm(self.softmax(x_h), x_h_value.permute(0, 2, 1)).permute(0, 2, 1)
        # x_h = self.act(x_h)

        x_h_key = self.key_conv(x_h)
        x_h_value = self.value_conv(x_h)

        x_w = x_w + self.pe[:, :, :x_w.size(-1)]

        # x_w_query = self.query_self(x_w)
        # x_w_key = self.key_self(x_w)
        # x_w_value = self.value_self(x_w)
        # x_w = torch.bmm(x_w_query.permute(0, 2, 1), x_w_key)
        # x_w = torch.bmm(self.softmax(x_w), x_w_value.permute(0, 2, 1)).permute(0, 2, 1)
        # x_w = self.act(x_w)

        x_w_key = self.key_conv(x_w)
        x_w_value = self.value_conv(x_w)

        for i in range(1):
            o_h = self.pool_h(out).squeeze(-1)
            o_w = self.pool_w(out).squeeze(-2)

            x = torch.cat([o_h, o_w], dim=2)
            x = self.act(self.conv_compress(x))
            o_h, o_w = torch.split(x, [h * self.ratio, w * self.ratio], dim=2)

            o_h = o_h + self.pe[:, :, :o_h.size(-1)]
            o_h_query = self.query_conv(o_h)
            o_w = o_w + self.pe[:, :, :o_w.size(-1)]
            o_w_query = self.query_conv(o_w)

            h_attn = torch.bmm(o_h_query.permute(0, 2, 1), x_h_key)
            h_attn = torch.bmm(self.softmax(h_attn), x_h_value.permute(0, 2, 1)).permute(0, 2, 1)

            w_attn = torch.bmm(o_w_query.permute(0, 2, 1), x_w_key)
            w_attn = torch.bmm(self.softmax(w_attn), x_w_value.permute(0, 2, 1)).permute(0, 2, 1)

            # x = self.conv_back(o_w.unsqueeze(-2) + o_h.unsqueeze(-1))
            x = self.conv_back_w(w_attn).unsqueeze(-2) + self.conv_back_h(h_attn).unsqueeze(-1)

            out = out + self.gamma * x
            out = self.act(out)

            return out


class RevolutionNaive22(nn.Module):
    def __init__(self,
                 channels,
                 mid_channels,
                 kernel_size,
                 stride,
                 ratio,
                 norm_cfg,
                 align_corners):
        super(RevolutionNaive22, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.align_corners = align_corners
        self.ratio = ratio
        self.mid = mid_channels

        self.conv_expand = nn.Conv1d(
            in_channels=self.mid,
            out_channels=self.mid * self.ratio,
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv_compress = ConvModule(
            in_channels=channels,
            out_channels=self.mid,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(self.mid * 2, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mid * 2, channels, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):

        out = resize(
            x,
            size=(x.shape[-2] * self.ratio, x.shape[-1] * self.ratio),
            mode='bilinear',
            align_corners=self.align_corners)

        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x = torch.cat([x_h, x_w], dim=2)
        x = self.act(self.conv_compress(x))
        x_h, x_w = torch.split(x, [h, w], dim=2)
        x_h = self.act(self.conv_expand(x_h.squeeze(-1))).unsqueeze(-1)
        if self.ratio != 1:
            x_h = rearrange(x_h, 'b (c n) w h -> b c (w n) h', n=self.ratio)
        x_w = self.act(self.conv_expand(x_w.squeeze(-1))).unsqueeze(-1)
        if self.ratio != 1:
            x_w = rearrange(x_w, 'b (c n) w h -> b c (w n) h', n=self.ratio)

        for i in range(2):
            o_h = self.pool_h(out)
            o_w = self.pool_w(out).permute(0, 1, 3, 2)

            x = torch.cat([o_h, o_w], dim=2)
            x = self.act(self.conv_compress(x))
            o_h, o_w = torch.split(x, [h * self.ratio, w * self.ratio], dim=2)

            o_h = torch.cat([x_h, o_h], dim=1)
            o_h = self.conv_h(o_h)

            o_w = torch.cat([x_w, o_w], dim=1)
            o_w = self.conv_w(o_w)
            o_w = o_w.permute(0, 1, 3, 2)

            x = o_w * o_h
            return self.act(self.gamma * x + out)


class RevolutionNaive4(nn.Module):
    def __init__(self,
                 channels,
                 mid_channels,
                 kernel_size,
                 stride,
                 ratio,
                 norm_cfg,
                 align_corners):
        super(RevolutionNaive4, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.align_corners = align_corners
        self.ratio = ratio
        self.mid = mid_channels

        self.conv_expand = nn.Conv1d(
            in_channels=self.mid,
            out_channels=self.mid * self.ratio,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            stride=1)
        self.conv_compress = ConvModule(
            in_channels=channels,
            out_channels=self.mid,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv_gate = ConvModule(
            in_channels=self.mid,
            out_channels=self.mid,
            kernel_size=1,
            padding=0,
            stride=1,
            act_cfg=None)

        # self.act = Mish()
        self.act = h_swish()

        self.conv_h = nn.Conv2d(self.mid, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mid, channels, kernel_size=1, stride=1, padding=0)

        # self.conv_h = nn.Conv2d(self.mid * 2, channels, kernel_size=1, stride=1, padding=0)
        # self.conv_w = nn.Conv2d(self.mid * 2, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        out = resize(
            x,
            size=(x.shape[-2] * self.ratio, x.shape[-1] * self.ratio),
            mode='bilinear',
            align_corners=self.align_corners)

        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        o_h = self.pool_h(out)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        o_w = self.pool_w(out).permute(0, 1, 3, 2)
        x = torch.cat([x_h, o_h, x_w, o_w], dim=2)
        x = self.act(self.conv_compress(x))
        x_h, o_h, x_w, o_w = torch.split(x, [h, h * self.ratio, w, w * self.ratio], dim=2)

        x_h = self.act(self.conv_expand(x_h.squeeze(-1))).unsqueeze(-1)
        if self.ratio != 1:
            x_h = rearrange(x_h, 'b (c n) w h -> b c (w n) h', n=self.ratio)
        x_g = self.conv_gate(x_h * o_h).sigmoid()
        x_h = x_g * x_h + (1 - x_g) * o_h
        x_h = self.conv_h(x_h)

        x_w = self.act(self.conv_expand(x_w.squeeze(-1))).unsqueeze(-1)
        if self.ratio != 1:
            x_w = rearrange(x_w, 'b (c n) w h -> b c (w n) h', n=self.ratio)
        x_g = self.conv_gate(x_w * o_w).sigmoid()
        x_w = x_g * x_w + (1 - x_g) * o_w
        x_w = self.conv_w(x_w)
        x_w = x_w.permute(0, 1, 3, 2)

        # x_h = self.conv_expand(x_h.squeeze(-1)).view(n, self.mid, -1, 1)
        # x_h = self.act(x_h)
        # x_h = torch.cat([x_h, o_h], dim=1)
        # x_h = self.conv_h(x_h)
        #
        # x_w = self.conv_expand(x_w.squeeze(-1)).view(n, self.mid, -1, 1)
        # x_w = self.act(x_w)
        # x_w = torch.cat([x_w, o_w], dim=1)
        # x_w = self.conv_w(x_w)
        # x_w = x_w.permute(0, 1, 3, 2)

        x = out * x_w.sigmoid() * x_h.sigmoid()
        out = self.act(x + out)
        return out


class RevolutionNaive1(nn.Module):
    def __init__(self,
                 channels,
                 mid_channels,
                 kernel_size,
                 stride,
                 ratio,
                 norm_cfg,
                 align_corners):
        super(RevolutionNaive1, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.align_corners = align_corners
        self.ratio = ratio

        mip = max(64, channels // 4)

        self.conv1 = ConvModule(
            in_channels=channels*2,
            out_channels=mip,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv0 = ConvModule(
            in_channels=channels,
            out_channels=channels * ratio,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.act = Mish()

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        out = resize(
            x,
            size=(x.shape[-2] * self.ratio, x.shape[-1] * self.ratio),
            mode='bilinear',
            align_corners=self.align_corners)

        n, c, h, w = x.size()
        x_h = self.conv0(self.pool_h(x))
        if self.ratio != 1:
            x_h = rearrange(x_h, 'b (c n) w h -> b c (w n) h', n=self.ratio)
        x_w = self.conv0(self.pool_w(x))
        if self.ratio != 1:
            x_w = rearrange(x_w, 'b (c n) w h -> b c w (h n)', n=self.ratio)
        x_h = torch.cat((x_h, self.pool_h(out)), dim=1)
        x_w = torch.cat((x_w, self.pool_w(out)), dim=1).permute(0, 1, 3, 2)

        x = torch.cat([x_h, x_w], dim=2)
        x = self.conv1(x)
        x = self.act(x)

        x_h, x_w = torch.split(x, [h * self.ratio, w * self.ratio], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w)

        x = out * x_w.sigmoid() * x_h.sigmoid()

        out = self.act(x + out)
        return out


class RevolutionNaiveOld(nn.Module):
    """ first method, same as CARAFE """
    def __init__(self, channels, kernel_size, stride, ratio, group_channels=16,
                 padding=None, norm_cfg=None, act_cfg=None):
        super(RevolutionNaiveOld, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.ratio = ratio
        self.new_size = int(ratio * stride)
        self.channels = channels
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels

        if padding:
            self.padding = padding
        else:
            self.padding = (kernel_size - 1) // 2
        self.unfold = torch.nn.Unfold(kernel_size, 1, self.padding, stride)

        # self.conv1 = ConvModule(
        #     in_channels=(self.channels * kernel_size * 2 +
        #                  self.groups * kernel_size * kernel_size) * 1,
        #     out_channels=self.channels * self.new_size * self.new_size // 4,
        #     kernel_size=1,
        #     padding=0,
        #     stride=1,
        #     norm_cfg=norm_cfg,
        #     act_cfg=None)
        # self.conv2 = ConvModule(
        #     in_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size // 4,
        #     out_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size,
        #     kernel_size=1,
        #     stride=1,
        #     act_cfg=None)

        self.conv1 = ConvModule(
            in_channels=(self.channels * kernel_size * 2 +
                         self.groups * kernel_size * kernel_size) * 1,
            out_channels=self.channels * self.new_size * self.new_size // 4,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv2 = ConvModule(
            in_channels=self.channels // 4,
            out_channels=self.channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv3 = ConvModule(
            in_channels=self.channels // 4,
            out_channels=self.groups * kernel_size * kernel_size,
            kernel_size=1,
            stride=1,
            act_cfg=None)
        self.bn = nn.SyncBatchNorm(self.channels * self.new_size * self.new_size // 4)

        # self.convn1 = ConvModule(
        #     in_channels=self.channels,
        #     out_channels=self.groups * self.new_size * self.new_size // 4,
        #     kernel_size=3,
        #     padding=1,
        #     stride=1,
        #     norm_cfg=norm_cfg,
        #     act_cfg=None)
        # self.convn2 = ConvModule(
        #     in_channels=(self.groups * self.new_size * self.new_size // 4) * kernel_size * kernel_size,
        #     out_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size,
        #     kernel_size=1,
        #     stride=1,
        #     act_cfg=None)

        self.activation = Mish()
        self.init()

    def forward(self, inputs):
        batch_size, channels, width, height = inputs.shape
        h = math.ceil((width + 2 * self.padding - self.kernel_size + 1) / self.stride)
        w = math.ceil((height + 2 * self.padding - self.kernel_size + 1) / self.stride)
        x = self.unfold(inputs).view(
            batch_size, self.groups, self.group_channels, self.kernel_size, self.kernel_size, h, w)

        weight = torch.cat(
            [torch.max(x, dim=i).values.view(batch_size, -1, h, w) for i in [2, 3, 4]], dim=1)
        weight = self.activation(self.conv1(weight))
        weight_new = rearrange(weight, 'b (c n1 n2) h w -> (b h w) c n1 n2', n1=self.new_size, n2=self.new_size)
        weight_new = self.conv2(weight_new)
        weight_new = rearrange(weight_new, '(b h w) c n1 n2 -> b (c n1 n2) h w', h=h, w=w)
        weight = weight * self.activation(self.bn(weight_new))
        weight = self.conv3(weight)

        weight = weight.view(batch_size, self.groups, 1, self.kernel_size * self.kernel_size,
                             self.new_size * self.new_size, h, w)
        weight = nn.functional.softmax(weight, dim=3)

        x = x.view(batch_size, self.groups, self.group_channels, self.kernel_size * self.kernel_size, 1, h, w)
        x = x * weight
        x = torch.sum(x, dim=3).view(batch_size, self.channels * self.new_size * self.new_size, h * w)
        x = nn.functional.fold(x, (self.new_size * h, self.new_size * w),
                               (self.new_size, self.new_size), stride=(self.new_size, self.new_size))
        return x

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


class ResizeCat(nn.Module):
    def __init__(self, in_channels, mid_channels, in_index, norm_cfg, act_cfg):
        super(ResizeCat, self).__init__()
        self.resize = nn.ModuleList()
        for i in in_index:
            self.resize.append(RevolutionNaive(
                channels=in_channels[i],
                mid_channels=in_channels[i] // 2,
                align_corners=False,
                kernel_size=[13, 9, 5, 3][i],
                stride=1,
                ratio=2 ** i,
                norm_cfg=norm_cfg))
            # self.resize.append(CARAFEPack(
            #     in_channels[i],
            #     2 ** i,
            #     up_kernel=5,
            #     up_group=1,
            #     encoder_kernel=3,
            #     encoder_dilation=1,
            #     compressed_channels=64))

            # Carafe: flops: 430.35   para: 13.28
            # attn ca: flops: 442.71 para: 12.39
            # base: 426.44 12.29


    def forward(self, x):
        # inputs = [x[i] for i in range(4)]
        # upsampled_inputs = [
        #     self.resize[i](inputs[i]) for i in range(4)
        # ]
        upsampled_inputs = []
        for i in range(4):
            inputs = self.resize[i](x[i])
            if inputs.shape[2:] != x[0].shape[2:]:
                inputs = resize(
                    inputs,
                    size=x[0].shape[2:],
                    mode='bilinear',
                    align_corners=False)
            upsampled_inputs.append(inputs)
        inputs = torch.cat(upsampled_inputs, dim=1)
        return inputs


"""
x        : B,C,W  ,H
unfold_x : B,C,k,k,W/k,H/k
weight   : B,C,k2,n,n,W/k,H/k
x'       : B,C,k2,1,1,W/k,H/k
x+w      : B,C,n,n,W/k,H/k
final    : B,C,W/k*n,H/k*n
target   : B,C,W/a,H/a
"""

"""
# test permute
a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8],
                  [9, 10, 11, 12], [13, 14, 15, 16]])
b = torch.tensor([[[[1, 3], [9, 11]], [[2, 4], [10, 12]]],
                  [[[5, 7], [13, 15]], [[6, 8], [14, 16]]]])

b = b.permute(2, 0, 3, 1).contiguous().view(4, 4)
"""
