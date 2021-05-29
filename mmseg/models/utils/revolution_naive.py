import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import math
from mmseg.ops import resize
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import constant_init, kaiming_init
from einops import rearrange, reduce, repeat
from mmseg.models.utils.involution_cuda import _involution_cuda


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x


class AlignCM(nn.Module):
    def __init__(self, channels):
        super(AlignCM, self).__init__()
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // 2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None)

        self.conv_t = ConvModule(
            in_channels=channels,
            out_channels=channels // 2,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv_x = ConvModule(
            in_channels=channels,
            out_channels=channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def forward(self, x, origin_x):
        x = self.conv_x(x)
        origin_x = self.conv_t(origin_x)

        target_x = resize(origin_x, size=x.size()[2:], mode='bilinear', align_corners=False)
        weight = torch.cat((x, target_x), dim=1)
        weight = self.conv2(self.conv1(weight)).permute(0, 2, 3, 1)
        weight = torch.tanh(weight)

        w_bias = torch.linspace(start=-1, end=-1, steps=x.size(-2)).cuda().unsqueeze(1).repeat(1, x.size(-1)).unsqueeze(0).unsqueeze(-1)
        h_bias = torch.linspace(start=-1, end=-1, steps=x.size(-1)).cuda().unsqueeze(0).repeat(x.size(-2), 1).unsqueeze(0).unsqueeze(-1)
        bias = torch.cat((w_bias, h_bias), dim=-1)
        weight = weight + bias

        x = nn.functional.grid_sample(x, weight, mode='bilinear', align_corners=False)

        return torch.cat((x, target_x), dim=1)


class AlignFA(nn.Module):
    def __init__(self, channels):
        super(AlignFA, self).__init__()
        self.conv1 = ConvModule(
            in_channels=channels * 2,
            out_channels=channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // 2,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None)

    def forward(self, x, origin_x):
        target_x = resize(origin_x, size=x.size()[2:], mode='bilinear', align_corners=False)
        weight = torch.cat((x, target_x), dim=1)
        weight = torch.tanh(self.conv2(self.conv1(weight)).permute(0, 2, 3, 1))

        w_bias = torch.linspace(start=-1, end=-1, steps=x.size(-2)).cuda().unsqueeze(1).repeat(1, x.size(-1)).unsqueeze(0).unsqueeze(-1)
        h_bias = torch.linspace(start=-1, end=-1, steps=x.size(-1)).cuda().unsqueeze(0).repeat(x.size(-2), 1).unsqueeze(0).unsqueeze(-1)
        bias = torch.cat((w_bias, h_bias, w_bias, h_bias), dim=-1)
        weight = weight + bias

        x = nn.functional.grid_sample(x, weight[:, :, :, :2], mode='bilinear', align_corners=True)
        target_x = nn.functional.grid_sample(target_x, weight[:, :, :, 2:], mode='bilinear', align_corners=True)

        return (x + target_x) / 2


class Position(nn.Module):
    def __init__(self, channels, norm_cfg, act_cfg):
        super(Position, self).__init__()
        self.conv1 = ConvModule(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, inputs, origin):
        origin = resize(origin, size=inputs.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((inputs, origin), dim=1)
        x = self.conv2(self.conv1(x))
        return x


class RevolutionNaive(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 ratio,
                 norm_cfg,
                 align_corners):
        super(RevolutionNaive, self).__init__()
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
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
            out_channels=channels * 2,
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
        x_h = self.conv0(self.pool_h(x)).view(n, c, -1, 1)
        x_w = self.conv0(self.pool_w(x)).view(n, c, 1, -1)
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


class RevolutionNaive0(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 ratio,
                 norm_cfg,
                 align_corners):
        super(RevolutionNaive0, self).__init__()
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.align_corners = align_corners
        self.ratio = ratio

        mip = max(64, channels // 4)

        self.conv1 = ConvModule(
            in_channels=channels * 2,
            out_channels=mip,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.act = Mish()

        self.conv_h = nn.Conv2d(mip, channels * 2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        out = resize(
            x,
            size=(x.shape[-2] * self.ratio, x.shape[-1] * self.ratio),
            mode='bilinear',
            align_corners=self.align_corners)

        n, c, h, w = x.size()
        x_h = torch.cat((
            self.pool_h(x),
            nn.functional.max_pool2d(out, (self.ratio, x.shape[-1] * self.ratio))), dim=1)
        x_w = torch.cat((
            self.pool_w(x),
            nn.functional.max_pool2d(out, (x.shape[-2] * self.ratio, self.ratio))), dim=1).permute(0, 1, 3, 2)

        x = torch.cat([x_h, x_w], dim=2)
        x = self.conv1(x)
        x = self.act(x)

        x_h, x_w = torch.split(x, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = rearrange(self.conv_h(x_h), 'b (c n) w h -> b c (w n) h', n=self.ratio).sigmoid()
        x_w = rearrange(self.conv_w(x_w), 'b (c n) w h -> b c w (h n)', n=self.ratio).sigmoid()

        x = out * x_w * x_h

        out = self.act(x + out)
        return out


class RevolutionNaive1(nn.Module):
    def __init__(self, channels, kernel_size, stride, ratio, group_channels=16,
                 padding=None, norm_cfg=None, act_cfg=None):
        super(RevolutionNaive1, self).__init__()
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

        # weight = x.permute(0, 5, 6, 1, 2, 3, 4).reshape(
        #     batch_size * h * w, self.groups * self.group_channels, self.kernel_size, self.kernel_size)
        # weight = self.activation(self.convn1(weight)).view(
        #     batch_size, h, w, -1, self.new_size ** 2, self.kernel_size ** 2)
        # weight = weight.permute(0, 3, 5, 4, 1, 2).reshape(batch_size, -1, h, w).contiguous()
        # weight = self.convn2(weight)

        # max
        # weight = torch.cat(
        #     [torch.max(x, dim=i).values.view(batch_size, -1, h, w) for i in [2, 3, 4]], dim=1)
        # weight = self.conv2(self.activation(self.conv1(weight)))

        weight = torch.cat(
            [torch.max(x, dim=i).values.view(batch_size, -1, h, w) for i in [2, 3, 4]], dim=1)
        weight = self.activation(self.conv1(weight))
        weight_new = rearrange(weight, 'b (c n1 n2) h w -> (b h w) c n1 n2', n1=self.new_size, n2=self.new_size)
        weight_new = self.conv2(weight_new)
        weight_new = rearrange(weight_new, '(b h w) c n1 n2 -> b (c n1 n2) h w', h=h, w=w)
        weight = weight * self.activation(self.bn(weight_new))
        weight = self.conv3(weight)
        # weight = rearrange(weight, '(b h w) (c k1 k2) n1 n2 -> b c 1 (k1 k2) (n1 n2) h w', b=batch_size, h=h, w=w,
        #                    k1=self.kernel_size, k2=self.kernel_size, n1=self.new_size, n2=self.new_size)

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
