import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import math
from mmseg.ops import resize
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import constant_init, kaiming_init


class AlignCM(nn.Module):
    def __init__(self, channels):
        super(AlignCM, self).__init__()
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

        x = nn.functional.grid_sample(x, weight, mode='bilinear', align_corners=False)

        return torch.cat((x, origin_x), dim=1)


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
    def __init__(self, channels, kernel_size, stride, ratio, group_channels=16,
                 padding=None, norm_cfg=None, act_cfg=None):
        super(RevolutionNaive, self).__init__()
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

        self.conv1 = ConvModule(
            in_channels=(self.channels * kernel_size * 2 +
                         self.channels * kernel_size * kernel_size // 32) * 1,
            out_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size // 4,
            kernel_size=1,
            padding=0,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size // 4,
            out_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size,
            kernel_size=1,
            stride=1,
            act_cfg=None)

        # self.conv3 = ConvModule(
        #     # in_channels=self.groups * (kernel_size * self.group_channels * 2 + kernel_size * kernel_size),
        #     in_channels=self.channels * (kernel_size * kernel_size),
        #     out_channels=self.channels // 8,
        #     kernel_size=1,
        #     padding=0,
        #     stride=1,
        #     conv_cfg=None,
        #     norm_cfg=dict(type='BN'),
        #     act_cfg=dict(type='ReLU'))
        # self.conv4 = ConvModule(
        #     in_channels=self.channels // 8,
        #     out_channels=self.channels,
        #     kernel_size=1,
        #     stride=1,
        #     act_cfg=None)

        # self.pos = Position(self.channels, norm_cfg, act_cfg)
        # self.alignfa = AlignFA(self.channels)
        self.init()

    def forward(self, inputs):
        batch_size, channels, width, height = inputs.shape
        h = math.ceil((width + 2 * self.padding - self.kernel_size + 1) / self.stride)
        w = math.ceil((height + 2 * self.padding - self.kernel_size + 1) / self.stride)

        x = self.unfold(inputs).view(
            batch_size, self.groups, self.group_channels, self.kernel_size, self.kernel_size, h, w)

        # max
        x1 = torch.max(x.view(batch_size, self.groups, -1, self.group_channels // 32, self.kernel_size,
                              self.kernel_size, h, w), dim=2).values.view(batch_size, -1, h, w)
        x2 = torch.max(x, dim=3).values.view(batch_size, -1, h, w)
        x3 = torch.max(x, dim=4).values.view(batch_size, -1, h, w)
        # x4 = torch.mean(x, dim=1, keepdim=True).view(batch_size, -1, x.shape[-2], x.shape[-1])
        # x5 = torch.mean(x, dim=3, keepdim=True).view(batch_size, -1, x.shape[-2], x.shape[-1])
        # x6 = torch.mean(x, dim=4, keepdim=True).view(batch_size, -1, x.shape[-2], x.shape[-1])
        weight = torch.cat((x1, x2, x3), dim=1)

        # channel weight
        # weight_channel = self.conv4(self.conv3(weight))
        # weight_channel = weight_channel.view(batch_size, self.groups, self.group_channels,
        #                                      self.kernel_size * self.kernel_size, 1, x.shape[-2], x.shape[-1])
        # weight_channel = torch.sigmoid(weight_channel) * 2

        weight = self.conv2(self.conv1(weight))
        weight = weight.view(batch_size, self.groups, 1, self.kernel_size * self.kernel_size,
                             self.new_size * self.new_size, h, w)
        weight = nn.functional.softmax(weight, dim=3)

        x = x.view(batch_size, self.groups, self.group_channels, self.kernel_size * self.kernel_size, 1, h, w)
        # x = x * weight_channel
        x = x * weight

        x = torch.sum(x, dim=3).view(batch_size, self.channels * self.new_size * self.new_size, h * w)
        # x = torch.mean(x, dim=3).view(batch_size, self.channels * self.new_size * self.new_size, h * w)

        x = nn.functional.fold(x, (self.new_size * h, self.new_size * w),
                               (self.new_size, self.new_size), stride=(self.new_size, self.new_size))

        # x = self.alignfa(x, inputs)
        # x = self.pos(x, inputs)
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
