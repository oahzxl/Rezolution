import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import math
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import constant_init, kaiming_init


class RevolutionNaive(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 ratio,
                 groups=16):
        super(RevolutionNaive, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.ratio = ratio
        self.new_size = int(ratio * stride)
        self.channels = channels
        self.group_channels = groups
        self.groups = self.channels // self.group_channels

        self.unfold = torch.nn.Unfold(
            kernel_size, 1, (kernel_size - 1) // 2, stride)
        self.conv1 = ConvModule(
            in_channels=self.groups * (kernel_size * self.group_channels * 2 + kernel_size * kernel_size),
            out_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size // 10,
            kernel_size=1,
            padding=0,
            stride=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size // 10,
            out_channels=self.groups * self.new_size * self.new_size * kernel_size * kernel_size,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

        self.init()

    def forward(self, x, target=None):
        batch_size, channels, width, height = x.shape

        x = self.unfold(x)
        x = x.view(batch_size, self.groups, self.group_channels, self.kernel_size, self.kernel_size,
                   math.ceil(width / self.stride), -1)

        x1 = torch.max(x, dim=2, keepdim=True)
        x1 = x1.values.view(batch_size, -1, x.shape[-2], x.shape[-1])
        x2 = torch.max(x, dim=3, keepdim=True)
        x2 = x2.values.view(batch_size, -1, x.shape[-2], x.shape[-1])
        x3 = torch.max(x, dim=4, keepdim=True)
        x3 = x3.values.view(batch_size, -1, x.shape[-2], x.shape[-1])
        weight = torch.cat((x1, x2, x3), dim=1)

        weight = self.conv2(self.conv1(weight))
        weight = weight.view(batch_size, self.groups, 1,
                             self.kernel_size * self.kernel_size,
                             self.new_size, self.new_size,
                             weight.shape[2], weight.shape[3])

        # mask = weight.view(batch_size, self.channels, -1,
        #                    weight.shape[-2], weight.shape[-1])
        # mask = torch.sum(mask, dim=2).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        # mask = torch.sigmoid(mask)
        # weight = nn.functional.softmax(weight, dim=2) * mask
        weight = nn.functional.softmax(weight, dim=2)

        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1,
                   x.shape[-2], x.shape[-1]).unsqueeze(-3).unsqueeze(-3)
        x = x * weight
        x = torch.mean(x, dim=3).view(x.shape[0], -1, x.shape[4], x.shape[5], x.shape[6], x.shape[7])

        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(
            batch_size, self.channels,
            x.shape[2] * x.shape[3],
            x.shape[4] * x.shape[5])

        if target and x.shape != target:
            # print(x.shape, target)
            x = x[:, :, :target[-2], :target[-1]]
        elif self.ratio == 1 and x.shape != (batch_size, channels, width, height):
            x = x[:batch_size, :channels, :width, :height]

        return x

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


# class RevolutionNaive(nn.Module):
#     def __init__(self,
#                  channels,
#                  kernel_size,
#                  stride,
#                  ratio):
#         super(RevolutionNaive, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.ratio = ratio
#         self.new_size = int(ratio * stride)
#         self.channels = channels
#
#         self.unfold = torch.nn.Unfold(
#             kernel_size, 1, (kernel_size - 1) // 2, stride)
#         self.conv1 = ConvModule(
#             in_channels=kernel_size * channels * 2 + kernel_size * kernel_size,
#             out_channels=channels * self.new_size * self.new_size * kernel_size * kernel_size,
#             kernel_size=1,
#             padding=0,
#             stride=1,
#             conv_cfg=None,
#             norm_cfg=dict(type='BN'),
#             act_cfg=dict(type='ReLU'))
#
#     def forward(self, x):
#         batch_size, channels, width, height = x.shape
#
#         x = self.unfold(x)
#         x = x.view(batch_size, self.channels, self.kernel_size, self.kernel_size,
#                    math.ceil(width / self.stride), -1)
#
#         x1 = torch.max(x, dim=1, keepdim=True)
#         x1 = x1.values.view(batch_size, -1, x.shape[-2], x.shape[-1])
#         x2 = torch.max(x, dim=2, keepdim=True)
#         x2 = x2.values.view(batch_size, -1, x.shape[-2], x.shape[-1])
#         x3 = torch.max(x, dim=3, keepdim=True)
#         x3 = x3.values.view(batch_size, -1, x.shape[-2], x.shape[-1])
#
#         weight = torch.cat((x1, x2, x3), dim=1)
#         weight = self.conv1(weight)
#         weight = weight.view(batch_size, self.channels,
#                              self.kernel_size * self.kernel_size,
#                              self.new_size, self.new_size,
#                              weight.shape[2], weight.shape[3])
#
#         x = x.view(x.shape[0], x.shape[1], -1,
#                    x.shape[-2], x.shape[-1]).unsqueeze(3).unsqueeze(3)
#         x = x * weight
#         x = torch.mean(x, dim=2)
#
#         x = x.permute(0, 1, 4, 2, 5, 3)
#         x = x.reshape(
#             batch_size, self.channels,
#             x.shape[2] * x.shape[3],
#             x.shape[4] * x.shape[5])
#
#         if self.ratio == 1 and x.shape != (batch_size, channels, width, height):
#             x = x[:batch_size, :channels, :width, :height]
#
#         return x
#
#     def init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                 constant_init(m, 1)

# B = 2
# W = 64
# H = 64
# C = 32
# kernel_size = 4
#
# new_size = 2
# stride = 4
#
# unfold = torch.nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)
# conv = torch.nn.Conv2d(kernel_size * C * 2 + kernel_size * kernel_size,
#                        C * new_size * new_size * kernel_size * kernel_size,
#                        kernel_size=1, padding=0, stride=1)
#
# x = torch.ones((B, C, W, H))
# x_shape = x.shape
# unfold_x = unfold(x)
# unfold_x = unfold_x.view(B, C, kernel_size, kernel_size,
#                          int(unfold_x.shape[-1] ** 0.5),
#                          int(unfold_x.shape[-1] ** 0.5))
# unfold_x_shape = unfold_x.shape
#
# x1 = torch.max(unfold_x, dim=1, keepdim=True)
# x1 = x1.values.view(B, -1, unfold_x.shape[-1], unfold_x.shape[-1])
# x2 = torch.max(unfold_x, dim=2, keepdim=True)
# x2 = x2.values.view(B, -1, unfold_x.shape[-1], unfold_x.shape[-1])
# x3 = torch.max(unfold_x, dim=3, keepdim=True)
# x3 = x3.values.view(B, -1, unfold_x.shape[-1], unfold_x.shape[-1])
#
# weight = torch.cat((x1, x2, x3), dim=1)
# weight = conv(weight)
# weight = weight.view(B, C, kernel_size * kernel_size,
#                      new_size, new_size, weight.shape[2], weight.shape[3])
#
# x = unfold_x.view(unfold_x.shape[0], unfold_x.shape[1], -1,
#                   unfold_x.shape[-2], unfold_x.shape[-1]).unsqueeze(3).unsqueeze(3)
# x = x * weight
# x = torch.mean(x, dim=2)
#
# x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C, int(W / stride * new_size), int(W / stride * new_size))
# print(1)


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
