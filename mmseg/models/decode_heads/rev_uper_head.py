import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops.carafe import CARAFEPack
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils.revolution_naive import RevolutionNaive


class PPM(nn.Module):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, revolution):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.ppm = nn.ModuleList()
        for pool_scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ))

        # small kernel
        self.revolution = revolution
        # self.revolution = RevolutionNaive(
            # channels=self.channels,
            # kernel_size=3,
            # stride=1,
            # ratio=2,
            # # group_channels=64,
            # norm_cfg=self.norm_cfg,
            # # act_cfg=self.act_cfg
            # align_corners=self.align_corners)

    def forward(self, x):
        """Forward function."""

        ppm_outs = []
        for n, ppm in enumerate(self.ppm):
            ppm_out = ppm(x)
            ppm_out = resize(
                ppm_out,
                size=(x.size(-2) // 2, x.size(-1) // 2),
                mode='bilinear',
                align_corners=self.align_corners)
            upsampled_ppm_out = self.revolution(ppm_out)
            if upsampled_ppm_out.shape[2:] != x.shape[2:]:
                upsampled_ppm_out = resize(
                    upsampled_ppm_out,
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class RevUPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(RevUPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.revolution2x = RevolutionNaive(
            channels=self.channels,
            kernel_size=3,
            stride=1,
            ratio=2,
            # group_channels=64,
            norm_cfg=self.norm_cfg,
            # act_cfg=self.act_cfg
            align_corners=self.align_corners
        )
        # self.carafe = CARAFEPack(
        #     self.channels,
        #     2,
        #     up_kernel=5,
        #     up_group=1,
        #     encoder_kernel=3,
        #     encoder_dilation=1,
        #     compressed_channels=64)

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
            revolution=self.revolution2x
            # revolution=self.carafe,
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            new = self.revolution2x(laterals[i])
            # new = self.carafe(laterals[i])
            if new.shape[2:] != laterals[i - 1].shape[2:]:
                new = resize(
                    new,
                    size=laterals[i - 1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            laterals[i - 1] += new

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)

        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output
