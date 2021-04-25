import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

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
                 act_cfg, align_corners):
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
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),

                    # origin
                    # RevolutionNaive(
                    #     channels=self.channels,
                    #     kernel_size={1: 38, 2: 19, 3: 12, 6: 8}[pool_scale],
                    #     stride={1: 32, 2: 16, 3: 10, 6: 6}[pool_scale],
                    #     ratio={1: 1.0 / 32, 2: 1.0 / 16, 3: 1.0 / 10, 6: 1.0 / 6}[pool_scale],
                    #     group_channels=channels // 4),
                    
                    # 1rev
                    RevolutionNaive(
                        channels=self.channels,
                        kernel_size={1: 20, 2: 10, 3: 7, 6: 4}[pool_scale],
                        stride={1: 16, 2: 8, 3: 5, 6: 3}[pool_scale],
                        ratio={1: 1.0/16, 2: 1.0/8, 3: 1.0/5, 6: 1.0/3}[pool_scale],
                        group_channels=channels // 16),

                    # 2rev
                    # RevolutionNaive(
                    #     channels=self.channels,
                    #     kernel_size={1: 12, 2: 8, 3: 6, 6: 3}[pool_scale],
                    #     stride={1: 8, 2: 6, 3: 4, 6: 2}[pool_scale],
                    #     ratio={1: 1.0 / 8, 2: 1.0 / 6, 3: 1.0 / 4, 6: 1.0 / 2}[pool_scale],
                    #     group_channels=channels // 16),
                    # RevolutionNaive(
                    #     channels=self.channels,
                    #     kernel_size={1: 12, 2: 8, 3: 6, 6: 3}[pool_scale],
                    #     stride={1: 8, 2: 6, 3: 4, 6: 2}[pool_scale],
                    #     ratio={1: 1.0 / 8, 2: 1.0 / 6, 3: 1.0 / 4, 6: 1.0 / 2}[pool_scale],
                    #     group_channels=channels // 16),

                    ConvModule(
                        self.channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ))

    def forward(self, x):
        """Forward function."""

        ppm_outs = []
        for n, ppm in enumerate(self.ppm):
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class RevPSPDownHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(RevPSPDownHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.cls_seg(output)
        return output
