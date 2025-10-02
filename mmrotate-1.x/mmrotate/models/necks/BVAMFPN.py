# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmrotate.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

import logging
from typing import Optional
import torch
from .window_att import Swin
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .transnext import get_relative_position_cpb ,BVAM
from .transnext import OverlapPatchEmbed

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=1024, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class Fpndecoder(nn.Module):
    def __init__(self, input_resolution, query_size, key_size, pretrain_size):
        super(Fpndecoder, self).__init__()
        self.BVAM = BVAM(dim=256, input_resolution=input_resolution)
        self.ffn = FFNLayer(d_model=256)
        # self.patch_embed = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=256, embed_dim=256)
        self.relative_pos_index, self.relative_coords_table = get_relative_position_cpb(
            query_size=to_2tuple(query_size),
            key_size=to_2tuple(key_size),
            pretrain_size=to_2tuple(pretrain_size))
        self.dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(256)
        self.activation = _get_activation_fn("relu")
        self.attention = AggregatedAttention(dim=256, input_resolution=input_resolution)

    def forward(self, q, kv):
        x = q
        b, n, h, w = q.shape
        pH = h//4
        pW = h//4
        q = self.dropout(self.activation(F.interpolate(q, size=[pH, pW], mode="bilinear", align_corners=False)))
        kv = self.dropout(self.activation(F.interpolate(kv, size=[pH,pW], mode="bilinear", align_corners=False)))
        q = q.view(b, n, -1).transpose(1,2)
        kv = kv.view(b, n, -1).transpose(1, 2)
        q = self.ffn(self.BVAM(q, pH, pW, self.relative_pos_index, self.relative_coords_table, kv))
        q = q.transpose(1, 2).reshape(-1, 256, pH, pW)
        q = F.interpolate(q, size=[h,w], mode="bilinear", align_corners=False)
        q = x + self.dropout(self.activation(q))
        return q

c3 = torch.rand(1, 256, 128, 128)
c4 = torch.rand(1, 256, 64, 64)
c5 = torch.rand(1, 256, 32, 32)


@MODELS.register_module()
class BVAMFPN(BaseModule):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.decoder_c5_c4 = Fpndecoder(input_resolution=to_2tuple(8),query_size=8, key_size=8, pretrain_size=8)
        self.decoder_c5_c3 = Fpndecoder(input_resolution=to_2tuple(8),query_size=8, key_size=8, pretrain_size=8)
        self.decoder_c4_c5 = Fpndecoder(input_resolution=to_2tuple(16),query_size=16, key_size=16, pretrain_size=16)
        self.decoder_c3_c5 = Fpndecoder(input_resolution=to_2tuple(32),query_size=32, key_size=32, pretrain_size=32)



    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        
        

        outs[0] = self.decoder_c3_c5(outs[0], outs[2])
        outs[1] = self.decoder_c4_c5(outs[1], outs[2])
        outs[2] = self.decoder_c5_c4(outs[2], outs[1])
        outs[2] = self.decoder_c5_c3(outs[2], outs[0])

        outs[0], outs[1], outs[2] = outs[0].contiguous(), outs[1].contiguous(), outs[2].contiguous()
        
        return tuple(outs)
