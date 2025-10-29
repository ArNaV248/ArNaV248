#!/usr/bin/env python3
"""
D-FINE Label Studio ML Backend
===============================

Label Studio ML Backend for D-FINE Object Detection
Workflow: Label Studio → S3 → D-FINE Inference → Label Studio UI

Features:
- Loads images from Amazon S3
- Runs D-FINE inference
- Returns predictions to Label Studio for visualization
- No file outputs - everything rendered in Label Studio UI

Usage:
    label-studio-ml start dfine_labelstudio_backend --port 9090
"""

from typing import List, Dict, Optional
import logging
import os
import re
import math
import copy
import functools
from pathlib import Path
from collections import OrderedDict

import base64
import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse

# Label Studio
from label_studio_ml.model import LabelStudioMLBase

# Torchvision
import torchvision.transforms as T
from torchvision.ops import nms, box_convert, box_area

# ==============================================================================
# CONFIGURATION
# ==============================================================================

LABEL_STUDIO_URL = 'http://127.0.0.1:8080/'
LABEL_STUDIO_API_KEY = '100926765214fd262cf9243b620e2ec72c9219ad'

# Model Configuration
MODEL_PATH = os.getenv('MODEL_PATH', "/app/models/model_2.pt")  # Docker path, override with env var
MODEL_NAME = "l"  # DFINE-Large
INPUT_SIZE = (1280, 448)  # width, height
NUM_CLASSES = 35

CLASS_NAMES = [
    "0_Adhering_Skin", "1_Blanched", "2_Broken_Blanched", "3_Mission", "4_Carmel",
    "5_Chip_Scratch_1_4", "6_Discolor", "7_Doubles", "8_Foreign_Material_Hull",
    "9_NonPareil", "10_OD_Brownspot", "11_SD_Insect_Damage", "12_Specks",
    "13_Split_Broken", "14_LooseSkin_Dust_Particle", "15_OD_Gummy", "16_SD_Pinhole",
    "17_Inshell", "18_Embeddedshell", "19_FM_Other", "20_FM_Rock_Dirtball",
    "21_FM_Pistachio", "22_FM_Walnut", "23_FM_Plastic", "24_FM_Metal",
    "25_FM_Glass", "26_SD_Mold", "27_SD_Decay", "28_SD_Other", "29_SD_Frass",
    "30_OD_Shrivel", "31_OD_Discolor", "32_Chip_Scratch_1_8", "33_California",
    "34_Fold_Deformed",
]

# S3 Configuration
FILEPATH_REGEX = re.compile(r"^s3://([^/]+)/(.+)$")
s3 = boto3.client('s3')

logger = logging.getLogger(__name__)

# ==============================================================================
# EMBEDDED D-FINE SOURCE CODE - UTILITIES
# ==============================================================================

def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def bias_init_with_prob(prior_prob=0.01):
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def get_activation(act: str, inpace: bool = True):
    if act is None:
        return nn.Identity()
    elif isinstance(act, nn.Module):
        return act

    act = act.lower()
    if act == "silu" or act == "swish":
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unknown activation: {act}")

    if hasattr(m, "inplace"):
        m.inplace = inpace
    return m


def distance2bbox(points, distance, reg_scale):
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return box_xyxy_to_cxcywh(bboxes)


def weighting_function(reg_max, up, reg_scale, deploy=False):
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.cat(values, 0)


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
):
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5
            ).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)
            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)


# ==============================================================================
# FROZEN BATCH NORM
# ==============================================================================

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# ==============================================================================
# HGNETV2 BACKBONE (Embedded)
# ==============================================================================

class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1, padding="", use_act=True, use_lab=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == "same":
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False),
            )
        else:
            self.conv = nn.Conv2d(
                in_chs, out_chs, kernel_size, stride,
                padding=(kernel_size - 1) // 2, groups=groups, bias=False,
            )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.ReLU() if self.use_act else nn.Identity()
        self.lab = LearnableAffineBlock() if (self.use_act and self.use_lab) else nn.Identity()

    def forward(self, x):
        return self.lab(self.act(self.bn(self.conv(x))))


class LightConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, groups=1, use_lab=False):
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=kernel_size, groups=out_chs, use_act=True, use_lab=use_lab)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        return self.stem4(x)


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False, light_block=False, use_lab=False, agg="ese", drop_path=0.0):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(LightConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, kernel_size=kernel_size, use_lab=use_lab))
            else:
                self.layers.append(ConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, kernel_size=kernel_size, stride=1, use_lab=use_lab))

        total_chs = in_chs + layer_num * mid_chs
        if agg == "se":
            aggregation_squeeze_conv = ConvBNAct(total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab)
            aggregation_excitation_conv = ConvBNAct(out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
            self.aggregation = nn.Sequential(aggregation_squeeze_conv, aggregation_excitation_conv)
        else:
            aggregation_conv = ConvBNAct(total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg="se", drop_path=0.0):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(in_chs, in_chs, kernel_size=3, stride=2, groups=in_chs, use_act=False, use_lab=use_lab)
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs, mid_chs, out_chs, layer_num,
                    residual=False if i == 0 else True, kernel_size=kernel_size,
                    light_block=light_block, use_lab=use_lab, agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        return self.blocks(self.downsample(x))


class HGNetv2(nn.Module):
    arch_configs = {
        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
        },
    }

    def __init__(self, name, use_lab=False, return_idx=[1, 2, 3], freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=False, local_model_dir="weight/hgnetv2/"):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]["stem_channels"]
        stage_config = self.arch_configs[name]["stage_config"]

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        self.stem = StemBlock(in_chs=stem_channels[0], mid_chs=stem_channels[1], out_chs=stem_channels[2], use_lab=use_lab)

        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]
            self.stages.append(
                HG_Stage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, light_block, kernel_size, use_lab)
            )

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


# ==============================================================================
# HYBRID ENCODER (Embedded) - Including all remaining components
# ==============================================================================

class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu", bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=get_activation(act)) for _ in range(num_blocks)]
        )
        self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


class HybridEncoder(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.0, enc_act="gelu", use_encoder_idx=[2], num_encoder_layers=1, pe_temperature=10000, expansion=1.0, depth_mult=1.0, act="silu", eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(
                OrderedDict([
                    ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ("norm", nn.BatchNorm2d(hidden_dim)),
                ])
            )
            self.input_proj.append(proj)

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=enc_act)
        self.encoder = nn.ModuleList([TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))])

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult)))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2)))
            self.pan_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult)))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)
                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


# ==============================================================================
# DFINE DECODER AND TRANSFORMER COMPONENTS
# ==============================================================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, method="default", offset_scale=0.5):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, value: torch.Tensor, value_spatial_shapes: List[int]):
        bs, Len_q = query.shape[:2]
        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.")

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)
        return output


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0, activation="relu", n_levels=4, n_points=4, cross_attn_method="default", layer_scale=None):
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes)
        target = self.gateway(target, self.dropout2(target2))
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))
        return target


class Integral(nn.Module):
    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up, eval_idx=-1, layer_scale=2):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def forward(self, target, ref_points_unact, memory, spatial_shapes, bbox_head, score_head, query_pos_head, pre_bbox_head, integral, up, reg_scale, attn_mask=None, memory_mask=None, dn_meta=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class DFINETransformer(nn.Module):
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300, feat_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], num_levels=3, num_points=4, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0, activation="relu", num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0, learn_query_content=False, eval_spatial_size=None, eval_idx=-1, eps=1e-2, aux_loss=True, cross_attn_method="default", query_select_method="default", reg_max=32, reg_scale=4.0, layer_scale=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale)

        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead, reg_max, self.reg_scale, self.up, eval_idx, layer_scale)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        self.enc_output = nn.Sequential(OrderedDict([("proj", nn.Linear(hidden_dim, hidden_dim)), ("norm", nn.LayerNorm(hidden_dim))]))

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)] + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3) for _ in range(self.eval_idx + 1)] + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3) for _ in range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)

        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(feat_channels)

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Sequential(OrderedDict([("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), ("norm", nn.BatchNorm2d(self.hidden_dim))])))

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Sequential(OrderedDict([("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)), ("norm", nn.BatchNorm2d(self.hidden_dim))])))
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])

        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory: torch.Tensor, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)
        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_anchors = outputs_anchors_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(init_ref_contents, init_ref_points_unact, memory, spatial_shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, self.pre_bbox_head, self.integral, self.up, self.reg_scale, attn_mask=attn_mask, dn_meta=dn_meta)

        return {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}


# ==============================================================================
# MAIN DFINE MODEL
# ==============================================================================

class DFINE(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x


# ==============================================================================
# MODEL BUILDING AND LOADING
# ==============================================================================

def get_model_config(model_name="l"):
    base_cfg = {
        "HGNetv2": {"pretrained": False, "freeze_stem_only": True},
        "HybridEncoder": {"num_encoder_layers": 1, "nhead": 8, "dropout": 0.0, "enc_act": "gelu", "act": "silu"},
        "DFINETransformer": {
            "eval_idx": -1, "num_queries": 300, "num_denoising": 100, "label_noise_ratio": 0.5,
            "box_noise_scale": 1.0, "reg_max": 32, "layer_scale": 1, "cross_attn_method": "default",
            "query_select_method": "default",
        },
    }

    model_l_cfg = {
        "HGNetv2": {"name": "B4", "return_idx": [1, 2, 3], "freeze_at": 0, "freeze_norm": True, "use_lab": False},
        "HybridEncoder": {
            "in_channels": [512, 1024, 2048], "feat_strides": [8, 16, 32], "hidden_dim": 256,
            "use_encoder_idx": [2], "dim_feedforward": 1024, "expansion": 1.0, "depth_mult": 1.0,
        },
        "DFINETransformer": {
            "feat_channels": [256, 256, 256], "feat_strides": [8, 16, 32], "hidden_dim": 256,
            "dim_feedforward": 1024, "num_levels": 3, "num_layers": 6, "reg_scale": 4, "num_points": [3, 6, 3],
        },
    }

    def merge_configs(base, specific):
        result = {**base}
        for key, value in specific.items():
            if key in result and isinstance(result[key], dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    return merge_configs(base_cfg, model_l_cfg)


def build_model(model_name, num_classes, device, img_size=None):
    model_cfg = get_model_config(model_name)
    model_cfg["HybridEncoder"]["eval_spatial_size"] = img_size
    model_cfg["DFINETransformer"]["eval_spatial_size"] = img_size

    backbone = HGNetv2(**model_cfg["HGNetv2"])
    encoder = HybridEncoder(**model_cfg["HybridEncoder"])
    decoder = DFINETransformer(num_classes=num_classes, **model_cfg["DFINETransformer"])

    model = DFINE(backbone, encoder, decoder)
    return model.to(device)


def matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
    """Filter state dict to only include matching parameters."""
    missed_list = []
    unmatched_list = []
    matched_state_dict = {}
    for k, v in state.items():
        if k in params:
            if v.shape == params[k].shape:
                matched_state_dict[k] = params[k]
            else:
                unmatched_list.append(k)
        else:
            missed_list.append(k)

    return matched_state_dict, {"missed": missed_list, "unmatched": unmatched_list}


def load_checkpoint(model, checkpoint_path):
    """Load checkpoint for D-FINE model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Detect checkpoint format
    if isinstance(checkpoint, dict):
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    matched_dict, infos = matched_state(model.state_dict(), state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(matched_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected {len(unexpected_keys)} keys")

    model.eval()
    logger.info("Checkpoint loaded successfully")
    return model


# ==============================================================================
# LABEL STUDIO ML BACKEND
# ==============================================================================

class DFineModel(LabelStudioMLBase):
    """D-FINE Label Studio ML Backend"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.info(f"Initializing D-FINE model from {MODEL_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        # Build and load model (eval_spatial_size=None for dynamic shapes)
        self.model = build_model(MODEL_NAME, NUM_CLASSES, self.device, img_size=None)
        self.model = load_checkpoint(self.model, MODEL_PATH)
        self.model.eval()

        # Preprocessing transform (matches D-FINE standalone script)
        # INPUT_SIZE = (1280, 448) = (width, height)
        # T.Resize expects (height, width)
        resize_hw = (INPUT_SIZE[1], INPUT_SIZE[0])  # (448, 1280)
        self.transform = T.Compose([
            T.Resize(resize_hw),
            T.ToTensor()
        ])

        logger.info(f"Model initialized successfully")
        logger.info(f"Input size: {INPUT_SIZE} (W x H)")
        logger.info(f"Number of classes: {NUM_CLASSES}")

    def _get_image_from_s3(self, s3_url: str) -> str:
        """Download image from S3 and return base64-encoded string."""
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc
        key = parsed_url.path.lstrip('/')

        logger.info(f"Downloading from S3: bucket={bucket_name}, key={key}")
        _img_bytes = s3.get_object(Bucket=bucket_name, Key=key).get('Body').read()
        return base64.b64encode(_img_bytes).decode('utf-8')

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        """
        Run D-FINE inference on images from S3.
        Returns predictions in Label Studio format.
        """
        # Default inference parameters (can be overridden via context)
        confidence_threshold = 0.3
        nms_threshold = 1.0  # NMS disabled by default

        if context and 'result' in context:
            confidence_threshold = context.get('confidence_threshold', confidence_threshold)
            nms_threshold = context.get('nms_threshold', nms_threshold)

        all_predictions = []

        for task in tasks:
            try:
                image_url = task['data']['image']
                logger.info(f"Processing image: {image_url}")

                # Download from S3
                image_b64 = self._get_image_from_s3(image_url)

                # Decode to PIL Image
                img_bytes = base64.b64decode(image_b64)
                img_pil = Image.open(BytesIO(img_bytes)).convert('RGB')
                original_width, original_height = img_pil.size

                # Preprocess
                image_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

                # Inference
                with torch.no_grad():
                    outputs = self.model(image_tensor)

                # Postprocess
                pred_logits = outputs['pred_logits']
                pred_boxes = outputs['pred_boxes']

                if isinstance(pred_logits, (list, tuple)):
                    pred_logits = pred_logits[-1]
                if isinstance(pred_boxes, (list, tuple)):
                    pred_boxes = pred_boxes[-1]

                pred_logits = pred_logits[0]  # Remove batch dimension
                pred_boxes = pred_boxes[0]

                # Convert to probabilities (softmax)
                probs = F.softmax(pred_logits, dim=-1)
                max_scores, labels = probs.max(dim=-1)

                # Filter by confidence
                keep_mask = max_scores > confidence_threshold

                if keep_mask.sum() == 0:
                    logger.info("No detections above threshold")
                    all_predictions.append({
                        'result': [],
                        'model_version': 'dfine_v1'
                    })
                    continue

                filtered_boxes = pred_boxes[keep_mask]
                filtered_scores = max_scores[keep_mask]
                filtered_labels = labels[keep_mask]

                # Convert boxes from cxcywh to xyxy
                boxes_cxcywh = filtered_boxes.cpu()
                boxes_xyxy = box_convert(boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')

                # Scale to original image size
                boxes_xyxy[:, [0, 2]] *= original_width
                boxes_xyxy[:, [1, 3]] *= original_height
                boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, original_width)
                boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, original_height)

                # Apply NMS if enabled
                if nms_threshold < 1.0:
                    keep_indices = nms(boxes_xyxy, filtered_scores.cpu(), nms_threshold)
                    boxes_xyxy = boxes_xyxy[keep_indices]
                    filtered_scores = filtered_scores[keep_indices]
                    filtered_labels = filtered_labels[keep_indices]

                # Convert to Label Studio format
                predictions = []
                for box, score, label in zip(boxes_xyxy, filtered_scores, filtered_labels):
                    x1, y1, x2, y2 = box.tolist()

                    # Fix coordinate order
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f'class_{int(label)}'

                    # Label Studio expects percentages
                    prediction = {
                        'from_name': 'label',
                        'to_name': 'image',
                        'type': 'rectanglelabels',
                        'value': {
                            'x': x1 / original_width * 100,
                            'y': y1 / original_height * 100,
                            'width': (x2 - x1) / original_width * 100,
                            'height': (y2 - y1) / original_height * 100,
                            'rectanglelabels': [class_name]
                        },
                        'score': float(score)
                    }
                    predictions.append(prediction)

                    logger.info(f"Detection: {class_name}, conf={score:.3f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

                logger.info(f"Total detections: {len(predictions)}")

                all_predictions.append({
                    'result': predictions,
                    'model_version': 'dfine_v1'
                })

            except Exception as e:
                logger.error(f"Error processing task: {e}", exc_info=True)
                all_predictions.append({
                    'result': [],
                    'model_version': 'dfine_v1'
                })

        return all_predictions

    def fit(self, event, data, **kwargs):
        """
        Training callback (optional).
        Called when annotations are created/updated in Label Studio.
        """
        logger.info(f"fit() called with event: {event}")

        # Retrieve cached data
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        logger.info(f'Old data: {old_data}')
        logger.info(f'Old model version: {old_model_version}')

        # Store new data
        self.set('my_data', 'dfine_data_value')
        self.set('model_version', 'dfine_v1')

        logger.info('fit() completed successfully')
