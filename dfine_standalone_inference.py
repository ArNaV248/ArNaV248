#!/usr/bin/env python3
"""
D-FINE Standalone Inference Script
===================================

100% Self-Contained D-FINE Object Detection Inference
No external dependencies except: torch, torchvision, PIL, numpy

Model: DFINE-L (Large)
Input Size: 1280x448
Classes: 35 almond defect types
Model File: model_2.pt (must be in same directory or specify path)

All D-FINE source code is embedded in this file.
No need for src/, config.yaml, or any other files!

Usage:
    python dfine_standalone_inference.py --input image.jpg
    python dfine_standalone_inference.py --input /path/to/images/ --conf 0.3
"""

import os
import sys
import json
import argparse
import math
import copy
import functools
from pathlib import Path
from typing import List, Dict, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch import Tensor

import torchvision.transforms as T
from torchvision.ops import nms, box_convert, box_area
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  🔧 MODEL PATH - CHANGE THIS WHEN COPYING TO ANOTHER LOCATION           ║
# ║  Default: /Users/borde/arnav/model_2.pt                                 ║
# ║  If you copy this script elsewhere, update the path below:               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

MODEL_PATH = "/Users/borde/arnav/model_2.pt"  # ← CHANGE THIS PATH IF NEEDED

# ═══════════════════════════════════════════════════════════════════════════

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

VIS_DIR = "visualize"
JSON_DIR = "json"

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


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        dn_meta = {"dn_positive_idx": None, "dn_num_group": 0, "dn_num_split": [0, num_queries]}
        return None, None, None, dn_meta

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    bs = len(num_gts)
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1

    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox[input_query_bbox < 0] *= -1
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)
    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True

    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
        else:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta


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
# HYBRID ENCODER (Embedded) - Continuing in next part
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


# Continue with DFINE Decoder, Matcher, and Main Model in next message due to length...
# This file will be completed with all remaining components


# ==============================================================================
# DFINE DECODER AND TRANSFORMER COMPONENTS (Continued)
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

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(targets, self.num_classes, self.num_queries, self.denoising_class_embed, num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio, box_noise_scale=1.0)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(init_ref_contents, init_ref_points_unact, memory, spatial_shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, self.pre_bbox_head, self.integral, self.up, self.reg_scale, attn_mask=attn_mask, dn_meta=dn_meta)

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

        return {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}


# ==============================================================================
# HUNGARIAN MATCHER (Embedded)
# ==============================================================================

class HungarianMatcher(nn.Module):
    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False):
        # Implementation omitted for inference-only script
        pass


# ==============================================================================
# MAIN DFINE MODEL (Embedded)
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
# MODEL CONFIGURATION
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


def map_class_weights(cur_tensor, pretrain_tensor):
    """Map class weights from pretrain model to current model based on class IDs."""
    if pretrain_tensor.size() == cur_tensor.size():
        return pretrain_tensor

    # If sizes don't match, just return current tensor (keep initialized weights)
    # The original function uses obj365_ids mapping which is dataset-specific
    # For our case, if sizes match we use pretrained weights, otherwise keep current
    print(f"[Warning] Weight size mismatch: pretrain {pretrain_tensor.size()} vs current {cur_tensor.size()}")
    return None


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


def adjust_head_parameters(cur_state_dict, pretrain_state_dict):
    """Adjust head parameters between datasets."""
    # Remove denoising_class_embed if size mismatch
    if (
        "decoder.denoising_class_embed.weight" in pretrain_state_dict
        and "decoder.denoising_class_embed.weight" in cur_state_dict
        and pretrain_state_dict["decoder.denoising_class_embed.weight"].size()
        != cur_state_dict["decoder.denoising_class_embed.weight"].size()
    ):
        del pretrain_state_dict["decoder.denoising_class_embed.weight"]

    # List of head parameters to adjust
    head_param_names = ["decoder.enc_score_head.weight", "decoder.enc_score_head.bias"]
    for i in range(8):
        head_param_names.append(f"decoder.dec_score_head.{i}.weight")
        head_param_names.append(f"decoder.dec_score_head.{i}.bias")

    adjusted_params = []

    for param_name in head_param_names:
        if param_name in cur_state_dict and param_name in pretrain_state_dict:
            cur_tensor = cur_state_dict[param_name]
            pretrain_tensor = pretrain_state_dict[param_name]
            adjusted_tensor = map_class_weights(cur_tensor, pretrain_tensor)
            if adjusted_tensor is not None:
                pretrain_state_dict[param_name] = adjusted_tensor
                adjusted_params.append(param_name)
            else:
                print(f"[Warning] Cannot adjust parameter '{param_name}' due to size mismatch.")

    if adjusted_params:
        print(f"[Model] Adjusted {len(adjusted_params)} head parameters")

    return pretrain_state_dict


def load_checkpoint(model, checkpoint_path):
    """Load checkpoint with proper head parameter adjustment (matches working script)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    print(f"[Model] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Detect checkpoint format and extract state_dict
    checkpoint_type = "unknown"
    if isinstance(checkpoint, dict):
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']['module']
            checkpoint_type = "EMA"
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            checkpoint_type = "model"
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            checkpoint_type = "state_dict"
        else:
            state_dict = checkpoint
            checkpoint_type = "dict"
    else:
        state_dict = checkpoint
        checkpoint_type = "raw"

    print(f"[Model] Checkpoint type: {checkpoint_type}")

    # CRITICAL FIX: Adjust head parameters (this was missing!)
    # This matches the working script's load_tuning_state() function
    try:
        adjusted_state_dict = adjust_head_parameters(model.state_dict(), state_dict)
        matched_dict, infos = matched_state(model.state_dict(), adjusted_state_dict)
        state_dict_to_load = matched_dict
        print(f"[Model] Applied head parameter adjustment (matches working script)")
    except Exception as e:
        print(f"[Warning] Head adjustment failed: {e}")
        print(f"[Model] Falling back to direct state dict loading")
        matched_dict, infos = matched_state(model.state_dict(), state_dict)
        state_dict_to_load = matched_dict

    # Load the adjusted state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)

    if missing_keys:
        print(f"[Warning] Missing {len(missing_keys)} keys (may affect performance)")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"          - {key}")
    if unexpected_keys:
        print(f"[Warning] Unexpected {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"          - {key}")

    # CRITICAL: Set model to evaluation mode
    model.eval()
    print("[Model] Set to evaluation mode (model.eval())")
    print("[Model] Checkpoint loaded successfully")
    return model


# ==============================================================================
# INFERENCE PIPELINE
# ==============================================================================

def select_device():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            device = torch.device("cuda:1")
            print(f"[Device] Using cuda:1 (detected {gpu_count} GPUs)")
        else:
            device = torch.device("cuda:0")
            print(f"[Device] Using cuda:0 (detected {gpu_count} GPU)")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU (no GPU detected)")
    return device


def preprocess_image(image_path: str, target_size: Tuple[int, int] = INPUT_SIZE):
    """
    Preprocessing that matches the working script EXACTLY.
    Working script uses: T.Resize((448, 1280)) + T.ToTensor()
    """
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # Convert (width, height) to (height, width) for T.Resize
    # INPUT_SIZE = (1280, 448) = (width, height)
    # T.Resize expects (height, width) = (448, 1280)
    resize_hw = (target_size[1], target_size[0])

    transform = T.Compose([
        T.Resize(resize_hw),  # (height=448, width=1280)
        T.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)

    print(f"[Preprocessing] Original image size: {original_size} (W x H)")
    print(f"[Preprocessing] Using T.Resize({resize_hw}) = (H x W)")
    print(f"[Preprocessing] Output tensor shape: {tensor.shape} (B, C, H, W)")
    print(f"[Preprocessing] Tensor range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")

    return tensor, image, original_size


@torch.no_grad()
def run_inference(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    return outputs


def postprocess_outputs(outputs: Dict, original_size: Tuple[int, int], confidence_threshold: float = 0.3, nms_threshold: float = 1.0, verbose: bool = True):
    pred_logits = outputs['pred_logits']
    pred_boxes = outputs['pred_boxes']

    if isinstance(pred_logits, (list, tuple)):
        pred_logits = pred_logits[-1]
    if isinstance(pred_boxes, (list, tuple)):
        pred_boxes = pred_boxes[-1]

    pred_logits = pred_logits[0]
    pred_boxes = pred_boxes[0]

    # DIAGNOSTIC: Check actual model output shape
    num_output_classes = pred_logits.shape[-1]
    total_queries = pred_logits.shape[0]

    if verbose:
        print(f"\n[Diagnostic] Model output shape: {pred_logits.shape}")
        print(f"[Diagnostic] Detected {num_output_classes} output classes")

    # EXACT MATCH TO WORKING SCRIPT:
    # Convert logits to probabilities using softmax
    # Take max across ALL classes (doesn't exclude background)
    probs = F.softmax(pred_logits, dim=-1)
    max_scores, labels = probs.max(dim=-1)

    if verbose:
        print(f"[Diagnostic] Using SOFTMAX on all classes (matches working script)")
        print(f"[Diagnostic] Total detection queries: {total_queries}")

        # Show score statistics BEFORE filtering
        print(f"\n[Diagnostic] Score statistics (before filtering):")
        print(f"             Min score: {max_scores.min().item():.6f}")
        print(f"             Max score: {max_scores.max().item():.6f}")
        print(f"             Mean score: {max_scores.mean().item():.6f}")

        # Count how many scores are exactly 0.10
        scores_at_010 = (torch.abs(max_scores - 0.10) < 0.001).sum().item()
        if scores_at_010 > 0:
            print(f"             ⚠️  {scores_at_010} scores are exactly ~0.10 (suspicious!)")

        # Show top 20 scores
        sorted_scores, _ = torch.sort(max_scores, descending=True)
        print(f"             Top 20 scores: {sorted_scores[:20].cpu().numpy()}")

        # Show confidence score distribution BEFORE filtering
        sorted_scores, _ = torch.sort(max_scores, descending=True)
        top_scores = sorted_scores[:50].cpu().numpy()  # Top 50 scores

        # Count scores in different ranges
        ranges = [
            (0.8, 1.0, "0.80-1.00 (Very High)"),
            (0.5, 0.8, "0.50-0.80 (High)"),
            (0.3, 0.5, "0.30-0.50 (Medium)"),
            (0.15, 0.3, "0.15-0.30 (Low)"),
            (0.05, 0.15, "0.05-0.15 (Very Low)"),
            (0.0, 0.05, "0.00-0.05 (Noise)")
        ]

        print(f"\n[Diagnostic] Confidence score distribution (before filtering):")
        for low, high, label in ranges:
            count = ((max_scores >= low) & (max_scores < high)).sum().item()
            if count > 0:
                print(f"             {label}: {count} detections")

        print(f"\n[Diagnostic] Top 10 confidence scores: {top_scores[:10]}")
        print(f"[Diagnostic] Your threshold: {confidence_threshold:.2f}")

    keep_mask = max_scores > confidence_threshold
    num_after_conf = keep_mask.sum().item()

    if verbose:
        num_filtered_out = len(max_scores) - num_after_conf
        print(f"\n[Diagnostic] After confidence filter (>= {confidence_threshold:.2f}): {num_after_conf} detections kept, {num_filtered_out} filtered out")

    if keep_mask.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    filtered_boxes = pred_boxes[keep_mask]
    filtered_scores = max_scores[keep_mask]
    filtered_labels = labels[keep_mask]

    boxes_cxcywh = filtered_boxes.cpu()
    boxes_xyxy = box_convert(boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')

    img_w, img_h = original_size
    boxes_xyxy[:, [0, 2]] *= img_w
    boxes_xyxy[:, [1, 3]] *= img_h
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, img_w)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, img_h)

    keep_indices = nms(boxes_xyxy, filtered_scores.cpu(), nms_threshold)
    final_boxes = boxes_xyxy[keep_indices].numpy()
    final_scores = filtered_scores[keep_indices].cpu().numpy()
    final_labels = filtered_labels[keep_indices].cpu().numpy()

    # DIAGNOSTIC: Show NMS results and class distribution
    if verbose:
        num_after_nms = len(final_boxes)
        num_removed_by_nms = num_after_conf - num_after_nms
        print(f"[Diagnostic] After NMS (threshold {nms_threshold:.2f}): {num_after_nms} detections")
        if num_removed_by_nms > 0:
            print(f"[Diagnostic] NMS removed {num_removed_by_nms} overlapping boxes")

        # Show class distribution
        if len(final_labels) > 0:
            unique_labels, counts = np.unique(final_labels, return_counts=True)
            print(f"[Diagnostic] Detected classes:")
            for label, count in zip(unique_labels, counts):
                class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class_{int(label)}"
                print(f"             - {class_name}: {count} objects")

    return final_boxes, final_scores, final_labels


def visualize_detections(image_path: str, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, output_path: str):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box

        # Fix coordinate order to ensure x1 < x2 and y1 < y2
        # (PIL requires this for rectangle drawing)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Draw ALL boxes - even rotated/inclined almonds
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class_{int(label)}"
        text = f"{class_name}: {score:.2f}"  # Show as decimal (0.00)
        try:
            bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle(bbox, fill='red')
            draw.text((x1, y1 - 20), text, fill='white', font=font)
        except:
            draw.text((x1, y1 - 20), text, fill='yellow', font=font)

    image.save(output_path)
    print(f"[Output] Saved visualization: {output_path}")


def save_json_output(image_name: str, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, output_path: str):
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box

        # Fix coordinate order (for rotated/inclined objects)
        x1, x2 = float(min(x1, x2)), float(max(x1, x2))
        y1, y2 = float(min(y1, y2)), float(max(y1, y2))

        class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class_{int(label)}"
        detection = {
            "class": class_name,
            "class_id": int(label),
            "confidence": float(score),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }
        detections.append(detection)

    output = {"image": image_name, "num_detections": len(detections), "detections": detections}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"[Output] Saved JSON: {output_path}")


def process_single_image(model, device, image_path: str, output_dir: str, confidence_threshold: float, nms_threshold: float, save_vis: bool = True, save_json_out: bool = True):
    print(f"\n[Processing] {image_path}")
    print(f"[Settings] Confidence: {confidence_threshold:.2f} | NMS: {nms_threshold:.2f}")

    image_tensor, original_image, original_size = preprocess_image(image_path, INPUT_SIZE)
    outputs = run_inference(model, image_tensor, device)

    # DIAGNOSTIC: Save raw model logits for comparison
    pred_logits = outputs['pred_logits']
    if isinstance(pred_logits, (list, tuple)):
        pred_logits = pred_logits[-1]
    pred_logits_cpu = pred_logits[0].cpu()  # [num_queries, num_classes]

    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    logits_path = os.path.join(output_dir, f"logits_{base_name}.pt")
    torch.save({
        'logits': pred_logits_cpu,
        'shape': pred_logits_cpu.shape,
        'image': image_name,
        'first_query_logits': pred_logits_cpu[0, :].tolist(),  # First query, all classes
        'first_10_queries_first_10_classes': pred_logits_cpu[:10, :10].tolist()  # First 10 queries, first 10 classes
    }, logits_path)
    print(f"[Diagnostic] Saved raw logits to: {logits_path}")
    print(f"[Diagnostic] Logits shape: {pred_logits_cpu.shape}")
    print(f"[Diagnostic] First query, first 10 class logits: {pred_logits_cpu[0, :10].tolist()}")

    boxes, scores, labels = postprocess_outputs(outputs, original_size, confidence_threshold, nms_threshold)

    print(f"\n[Results] Final detections: {len(boxes)} objects")

    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    if save_vis:
        vis_path = os.path.join(output_dir, VIS_DIR, image_name)
        visualize_detections(image_path, boxes, scores, labels, vis_path)

    if save_json_out:
        json_path = os.path.join(output_dir, JSON_DIR, f"{base_name}.json")
        save_json_output(image_name, boxes, scores, labels, json_path)

    if len(boxes) > 0:
        print(f"\nDetections:")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels), 1):
            x1, y1, x2, y2 = box

            # Fix coordinate order (for rotated/inclined objects)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            class_name = CLASS_NAMES[int(label)]
            print(f"  {i}. {class_name}: {score:.2f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        print(f"\n💡 Tips:")
        print(f"   - Too many unwanted boxes? Increase confidence: --conf 0.5")
        print(f"   - Duplicate boxes on same object? Enable NMS: --nms 0.5 or --nms 0.3")
        print(f"   - Missing objects? Lower confidence: --conf 0.1")
    else:
        print("\n⚠️  No detections found!")
        print("\n💡 Troubleshooting tips:")
        print("   1. Try MUCH lower confidence: 0.01 or 0.05")
        print("   2. Check if model file is correct: /Users/borde/arnav/model_2.pt")
        print("   3. Verify image contains objects the model was trained on")
        print(f"\n   Re-run with: python3 dfine_standalone_inference.py --input {image_path} --conf 0.01")


def process_directory(model, device, input_dir: str, output_dir: str, confidence_threshold: float, nms_threshold: float, save_vis: bool = True, save_json_out: bool = True):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []

    for file_name in sorted(os.listdir(input_dir)):
        if os.path.splitext(file_name)[1].lower() in image_extensions:
            image_files.append(os.path.join(input_dir, file_name))

    print(f"\n[Directory] Found {len(image_files)} images in {input_dir}")

    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*70}")
        print(f"Image {i}/{len(image_files)}")
        print(f"{'='*70}")
        try:
            process_single_image(model, device, image_path, output_dir, confidence_threshold, nms_threshold, save_vis, save_json_out)
        except Exception as e:
            print(f"[Error] Failed to process {image_path}: {e}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='D-FINE Standalone Inference for Almond Defect Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (no arguments)
  python dfine_standalone_inference.py

  # Command-line mode
  python dfine_standalone_inference.py --input image.jpg --conf 0.3
  python dfine_standalone_inference.py --input /path/to/images/ --conf 0.3 --nms 0.5
  python dfine_standalone_inference.py --input image.jpg --model /path/to/model_2.pt
        """
    )

    parser.add_argument('--input', '-i', type=str, required=False, help='Path to input image or directory')
    parser.add_argument('--output', '-o', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--conf', '-c', type=float, default=None, help='Confidence threshold (default: 0.3)')
    parser.add_argument('--nms', '-n', type=float, default=1.0, help='NMS IoU threshold (default: 1.0 = disabled, lower = remove more overlapping boxes)')
    parser.add_argument('--model', '-m', type=str, default=MODEL_PATH, help=f'Path to model checkpoint (default: {MODEL_PATH})')
    parser.add_argument('--no-vis', action='store_true', help='Skip saving visualizations')
    parser.add_argument('--no-json', action='store_true', help='Skip saving JSON outputs')

    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════════════════
    # INTERACTIVE MODE: Ask for input if not provided via command-line
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*18 + "D-FINE STANDALONE INFERENCE" + " "*23 + "║")
    print("║" + " "*14 + "Almond Defect Detection System" + " "*24 + "║")
    print("╚" + "═"*68 + "╝\n")

    # Ask for image path if not provided
    if args.input is None:
        print("📂 Please provide the path to your image or folder:")
        print("   Examples:")
        print("   - Single image: /Users/borde/Downloads/image.jpg")
        print("   - Folder:       /Users/borde/Downloads/almond_images/")
        print()
        args.input = input("Enter path: ").strip()

        if not args.input:
            print("\n❌ Error: No input path provided. Exiting.")
            sys.exit(1)

    # Ask for confidence threshold if not provided
    if args.conf is None:
        print("\n🎯 Confidence threshold (0.0 to 1.0):")
        print("   - 0.05-0.10: VERY LOW - Maximum detections (many false positives)")
        print("   - 0.15-0.20: LOW - More detections (some false positives)")
        print("   - 0.30:      MEDIUM - Balanced (recommended)")
        print("   - 0.50+:     HIGH - Only very confident detections")
        print("\n   ⚠️  Objects missing? Try these in order:")
        print("       1. Start with 0.10 (shows almost everything)")
        print("       2. If too many false positives, increase to 0.15")
        print("       3. Gradually increase until you find the sweet spot")
        print()
        conf_input = input("Enter confidence threshold [default: 0.30]: ").strip()

        if conf_input:
            try:
                args.conf = float(conf_input)
                if not (0.0 <= args.conf <= 1.0):
                    print("⚠️  Warning: Confidence should be between 0.0 and 1.0. Using default: 0.3")
                    args.conf = 0.3
            except ValueError:
                print("⚠️  Warning: Invalid input. Using default confidence: 0.3")
                args.conf = 0.3
        else:
            args.conf = 0.3

    print("\n" + "─"*70)
    print(f"📋 Configuration:")
    print(f"   Model:       {args.model}")
    print(f"   Input:       {args.input}")
    print(f"   Output:      {args.output}")
    print(f"   Confidence:  {args.conf}")
    print(f"   NMS:         {args.nms}")
    print("─"*70 + "\n")

    os.makedirs(os.path.join(args.output, VIS_DIR), exist_ok=True)
    os.makedirs(os.path.join(args.output, JSON_DIR), exist_ok=True)

    device = select_device()

    print(f"\n📊 Model Info:")
    print(f"   Architecture: DFINE-{MODEL_NAME.upper()}")
    print(f"   Input Size:   {INPUT_SIZE[0]}x{INPUT_SIZE[1]}")
    print(f"   Classes:      {NUM_CLASSES}")
    print()

    print("\n[Model] Building D-FINE model...")
    # CRITICAL: Pass None for img_size to match working script (eval_spatial_size = None)
    model = build_model(MODEL_NAME, NUM_CLASSES, device, None)
    model = load_checkpoint(model, args.model)
    model.eval()
    print("[Model] Model ready for inference\n")

    if os.path.isfile(args.input):
        process_single_image(model, device, args.input, args.output, args.conf, args.nms, save_vis=not args.no_vis, save_json_out=not args.no_json)
    elif os.path.isdir(args.input):
        process_directory(model, device, args.input, args.output, args.conf, args.nms, save_vis=not args.no_vis, save_json_out=not args.no_json)
    else:
        print(f"[Error] Input path not found: {args.input}")
        sys.exit(1)

    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*22 + "✅ INFERENCE COMPLETE" + " "*25 + "║")
    print("╚" + "═"*68 + "╝\n")
    print(f"📁 Outputs saved to: {args.output}")
    if not args.no_vis:
        print(f"   🖼️  Visualizations: {os.path.join(args.output, VIS_DIR)}")
    if not args.no_json:
        print(f"   📄 JSON files:      {os.path.join(args.output, JSON_DIR)}")
    print("\n💡 Tip: Open visualize folder to see detection results!\n")


if __name__ == '__main__':
    main()
