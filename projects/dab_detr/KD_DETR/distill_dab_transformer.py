# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
)
from detrex.utils import inverse_sigmoid
from ..modeling import DabDetrTransformer

class DistillDabDetrTransformer(DabDetrTransformer):
    """Transformer module for Deformable DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 300.
            Only used when as_two_stage is True.
    """
    def forward_teacher(self,x, mask, anchor_box_embed, pos_embed,aux_box_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)  # (c, bs, num_queries)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        anchor_box_embed = anchor_box_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )
        num_queries = anchor_box_embed.shape[0]

        if self.num_patterns == 0:
            target = torch.zeros(num_queries, bs, self.embed_dim, device=anchor_box_embed.device)
        else:
            target = self.patterns.weight[:, None, None, :].repeat(1, num_queries, bs, 1).flatten(0, 1)
            anchor_box_embed = anchor_box_embed.repeat(self.num_patterns, 1, 1)

        aux_box_embed = aux_box_embed.unsqueeze(1).repeat(1, bs, 1)
        aux_num_queries = aux_box_embed.shape[0]
        if self.num_patterns == 0:
            aux_target = torch.zeros(aux_num_queries, bs, self.embed_dim, device=aux_box_embed.device)
        else:
            aux_target = self.patterns.weight[:, None, None, :].repeat(1, aux_num_queries, bs, 1).flatten(0, 1)
            aux_box_embed = aux_box_embed.repeat(self.num_patterns, 1, 1)

        # decoder
        hs, references = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=anchor_box_embed,
        )
        # aux query
        aux_hs, aux_references = self.decoder(
            query=aux_target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=aux_box_embed,
        )

        return hs, references, aux_hs, aux_references
    
    def forward_student(self,x, mask, anchor_box_embed, pos_embed,aux_box_embed,teacher_box_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)  # (c, bs, num_queries)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        anchor_box_embed = anchor_box_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )
        num_queries = anchor_box_embed.shape[0]

        if self.num_patterns == 0:
            target = torch.zeros(num_queries, bs, self.embed_dim, device=anchor_box_embed.device)
        else:
            target = self.patterns.weight[:, None, None, :].repeat(1, num_queries, bs, 1).flatten(0, 1)
            anchor_box_embed = anchor_box_embed.repeat(self.num_patterns, 1, 1)

        # aux group
        aux_box_embed = aux_box_embed.unsqueeze(1).repeat(1, bs, 1)
        aux_num_queries = aux_box_embed.shape[0]
        if self.num_patterns == 0:
            aux_target = torch.zeros(aux_num_queries, bs, self.embed_dim, device=aux_box_embed.device)
        else:
            aux_target = self.patterns.weight[:, None, None, :].repeat(1, aux_num_queries, bs, 1).flatten(0, 1)
            aux_box_embed = aux_box_embed.repeat(self.num_patterns, 1, 1)

        # teacher group
        teacher_box_embed = teacher_box_embed.unsqueeze(1).repeat(1, bs, 1)
        teacher_num_queries = teacher_box_embed.shape[0]
        if self.num_patterns == 0:
            teacher_target = torch.zeros(teacher_num_queries, bs, self.embed_dim, device=teacher_box_embed.device)
        else:
            teacher_target = self.patterns.weight[:, None, None, :].repeat(1, teacher_num_queries, bs, 1).flatten(0, 1)
            teacher_box_embed = teacher_box_embed.repeat(self.num_patterns, 1, 1)

        # decoder
        hs, references = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=anchor_box_embed,
        )
        # aux query
        aux_hs, aux_references = self.decoder(
            query=aux_target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=aux_box_embed,
        )
        # teacher query
        tea_hs, tea_references = self.decoder(
            query=teacher_target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=teacher_box_embed,
        )

        return hs,references,aux_hs,aux_references,tea_hs,tea_references
