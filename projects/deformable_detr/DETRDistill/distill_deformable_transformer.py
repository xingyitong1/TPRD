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
from ..modeling import DeformableDetrTransformer

class DistillDeformableDetrTransformer(DeformableDetrTransformer):
    """Transformer module for Deformable DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 300.
            Only used when as_two_stage is True.
    """
    def forward_teacher(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        **kwargs,
    ):
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points
        
        # 用于传输给student
        teacher_query_pos = query_pos
        teacher_query = query
        teacher_refpoints = reference_points

        # decoder
        inter_states, inter_references = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=None,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            **kwargs,
        )


        inter_references_out = inter_references
        if self.as_two_stage:
            return (
                inter_states,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
                (teacher_query_pos, teacher_query,teacher_refpoints)
            )
        return inter_states, init_reference_out, inter_references_out,None ,None,(teacher_query_pos, teacher_query,teacher_refpoints)
    
    def forward_student(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        teacher_query,
        **kwargs,
    ):
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points
        
        # teacher query
        teacher_query_pos,teacher_query,teacher_refpoints = teacher_query

        # decoder
        inter_states, inter_references = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=None,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            **kwargs,
        )

        # teacher query
        inter_states_tea, inter_references_tea = self.decoder(
            query=teacher_query,  # bs, num_queries, embed_dims
            key=None,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=teacher_query_pos,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=teacher_refpoints,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            **kwargs,
        )

        inter_references_out = inter_references
        if self.as_two_stage:
            return (
                inter_states,
                init_reference_out,
                inter_references_out,
                inter_states_tea,
                teacher_refpoints,
                inter_references_tea,
                enc_outputs_class,
                enc_outputs_coord_unact
            )
        return inter_states, init_reference_out, inter_references_out,inter_states_tea,teacher_refpoints,inter_references_tea,None,None
