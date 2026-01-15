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

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import cv2 
from typing import Tuple

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from modeling.deformable_detr import DeformableDETR
from detectron2.utils.events import get_event_storage


class DistillDeformableDETR(DeformableDETR):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """
    def forward_encoder(self,images, img_masks):
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in deformable DETR
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        (
            memory,
            mask_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios
        ) = self.transformer.forward_memory(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings
        )

        return memory,mask_flatten,spatial_shapes,level_start_index,valid_ratios


    def forward_decoder_student(self,memory_s,mask_flatten,spatial_shapes,level_start_index,valid_ratios):
        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact
        ) = self.transformer.forward_student(
            memory_s,mask_flatten,spatial_shapes,level_start_index,valid_ratios,query_embeds
        )

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            
        
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return output
    
    def forward_decoder_teacher(self, memory_t,memory_s,mask_flatten,spatial_shapes,level_start_index,valid_ratios):
        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        # enc teacher dec teacher
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact
        ) = self.transformer.forward_teacher(
            memory_t,mask_flatten,spatial_shapes,level_start_index,valid_ratios,query_embeds
        )
        
        # enc student dec teacher
        (
            inter_states_s,
            init_reference_s,
            inter_references_s,
            enc_outputs_class_s,
            enc_outputs_coord_unact_s
        ) = self.transformer.forward_teacher(
            memory_s,mask_flatten,spatial_shapes,level_start_index,valid_ratios,query_embeds
        )

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        outputs_classes_s = []
        outputs_coords_s = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
                reference_s = init_reference_s
            else:
                reference = inter_references[lvl - 1]
                reference_s = inter_references_s[lvl - 1]
            reference = inverse_sigmoid(reference)
            reference_s = inverse_sigmoid(reference_s)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            outputs_class_s = self.class_embed[lvl](inter_states_s[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            tmp_s = self.bbox_embed[lvl](inter_states_s[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
                tmp_s += reference_s
            else:
                assert reference.shape[-1] == 2 and reference_s.shape[-1] == 2
                tmp[..., :2] += reference
                tmp_s[..., :2] += reference_s
            outputs_coord = tmp.sigmoid()
            outputs_coord_s = tmp_s.sigmoid()
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            
            outputs_classes_s.append(outputs_class_s)
            outputs_coords_s.append(outputs_coord_s)
        
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        outputs_class_s = torch.stack(outputs_classes_s)
        outputs_coord_s = torch.stack(outputs_coords_s)


        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }

        predictions_esdt = {
            "pred_logits": outputs_class_s[-1],
            "pred_boxes" : outputs_coord_s[-1]
        }
        
        if self.aux_loss:
            predictions_esdt["aux_outputs"] = self._set_aux_loss(outputs_class_s,outputs_coord_s)
        if self.as_two_stage:
            enc_outputs_coord_s = enc_outputs_coord_unact_s.sigmoid()
            predictions_esdt["enc_outputs"] = {
                "pred_logits": enc_outputs_class_s,
                "pred_boxes": enc_outputs_coord_s,
            }
        output["predictions_encstu_dectea"] = predictions_esdt
        return output