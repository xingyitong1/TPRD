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

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from ..modeling import DeformableDETR


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

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    # TODO 加上了auxiliary_refpoints和teacher_query
    def forward_student(self, images, img_masks,auxiliary_query,teacher_query):
        # original features
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

        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        (
            inter_states,
            init_reference,
            inter_references,
            inter_states_aux,
            auxiliary_reference_points,
            inter_references_aux,
            inter_states_tea,
            teacher_refpoints,
            inter_references_tea,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer.forward_student(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds, auxiliary_query,teacher_query
        )

        # Calculate output coordinates and classes.
        outputs_class,outputs_coord = self.get_outputs(inter_states,inter_references,init_reference)
        outputs_class_aux,outputs_coord_aux = self.get_outputs(inter_states_aux,inter_references_aux,auxiliary_reference_points)
        outputs_class_tea,outputs_coord_tea = self.get_outputs(inter_states_tea,inter_references_tea,teacher_refpoints)

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
        output['auxiliary_predictions'] = {
            "pred_logits": outputs_class_aux[-1],
            "pred_boxes": outputs_coord_aux[-1],
            "aux_outputs": self._set_aux_loss(outputs_class_aux, outputs_coord_aux)
        }
        output['teacher_predictions'] = {
            "pred_logits": outputs_class_tea[-1],
            "pred_boxes": outputs_coord_tea[-1],
            "aux_outputs": self._set_aux_loss(outputs_class_tea, outputs_coord_tea)
        }
        return output
    
    def forward_teacher(self, images, img_masks,auxiliary_query):
        # original features
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

        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        (
            inter_states,
            init_reference,
            inter_references,
            inter_states_aux,
            auxiliary_reference_points,
            inter_references_aux,
            enc_outputs_class,
            enc_outputs_coord_unact,
            teacher_query # 用于输入到student的模型中
        ) = self.transformer.forward_teacher(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds, auxiliary_query
        )

        # Calculate output coordinates and classes.
        outputs_class,outputs_coord = self.get_outputs(inter_states,inter_references,init_reference)
        outputs_class_aux,outputs_coord_aux = self.get_outputs(inter_states_aux,inter_references_aux,auxiliary_reference_points)

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
        output["auxiliary_predictions"] = {
            "pred_logits": outputs_class_aux[-1],
            "pred_boxes": outputs_coord_aux[-1],
            "aux_outputs": self._set_aux_loss(outputs_class_aux, outputs_coord_aux)
        }
        return output,teacher_query
    
    # TODO preprocess for 一致点采样
    def preprocess(self,num_queries,num_auxilary_queries,num_teacher_queries=0):
        total_num = num_queries + num_auxilary_queries+ num_teacher_queries
        
        attn_mask = torch.ones(total_num,total_num).to('cuda') > 0
        # 原始组
        attn_mask[:num_queries,:num_queries] = False
        # auxiliary 组
        attn_mask[num_queries:num_queries+num_auxilary_queries,num_queries:num_queries+num_auxilary_queries] = False
        # teacher 组
        attn_mask[num_queries+num_auxilary_queries:,num_queries+num_auxilary_queries:] = False
        
        return attn_mask

    # TODO postprocess for 一致点采样
    def postprocess(self,outputs_class,outputs_coords,num_queries,num_auxilary_queries,num_teacher_queries=0):
        auxiliary_class = outputs_class[:,:,num_queries:num_queries+num_auxilary_queries,:]
        auxiliary_coord = outputs_coords[:,:,num_queries:num_queries+num_auxilary_queries,:]
        
        auxiliary_predictions = {
            'pred_logits': auxiliary_class[-1],
            'pred_boxes':auxiliary_coord[-1]
        }
        auxiliary_predictions['aux_outputs'] = (self._set_aux_loss(auxiliary_class,auxiliary_coord))

        teacher_predictions = None
        if num_teacher_queries != 0:
            teacher_class = outputs_class[:,:,num_queries+num_auxilary_queries:,:]
            teacher_coord = outputs_coords[:,:,num_queries+num_auxilary_queries:,:]
            teacher_predictions = {
            'pred_logits': teacher_class[-1],
            'pred_boxes':teacher_coord[-1]
            }
            teacher_predictions['aux_outputs'] = (self._set_aux_loss(teacher_class,teacher_coord))
        
        outputs_class = outputs_class[:,:,:num_queries,:]
        outputs_coords = outputs_coords[:,:,:num_queries,:]

        return outputs_class,outputs_coords,auxiliary_predictions,teacher_predictions
    
    def get_outputs(self,inter_states,inter_references,init_reference):
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
        return outputs_class,outputs_coord


        
    

