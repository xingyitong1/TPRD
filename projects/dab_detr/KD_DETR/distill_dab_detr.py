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
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.layers.mlp import MLP
from detrex.utils.misc import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from ..modeling import DABDETR


class DistillDABDETR(DABDETR):
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

    def forward_student(self, images, img_masks,aux_box_embed,teacher_box_embed):
        # only use last level feature in DAB-DETR
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        # dynamic anchor boxes
        dynamic_anchor_boxes = self.anchor_box_embed.weight

        # hidden_states: transformer output hidden feature
        # reference_boxes: the refined dynamic anchor boxes in format (x, y, w, h)
        # with normalized coordinates in range of [0, 1].
        hs, references,aux_hs,aux_references,tea_hs,tea_references = self.transformer.forward_student(
            features, img_masks, dynamic_anchor_boxes, pos_embed,aux_box_embed,teacher_box_embed
        )

        # Calculate output coordinates and classes.
        outputs_class,outputs_coord = self.get_outputs(hs,references)
        outputs_class_aux,outputs_coord_aux = self.get_outputs(aux_hs,aux_references)
        outputs_class_tea,outputs_coord_tea = self.get_outputs(tea_hs,tea_references)

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

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
    
    def forward_teacher(self, images, img_masks,aux_box_embed):
        # only use last level feature in DAB-DETR
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        # dynamic anchor boxes
        dynamic_anchor_boxes = self.anchor_box_embed.weight

        # hidden_states: transformer output hidden feature
        # reference_boxes: the refined dynamic anchor boxes in format (x, y, w, h)
        # with normalized coordinates in range of [0, 1].
        hs, references,aux_hs,aux_references = self.transformer.forward_teacher(
            features, img_masks, dynamic_anchor_boxes, pos_embed,aux_box_embed
        )

        # Calculate output coordinates and classes.
        outputs_class,outputs_coord = self.get_outputs(hs,references)
        outputs_class_aux,outputs_coord_aux = self.get_outputs(aux_hs,aux_references)

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        output['auxiliary_predictions'] = {
            "pred_logits": outputs_class_aux[-1],
            "pred_boxes": outputs_coord_aux[-1],
            "aux_outputs": self._set_aux_loss(outputs_class_aux, outputs_coord_aux)
        }
        return output,dynamic_anchor_boxes
    
    def get_outputs(self,hs,references):
        references = inverse_sigmoid(references)
        anchor_box_offsets = self.bbox_embed(hs)
        outputs_coord = (references + anchor_box_offsets).sigmoid()
        outputs_class = self.class_embed(hs)
        return outputs_class,outputs_coord


        
    

