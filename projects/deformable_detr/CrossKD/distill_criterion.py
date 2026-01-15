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
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.modeling import SetCriterion
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou,box_iou
from torchvision.ops.boxes import box_area
from detrex.utils import get_world_size, is_dist_avail_and_initialized

class DistillCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """
    def __init__(self,distill_matcher=None, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.distill_matcher = distill_matcher

    def loss_labels_distill(self, outputs, targets, indices, avg_factor):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_logits = outputs["pred_logits"][src_idx]  # Student network prediction
        tgt_logits = targets["pred_logits"][tgt_idx]  # Teacher network prediction
        
        # Apply softmax to logits
        tgt_prob = F.sigmoid(tgt_logits)
        
        # Compute binary cross entropy loss
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, tgt_prob, reduction="none").sum()
        # Similar to focal loss, compute weight for each point
        loss_ce = loss_ce / avg_factor
        
        return {'loss_class': loss_ce}

    def loss_boxes_distill(self, outputs, targets, indices, avg_factor):
        """
        Compute L1 loss and GIoU loss for bounding boxes
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][src_idx] # [batch*num_queries,4]
        target_boxes = targets["pred_boxes"][tgt_idx]

        # Compute L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum()
        loss_bbox = loss_bbox / avg_factor

        # Compute GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        loss_giou = loss_giou.sum() / avg_factor

        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }
    
    def get_loss_distill(self, loss, outputs, targets, indices, avg_factor, **kwargs):
        loss_map = {
            "class": self.loss_labels_distill,
            "boxes": self.loss_boxes_distill,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices,avg_factor, **kwargs)

    def forward(self, outputs_s, outputs_t,targets):
        predictions_encstu_dectea = outputs_t["predictions_encstu_dectea"]

        # Matching result between teacher and student queries
        indices = self.distill_matcher(predictions_encstu_dectea, outputs_t)
        
        avg_factor = predictions_encstu_dectea['pred_logits'].shape[0] * predictions_encstu_dectea["pred_logits"].shape[1] # batch_size * num_quries

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss_distill(loss, predictions_encstu_dectea, outputs_t, indices, avg_factor))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in predictions_encstu_dectea:
            for i, (aux_outputs_encstu_dectea,aux_outputs_t) in enumerate(zip(predictions_encstu_dectea["aux_outputs"],outputs_t["aux_outputs"])):
                indices = self.distill_matcher(aux_outputs_encstu_dectea, aux_outputs_t)
                for loss in self.losses:
                    l_dict = self.get_loss_distill(loss, aux_outputs_encstu_dectea, aux_outputs_t, indices, avg_factor)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        losses = {"distill_" + k : v for k,v in losses.items()}

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
