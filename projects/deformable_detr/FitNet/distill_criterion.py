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
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized


class DistillCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """
    def __init__(self,dim_in,dim_out,num_features, *args,**kwargs):
        super().__init__(*args,**kwargs)
        # Used to align teacher and student models
        self.align = nn.ModuleList()
        for _ in range(num_features):
            self.align.append(nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0))
        
    def loss_labels(self, outputs, targets, indices, num_boxes, weights=None):
        
        src_logits = outputs["pred_logits"].reshape(-1, self.num_classes)  # Student network prediction
        tgt_logits = targets["pred_logits"].reshape(-1, self.num_classes)  # Teacher network prediction
        
        # Apply softmax to logits
        src_prob = F.sigmoid(src_logits)
        tgt_prob = F.sigmoid(tgt_logits)
        
        # Calculate binary cross entropy loss
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, tgt_prob, reduction="none") # [batch_size*num_queries,num_classes]
        # Analogy to focal loss, calculate weight for each point
        distill_weights = torch.abs(src_prob - tgt_prob)
        loss_ce = loss_ce * distill_weights
        loss_ce = loss_ce.mean(dim=-1) # [batch_size*num_queries] Average over classes dimension
        
        # If weights are provided, perform weighted average
        if weights is not None: 
            # Ensure weight dimensions match
            weights = weights.view(-1, 1)  # Expand dimensions for broadcasting
            loss_ce = loss_ce * weights
            
        # Sum over all dimensions and normalize by num_boxes    
        loss_ce = loss_ce.sum() / num_boxes
        
        losses = {'loss_class': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, weights=None):
        """
        Compute L1 loss and GIoU loss for bounding boxes
        """
        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"].reshape(-1, 4)
        target_boxes = targets["pred_boxes"].reshape(-1, 4)

        # Calculate L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        
        # If weights are provided, apply weighting
        if weights is not None:
            weights = weights.view(-1, 1)  # Expand dimensions for broadcasting
            loss_bbox = loss_bbox * weights
        
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # Calculate GIoU loss
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        
        # If weights are provided, apply weighting to GIoU loss as well
        if weights is not None:
            weights = weights.squeeze(-1)  # Ensure dimensions match
            loss_giou = loss_giou * weights
            
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses
    
    def get_dis_loss(self, preds_S, preds_T,index):
        preds_S = self.align[index](preds_S) # Align teacher and student models first
        
        loss_mse = nn.MSELoss(reduction='mean')
        N, C, H, W = preds_T.shape

        dis_loss = loss_mse(preds_S, preds_T) / N

        return dis_loss
    
    def get_loss_2(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": super().loss_labels,
            "boxes": super().loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs_s, outputs_t,targets):
        losses = {}
        # Calculate neck feature distillation loss
        loss_dis = 0
        for i, (features_s, features_t) in enumerate(zip(outputs_s["multi_level_features"], outputs_t["multi_level_features"])):
            # Calculate teacher model's discriminative score
            loss_dis += self.get_dis_loss(features_s,features_t,i)
        losses["loss_distill_dis"] = loss_dis / len(outputs_s["multi_level_features"])

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
