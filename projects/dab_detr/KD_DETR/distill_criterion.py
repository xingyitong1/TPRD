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


class DistillCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(self, temperature=10.0, **kwargs):
        super().__init__(**kwargs)
        self.T = temperature

    def _normalize_weights(self, weights, dim=-1):
        """
        Auxiliary function for normalizing weights
        """
        if weights is not None:
            return weights / weights.sum(dim=dim, keepdim=True)
        return None

    def _weighted_mean(self, loss, weights, dim=1):
        """
        Auxiliary function for computing weighted mean loss
        """
        if weights is not None:
            norm_weights = self._normalize_weights(weights,dim=1)
            return (loss * norm_weights).sum() 
        return loss.mean(dim=dim).sum()

    def loss_labels_distill(self, outputs, targets, indices, num_boxes, weights=None):
        """
        Compute KL divergence loss between student and teacher outputs
        Args:
            outputs: Dictionary containing predicted logits
            targets: Dictionary containing target logits
            indices: Matching information (not used here)
            num_boxes: Number of boxes for loss normalization
            weights: Weights for each sample, shape should be same as batch dimension of logits
        Returns:
            Dictionary containing classification loss
        """
        src_logits = outputs["pred_logits"]  # student network prediction
        tgt_logits = targets["pred_logits"]  # teacher network prediction
        
        # Apply softmax to logits
        tgt_prob = F.softmax(tgt_logits / self.T, dim=-1)
        
        # Compute KL divergence loss
        loss_ce = F.kl_div(F.log_softmax(src_logits / self.T, dim=-1), tgt_prob, reduction="none").sum(dim= -1) * self.T * self.T # [batch_size，num_queries]
        # Analogous to focal loss, compute weight for each point
        loss_ce = self._weighted_mean(loss_ce, weights) / num_boxes
        
        return {'loss_class': loss_ce}

    def loss_boxes_distill(self, outputs, targets, indices, num_boxes, weights=None):
        """
        Compute L1 loss and GIoU loss for bounding boxes
        """
        src_boxes = outputs["pred_boxes"]
        target_boxes = targets["pred_boxes"]

        # Compute L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum(dim=-1)
        loss_bbox = self._weighted_mean(loss_bbox, weights) / num_boxes

        # Compute GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes.reshape(-1, 4))
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes.reshape(-1, 4))
        
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        loss_giou = loss_giou.reshape(src_boxes.shape[0], src_boxes.shape[1])
        loss_giou = self._weighted_mean(loss_giou, weights) / num_boxes

        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels_distill,
            "boxes": self.loss_boxes_distill,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs_s, outputs_t):
        # auxiliary predictions
        outputs_s_auxilary = outputs_s["auxiliary_predictions"]
        outputs_t_auxilary = outputs_t["auxiliary_predictions"]
        
        num_boxes = outputs_s_auxilary["pred_logits"].shape[0]
        weights = F.sigmoid(outputs_t_auxilary["pred_logits"].max(-1)[0]) # [batch_size * num_queries]

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {"weights":weights}
            losses.update(self.get_loss(loss, outputs_s_auxilary, outputs_t_auxilary, None, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs_s_auxilary:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s_auxilary["aux_outputs"],outputs_t_auxilary["aux_outputs"])):
                weights = F.sigmoid(aux_outputs_t["pred_logits"].max(-1)[0])
                for loss in self.losses:
                    kwargs = {"weights":weights}
                    l_dict = self.get_loss(loss, aux_outputs_s, aux_outputs_t, None , num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses = {"auxilary_" + k: v for k,v in losses.items()}    

        # Compute loss for teacher query
        outputs_s_teacher = outputs_s["teacher_predictions"]
        num_boxes = outputs_s_teacher["pred_logits"].shape[0]
        # Compute weight for each sample
        weights = F.sigmoid(outputs_t["pred_logits"].max(-1)[0]) # [batch_size * num_queries]
        
        teacher_losses = {}
        for loss in self.losses:
            kwargs = {"weights":weights}
            teacher_losses.update(self.get_loss(loss, outputs_s_teacher, outputs_t, None, num_boxes, **kwargs))
        
        if "aux_outputs" in outputs_s_teacher:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s_teacher["aux_outputs"],outputs_t["aux_outputs"])):
                weights = F.sigmoid(aux_outputs_t["pred_logits"].max(-1)[0])
                for loss in self.losses:
                    kwargs = {"weights":weights}
                    l_dict = self.get_loss(loss, aux_outputs_s, aux_outputs_t, None , num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    teacher_losses.update(l_dict)
        
        teacher_losses = {"teacher_" + k : v for k,v in teacher_losses.items()}
        losses.update(teacher_losses)
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
    

