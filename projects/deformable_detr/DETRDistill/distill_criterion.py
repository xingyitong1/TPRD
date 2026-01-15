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

from audioop import avg
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

    def __init__(self, distill_matcher=None, **kwargs):
        super().__init__(**kwargs)
        self.distill_matcher = distill_matcher

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

    def loss_labels_distill(self, outputs, targets, indices, avg_factor):
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_logits = outputs["pred_logits"][src_idx]  # Student network prediction
        tgt_logits = targets["pred_logits"][tgt_idx]  # Teacher network prediction
        
        # Apply sigmoid to logits
        tgt_prob = F.sigmoid(tgt_logits)
        
        # Compute binary cross entropy loss
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, tgt_prob, reduction="none").sum()
        # Compute weight for each point, analogous to focal loss
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
        # Matching results between teacher and student queries
        indices = self.distill_matcher(outputs_s, outputs_t)
        
        avg_factor = outputs_s['pred_logits'].shape[0] * outputs_s["pred_logits"].shape[1] # batch_size * num_quries

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss_distill(loss, outputs_s, outputs_t, indices, avg_factor))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs_s:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s["aux_outputs"],outputs_t["aux_outputs"])):
                indices = self.distill_matcher(aux_outputs_s, aux_outputs_t)
                for loss in self.losses:
                    l_dict = self.get_loss_distill(loss, aux_outputs_s, aux_outputs_t, indices, avg_factor)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        losses = {"distill_" + k: v for k, v in losses.items()}
        
        # Compute target-aware feature distillation
        all_stage_cls_iou_score = outputs_t['all_stage_cls_iou_score']
        all_stage_weight_query = outputs_t['all_stage_weight_query']
        stages = len(all_stage_cls_iou_score)
        loss_mse = nn.MSELoss(reduction='mean')
        for stage in range(stages):
            stage_cls_iou_score = all_stage_cls_iou_score[stage]
            stage_cls_iou_score = torch.stack(stage_cls_iou_score, dim=0)
            teacher_stage_det_querys = all_stage_weight_query[stage]
            teacher_stage_det_querys = torch.stack(teacher_stage_det_querys, dim=0)

            stage_loss = 0
            c_querys = torch.nn.functional.normalize(teacher_stage_det_querys, dim=-1)
            for fea_tea, fea_stu in zip(outputs_t["features"], outputs_s["features"]):

                c_feats = torch.nn.functional.normalize(fea_tea, dim=1)
                mat = torch.einsum('bnc,bchw->bnhw', [c_querys, c_feats])

                mask = torch.einsum('bnhw,bn->bhw', [mat, stage_cls_iou_score]).clamp(min=1e-2)

                max_shu = torch.max(mask.flatten(1,2),dim=-1)[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                mask = (mask / max_shu).unsqueeze(dim=1)

                stage_loss += loss_mse(mask * fea_tea, mask * fea_stu)
                if stage == 5:
                    losses['target_aware_loss'] = stage_loss
                else:
                    losses[f'target_aware_loss_{stage}'] = stage_loss
        
        # Compute teacher prior assignment loss
        outputs_s_teacher = outputs_s["teacher_predictions"]
        teacher_losses = {}
        indices = self.matcher(outputs_t,targets)
        
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs_t.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        for loss in self.losses:
            teacher_losses.update(self.get_loss(loss, outputs_s_teacher,targets, indices, num_boxes))
        
        if "aux_outputs" in outputs_s_teacher:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s_teacher["aux_outputs"],outputs_t["aux_outputs"])):
                indices = self.matcher(aux_outputs_t, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs_s, targets, indices , num_boxes)
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
    

