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
from re import T
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.modeling import SetCriterion
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou,box_iou
from torchvision.ops.boxes import box_area

def batch_box_iou(boxes1, boxes2):
    """IoU calculation function supporting batch processing
    
    Calculate IoU and union of two sets of boxes in batch dimension
    
    Args:
        boxes1: (torch.Tensor[B, N, 4]): The first set of bounding boxes, B is batch size, N is number of boxes per batch
        boxes2: (torch.Tensor[B, M, 4]): The second set of bounding boxes, B is batch size, M is number of boxes per batch
        
    Returns:
        Tuple: Tuple containing pairwise IoU and union, shape (torch.Tensor[B, N, M], torch.Tensor[B, N, M])
    """
    B, N = boxes1.shape[:2]
    M = boxes2.shape[1]
    
    # Calculate area, keeping batch dimension
    area1 = box_area(boxes1.reshape(-1, 4)).reshape(B, N)  # [B, N]
    area2 = box_area(boxes2.reshape(-1, 4)).reshape(B, M)  # [B, M]
    
    # [B, N, 1, 2] -> [B, N, M, 2]
    boxes1_expand = boxes1.unsqueeze(2).expand(B, N, M, 4)
    # [B, 1, M, 2] -> [B, N, M, 2]
    boxes2_expand = boxes2.unsqueeze(1).expand(B, N, M, 4)
    
    # Calculate top-left and bottom-right of intersection
    lt = torch.max(boxes1_expand[..., :2], boxes2_expand[..., :2])  # [B, N, M, 2]
    rb = torch.min(boxes1_expand[..., 2:], boxes2_expand[..., 2:])  # [B, N, M, 2]
    
    # Calculate intersection area
    wh = (rb - lt).clamp(min=0)  # [B, N, M, 2]
    inter = wh[..., 0] * wh[..., 1]  # [B, N, M]
    
    # Calculate union area
    union = area1.unsqueeze(2) + area2.unsqueeze(1) - inter  # [B, N, M]
    
    # Calculate IoU
    iou = inter / (union + 1e-6)
    
    return iou, union


class DistillCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(self, temperature=10.0, **kwargs):
        super().__init__(**kwargs)
        self.T = temperature

    def _normalize_weights(self, weights, dim=-1):
        """
        Helper function to normalize weights
        """
        if weights is not None:
            return weights / weights.sum(dim=dim, keepdim=True)
        return None

    def _weighted_mean(self, loss, weights, dim=1):
        """
        Helper function to compute weighted mean loss
        """
        if weights is not None:
            norm_weights = self._normalize_weights(weights,dim=1)
            return (loss * norm_weights).sum() 
        return loss.mean(dim=dim).sum()

    def loss_labels_distill(self, outputs, outputs_t, indices, num_boxes, weights=None,decoupled = False,targets = None):

        src_logits = outputs["pred_logits"]  # Student network prediction
        tgt_logits = outputs_t["pred_logits"]  # Teacher network prediction

        if not decoupled:
            # Softmax processing on logits
            tgt_prob = F.softmax(tgt_logits / self.T, dim=-1)

            # Compute binary cross entropy loss
            loss_ce = F.kl_div(F.log_softmax(src_logits / self.T, dim=-1), tgt_prob, reduction="none").sum(dim= -1) * self.T * self.T # [batch_size，num_queries]

        else:
            # Create one hot labels
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

            src_prob = F.softmax(src_logits, dim=-1)  # [B, Q, C] 
            tgt_prob = F.softmax(tgt_logits, dim=-1)  # [B, Q, C]

            # Create negative sample mask and get positive/negative sample probabilities
            neg_mask = torch.ones(src_logits.shape[:2], dtype=torch.bool)
            neg_mask[idx] = False

            pos_stu = src_prob[idx]  # [N+, C] 
            pos_tea = tgt_prob[idx]  # [N+, C]
            target_classes_onehot = F.one_hot(target_classes_o, num_classes=src_logits.shape[-1])  # [N+, C]

            pt_stu = torch.sum(pos_stu * target_classes_onehot, dim=-1)  # [N+]
            pt_tea = torch.sum(pos_tea * target_classes_onehot, dim=-1)  # [N+]

            # Calculate TCKD
            tckd = F.kl_div(
                torch.log(pt_stu), pt_tea, reduction="none"
            ) + F.kl_div(
                torch.log(1 - pt_stu), 1 - pt_tea, reduction="none"
            )  # [N+]

            # Calculate NCKD
            non_target_mask = ~target_classes_onehot.bool()
            pnct_stu = src_logits[idx][non_target_mask].view(pos_stu.size(0),-1) # [N+, C-1]
            pnct_tea = tgt_logits[idx][non_target_mask].view(pos_tea.size(0),-1) # [N+, C-1]

            nckd = F.kl_div(
                F.log_softmax(pnct_stu, dim=-1),
                F.softmax(pnct_tea, dim=-1),
                reduction="none"
            ).sum(dim=-1)  # [N+]

            # Calculate positive sample loss
            loss_ce = torch.zeros(src_logits.shape[:2], dtype=src_logits.dtype,device = src_logits.device)
            alpha = 1.0
            beta = 1.0  
            loss_ce[idx] = (alpha * tckd + beta * nckd) * (1 ** 2)

            # Calculate negative sample loss
            neg_loss = F.kl_div(
                F.log_softmax(src_logits[neg_mask], dim=-1),
                F.softmax(tgt_logits[neg_mask], dim=-1),
                reduction="none"
            ).sum(dim=-1) * (1 ** 2)

            neg_indices = torch.nonzero(neg_mask)
            loss_ce[neg_indices[:, 0], neg_indices[:, 1]] = neg_loss

        # Calculate weighted mean loss
        return {'loss_class': self._weighted_mean(loss_ce, weights) / num_boxes}

    def loss_boxes_distill(self, outputs, outputs_t, indices, num_boxes, weights=None,decoupled = False,targets = None):
        """
        Compute L1 loss and GIoU loss for bounding boxes
        """
        src_boxes = outputs["pred_boxes"]
        target_boxes = outputs_t["pred_boxes"]

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

    def optimize_single_stage(self, all_stages_output, current_stage_idx, targets, indices):
        """
        refining single stage prediction
        """
        num_stages = len(all_stages_output)
        batch_size = len(targets)
        device = all_stages_output[0]["pred_logits"].device
        num_queries = all_stages_output[0]["pred_logits"].shape[1]

        # Prepare predictions for all stages in advance
        all_stages_logits = torch.stack([stage["pred_logits"] for stage in all_stages_output], dim=0)  # [num_stages, B, Q, C]
        all_stages_boxes = torch.stack([stage["pred_boxes"] for stage in all_stages_output], dim=0)    # [num_stages, B, Q, 4]

        optimized_logits = all_stages_logits[current_stage_idx-1].clone()
        optimized_boxes = all_stages_boxes[current_stage_idx-1].clone()

        max_scores, pred_classes = torch.max(all_stages_logits, dim=-1)  # [num_stages, B, Q]

        # Get scores and classes of current stage as baseline
        current_stage_scores = max_scores[current_stage_idx-1]  # [B, Q]
        current_stage_classes = pred_classes[current_stage_idx-1]  # [B, Q]

        # Process all batches
        for b in range(batch_size):
            if len(targets[b]["labels"]) == 0:
                continue

            src_idx, tgt_idx = indices[b]
            negative_mask = torch.ones(num_queries, dtype=torch.bool, device=device)
            negative_mask[src_idx] = False

            if len(src_idx) > 0:
                # Process positive samples
                target_classes = targets[b]["labels"][tgt_idx]
                target_boxes = targets[b]["boxes"][tgt_idx]

                # Compute classification scores [num_stages, num_positive]
                pos_logits = all_stages_logits[:current_stage_idx, b, src_idx]  # [current_stage_idx, num_positive, num_classes]
                class_scores = torch.gather(F.sigmoid(pos_logits), 2, 
                                        target_classes.view(1, -1, 1).expand(current_stage_idx, -1, 1)).squeeze(-1)

                # Compute IoU
                pos_boxes = all_stages_boxes[:current_stage_idx, b, src_idx]  # [current_stage_idx, num_positive, 4]
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pos_boxes.reshape(-1, 4)).reshape(current_stage_idx, -1, 4)
                target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

                # Compute IoU of previous stages
                iou_scores = torch.zeros((current_stage_idx, len(src_idx)), device=device)
                for stage in range(current_stage_idx):
                    iou_scores[stage] = box_iou(pred_boxes_xyxy[stage], target_boxes_xyxy)[0].diag()

                # Get score of current stage as baseline
                current_iou = iou_scores[current_stage_idx-1]
                current_cls = class_scores[current_stage_idx-1]

                # Find better predictions
                better_mask = (iou_scores > current_iou.unsqueeze(0)) & (class_scores > current_cls.unsqueeze(0))

                # Find best stage for each positive sample
                for i, (query_idx, gt_idx) in enumerate(zip(src_idx, tgt_idx)):
                    better_stages = torch.where(better_mask[:, i])[0]
                    if len(better_stages) > 0:
                        best_stage = better_stages[torch.argmax(iou_scores[better_stages, i])]
                        optimized_boxes[b, query_idx] = all_stages_boxes[best_stage, b, query_idx]
                        optimized_logits[b, query_idx, target_classes[i]] = all_stages_logits[best_stage, b, query_idx, target_classes[i]]

            # Process negative samples
            negative_indices = torch.where(negative_mask)[0]
            if len(negative_indices) > 0:
                stage_scores = max_scores[:current_stage_idx, b, negative_indices]  # [num_stages, num_neg]
                stage_classes = pred_classes[:current_stage_idx, b, negative_indices]  # [num_stages, num_neg]
                current_neg_scores = current_stage_scores[b, negative_indices].unsqueeze(0)  # [1, num_neg]
                current_neg_classes = current_stage_classes[b, negative_indices].unsqueeze(0)  # [1, num_neg]

                valid_mask = (stage_classes == current_neg_classes) & (stage_scores < current_neg_scores)

                # Find best stage for each negative sample
                best_stages = torch.argmin(stage_scores.masked_fill(~valid_mask, float('inf')), dim=0)
                valid_updates = valid_mask.any(dim=0)

                if valid_updates.any():  # Only operate when there are samples needing update
                    update_indices = negative_indices[valid_updates]
                    best_valid_stages = best_stages[valid_updates]
                    update_classes = current_neg_classes[0, valid_updates]

                    optimized_logits[b, update_indices, update_classes] = all_stages_logits[
                        best_valid_stages, b, update_indices, update_classes
                    ]

        return {"pred_logits": optimized_logits, "pred_boxes": optimized_boxes}

    def get_loss(self, loss, outputs, outputs_t, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels_distill,
            "boxes": self.loss_boxes_distill,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs,outputs_t, indices, num_boxes, **kwargs)

    def forward(self, outputs_s, outputs_t,targets):
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

        if "aux_outputs" in outputs_s_auxilary:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s_auxilary["aux_outputs"],outputs_t_auxilary["aux_outputs"])):
                weights = F.sigmoid(aux_outputs_t["pred_logits"].max(-1)[0])
                for loss in self.losses:
                    kwargs = {"weights":weights}
                    l_dict = self.get_loss(loss, aux_outputs_s, aux_outputs_t, None , num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses = {"auxilary_" + k: v for k,v in losses.items()}    

        all_stage_outputs = outputs_t["aux_outputs"] + [{"pred_logits": outputs_t["pred_logits"], "pred_boxes": outputs_t["pred_boxes"]}]
        indices = self.matcher(all_stage_outputs[-1], targets)
        optimized_outputs_t = []
        optimized_outputs_t.append(self.optimize_single_stage(all_stage_outputs, 6, targets, indices))

        # Compute teacher query loss
        outputs_s_teacher = outputs_s["teacher_predictions"]
        num_boxes = outputs_s_teacher["pred_logits"].shape[0]
        # Compute weight for each sample
        weights = F.sigmoid(optimized_outputs_t[0]["pred_logits"].max(-1)[0]) # [batch_size * num_queries]

        teacher_losses = {}
        for loss in self.losses:
            kwargs = {"weights":weights,"decoupled":True,"targets":targets}
            teacher_losses.update(self.get_loss(loss, outputs_s_teacher, optimized_outputs_t[0], indices, num_boxes, **kwargs))

        if "aux_outputs" in outputs_s_teacher:
            for i, aux_outputs_s in enumerate(outputs_s_teacher["aux_outputs"]):
                indices = self.matcher(all_stage_outputs[i], targets)
                optimized_outputs_t.append(self.optimize_single_stage(all_stage_outputs, i+1, targets, indices))
                weights = F.sigmoid(optimized_outputs_t[i+1]["pred_logits"].max(-1)[0])
                for loss in self.losses:
                    kwargs = {"weights":weights,"decoupled":True,"targets":targets}
                    l_dict = self.get_loss(loss, aux_outputs_s, optimized_outputs_t[i+1], indices , num_boxes, **kwargs)
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
