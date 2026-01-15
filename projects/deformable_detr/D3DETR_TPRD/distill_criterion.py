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
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou,box_iou
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

    def get_new_teacher_indices(self, indices, teacher_indices):
        new_indices = []
        for b in range(len(indices)):
            stu_idx, tea_idx = indices[b][0], indices[b][1]
            new_tea_idx = []
            new_gt_idx = []

            for idx, gt_idx in zip(teacher_indices[b][0], teacher_indices[b][1]):
                # Find the position of idx in tea_idx
                if idx in tea_idx:
                    pos = (tea_idx == idx).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        # Use position in tea_idx directly
                        new_tea_idx.append(pos[0].item())
                        new_gt_idx.append(gt_idx)

            new_indices.append((torch.as_tensor(new_tea_idx, device=stu_idx.device,dtype=torch.int64), 
                              torch.as_tensor(new_gt_idx, device=stu_idx.device,dtype=torch.int64)))

        return new_indices

    def loss_labels_distill(self, outputs, targets, indices, avg_factor,GTs = None,teacher_indices = None):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        B,N,C = outputs["pred_logits"].shape

        src_logits = outputs["pred_logits"][src_idx].reshape(B,N,C)  # Student network prediction
        tgt_logits = targets["pred_logits"][tgt_idx].reshape(B,N,C)  # Teacher network prediction

        teacher_indices = self.get_new_teacher_indices(indices,teacher_indices)

        idx = self._get_src_permutation_idx(teacher_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(GTs, teacher_indices)])

        src_prob = F.softmax(src_logits, dim=-1)  # [B, Q, C] 
        tgt_prob = F.softmax(tgt_logits, dim=-1)  # [B, Q, C]

        # Create negative sample mask and get positive/negative sample probabilities
        neg_mask = torch.ones(src_logits.shape[:2], dtype=torch.bool)
        neg_mask[idx] = False

        pos_stu = src_prob[idx]  # [N+, C] 
        pos_tea = tgt_prob[idx]  # [N+, C]
        target_classes_onehot = F.one_hot(target_classes_o, num_classes=src_logits.shape[-1])  # [N+, C]

        # Compute target class probability and non-target class probability for positive samples
        pt_stu = torch.sum(pos_stu * target_classes_onehot, dim=-1)  # [N+]
        pt_tea = torch.sum(pos_tea * target_classes_onehot, dim=-1)  # [N+]

        # Compute TCKD
        tckd = F.kl_div(
            torch.log(pt_stu), pt_tea, reduction="none"
        ) + F.kl_div(
            torch.log(1 - pt_stu), 1 - pt_tea, reduction="none"
        )  # [N+]

        # Compute NCKD
        non_target_mask = ~target_classes_onehot.bool()

        pnct_stu = src_logits[idx][non_target_mask].view(pos_stu.size(0),-1) # [N+, C-1]
        pnct_tea = tgt_logits[idx][non_target_mask].view(pos_tea.size(0),-1) # [N+, C-1]

        nckd = F.kl_div(
            F.log_softmax(pnct_stu, dim=-1),
            F.softmax(pnct_tea, dim=-1),
            reduction="none"
        ).sum(dim=-1)  # [N+]

        # Compute positive sample loss
        loss_ce = torch.zeros(src_logits.shape[:2], dtype=src_logits.dtype,device = src_logits.device)
        alpha = 1.0
        beta = 1.0  
        loss_ce[idx] = (alpha * tckd + beta * nckd) * (1 ** 2) # temeprature = 1 here

        # Compute negative sample loss
        neg_loss = F.kl_div(
            F.log_softmax(src_logits[neg_mask], dim=-1),
            F.softmax(tgt_logits[neg_mask], dim=-1),
            reduction="none"
        ).sum(dim=-1) * (1 ** 2)

        neg_indices = torch.nonzero(neg_mask)
        loss_ce[neg_indices[:, 0], neg_indices[:, 1]] = neg_loss

        loss_ce = loss_ce.sum() / avg_factor

        return {'loss_class': loss_ce}

    def loss_boxes_distill(self, outputs, targets, indices, avg_factor,GTs = None,teacher_indices = None):
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

    def optimize_single_stage(self, all_stages_output, current_stage_idx, targets, indices):
        """
        refining single stage prediction
        """
        num_stages = len(all_stages_output)
        batch_size = len(targets)
        device = all_stages_output[0]["pred_logits"].device
        num_queries = all_stages_output[0]["pred_logits"].shape[1]

        # Prepare predictions of all stages in advance
        all_stages_logits = torch.stack([stage["pred_logits"] for stage in all_stages_output], dim=0)  # [num_stages, B, Q, C]
        all_stages_boxes = torch.stack([stage["pred_boxes"] for stage in all_stages_output], dim=0)    # [num_stages, B, Q, 4]

        # Create output container
        optimized_logits = all_stages_logits[current_stage_idx-1].clone()
        optimized_boxes = all_stages_boxes[current_stage_idx-1].clone()

        # Batch compute max scores and classes of all stages
        max_scores, pred_classes = torch.max(all_stages_logits, dim=-1)  # [num_stages, B, Q]

        # Get scores and classes of current stage as baseline
        current_stage_scores = max_scores[current_stage_idx-1]  # [B, Q]
        current_stage_classes = pred_classes[current_stage_idx-1]  # [B, Q]

        # Batch process all batches
        for b in range(batch_size):
            if len(targets[b]["labels"]) == 0:
                continue

            src_idx, tgt_idx = indices[b]
            negative_mask = torch.ones(num_queries, dtype=torch.bool, device=device)
            negative_mask[src_idx] = False

            if len(src_idx) > 0:
                # Handle positive samples
                target_classes = targets[b]["labels"][tgt_idx]
                target_boxes = targets[b]["boxes"][tgt_idx]

                # compute classification scores [num_stages, num_positive]
                pos_logits = all_stages_logits[:current_stage_idx, b, src_idx]  # [current_stage_idx, num_positive, num_classes]
                class_scores = torch.gather(F.sigmoid(pos_logits), 2, 
                                        target_classes.view(1, -1, 1).expand(current_stage_idx, -1, 1)).squeeze(-1)

                # compute IoU
                pos_boxes = all_stages_boxes[:current_stage_idx, b, src_idx]  # [current_stage_idx, num_positive, 4]
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pos_boxes.reshape(-1, 4)).reshape(current_stage_idx, -1, 4)
                target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

                # Compute IoU for previous stages
                iou_scores = torch.zeros((current_stage_idx, len(src_idx)), device=device)
                for stage in range(current_stage_idx):
                    iou_scores[stage] = box_iou(pred_boxes_xyxy[stage], target_boxes_xyxy)[0].diag()

                # Get score of current stage as baseline
                current_iou = iou_scores[current_stage_idx-1]
                current_cls = class_scores[current_stage_idx-1]

                # Find better predictions
                better_mask = (iou_scores > current_iou.unsqueeze(0)) & (class_scores > current_cls.unsqueeze(0))

                # For each positive sample, find the best stage
                for i, (query_idx, gt_idx) in enumerate(zip(src_idx, tgt_idx)):
                    better_stages = torch.where(better_mask[:, i])[0]
                    if len(better_stages) > 0:
                        best_stage = better_stages[torch.argmax(iou_scores[better_stages, i])]
                        optimized_boxes[b, query_idx] = all_stages_boxes[best_stage, b, query_idx]
                        optimized_logits[b, query_idx, target_classes[i]] = all_stages_logits[best_stage, b, query_idx, target_classes[i]]

            # Handle negative samples
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

                if valid_updates.any():  # Only proceed if there are samples to update
                    update_indices = negative_indices[valid_updates]
                    best_valid_stages = best_stages[valid_updates]
                    update_classes = current_neg_classes[0, valid_updates]

                    optimized_logits[b, update_indices, update_classes] = all_stages_logits[
                        best_valid_stages, b, update_indices, update_classes
                    ]

        return {"pred_logits": optimized_logits, "pred_boxes": optimized_boxes,"self_attn_maps":all_stages_output[current_stage_idx-1]["self_attn_maps"],"cross_attn_maps":all_stages_output[current_stage_idx-1]["cross_attn_maps"]}

    def loss_attn_distill(self,outputs,targets,indices,avg_factor,GTs = None,teacher_indices = None):
        """
        Compute attn distillation loss
        """
        stu_self_attn_map = outputs["self_attn_maps"] # [B,num_heads,N,N]
        tea_self_attn_map = targets["self_attn_maps"] # [B,num_heads,N,N]
        stu_cross_attn_map = outputs["cross_attn_maps"] # [B,num_heads,HW]
        tea_cross_attn_map = targets["cross_attn_maps"] # [B,num_heads,HW]

        # Align attn map of teacher model and student model
        batch_size = stu_self_attn_map.shape[0]

        total_self_attn_loss = 0.0
        total_cross_attn_loss = 0.0

        for b in range(batch_size):
            # Get matching index of current batch
            src_indices_b = indices[b][0]  # query index in student model
            tgt_indices_b = indices[b][1]  # query index in teacher model

            # Align self attention maps
            # Student model: Select attention corresponding to matched query [num_heads, matched_queries, matched_queries]
            stu_self_attn_b = stu_self_attn_map[b, :, src_indices_b, :][:, :, src_indices_b]
            # Teacher model: Select attention corresponding to matched query [num_heads, matched_queries, matched_queries]
            tea_self_attn_b = tea_self_attn_map[b, :, tgt_indices_b, :][:, :, tgt_indices_b]

            # Compute L1 loss
            self_attn_loss = F.mse_loss(stu_self_attn_b, tea_self_attn_b)
            cross_attn_loss = F.mse_loss(stu_cross_attn_map[b], tea_cross_attn_map[b])

            total_self_attn_loss += self_attn_loss
            total_cross_attn_loss += cross_attn_loss

        total_self_attn_loss = total_self_attn_loss / batch_size
        total_cross_attn_loss = total_cross_attn_loss / batch_size

        return {
            "loss_self_attn": total_self_attn_loss,
            "loss_cross_attn": total_cross_attn_loss
        }

    def get_loss_distill(self, loss, outputs, targets, indices, avg_factor, **kwargs):
        loss_map = {
            "class": self.loss_labels_distill,
            "boxes": self.loss_boxes_distill,
            "attn": self.loss_attn_distill
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices,avg_factor, **kwargs)

    def forward(self, outputs_s, outputs_t,targets):
        all_stage_outputs = outputs_t["aux_outputs"] + [{"pred_logits": outputs_t["pred_logits"], "pred_boxes": outputs_t["pred_boxes"],"self_attn_maps":outputs_t["self_attn_maps"],"cross_attn_maps":outputs_t["cross_attn_maps"]}]

        teacher_indices = []
        optimized_outputs_t = []
        teacher_indices.append(self.matcher(all_stage_outputs[-1], targets))
        optimized_outputs_t.append(self.optimize_single_stage(all_stage_outputs, 6, targets, teacher_indices[0]))

        avg_factor = outputs_s["pred_logits"].shape[0] * outputs_s["pred_logits"].shape[1]
        # Matching result between teacher and student queries
        indices = self.distill_matcher(outputs_s, optimized_outputs_t[0])

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {"teacher_indices":teacher_indices[0],"GTs":targets}
            losses.update(self.get_loss_distill(loss, outputs_s, optimized_outputs_t[0], indices, avg_factor,**kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs_s:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s["aux_outputs"],outputs_t["aux_outputs"])):
                teacher_indices.append(self.matcher(aux_outputs_t, targets))
                optimized_outputs_t.append(self.optimize_single_stage(all_stage_outputs, i+1, targets, teacher_indices[i+1]))
                indices = self.distill_matcher(aux_outputs_s, optimized_outputs_t[i+1])
                for loss in self.losses:
                    kwargs = {"teacher_indices":teacher_indices[i+1],"GTs":targets}
                    l_dict = self.get_loss_distill(loss, aux_outputs_s, optimized_outputs_t[i+1], indices, avg_factor,**kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses = {"distill_" + k: v for k, v in losses.items()}

        # Compute fixed matching loss
        outputs_s_teacher = outputs_s["teacher_predictions"]
        batch_size, num_queries = outputs_s_teacher["pred_logits"].shape[:2]
        num_boxes = batch_size * num_queries
        teacher_losses = {}
        indices = [] # Build one-to-one correspondence between teacher model and student model
        for i in range(batch_size):
            src_idx = torch.arange(0,num_queries).long().cuda()
            tgt_idx = src_idx
            indices.append((src_idx,tgt_idx))

        for loss in self.losses:
            kwargs = {"teacher_indices":teacher_indices[0],"GTs":targets}
            teacher_losses.update(self.get_loss_distill(loss, outputs_s_teacher,optimized_outputs_t[0], indices, num_boxes,**kwargs))

        if "aux_outputs" in outputs_s_teacher:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s_teacher["aux_outputs"],outputs_t["aux_outputs"])):
                kwargs = {"teacher_indices":teacher_indices[i+1],"GTs":targets}
                for loss in self.losses:
                    l_dict = self.get_loss_distill(loss, aux_outputs_s,optimized_outputs_t[i+1], indices , num_boxes,**kwargs)
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
