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

    def get_new_teacher_indices(self, indices, teacher_indices):
        new_indices = []
        current_idx = 0
        for b in range(len(indices)):
            stu_idx, tea_idx = indices[b][0], indices[b][1]
            new_tea_idx = []
            new_gt_idx = []

            for idx, gt_idx in zip(teacher_indices[b][0], teacher_indices[b][1]):
                # Find the position of idx in tea_idx
                if idx in tea_idx:
                    pos = (tea_idx == idx).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        # Use the position in tea_idx directly
                        new_tea_idx.append(current_idx + pos[0].item())
                        new_gt_idx.append(gt_idx)

            new_indices.append((torch.as_tensor(new_tea_idx, device=stu_idx.device,dtype=torch.int64), 
                              torch.as_tensor(new_gt_idx, device=stu_idx.device,dtype=torch.int64)))
            current_idx += len(tea_idx)
        return new_indices

    def loss_labels_distill(self, outputs, targets, indices, avg_factor,GTs = None,teacher_indices = None):

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # One-to-one output after matching
        src_logits = outputs["pred_logits"][src_idx] # Student network prediction
        tgt_logits = targets["pred_logits"][tgt_idx]  # Teacher network prediction

        teacher_indices = self.get_new_teacher_indices(indices,teacher_indices)

        idx = torch.cat([t[0] for t in teacher_indices]) # [N+]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(GTs, teacher_indices)])

        src_prob = F.softmax(src_logits, dim=-1)  # [B, Q, C] 
        tgt_prob = F.softmax(tgt_logits, dim=-1)  # [B, Q, C]

        neg_mask = torch.ones(src_logits.shape[0], dtype=torch.bool)
        neg_mask[idx] = False

        pos_stu = src_prob[idx]  # [N+, C] 
        pos_tea = tgt_prob[idx]  # [N+, C]
        target_classes_onehot = F.one_hot(target_classes_o, num_classes=src_logits.shape[-1])  # [N+, C]

        # Calculate positive sample target class probability and non-target class probability
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
        loss_ce = torch.zeros(src_logits.shape[0], dtype=src_logits.dtype,device = src_logits.device)
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
        loss_ce[neg_indices[:, 0]] = neg_loss 

        loss_ce = loss_ce.mean()

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
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum(dim=-1)
        loss_bbox = loss_bbox.mean()

        # Compute GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        loss_giou = loss_giou.mean()

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

    def select_hard_negative_query(self, outputs, indices, targets, giou_threshold=0):

        pred_boxes = outputs["pred_boxes"]  # [B,num_queries,4]
        B, num_queries, _ = pred_boxes.shape
        device = pred_boxes.device

        all_selected_indices = []
        all_selected_gious = []
        all_neg_indices = []

        # Compute GIoU for all batches to reduce repetitive computation
        for b in range(B):
            positive_indices = indices[b][0]  # Indices matched to positive samples
            positive_tgt_indices = indices[b][1]  # Corresponding GT indices
            target_boxes = targets[b]["boxes"]  # [num_gt,4]
            num_targets = len(target_boxes)

            if num_targets == 0:
                empty_tensor_long = torch.tensor([], device=device, dtype=torch.long)
                empty_tensor_float = torch.tensor([], device=device, dtype=torch.float)
                all_selected_indices.append(empty_tensor_long)
                all_selected_gious.append(empty_tensor_float)
                all_neg_indices.append(empty_tensor_long)
                continue

            # Compute GIoU matrix
            gious = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[b]),
                box_cxcywh_to_xyxy(target_boxes)
            )  # [num_queries, num_targets]

            # Collect positive sample GIoU scores
            if len(positive_indices) > 0:
                positive_gious = gious[positive_indices, positive_tgt_indices]
            else:
                positive_gious = torch.tensor([], device=device, dtype=torch.float)

            # Create negative sample mask
            neg_mask = torch.ones(num_queries, dtype=torch.bool, device=device)
            if len(positive_indices) > 0:
                neg_mask[positive_indices] = False

            if neg_mask.any():
                # Get max GIoU and corresponding target index for negative samples
                max_gious, max_target_idx = gious.max(dim=1)  # [num_queries]
                neg_max_gious = max_gious[neg_mask]
                neg_max_target_idx = max_target_idx[neg_mask]
                neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

                hard_mask = neg_max_gious > giou_threshold
                if hard_mask.any():
                    hard_neg_indices = neg_indices[hard_mask]
                    hard_neg_gious = neg_max_gious[hard_mask]
                else:
                    hard_neg_indices = torch.tensor([], device=device, dtype=torch.long)
                    hard_neg_gious = torch.tensor([], device=device, dtype=torch.float)
            else:
                hard_neg_indices = torch.tensor([], device=device, dtype=torch.long)
                hard_neg_gious = torch.tensor([], device=device, dtype=torch.float)

            # Combine positive samples and hard negative samples
            if len(positive_indices) > 0 and len(hard_neg_indices) > 0:
                all_indices = torch.cat([positive_indices.to(device), hard_neg_indices.to(device)])
                all_gious = torch.cat([positive_gious.to(device), hard_neg_gious.to(device)])
            elif len(positive_indices) > 0:
                all_indices = positive_indices
                all_gious = positive_gious
            elif len(hard_neg_indices) > 0:
                all_indices = hard_neg_indices
                all_gious = hard_neg_gious
            else:
                all_indices = torch.tensor([], device=device, dtype=torch.long)
                all_gious = torch.tensor([], device=device, dtype=torch.float)

            all_selected_indices.append(all_indices)
            all_selected_gious.append(all_gious)
            all_neg_indices.append(hard_neg_indices)

        return all_selected_indices, all_selected_gious, all_neg_indices

    def get_teacher_student_matching(self, outputs_s, outputs_t, targets):

        device = outputs_s["pred_logits"].device
        batch_size = outputs_s["pred_logits"].shape[0]

        # Get positive sample matching (GT matching)
        stu_pos_indices = self.matcher(outputs_s, targets)
        tea_pos_indices = self.matcher(outputs_t, targets)

        # Get hard negative samples
        tea_all_selected_indices, tea_all_selected_gious, tea_hard_neg_indices = self.select_hard_negative_query(outputs_t, tea_pos_indices, targets)
        _, _, stu_hard_neg_indices = self.select_hard_negative_query(outputs_s, stu_pos_indices, targets)

        # Construct final matching relationship
        indices = []

        for b in range(batch_size):
            # Get positive sample matching for current batch
            stu_pos_src = stu_pos_indices[b][0]  # Student model positive sample query index
            stu_pos_tgt = stu_pos_indices[b][1]  # GT index
            tea_pos_src = tea_pos_indices[b][0]  # Teacher model positive sample query index
            tea_pos_tgt = tea_pos_indices[b][1]  # GT index

            # Construct of one-to-one correspondence for positive samples
            pos_src_indices = []
            pos_tgt_indices = []

            if len(stu_pos_tgt) > 0 and len(tea_pos_tgt) > 0:
                # Find matching GT indices
                stu_pos_tgt_expanded = stu_pos_tgt.unsqueeze(1)  # [N, 1]
                tea_pos_tgt_expanded = tea_pos_tgt.unsqueeze(0)  # [1, M]
                matches = (stu_pos_tgt_expanded == tea_pos_tgt_expanded)  # [N, M]

                # Find matched indices
                stu_match_idx, tea_match_idx = torch.where(matches)

                if len(stu_match_idx) > 0:
                    pos_src_indices = stu_pos_src[stu_match_idx].tolist()
                    pos_tgt_indices = tea_pos_src[tea_match_idx].tolist()

            # Get hard negative samples for current batch
            stu_neg_indices = stu_hard_neg_indices[b]
            tea_neg_indices = tea_hard_neg_indices[b]

            # Bipartite matching for hard negative samples
            if len(stu_neg_indices) > 0 and len(tea_neg_indices) > 0:
                neg_src_indices, neg_tgt_indices = self._bipartite_match_negatives(
                    outputs_s, outputs_t, stu_neg_indices, tea_neg_indices, b
                )
            else:
                neg_src_indices = []
                neg_tgt_indices = []

            # Combine matching results of positive and negative samples
            all_src_indices = pos_src_indices + neg_src_indices
            all_tgt_indices = pos_tgt_indices + neg_tgt_indices

            if len(all_src_indices) > 0:
                src_tensor = torch.tensor(all_src_indices, dtype=torch.int64, device=device)
                tgt_tensor = torch.tensor(all_tgt_indices, dtype=torch.int64, device=device)
                indices.append((src_tensor, tgt_tensor))
            else:
                # If no matching, create empty matching
                empty_tensor = torch.tensor([], dtype=torch.int64, device=device)
                indices.append((empty_tensor, empty_tensor))

        return indices, tea_all_selected_indices, tea_all_selected_gious

    def _bipartite_match_negatives(self, outputs_s, outputs_t, stu_neg_indices, tea_neg_indices, batch_idx):
        if len(stu_neg_indices) == 0 or len(tea_neg_indices) == 0:
            return [], []

        # Construct output of student model hard negative samples
        stu_neg_outputs = {
            "pred_logits": outputs_s["pred_logits"][batch_idx:batch_idx+1, stu_neg_indices],
            "pred_boxes": outputs_s["pred_boxes"][batch_idx:batch_idx+1, stu_neg_indices],
        }

        # Construct output of teacher model hard negative samples
        tea_neg_outputs = {
            "pred_logits": outputs_t["pred_logits"][batch_idx:batch_idx+1, tea_neg_indices],
            "pred_boxes": outputs_t["pred_boxes"][batch_idx:batch_idx+1, tea_neg_indices],
        }

        # Use distill_matcher to perform matching
        neg_indices = self.distill_matcher(stu_neg_outputs, tea_neg_outputs)

        if len(neg_indices) == 0 or len(neg_indices[0][0]) == 0:
            return [], []

        # Get matching results and convert to absolute indices
        src_matched_indices = neg_indices[0][0]
        tgt_matched_indices = neg_indices[0][1]

        matched_src_indices = stu_neg_indices[src_matched_indices].tolist()
        matched_tgt_indices = tea_neg_indices[tgt_matched_indices].tolist()

        return matched_src_indices, matched_tgt_indices

    def get_loss_distill(self, loss, outputs, targets, indices, avg_factor, **kwargs):
        loss_map = {
            "class": self.loss_labels_distill,
            "boxes": self.loss_boxes_distill,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices,avg_factor, **kwargs)

    def forward(self, outputs_s, outputs_t,targets):
        # Optimize teacher model output
        all_stage_outputs = outputs_t["aux_outputs"] + [{"pred_logits": outputs_t["pred_logits"], "pred_boxes": outputs_t["pred_boxes"]}]

        teacher_indices = self.matcher(all_stage_outputs[-1], targets)
        optimized_outputs_t = self.optimize_single_stage(all_stage_outputs, 6, targets, teacher_indices)

        all_stage_indices = []
        all_stage_gious = []
        indices,all_selected_indices,all_selected_gious = self.get_teacher_student_matching(outputs_s, outputs_t, targets)
        all_stage_indices.append(all_selected_indices)
        all_stage_gious.append(all_selected_gious)

        avg_factor = outputs_s['pred_logits'].shape[0] * outputs_s["pred_logits"].shape[1] # batch_size * num_quries

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {"teacher_indices":teacher_indices,"GTs":targets}
            losses.update(self.get_loss_distill(loss, outputs_s, optimized_outputs_t, indices, avg_factor,**kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs_s:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s["aux_outputs"],outputs_t["aux_outputs"])):
                teacher_indices = self.matcher(aux_outputs_t, targets)
                optimized_outputs_t = self.optimize_single_stage(all_stage_outputs, i+1, targets, teacher_indices)
                indices,hard_neg_indices,hard_neg_gious = self.get_teacher_student_matching(aux_outputs_s, aux_outputs_t, targets)
                all_stage_indices.append(hard_neg_indices)
                all_stage_gious.append(hard_neg_gious)
                for loss in self.losses:
                    kwargs = {"teacher_indices":teacher_indices,"GTs":targets}
                    l_dict = self.get_loss_distill(loss, aux_outputs_s, optimized_outputs_t, indices, avg_factor,**kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses = {"distill_" + k: v for k, v in losses.items()}

        losses.update(self.compute_target_aware_loss(outputs_s, outputs_t, all_stage_indices, all_stage_gious))

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

    def generate_weighted_cross_attn_mask(self, hard_neg_indices, hard_neg_gious, cross_attn_maps):

        batch_size = cross_attn_maps.shape[0]
        hw = cross_attn_maps.shape[2]
        device = cross_attn_maps.device

        # Initialize mask
        weighted_masks = torch.zeros(batch_size, hw, device=device)

        for b in range(batch_size):
            batch_selected_indices = hard_neg_indices[b]
            batch_selected_gious = hard_neg_gious[b]

            if len(batch_selected_indices) == 0:
                continue

            # Compute weights and weighted attention
            weights = 1.0 + batch_selected_gious  # [num_selected]
            selected_cross_attn = cross_attn_maps[b, batch_selected_indices, :]  # [num_selected, HW]

            weighted_cross_attn = selected_cross_attn * weights.unsqueeze(-1)  # [num_selected, HW]

            # Average weighted attention for all selected queries
            weighted_masks[b] = weighted_cross_attn.mean(dim=0)  # [HW]

        return weighted_masks

    def compute_target_aware_loss(self, outputs_s, outputs_t, all_stage_indices, all_stage_gious):
        """
        Compute target-aware loss using weighted cross attention mask
        """
        losses = {}

        # Check necessary inputs in advance
        if ("features" not in outputs_t or "features" not in outputs_s or 
            "cross_attn_maps" not in outputs_t):
            return losses

        # Get multi-level features
        fea_tea = outputs_t["features"]
        fea_stu = outputs_s["features"]

        tea_cross_attn = outputs_t["cross_attn_maps"]  # [num_layers, B, num_queries, HW]
        stages = len(all_stage_indices)

        # Pre-allocate loss computation function
        loss_mse = nn.MSELoss(reduction='mean')

        for stage in range(stages):
            stage_hard_neg_indices = all_stage_indices[stage]
            stage_hard_neg_gious = all_stage_gious[stage]

            # Generate weighted cross-attention mask
            weighted_mask = self.generate_weighted_cross_attn_mask(
                stage_hard_neg_indices, stage_hard_neg_gious, tea_cross_attn[stage]
            )  # [B, HW_total]

            # Compute losses for all levels
            stage_losses = []
            B, HW,C = fea_tea.shape

            # Normalize weights
            max_vals = weighted_mask.max(dim=-1)[0]  # [B]
            max_vals = torch.clamp(max_vals, min=1e-2)
            weighted_mask = weighted_mask / max_vals.view(B, 1)  # [B, H*W]

            # Apply mask and compute loss
            mask_expanded = weighted_mask.unsqueeze(2)  # [B, H*W, 1]

            # Compute masked features
            masked_fea_tea = fea_tea * mask_expanded
            masked_fea_stu = fea_stu * mask_expanded

            # Compute MSE loss
            level_loss = loss_mse(masked_fea_tea, masked_fea_stu)
            stage_losses.append(level_loss)

            # Accumulate losses of all levels
            stage_loss = sum(stage_losses)

            # Save loss
            if stage == stages - 1:  # Last stage
                losses['target_aware_loss'] = stage_loss
            else:
                losses[f'target_aware_loss_{stage}'] = stage_loss

        return losses
