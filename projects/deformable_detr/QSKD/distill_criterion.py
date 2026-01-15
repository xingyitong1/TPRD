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

    # Rewrite loss_labels because KL divergence loss is used here
    def loss_labels_distill(self, outputs, targets, indices, avg_factor):
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_logits = outputs["pred_logits"][src_idx]  # Student network prediction
        tgt_logits = targets["pred_logits"][tgt_idx]  # Teacher network prediction
        
        # Apply sigmoid to logits
        tgt_prob = F.sigmoid(tgt_logits)
        
        # Compute binary cross entropy loss
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, tgt_prob, reduction="none").sum(dim=-1)# sum over class dimension
        loss_ce = loss_ce.mean() # average over batch dimension
        
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

    def select_hard_negative_query(self, outputs, indices, targets, giou_threshold=0):
        """
        Select hard negative samples and positive samples for relation matrix distillation - Optimized version
        
        Args:
            indices: Matching results containing positive sample indices
            outputs: Model prediction results
            targets: Ground truth labels
            giou_threshold: GIoU threshold, negative samples above this threshold are considered hard negatives
        
        Returns:
            all_selected_indices: List of all selected indices (including positive samples and hard negative samples)
            all_selected_gious: List of corresponding GIoU scores
            all_neg_indices: List of hard negative sample indices
        """
        pred_boxes = outputs["pred_boxes"]  # [B,num_queries,4]
        B, num_queries, _ = pred_boxes.shape
        device = pred_boxes.device
        
        all_selected_indices = []
        all_selected_gious = []
        all_neg_indices = []
        
        #  compute GIoU for all batches to reduce redundant calculations
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
            
            if len(positive_indices) > 0:
                positive_gious = gious[positive_indices, positive_tgt_indices]
            else:
                positive_gious = torch.tensor([], device=device, dtype=torch.float)
            
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
            
            # Merge positive samples and hard negative samples
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
            
            # Construct one-to-one correspondence for positive samples
            pos_src_indices = []
            pos_tgt_indices = []
            
            if len(stu_pos_tgt) > 0 and len(tea_pos_tgt) > 0:
                #  find matching GT indices
                stu_pos_tgt_expanded = stu_pos_tgt.unsqueeze(1)  # [N, 1]
                tea_pos_tgt_expanded = tea_pos_tgt.unsqueeze(0)  # [1, M]
                matches = (stu_pos_tgt_expanded == tea_pos_tgt_expanded)  # [N, M]
                
                # Find matching indices
                stu_match_idx, tea_match_idx = torch.where(matches)
                
                if len(stu_match_idx) > 0:
                    pos_src_indices = stu_pos_src[stu_match_idx].tolist()
                    pos_tgt_indices = tea_pos_src[tea_match_idx].tolist()
            
            # Get hard negative samples for current batch
            stu_neg_indices = stu_hard_neg_indices[b]
            tea_neg_indices = tea_hard_neg_indices[b]
            
            # Perform bipartite matching for hard negative samples
            if len(stu_neg_indices) > 0 and len(tea_neg_indices) > 0:
                neg_src_indices, neg_tgt_indices = self._bipartite_match_negatives(
                    outputs_s, outputs_t, stu_neg_indices, tea_neg_indices, b
                )
            else:
                neg_src_indices = []
                neg_tgt_indices = []
            
            # Merge matching results of positive samples and negative samples
            all_src_indices = pos_src_indices + neg_src_indices
            all_tgt_indices = pos_tgt_indices + neg_tgt_indices
            
            if len(all_src_indices) > 0:
                src_tensor = torch.tensor(all_src_indices, dtype=torch.int64, device=device)
                tgt_tensor = torch.tensor(all_tgt_indices, dtype=torch.int64, device=device)
                indices.append((src_tensor, tgt_tensor))
            else:
                # If no match, create empty match
                empty_tensor = torch.tensor([], dtype=torch.int64, device=device)
                indices.append((empty_tensor, empty_tensor))
        
        return indices, tea_all_selected_indices, tea_all_selected_gious

    def _bipartite_match_negatives(self, outputs_s, outputs_t, stu_neg_indices, tea_neg_indices, batch_idx):
        
        if len(stu_neg_indices) == 0 or len(tea_neg_indices) == 0:
            return [], []
        
        # Construct output for student model hard negative samples
        stu_neg_outputs = {
            "pred_logits": outputs_s["pred_logits"][batch_idx:batch_idx+1, stu_neg_indices],
            "pred_boxes": outputs_s["pred_boxes"][batch_idx:batch_idx+1, stu_neg_indices],
        }
        
        # Construct output for teacher model hard negative samples
        tea_neg_outputs = {
            "pred_logits": outputs_t["pred_logits"][batch_idx:batch_idx+1, tea_neg_indices],
            "pred_boxes": outputs_t["pred_boxes"][batch_idx:batch_idx+1, tea_neg_indices],
        }
        
        # Use distill_matcher for matching
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
        all_stage_indices = []
        all_stage_gious = []
        indices,all_selected_indices,all_selected_gious = self.get_teacher_student_matching(outputs_s, outputs_t, targets)
        all_stage_indices.append(all_selected_indices)
        all_stage_gious.append(all_selected_gious)

        avg_factor = outputs_s['pred_logits'].shape[0] * outputs_s["pred_logits"].shape[1] # batch_size * num_quries

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss_distill(loss, outputs_s, outputs_t, indices, avg_factor))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs_s:
            for i, (aux_outputs_s,aux_outputs_t) in enumerate(zip(outputs_s["aux_outputs"],outputs_t["aux_outputs"])):
                indices,hard_neg_indices,hard_neg_gious = self.get_teacher_student_matching(aux_outputs_s, aux_outputs_t, targets)
                all_stage_indices.append(hard_neg_indices)
                all_stage_gious.append(hard_neg_gious)
                for loss in self.losses:
                    l_dict = self.get_loss_distill(loss, aux_outputs_s, aux_outputs_t, indices, avg_factor)
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
                
            # Calculate weights and weighted attention
            weights = 1.0 + batch_selected_gious  # [num_selected]
            selected_cross_attn = cross_attn_maps[b, batch_selected_indices, :]  # [num_selected, HW]
            
            # Apply weights using broadcasting
            weighted_cross_attn = selected_cross_attn * weights.unsqueeze(-1)  # [num_selected, HW]
            
            # Average weighted attention over all selected queries
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
        
        # Pre-assign loss calculation function
        loss_mse = nn.MSELoss(reduction='mean')
        
        for stage in range(stages):
            stage_hard_neg_indices = all_stage_indices[stage]
            stage_hard_neg_gious = all_stage_gious[stage]
            
            # Generate weighted cross-attention mask
            weighted_mask = self.generate_weighted_cross_attn_mask(
                stage_hard_neg_indices, stage_hard_neg_gious, tea_cross_attn[stage]
            )  # [B, HW_total]

            # Calculate losses for all levels
            stage_losses = []
            B, HW,C = fea_tea.shape
                
            # Normalize weights: avoid loops
            max_vals = weighted_mask.max(dim=-1)[0]  # [B]
            max_vals = torch.clamp(max_vals, min=1e-2)
            weighted_mask = weighted_mask / max_vals.view(B, 1)  # [B, H*W]

            # Apply mask and compute loss
            mask_expanded = weighted_mask.unsqueeze(2)  # [B, H*W,1]
            
            # Calculate masked features
            masked_fea_tea = fea_tea * mask_expanded
            masked_fea_stu = fea_stu * mask_expanded

            # Compute MSE loss
            level_loss = loss_mse(masked_fea_tea, masked_fea_stu)
            stage_losses.append(level_loss)
            
            # Accumulate losses from all levels
            stage_loss = sum(stage_losses)
            
            # Save loss
            if stage == stages - 1:  # Last stage
                losses['target_aware_loss'] = stage_loss
            else:
                losses[f'target_aware_loss_{stage}'] = stage_loss
        
        return losses


