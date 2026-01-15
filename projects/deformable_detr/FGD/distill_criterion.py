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
import torch.nn.init as init

from detrex.modeling import SetCriterion
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized

class FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))

        self._reset_parameters()


    def forward(self,
                preds_S,
                preds_T,
                targets):
        
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        N,C,H,W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin,wmax,hmin,hmax = [],[],[],[]
        
        for i in range(N):
            
            boxes = targets[i]["boxes"]  # [num_boxes, 4], normalized cxcywh
            
            if "image_size" in targets[i]:
                img_h, img_w = targets[i]["image_size"]
            else:
                
                img_h, img_w = H, W
            
            # Convert cxcywh to xyxy (still normalized relative to original image)
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            
            # First scale to original image size (in pixels)
            boxes_pixels = boxes_xyxy * torch.tensor([img_w, img_h, img_w, img_h], 
                                                       device=boxes_xyxy.device, 
                                                       dtype=boxes_xyxy.dtype)

            if "padded_image_size" in targets[i]:
                padded_h, padded_w = targets[i]["padded_image_size"]
            else:
                #  assume no padding (will be incorrect if there is padding)
                padded_h, padded_w = img_h, img_w
            
            # Now scale from original image pixels to feature map coordinates
            # boxes_pixels are in original image coordinate
            # feature map H, W correspond to padded_h, padded_w after downsampling
            # So the scale factor is: W / padded_w for width, H / padded_h for height
            scale_w = W / padded_w
            scale_h = H / padded_h
            
            new_boxxes = boxes_pixels * torch.tensor([scale_w, scale_h, scale_w, scale_h],
                                                       device=boxes_xyxy.device,
                                                       dtype=boxes_xyxy.dtype)

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            # Clamp to valid feature map range
            wmin[-1] = torch.clamp(wmin[-1], 0, W-1)
            wmax[-1] = torch.clamp(wmax[-1], 0, W-1)
            hmin[-1] = torch.clamp(hmin[-1], 0, H-1)
            hmax[-1] = torch.clamp(hmax[-1], 0, H-1)

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))

            for j in range(len(targets[i]["boxes"])):
                Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, 
                           C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)


        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
            
        return loss


    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention


    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss


    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)

        return mask_loss
     
    
    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context


    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss


    def _reset_parameters(self):
        """Initialize parameters."""
        # Initialize conv_mask layers with kaiming initialization
        for m in [self.conv_mask_s, self.conv_mask_t]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        
        # Initialize channel_add_conv layers - set last layer to zero
        for conv_seq in [self.channel_add_conv_s, self.channel_add_conv_t]:
            # Last conv layer (index -1) should be initialized to zero
            init.constant_(conv_seq[-1].weight, 0)
            if conv_seq[-1].bias is not None:
                init.constant_(conv_seq[-1].bias, 0)


class DistillCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """
    def __init__(self,dim_in,dim_out,num_features, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.feature_loss = nn.ModuleList()
        for _ in range(num_features):
            self.feature_loss.append(FeatureLoss(dim_in, dim_out))
        
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
    
    def get_dis_loss(self, preds_S, preds_T):
        preds_S = self.align(preds_S) # Align teacher and student models first
        
        loss_mse = nn.MSELoss(reduction='mean')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat > 1-self.mask_ratio, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

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
            loss_dis += self.feature_loss[i](features_s,features_t,targets)
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
