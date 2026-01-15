import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from .distill_deformable_detr_r18 import model as student_model
from .distill_deformable_detr_r50 import model as teacher_model

from projects.deformable_detr.DETRDistill_TPRD import (
    Distiller,
    DistillCriterion,
    DistillHungarianMatcher
)

model = L(Distiller)(
    student = student_model,
    teacher = teacher_model,
    distill_criterion=L(DistillCriterion)(
        num_classes=80,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        distill_matcher = L(DistillHungarianMatcher)(
            cost_class=1.0,
            cost_bbox=5.0,
            cost_giou=2.0
        ),
        weight_dict={
            # distill loss
            "distill_loss_class": 4.0,
            "distill_loss_bbox": 5.0,
            "distill_loss_giou": 2.0,
            # teacher query loss
            "teacher_loss_class": 1.0,
            "teacher_loss_bbox": 5.0,
            "teacher_loss_giou" : 2.0,
            # target-aware feature loss
            "target_aware_loss": 2.0
        },
    ),
    max_per_img = 100,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    aux_loss=True,
    device="cuda",
)

# set aux loss weight dict
if model.aux_loss:
    weight_dict = model.distill_criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.student.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.distill_criterion.weight_dict = weight_dict