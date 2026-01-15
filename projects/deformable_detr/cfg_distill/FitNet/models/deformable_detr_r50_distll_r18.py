import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from .distill_deformable_detr_r18 import model as student_model
from .distill_deformable_detr_r50 import model as teacher_model

from projects.deformable_detr.FitNet import (
    Distiller,
    DistillCriterion
)

model = L(Distiller)(
    student = student_model,
    teacher = teacher_model,
    distill_criterion=L(DistillCriterion)(
        num_classes=80,
        matcher=None,
        weight_dict={
            "loss_distill_dis" : 10.0
        },
        dim_in=256,
        dim_out=256,
        num_features=4,
    ),
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