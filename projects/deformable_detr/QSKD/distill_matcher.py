import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

from detrex.layers.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def distill_classification_cost(cls_pred, gt_labels):
    
    cls_pred = cls_pred.flatten(1).float()
    gt_labels = gt_labels.flatten(1).float()
    
    pos_cost = F.binary_cross_entropy_with_logits(
        cls_pred, 
        torch.ones_like(cls_pred), 
        reduction='none'
    )
    neg_cost = F.binary_cross_entropy_with_logits(
        cls_pred, 
        torch.zeros_like(cls_pred), 
        reduction='none'
    )
    
    cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
               torch.einsum('nc,mc->nm', neg_cost, 1 - gt_labels)
               
    return cls_cost

class DistillHungarianMatcher(nn.Module):
    """HungarianMatcher which computes an assignment between targets and predictions.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class (float): The relative weight of the classification error
            in the matching cost. Default: 1.
        cost_bbox (float): The relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: 1.
        cost_giou (float): This is the relative weight of the giou loss of
            the bounding box in the matching cost. Default: 1.
        cost_class_type (str): How the classification error is calculated.
            Choose from ``["ce_cost", "focal_loss_cost"]``. Default: "focal_loss_cost".
        alpha (float): Weighting factor in range (0, 1) to balance positive vs
            negative examples in focal loss. Default: 0.25.
        gamma (float): Exponent of modulating factor (1 - p_t) to balance easy vs
            hard examples in focal loss. Default: 2.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Forward function for `HungarianMatcher` which performs the matching.

        Args:
            outputs (Dict[str, torch.Tensor]): This is a dict that contains at least these entries:

                - ``"pred_logits"``: Tensor of shape (bs, num_queries, num_classes) with the classification logits.
                - ``"pred_boxes"``: Tensor of shape (bs, num_queries, 4) with the predicted box coordinates.

            targets (List[Dict[str, torch.Tensor]]): This is a list of targets (len(targets) = batch_size),
                where each target is a dict containing:

                - ``"labels"``: Tensor of shape (num_target_boxes, ) (where num_target_boxes is the number of ground-truth objects in the target) containing the class labels.  # noqa
                - ``"boxes"``: Tensor of shape (num_target_boxes, 4) containing the target box coordinates.

        Returns:
            list[torch.Tensor]: A list of size batch_size, containing tuples of `(index_i, index_j)` where:

                - ``index_i`` is the indices of the selected predictions (in order)
                - ``index_j`` is the indices of the corresponding selected targets (in order)

            For each batch element, it holds: `len(index_i) = len(index_j) = min(num_queries, num_target_boxes)`
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):

            out_bbox = outputs["pred_boxes"][b]
            out_bbox = torch.clamp(out_bbox, 0, 1)
            
            tgt_bbox = targets["pred_boxes"][b]

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), 
                box_cxcywh_to_xyxy(tgt_bbox)
            )


            out_prob = outputs["pred_logits"][b] 
            tgt_labels = targets["pred_logits"][b].sigmoid() # soft label
            cost_class = distill_classification_cost(
                out_prob,
                tgt_labels,
            )
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_bbox: {}".format(self.cost_bbox),
            "cost_giou: {}".format(self.cost_giou),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)