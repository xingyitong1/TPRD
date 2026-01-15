import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from detrex.layers import MultiScaleDeformableAttention
from torch.nn.init import constant_, xavier_uniform_

# helpers
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0

class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = _C.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _C.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attention_weights):
    # sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 2]
    # attention_weights:  [N, n_layers, Len_q, n_heads, n_levels, n_points]
    N, n_layers, len_q, n_heads, n_levels, n_points = attention_weights.shape

    # 将 (level, point, head) 聚合到同一维，保持 len_q 不变
    sampling_locations = sampling_locations.permute(0, 1, 2, 4, 5, 3, 6).flatten(3, 5)
    # [N, n_layers, len_q, n_levels * n_points * n_heads, 2]
    attention_weights = attention_weights.permute(0, 1, 2, 4, 5, 3).flatten(3, 5)
    # [N, n_layers, len_q, n_levels * n_points * n_heads]

    # 将 (h, w) -> (w, h) (xy)
    rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)  # [n_levels, 2]

    # 关键修复：为每个 (level, point, head) 复制对应的 (w, h) 尺寸，避免广播维度不一致
    rep = n_points * n_heads  # 每个 level 有 rep 个 (point, head) 组合
    rev_spatial_shapes_exp = rev_spatial_shapes.repeat_interleave(rep, dim=0)  # [n_levels*rep, 2]

    # 计算浮点网格坐标
    col_row_float = sampling_locations * rev_spatial_shapes_exp.view(1, 1, 1, -1, 2)

    # 四邻域整数坐标
    col_row_ll = col_row_float.floor().to(torch.int64)
    zero = torch.zeros_like(col_row_ll[..., 0])
    one = torch.ones_like(col_row_ll[..., 0])
    col_row_lh = torch.stack([col_row_ll[..., 0], col_row_ll[..., 1] + 1], dim=-1)
    col_row_hl = torch.stack([col_row_ll[..., 0] + 1, col_row_ll[..., 1]], dim=-1)
    col_row_hh = col_row_ll + 1

    # 双线性插值权重（边角“面积”）
    margin_ll = (col_row_float - col_row_ll).prod(dim=-1)
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1)

    # 有效性掩码需要使用扩展后的 rev_spatial_shapes_exp
    def in_bound(cr, wh):
        return torch.logical_and(
            torch.logical_and(cr[..., 0] >= 0, cr[..., 0] < wh[:, 0]),
            torch.logical_and(cr[..., 1] >= 0, cr[..., 1] < wh[:, 1]),
        )

    # 计算每个扁平化项对应的 level id，用于构造平面索引
    level_ids = torch.arange(n_levels, device=spatial_shapes.device).repeat_interleave(rep)  # [n_levels*rep]
    widths = spatial_shapes[level_ids, 1]            # [n_levels*rep]
    level_starts = level_start_index[level_ids]      # [n_levels*rep]

    spatial_size = int((spatial_shapes[..., 0] * spatial_shapes[..., 1]).sum())
    flat_grid = torch.zeros((N, n_layers, len_q, spatial_size), dtype=torch.float32, device=attention_weights.device)

    zipped = [
        (col_row_ll, margin_hh),
        (col_row_lh, margin_hl),
        (col_row_hl, margin_lh),
        (col_row_hh, margin_ll),
    ]
    for col_row, margin in zipped:
        valid_mask = in_bound(col_row, rev_spatial_shapes_exp)  # [N, n_layers, len_q, n_levels*rep]

        # 计算每个 (x, y) 在整幅平面上的索引：y * W[level] + x + level_start[level]
        # 注意 widths/level_starts 需要在最后一维对齐
        idx = (col_row[..., 1] * widths + col_row[..., 0] + level_starts).to(torch.int64)  # [N, n_layers, len_q, n_levels*rep]

        # 将无效位置的 idx 改到一个安全桶（如 0），并把对应权重清零，避免越界写入
        idx_safe = torch.where(valid_mask, idx, torch.zeros_like(idx))
        weights = (attention_weights * margin)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

        # 对每个 query 独立聚合
        for q_idx in range(len_q):
            flat_grid[:, :, q_idx].scatter_add_(2, idx_safe[:, :, q_idx], weights[:, :, q_idx])

    return flat_grid  # [N, n_layers, len_q, spatial_size]

class DistillMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    """Multi-Scale Deformable Attention Module used in Deformable-DETR

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dim (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): The number of attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query
            in each head. Default: 4.
        img2col_steps (int): The step used in image_to_column. Defualt: 64.
            dropout (float): Dropout layer used in output. Default: 0.1.
        batch_first (bool): if ``True``, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """
    
    def forward_distill(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:

        """Forward Function of MultiScaleDeformableAttention

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default: None. If None, `query` will be
                used.
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (torch.Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # value projection
        value = self.value_proj(value)
        # fill "0" for the padding part
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        # [bs, all hw, 256] -> [bs, all hw, 8, 32]
        value = value.view(bs, num_value, self.num_heads, -1)
        # [bs, all hw, 8, 4, 4, 2]: 8 heads, 4 level features, 4 sampling points, 2 offsets
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        # [bs, all hw, 8, 16]: 4 level 4 sampling points: 16 features total
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        # bs, num_query, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            
            # reference_points   [bs, all hw, 4, 2] -> [bs, all hw, 1, 4, 1, 2]
            # sampling_offsets   [bs, all hw, 8, 4, 4, 2]
            # offset_normalizer  [4, 2] -> [1, 1, 1, 4, 1, 2]
            # references_points + sampling_offsets
            
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        
        attn = attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations.unsqueeze(1), attention_weights.unsqueeze(1)) # [N,1,num_heads,sum(H*W)]
        attn = attn.squeeze(1) # [N,len_q,sum(H*W)]

        # the original impl for fp32 training
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value.to(torch.float32) if value.dtype==torch.float16 else value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        if value.dtype==torch.float16:
            output=output.to(torch.float16)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity,attn

def create_dummy_class(klass, dependency, message=""):
    """
    When a dependency of a class is not available, create a dummy class which throws ImportError
    when used.

    Args:
        klass (str): name of the class.
        dependency (str): name of the dependency.
        message: extra message to print
    Returns:
        class: a class object
    """
    err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, klass)
    if message:
        err = err + " " + message

    class _DummyMetaClass(type):
        # throw error on class attribute access
        def __getattr__(_, __):  # noqa: B902
            raise ImportError(err)

    class _Dummy(object, metaclass=_DummyMetaClass):
        # throw error on constructor
        def __init__(self, *args, **kwargs):
            raise ImportError(err)

    return _Dummy


def create_dummy_func(func, dependency, message=""):
    """
    When a dependency of a function is not available, create a dummy function which throws
    ImportError when used.

    Args:
        func (str): name of the function.
        dependency (str or list[str]): name(s) of the dependency.
        message: extra message to print
    Returns:
        function: a function object
    """
    err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, func)
    if message:
        err = err + " " + message

    if isinstance(dependency, (list, tuple)):
        dependency = ",".join(dependency)

    def _dummy(*args, **kwargs):
        raise ImportError(err)

    return _dummy

try:
    from detrex import _C
except ImportError:
    # TODO: register ops natively so there is no need to import _C.
    _msg = "detrex is not compiled successfully, please build following the instructions!"
    _args = ("detrex._C", _msg)
    MultiScaleDeformableAttention = create_dummy_class(  # noqa
        "MultiScaleDeformableAttention", *_args
    )