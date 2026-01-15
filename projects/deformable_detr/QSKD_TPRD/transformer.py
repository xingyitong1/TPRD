import copy
import warnings
from typing import List
import torch
import torch.nn as nn
from detrex.layers import BaseTransformerLayer

class DistillBaseTransformerLayer(BaseTransformerLayer):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.

    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """
    def forward_distill(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_masks: List[torch.Tensor] = None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query,cross_attn_map = self.attentions[attn_index].forward_distill(
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query,cross_attn_map