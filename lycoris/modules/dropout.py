"""
rank_dropout: a redefinition of Kohya's rank-dropout for LyCORIS.

In Kohya's LoRA, the forward pass has a cleanly separable rank dimension:

    g(x) = Wx + drop(B · rank_drop(Ax))        # bypass
    g(x) = (W + B · rank_drop(A))x             # rebuild (weight-applied)

Here A maps to the rank dimension, B maps back, and rank_drop drops entire
rank slices — a meaningful operation.

For newer LyCORIS algorithms (LoKr, LoHa, GLoRA, BOFT, OFT, etc.) the
internal decompositions are far more complex (Kronecker products, Tucker
decompositions, butterfly factorizations, block-diagonal rotations).  It is
often impractical to isolate and drop a "rank" dimension inside these
structures, so rank_dropout is approximated by applying it to the output
activation tensor instead:

   Δ = rank_drop(ΔW · x)     for any algorithm in non-bypass mode

   Δ = B · rank_drop(A · x)  for LoCon/LoRA in bypass mode  (exact rank-drop)
   Δ = rank_drop(ΔW · x)     for other algorithms in bypass mode (approximation)

In both approximation cases, dim¹ of the activation happens to be the output
feature/channel axis, so the effect is closer to output-channel dropout than
to true rank dropout.  This is a deliberate trade-off: you lose the precise
semantics but gain a simple, uniform regularization mechanism that works
across all algorithm types.
"""

import torch
import torch.nn as nn

from torch import Tensor
from torch.overrides import (
    handle_torch_function,
    has_torch_function_unary,
)


def rank_dropout(
    input: Tensor,
    p: float = 0.5,
    rank_dropout_scale: bool = False,
    per_sample: bool = False,
    training: bool = True,
) -> Tensor:
    """Apply dropout along dimension 1 of the input tensor.

    Creates a broadcastable mask of shape ``[B, D, ...]`` so that entire slices
    along dim=1 are either kept or dropped together.  When *rank_dropout_scale*
    is `True` the surviving mask entries are re-scaled to preserve the expected
    sum.

    Setting `per_sample` to `True` will apply different dropout to each sample
    of the batch; instead of reducing the effective rank and building redundancy
    into the ranks, it will instead act as a regularizer, similar to network
    dropout.
    """

    if has_torch_function_unary(input):
        return handle_torch_function(
            rank_dropout, (input,), input, p=p, training=training
        )

    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

    if not training:
        return input

    mask_shape = [1] * input.ndim
    mask_shape[1] = input.shape[1]
    if per_sample:
        mask_shape[0] = input.shape[0]

    drop = torch.empty(mask_shape, device=input.device, dtype=input.dtype).bernoulli_(1.0 - p)
    if rank_dropout_scale:
        drop /= drop.mean()
    return input * drop


class SkipDropout(nn.Identity):
    def __init__(self) -> None:
        super().__init__()

class NetworkDropout(nn.Dropout):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p, False)

class ConvDropout(nn.Module):
    __constants__ = ["p", "scale"]
    p: float
    scale: bool

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace=False"

    def forward(self, input: Tensor) -> Tensor:
        return rank_dropout(input, self.p, False, True, self.training)

class RankDropout(nn.Module):
    __constants__ = ["p", "scale"]
    p: float
    scale: bool

    def __init__(self, p: float = 0.5, rank_dropout_scale: bool = False) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.scale = rank_dropout_scale

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace=False"

    def forward(self, input: Tensor) -> Tensor:
        return rank_dropout(input, self.p, self.scale, False, self.training)
