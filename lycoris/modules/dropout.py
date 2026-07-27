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
    scale: bool = False,
    training: bool = True,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            rank_dropout, (input,), input,
            p=p, scale=scale, training=training
        )

    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )

    if not training or p == 0.0:
        return input

    mask = (
        torch.empty(input.shape[0], device=input.device, dtype=input.dtype)
        .bernoulli_(1.0 - p)
    )
    if scale:
        mask /= mask.mean()

    return input * mask.view(-1, *[1] * (input.ndim - 1))

def rank_dropout_with_bias(
    weight: Tensor,
    bias: Tensor | None,
    p: float = 0.5,
    scale: bool = False,
    training: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Applies the same dropout mask to both `weight` and `bias`."""

    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )

    if not training or p == 0.0:
        return weight, bias

    mask = (
        torch.empty(weight.shape[0], device=weight.device, dtype=weight.dtype)
        .bernoulli_(1.0 - p)
    )
    if scale:
        mask = mask / mask.mean()

    weight = weight * mask.view(-1, *[1] * (weight.ndim - 1))
    if bias is not None:
        bias = bias * mask
    return weight, bias


class SkipDropout(nn.Identity):
    """A subclass of ``nn.Identity`` which does nothing, but has a clearer name
    to assist debugging.
    """

    def __init__(self) -> None:
        super().__init__()

class NetworkDropout(nn.Dropout):
    """A subclass of ``nn.Dropout`` that restricts the dropout from being performed
    in-place, which is almost never desirable while training.

    This module's clear name may also assist debugging.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p, False)

class RankDropout(nn.Module):
    """A module that applies a dropout mask to the output channels of the input
    tensor.  Most commonly used to approximate Kohya's rank dropout when applied
    to an algorithm's diff weights.

    This module's clear name may also assist debugging.
    """

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
        return f"p={self.p}, rank_dropout_scale={self.scale}"

    def forward(self, input: Tensor) -> Tensor:
        return rank_dropout(input, self.p, self.scale, self.training)

class BatchRankDropout(nn.Module):
    """RankDropout that operates on a user-specified dimension.

    The standard ``RankDropout`` always applies its mask along dimension 0,
    which is correct for weight tensors shaped ``(out_channels, ...)``.
    In the bypass forward path, activations are shaped ``(batch, channels, ...)``,
    so dimension 0 is the batch axis, not the channel axis.

    This wrapper transposes the target channel dimension to position 0 before
    delegating to ``RankDropout``, then transposes back.
    """

    __constants__ = ["p", "scale", "channel_dim"]
    p: float
    scale: bool
    channel_dim: int

    def __init__(
        self,
        p: float = 0.5,
        rank_dropout_scale: bool = False,
        channel_dim: int = 1,
    ) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.scale = rank_dropout_scale
        self.channel_dim = channel_dim

    def extra_repr(self) -> str:
        return (
            f"p={self.p}, rank_dropout_scale={self.scale}, "
            f"channel_dim={self.channel_dim}"
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.channel_dim != 0:
            input = input.transpose(0, self.channel_dim)
        input = rank_dropout(input, self.p, self.scale, self.training)
        if self.channel_dim != 0:
            input = input.transpose(0, self.channel_dim)
        return input
