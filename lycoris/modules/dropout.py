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
    def __init__(self) -> None:
        super().__init__()

class NetworkDropout(nn.Dropout):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p, False)

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
        return f"p={self.p}, rank_dropout_scale={self.scale}"

    def forward(self, input: Tensor) -> Tensor:
        return rank_dropout(input, self.p, self.scale, self.training)
