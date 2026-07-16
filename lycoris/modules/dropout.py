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
    training: bool = True,
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            rank_dropout, (input,), input,
            p=p, rank_dropout_scale=rank_dropout_scale, training=training
        )

    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

    if not training:
        return input

    mask_shape = [1] * input.ndim
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
