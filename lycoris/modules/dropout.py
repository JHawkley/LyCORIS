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
    """Apply dropout along dimension 1 of the input tensor.

    Creates a broadcastable mask of shape ``[1, size_dim1, 1, ...]`` so that
    entire slices along dim=1 are either kept or dropped together.  When
    *rank_dropout_scale* is True the surviving mask entries are re-scaled to
    preserve the expected sum.

    This is used in two contexts:
    - **Intermediate activations** (e.g. ``lora_down(x)``) where dim=1 is the
      *rank* dimension.
    - **Output activations** (e.g. ``delta = op(x, delta_weight)``) where
      dim=1 is the *output* / channel dimension.
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
    """Module wrapper around :func:`rank_dropout`."""

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
        return rank_dropout(input, self.p, self.scale, self.training)
