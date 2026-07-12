import torch
import torch.nn as nn

from torch import Tensor
from torch.overrides import (
    handle_torch_function,
    has_torch_function_unary,
)

def lora_rank_dropout(
    input: Tensor,
    p: float = 0.5,
    rank_dropout_scale: bool = False,
    training: bool = True,
) -> Tensor:
    """Apply dropout to LoRA rank dimension (dim=1).
    
    Use this for intermediate activations in LoRA forward pass.
    """

    if has_torch_function_unary(input):
        return handle_torch_function(
            lora_rank_dropout, (input,), input, p=p, training=training
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

def output_rank_dropout(
    input: Tensor,
    p: float = 0.5,
    rank_dropout_scale: bool = False,
    training: bool = True,
) -> Tensor:
    """Apply dropout to output dimension (dim=0).
    
    Use this for merged weight tensors.
    """

    if has_torch_function_unary(input):
        return handle_torch_function(
            output_rank_dropout, (input,), input, p=p, training=training
        )

    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

    if not training:
        return input

    mask_shape = [1] * input.ndim
    mask_shape[1] = input.size(1)
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

class LoraRankDropout(nn.Module):
    __constants__ = ["p", "scale"]
    p: float
    scale: bool

    def __init__(self, p: float = 0.5, rank_dropout_scale = False) -> None:
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
        return lora_rank_dropout(input, self.p, self.scale, self.training)

class OutputRankDropout(nn.Module):
    __constants__ = ["p", "scale"]
    p: float
    scale: bool

    def __init__(self, p: float = 0.5, rank_dropout_scale = False) -> None:
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
        return output_rank_dropout(input, self.p, self.scale, self.training)
