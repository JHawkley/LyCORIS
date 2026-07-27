import math
import random

import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..utils import product


class DyLoraModule(LycorisBaseModule):
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        block_size=4,
        use_scalar=False,
        rank_dropout_scale=False,
        weight_decompose=False,
        bypass_mode=None,
        rs_lora=False,
        train_on_input=False,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            rank_dropout_scale,
            bypass_mode,
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in IA^3 algo.")
        assert lora_dim % block_size == 0, "lora_dim must be a multiple of block_size"
        self.block_count = lora_dim // block_size
        self.block_size = block_size

        shape = (
            self.shape[0],
            product(self.shape[1:]),
        )

        self.lora_dim = lora_dim
        self.up_list = nn.ParameterList(
            [torch.empty(shape[0], self.block_size) for i in range(self.block_count)]
        )
        self.down_list = nn.ParameterList(
            [torch.empty(self.block_size, shape[1]) for i in range(self.block_count)]
        )

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # Need more experiences on init method
        for v in self.down_list:
            torch.nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        for v in self.up_list:
            torch.nn.init.zeros_(v)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        destination["lora_up.weight"] = nn.Parameter(
            torch.concat(list(self.up_list), dim=1)
        )
        destination["lora_down.weight"] = nn.Parameter(
            torch.concat(list(self.down_list)).reshape(
                self.lora_dim, -1, *self.shape[2:]
            )
        )
        return destination

    def _get_weight_parts(self, rank):
        """Internal helper: return (down, up, gamma) for the given rank."""
        b = math.ceil(rank / self.block_size)
        down = torch.concat(
            list(i.data for i in self.down_list[:b]) + list(self.down_list[b : (b + 1)])
        )
        up = torch.concat(
            list(i.data for i in self.up_list[:b]) + list(self.up_list[b : (b + 1)]),
            dim=1,
        )
        return down, up, self.alpha / (b + 1)

    def make_weight(self, rank, scale=1, device=None, diff=False):
        # NOTE: Computing the diff weight (diff=True) is faster than computing
        # the merged weight, since it avoids adding the original weight.
        down, up, gamma = self._get_weight_parts(rank)
        w = up @ (down * (gamma * scale))
        if device is not None:
            w = w.to(device)
        w = w.view(self.shape)
        if diff:
            return w
        return self.org_weight + w

    def get_diff_weight(self, multiplier=1, shape=None, device=None, rank=None):
        if rank is None:
            rank = random.randint(0, self.block_count - 1) * self.block_size
        diff = self.make_weight(rank, scale=multiplier, device=device, diff=True)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None, rank=None):
        if rank is None:
            rank = random.randint(0, self.block_count - 1) * self.block_size
        merged = self.make_weight(rank, scale=multiplier, device=device, diff=False)
        if shape is not None:
            merged = merged.view(shape)
        return merged, None

    def bypass_forward_diff(self, x, scale=1, rank=None):
        if rank is None:
            down, up, gamma = self._get_weight_parts(
                random.randint(0, self.block_count - 1) * self.block_size
            )
        else:
            down, up, gamma = self._get_weight_parts(rank)
        down = down.view(self.lora_dim, -1, *self.shape[2:])
        up = up.view(-1, self.lora_dim, *(1 for _ in self.shape[2:]))
        return self.op(self.op(x, down, **self.kw_dict), up) * gamma * scale

    def bypass_forward(self, x, scale=1, rank=None):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale, rank)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)
        if self.bypass_mode:
            return self.bypass_forward(x, self.multiplier)
        else:
            base = self.org_forward(x, *args, **kwargs)
            diff_weight = self.make_weight(
                random.randint(0, self.block_count - 1) * self.block_size,
                scale=self.multiplier,
                diff=True,
            ).to(x.dtype)
            delta = self.op(x, diff_weight, None, **self.kw_dict)
            return base + delta
