from functools import cache
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import LycorisBaseModule
from .dropout import BatchRankDropout
from .weight_decompose import (
    WeightDecomposeOnInput,
    WeightDecomposeOnOutput,
    infer_wd_on_out,
    log_wd_mode_mismatch,
    parse_weight_decompose,
    pop_wd_for_diff_key,
    remap_dora_scale_key,
)
from ..functional import power2factorization
from ..logging import logger


@cache
def log_butterfly_factorize(dim, factor, result):
    logger.info(
        f"Use BOFT({int(log2(result[1]))}, {result[0]//2})"
        f" (equivalent to factor={result[0]}) "
        f"for {dim=} and {factor=}"
    )


def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
    m, n = power2factorization(dimension, factor)

    if n == 0:
        raise ValueError(
            f"It is impossible to decompose {dimension} with factor {factor} under BOFT constraints."
        )

    log_butterfly_factorize(dimension, factor, (m, n))
    return m, n


class ButterflyOFTModule(LycorisBaseModule):
    name = "boft"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "oft_blocks",
        "rescale",
        "alpha",
        "dora_scale",
        "wd_for_diff",
    ]
    weight_list_det = ["oft_blocks"]
    # make_weight produces the merged weight fastest (the diff requires an
    # extra subtraction pass), so weight_decompose="auto" resolves to merged.
    wd_auto_mode = "merged"

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
        use_scalar=False,
        rank_dropout_scale=False,
        constraint=0,
        rescaled=False,
        weight_decompose=False,
        wd_on_out=True,
        bypass_mode=None,
        **kwargs,
    ):
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
            raise ValueError(f"{self.module_type} is not supported in BOFT algo.")

        out_dim = self.dim
        b, m_exp = butterfly_factor(out_dim, lora_dim)
        self.block_size = b
        self.block_num = m_exp
        # BOFT(m, b)
        self.boft_b = b
        self.boft_m = sum(int(i) for i in f"{m_exp-1:b}") + 1
        # block_num > block_size
        self.rescaled = rescaled
        self.constraint = constraint * out_dim
        self.register_buffer("alpha", torch.tensor(constraint))
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.boft_m, self.block_num, self.block_size, self.block_size)
        )
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

        self.wd, self.wd_for_diff = parse_weight_decompose(
            weight_decompose, self.wd_auto_mode
        )
        self.wd_on_out = wd_on_out
        self.wd_module = (
            None if not self.wd else
            WeightDecomposeOnOutput(org_module.weight, for_diff=self.wd_for_diff) if wd_on_out else
            WeightDecomposeOnInput(org_module.weight, for_diff=self.wd_for_diff)
        )

        # Diff-weight decomposition requires a non-zero initial diff: the
        # decomposition normalizes the diff weight, which is undefined (and
        # blocks gradients) when it is exactly zero.  Combined with the
        # zero-initialized dora_scale the module still starts as an exact
        # identity.
        if self.wd_for_diff:
            nn.init.uniform_(self.oft_blocks, a=-0.01, b=0.01)

        if self.bypass_mode and self.rank_dropout > 0:
            self.rank_drop = BatchRankDropout(self.rank_dropout, self.rank_dropout_scale)

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        if f"{lora_name}.oft_blocks" in state_dict:
            oft_blocks = state_dict[f"{lora_name}.oft_blocks"]
            if oft_blocks.ndim == 4:
                return True
        return False

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, oft_blocks, rescale, alpha,
        dora_scale=None, wd_for_diff=None,
    ):
        m, n, s, _ = oft_blocks.shape
        module = cls(
            lora_name,
            orig_module,
            1,
            lora_dim=s,
            constraint=float(alpha),
            rescaled=rescale is not None,
            weight_decompose=(
                False if dora_scale is None
                else "diff" if wd_for_diff is not None and bool(wd_for_diff)
                else True
            ),
            wd_on_out=(
                dora_scale is None
                or infer_wd_on_out(dora_scale, orig_module.weight)
            ),
        )
        module.oft_blocks.copy_(oft_blocks)
        if rescale is not None:
            module.rescale.copy_(rescale)
        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    def load_weight_prehook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.wd:
            remap_dora_scale_key(state_dict, prefix)
            ckpt_for_diff = pop_wd_for_diff_key(state_dict, prefix)
            if ckpt_for_diff != self.wd_for_diff:
                log_wd_mode_mismatch(self.lora_name, ckpt_for_diff, self.wd_for_diff)

    def custom_state_dict(self):
        destination = {}
        destination["oft_blocks"] = self.oft_blocks
        if self.rescaled:
            destination["rescale"] = self.rescale
        destination["alpha"] = self.alpha
        if self.wd:
            destination["dora_scale"] = self.dora_scale
            if self.wd_for_diff:
                destination["wd_for_diff"] = torch.tensor(True)
        return destination

    def apply_weight_decompose(self, weight, multiplier=1):
        """Backward-compatible alias for the weight-decomposition submodule.

        The decomposition logic now lives in ``self.wd_module``; see
        ``lycoris/modules/weight_decompose.py``.
        """
        return self.wd_module(weight, multiplier)

    @property
    def I(self):
        return torch.eye(self.block_size, device=self.device)

    def get_r(self):
        I = self.I
        # for Q = -Q^T
        q = self.oft_blocks - self.oft_blocks.transpose(-1, -2)
        normed_q = q
        # Diag OFT style constrain
        if self.constraint > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constraint:
                normed_q = q * self.constraint / q_norm
        # use float() to prevent unsupported type
        r = (I + normed_q) @ (I - normed_q).float().inverse()
        return r

    def make_weight(self, scale=1, device=None, diff=False):
        # NOTE: Computing the merged weight (diff=False) is faster than computing
        # the diff weight, since diff=True requires an additional subtraction pass.
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        inp = org = self.org_weight.to(device, dtype=r.dtype)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            g = 2
            k = 2**i * r_b
            if scale != 1:
                bi = bi * scale + (1 - scale) * self.I
            inp = (
                inp.unflatten(0, (-1, g, k))
                .transpose(1, 2)
                .flatten(0, 2)
                .unflatten(0, (-1, b))
            )
            inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)
            inp = (
                inp.flatten(0, 1).unflatten(0, (-1, k, g)).transpose(1, 2).flatten(0, 2)
            )

        if self.rescaled:
            inp = inp * self.rescale

        if diff:
            inp = inp - org

        return inp.to(self.oft_blocks.dtype)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        if self.wd and self.wd_for_diff:
            # The decomposition targets the diff weight directly.
            diff = self.make_weight(scale=1, device=device, diff=True)
            diff = self.wd_module(diff, multiplier)
        elif self.wd:
            # Weight decomposition affects the final weight, so the diff must be
            # computed from the fully decomposed merged weight.
            merged = self.make_weight(scale=1, device=device, diff=False)
            merged = self.wd_module(merged, multiplier)
            org = self.org_weight.to(device, dtype=merged.dtype) if device else self.org_weight.to(dtype=merged.dtype)
            diff = merged - org
        else:
            diff = self.make_weight(scale=multiplier, device=device, diff=True)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        if self.wd and self.wd_for_diff:
            # The decomposition targets the diff weight directly; the merged
            # weight is the original weight plus the decomposed diff.
            diff = self.make_weight(scale=1, device=device, diff=True)
            diff = self.wd_module(diff, multiplier)
            org = self.org_weight.to(device, dtype=diff.dtype) if device else self.org_weight.to(dtype=diff.dtype)
            merged = org + diff
        elif self.wd:
            merged = self.make_weight(scale=1, device=device, diff=False)
            merged = self.wd_module(merged, multiplier)
        else:
            merged = self.make_weight(scale=multiplier, device=device, diff=False)
        if shape is not None:
            merged = merged.view(shape)
        return merged, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.oft_blocks.to(device).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired / norm

        scaled = norm != desired
        if scaled:
            self.oft_blocks *= ratio

        return scaled, orig_norm * ratio

    def _bypass_forward(self, x, scale=1, diff=False):
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        inp = org = self.org_forward(x)
        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            inp = inp.transpose(1, -1)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            g = 2
            k = 2**i * r_b
            if scale != 1:
                bi = bi * scale + (1 - scale) * self.I
            inp = (
                inp.unflatten(-1, (-1, g, k))
                .transpose(-2, -1)
                .flatten(-3)
                .unflatten(-1, (-1, b))
            )
            inp = torch.einsum("b i j, ... b j -> ... b i", bi, inp)
            inp = (
                inp.flatten(-2).unflatten(-1, (-1, k, g)).transpose(-2, -1).flatten(-3)
            )

        if self.rescaled:
            inp = inp * self.rescale.transpose(0, -1)

        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            inp = inp.transpose(1, -1)

        # We can take a fast-path if no dropout is being applied.
        if self.dropout == 0 and self.rank_dropout == 0:
            return inp - org if diff else inp

        delta = inp - org
        delta = self.drop(delta)
        delta = self.rank_drop(delta)
        inp = delta if diff else org + delta
        return inp

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=False)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)
        scale = self.multiplier

        if self.bypass_mode:
            return self.bypass_forward(x, scale)

        base = self.org_forward(x, *args, **kwargs)
        base_weight = self._current_weight().to(x.device)
        base_dtype = base_weight.dtype

        if self.wd and self.wd_for_diff:
            # Decompose the diff weight directly, skipping the
            # merge-decompose-subtract round trip through the base weight.
            diff_weight = self.make_weight(scale=1, device=x.device, diff=True)
            diff_weight = diff_weight.to(base_dtype)
            delta_weight = self.wd_module(diff_weight, scale)
        elif self.wd:
            new_weight = self.make_weight(scale=1, device=x.device, diff=False)
            new_weight = new_weight.to(base_dtype)
            new_weight = self.wd_module(new_weight, scale)
            delta_weight = new_weight - base_weight
        else:
            new_weight = self.make_weight(scale=scale, device=x.device)
            new_weight = new_weight.to(base_dtype)
            delta_weight = new_weight - base_weight

        delta_weight = self.drop(delta_weight)
        delta_weight = self.rank_drop(delta_weight)
        delta = self.op(x, weight=delta_weight, bias=None, **self.kw_dict)
        return base + delta
