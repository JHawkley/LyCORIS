import math

import torch
import torch.nn as nn

from .base import LycorisBaseModule
from .weight_decompose import (
    WeightDecomposeOnInput,
    WeightDecomposeOnOutput,
    infer_wd_on_out,
    log_wd_mode_mismatch,
    parse_weight_decompose,
    pop_wd_for_diff_key,
    remap_dora_scale_key,
)
from ..functional.loha import diff_weight as loha_diff_weight


class LohaModule(LycorisBaseModule):
    name = "loha"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "hada_t1",
        "hada_t2",
        "alpha",
        "dora_scale",
        "wd_for_diff",
    ]
    weight_list_det = ["hada_w1_a"]

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
        weight_decompose=False,
        wd_on_out=True,
        bypass_mode=None,
        rs_lora=False,
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
            raise ValueError(f"{self.module_type} is not supported in LoHa algo.")
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.tucker = False
        self.rs_lora = rs_lora

        w_shape = self.shape
        if self.module_type.startswith("conv"):
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.shape = (out_dim, in_dim, *k_size)
            self.tucker = use_tucker and any(i != 1 for i in k_size)
            if self.tucker:
                w_shape = (out_dim, in_dim, *k_size)
            else:
                w_shape = (out_dim, in_dim * torch.tensor(k_size).prod().item())

        if self.tucker:
            self.hada_t1 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w1_a = nn.Parameter(
                torch.empty(lora_dim, w_shape[0])
            )  # out_dim, 1-mode
            self.hada_w1_b = nn.Parameter(
                torch.empty(lora_dim, w_shape[1])
            )  # in_dim , 2-mode

            self.hada_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w2_a = nn.Parameter(
                torch.empty(lora_dim, w_shape[0])
            )  # out_dim, 1-mode
            self.hada_w2_b = nn.Parameter(
                torch.empty(lora_dim, w_shape[1])
            )  # in_dim , 2-mode
        else:
            self.hada_w1_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

            self.hada_w2_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

        self.wd, self.wd_for_diff = parse_weight_decompose(
            weight_decompose, self.wd_auto_mode
        )
        self.wd_on_out = wd_on_out
        self.wd_module = (
            None if not self.wd else
            WeightDecomposeOnOutput(org_module.weight, for_diff=self.wd_for_diff) if wd_on_out else
            WeightDecomposeOnInput(org_module.weight, for_diff=self.wd_for_diff)
        )

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
        # Need more experiments on init method
        if self.tucker:
            torch.nn.init.normal_(self.hada_t1, std=0.1)
            torch.nn.init.normal_(self.hada_t2, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1)
        torch.nn.init.normal_(self.hada_w1_a, std=0.1)
        torch.nn.init.normal_(self.hada_w2_b, std=1)
        if use_scalar or self.wd_for_diff:
            # Diff-weight decomposition requires a non-zero initial diff: the
            # decomposition normalizes the diff weight, which is undefined
            # (and blocks gradients) when it is exactly zero.  Combined with
            # the zero-initialized dora_scale the module still starts as an
            # exact identity.
            torch.nn.init.normal_(self.hada_w2_a, std=0.1)
        else:
            torch.nn.init.constant_(self.hada_w2_a, 0)

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, w1a, w1b, w2a, w2b, t1, t2, alpha, dora_scale, wd_for_diff
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            w1b.size(0),
            float(alpha),
            use_tucker=t1 is not None,
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
        module.hada_w1_a.copy_(w1a)
        module.hada_w1_b.copy_(w1b)
        module.hada_w2_a.copy_(w2a)
        module.hada_w2_b.copy_(w2b)
        if t1 is not None:
            module.hada_t1.copy_(t1)
            module.hada_t2.copy_(t2)
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

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

    def make_weight(self, scale=1, device=None, diff=False):
        # NOTE: Computing the diff weight (diff=True) is faster than computing
        # the merged weight, since it avoids the addition of the original weight.
        scale_t = torch.tensor(
            self.scale * scale,
            dtype=self.hada_w1_b.dtype,
            device=self.hada_w1_b.device,
        )
        if device is not None:
            scale_t = scale_t.to(device)
            w1b = self.hada_w1_b.to(device)
            w1a = self.hada_w1_a.to(device)
            w2b = self.hada_w2_b.to(device)
            w2a = self.hada_w2_a.to(device)
            t1 = self.hada_t1.to(device) if self.tucker else None
            t2 = self.hada_t2.to(device) if self.tucker else None
        else:
            w1b = self.hada_w1_b
            w1a = self.hada_w1_a
            w2b = self.hada_w2_b
            w2a = self.hada_w2_a
            t1 = self.hada_t1 if self.tucker else None
            t2 = self.hada_t2 if self.tucker else None

        if self.tucker:
            weight = loha_diff_weight(
                w1b, w1a, w2b, w2a, t1, t2, gamma=scale_t,
            )
        else:
            weight = loha_diff_weight(
                w1b, w1a, w2b, w2a, None, None, gamma=scale_t,
            )

        weight = weight * self.scalar

        # Reshape to the target weight shape (handles conv flattened diff)
        weight = weight.view(self.shape)

        if diff:
            return weight

        # diff=False: return merged weight (original + diff)
        org = self.org_weight.to(device, dtype=weight.dtype) if device else self.org_weight.to(dtype=weight.dtype)
        return org + weight

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
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

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
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

    def apply_weight_decompose(self, weight, multiplier=1):
        """Backward-compatible alias for the weight-decomposition submodule.

        The decomposition logic now lives in ``self.wd_module``; see
        ``lycoris/modules/weight_decompose.py``.
        """
        return self.wd_module(weight, multiplier)

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        if self.wd:
            destination["dora_scale"] = self.dora_scale
            if self.wd_for_diff:
                # Marker for third-party consumers: dora_scale decomposes the
                # diff weight rather than the merged weight.  Omitted for
                # merged mode so those checkpoints keep the historical format.
                destination["wd_for_diff"] = torch.tensor(True)
        destination["hada_w1_a"] = self.hada_w1_a * self.scalar
        destination["hada_w1_b"] = self.hada_w1_b
        destination["hada_w2_a"] = self.hada_w2_a
        destination["hada_w2_b"] = self.hada_w2_b
        if self.tucker:
            destination["hada_t1"] = self.hada_t1
            destination["hada_t2"] = self.hada_t2
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device=device, diff=True).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio

        return scaled, orig_norm * ratio

    def bypass_forward_diff(self, x, scale=1):
        diff_weight = self.make_weight(scale=scale, diff=True)
        diff_weight = self.drop(diff_weight)
        diff_weight = self.rank_drop(diff_weight)
        return self.op(x, diff_weight, **self.kw_dict)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        base_weight = self._current_weight().to(x.device)
        diff_weight = self.make_weight(scale=1, device=x.device, diff=True).to(
            base_weight.dtype
        )

        if self.wd and self.wd_for_diff:
            # Decompose the diff weight directly, skipping the
            # merge-decompose-subtract round trip through the base weight.
            delta_weight = self.wd_module(diff_weight, self.multiplier)
        elif self.wd:
            new_weight = self.wd_module(
                base_weight + diff_weight, self.multiplier
            )
            delta_weight = new_weight - base_weight
        else:
            delta_weight = diff_weight * self.multiplier

        delta_weight = self.drop(delta_weight)
        delta_weight = self.rank_drop(delta_weight)
        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta
