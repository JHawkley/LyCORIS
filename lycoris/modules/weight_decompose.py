"""DoRA-style weight decomposition modules.

These modules encapsulate the "magnitude" half of DoRA: a learnable per-slice
scale (``dora_scale``), initialized from the norm of the original weight, which
is reapplied to a renormalized weight tensor::

    weight * (dora_scale / ||weight||)

Historically this logic was copy-and-pasted into every algorithm module that
supported ``weight_decompose`` (LoCon/LoHa/LoKr) as an
``apply_weight_decompose`` method plus a ``dora_scale`` parameter.  It now lives
here, mirroring how ``dropout.py`` encapsulates the dropout variants: the
algorithm module chooses the concrete class once (based on ``wd_on_out``)
during initialization and afterwards simply calls the module.

Two variants exist, differing only in the dimension the norm is computed over:

- ``WeightDecomposeOnOutput``: norm over the output dimension (dim 0).  This is
  the classic DoRA behavior and the default.
- ``WeightDecomposeOnInput``: norm over the input dimension (dim 1).

A weight-decomposition module is intentionally agnostic about *which* tensor it
rescales: algorithm modules currently pass the fully merged weight, but a diff
weight can be passed instead (e.g. for decomposition modes that operate on the
update itself).
"""

import torch
import torch.nn as nn
from torch import Tensor


# Attribute name algorithm modules use to hold their weight-decomposition
# submodule.  Centralized here so the state-dict remapping below cannot drift
# out of sync with the attribute used by the algorithm modules.
WD_MODULE_ATTR = "wd_module"


class WeightDecomposeBase(nn.Module):
    """Base class for weight decomposition (DoRA magnitude) modules.

    Owns the learnable ``dora_scale`` parameter and implements the shared
    rescaling logic in ``forward``; subclasses only define how the norm of the
    weight is computed (and the matching initial scale) via ``weight_norm``
    and ``init_scale``.
    """

    #: Whether this variant decomposes along the output dimension.  Useful for
    #: introspection without an isinstance chain.
    on_output: bool = NotImplemented

    def __init__(self, org_weight: Tensor) -> None:
        """Initialize ``dora_scale`` from the original (frozen) weight.

        Args:
            org_weight: The weight of the module being adapted.  Only its
                values (at init time) and shape are used.
        """
        super().__init__()
        org_weight = org_weight.detach().cpu().clone().float()
        self.dora_norm_dims = org_weight.dim() - 1
        self.dora_scale = nn.Parameter(self.init_scale(org_weight))

    def init_scale(self, org_weight: Tensor) -> Tensor:
        """Compute the initial magnitude from the original weight.

        Must return a float tensor broadcastable to ``org_weight``.
        """
        raise NotImplementedError

    def weight_norm(self, weight: Tensor) -> Tensor:
        """Norm of ``weight`` along the decomposition axis.

        Must return a tensor broadcastable to ``weight``.
        """
        raise NotImplementedError

    def forward(self, weight: Tensor, multiplier: float = 1.0) -> Tensor:
        """Renormalize ``weight`` and reapply the learned magnitude.

        Args:
            weight: The weight tensor to rescale (merged or diff weight).
            multiplier: Interpolates the rescaling towards the identity, so
                that ``multiplier=0`` returns ``weight`` unchanged.
        """
        weight = weight.to(self.dora_scale.dtype)
        scale = self.dora_scale.to(weight.device) / self.weight_norm(weight)
        if multiplier != 1:
            scale = multiplier * (scale - 1) + 1
        return weight * scale

    def extra_repr(self) -> str:
        return f"dora_norm_dims={self.dora_norm_dims}"


class WeightDecomposeOnOutput(WeightDecomposeBase):
    """Weight decomposition along the output dimension (classic DoRA).

    ``dora_scale`` holds one magnitude per output slice and has shape
    ``(out_dim, 1, ...)``.
    """

    on_output = True

    def init_scale(self, org_weight: Tensor) -> Tensor:
        return torch.norm(
            org_weight.reshape(org_weight.shape[0], -1),
            dim=1,
            keepdim=True,
        ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)

    def weight_norm(self, weight: Tensor) -> Tensor:
        return (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1)
            .reshape(weight.shape[0], *[1] * self.dora_norm_dims)
        ) + torch.finfo(weight.dtype).eps


class WeightDecomposeOnInput(WeightDecomposeBase):
    """Weight decomposition along the input dimension.

    ``dora_scale`` holds one magnitude per input slice and has shape
    ``(1, in_dim, 1, ...)``.
    """

    on_output = False

    def init_scale(self, org_weight: Tensor) -> Tensor:
        return (
            torch.norm(
                org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
                dim=1,
                keepdim=True,
            )
            .reshape(org_weight.shape[1], *[1] * self.dora_norm_dims)
            .transpose(1, 0)
        )

    def weight_norm(self, weight: Tensor) -> Tensor:
        return (
            weight.transpose(0, 1)
            .reshape(weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
            .transpose(0, 1)
        ) + torch.finfo(weight.dtype).eps


def infer_wd_on_out(dora_scale: Tensor, org_weight: Tensor) -> bool:
    """Infer whether a serialized ``dora_scale`` was trained on the output or
    the input dimension of the original weight.

    Checkpoints do not record the ``wd_on_out`` setting, but the two variants
    produce distinct shapes:

    - on output: ``(out_dim, 1, ...)``
    - on input:  ``(1, in_dim, 1, ...)``

    The fully-degenerate all-singleton shape is resolved with the original
    weight's shape: it can only stem from on-input training when
    ``in_dim == 1`` (an on-output scale would have kept ``out_dim`` in dim 0).
    In the remaining ambiguous cases the two variants are numerically
    identical, so on-output is returned.
    """
    out_dim, in_dim = org_weight.shape[0], org_weight.shape[1]
    if dora_scale.shape[0] != 1:
        return True
    if dora_scale.shape[1] != 1:
        return False
    return in_dim != 1 or out_dim == 1


def remap_dora_scale_key(state_dict, prefix: str, attr_name: str = WD_MODULE_ATTR) -> None:
    """Remap a top-level ``dora_scale`` entry of a state dict onto the child
    weight-decomposition module's parameter key, in place.

    ``custom_state_dict`` of the algorithm modules keeps writing ``dora_scale``
    at the top level (the historical on-disk format understood by third-party
    implementations), so loading must map it back onto
    ``<attr_name>.dora_scale``.
    """
    old_key = f"{prefix}dora_scale"
    new_key = f"{prefix}{attr_name}.dora_scale"
    if old_key in state_dict and new_key not in state_dict:
        state_dict[new_key] = state_dict.pop(old_key)
