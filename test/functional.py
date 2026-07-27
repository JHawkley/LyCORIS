import unittest
from itertools import product
from parameterized import parameterized

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.functional import locon, loha, lokr
from lycoris.functional.diag_oft import (
    weight_gen as diag_oft_weight_gen,
    diff_weight as diag_oft_diff_weight,
    bypass_forward_diff as diag_oft_bypass_forward_diff,
)
from lycoris.functional.boft import (
    weight_gen as boft_weight_gen,
    diff_weight as boft_diff_weight,
    bypass_forward_diff as boft_bypass_forward_diff,
)


EPS_DTYPE = {
    torch.float32: 5e-6,
    torch.float16: 5e-5,
    torch.bfloat16: 5e-4,
}


class _LoconLike:
    __name__ = "locon"

    def __init__(self):
        self.weight_gen = staticmethod(locon.weight_gen)
        self.diff_weight = staticmethod(locon.diff_weight)
        self.bypass_forward_diff = staticmethod(locon.bypass_forward_diff)


class _LohaLike:
    __name__ = "loha"

    def __init__(self):
        self.weight_gen = staticmethod(loha.weight_gen)
        self.diff_weight = staticmethod(loha.diff_weight)
        self.bypass_forward_diff = staticmethod(loha.bypass_forward_diff)


class _LokrLike:
    __name__ = "lokr"

    def __init__(self):
        self.weight_gen = staticmethod(lokr.weight_gen)
        self.diff_weight = staticmethod(lokr.diff_weight)
        self.bypass_forward_diff = staticmethod(lokr.bypass_forward_diff)


class _DiagOFTLike:
    __name__ = "diag_oft"

    def __init__(self):
        self.weight_gen = staticmethod(diag_oft_weight_gen)
        self.diff_weight = staticmethod(diag_oft_diff_weight)
        self.bypass_forward_diff = staticmethod(diag_oft_bypass_forward_diff)


class _BoftLike:
    __name__ = "boft"

    def __init__(self):
        self.weight_gen = staticmethod(boft_weight_gen)
        self.diff_weight = staticmethod(boft_diff_weight)
        self.bypass_forward_diff = staticmethod(boft_bypass_forward_diff)


modules = [_LoconLike(), _LohaLike(), _LokrLike(), _DiagOFTLike(), _BoftLike()]
base_module_and_input_adn_weight = [
    lambda dim: (F.linear, torch.randn(dim, dim), torch.randn(1, dim)),
    lambda dim: (F.conv1d, torch.randn(dim, dim, 3), torch.randn(1, dim, 16)),
    lambda dim: (F.conv2d, torch.randn(dim, dim, 3, 3), torch.randn(1, dim, 16, 16)),
    lambda dim: (
        F.conv3d,
        torch.randn(dim, dim, 3, 3, 3),
        torch.randn(1, dim, 16, 16, 16),
    ),
]
device_and_dtype = [
    (torch.device("cpu"), torch.float32),
]

if torch.cuda.is_available():
    device_and_dtype.append((torch.device("cuda"), torch.float32))
    device_and_dtype.append((torch.device("cuda"), torch.float16))
    device_and_dtype.append((torch.device("cuda"), torch.bfloat16))

if torch.backends.mps.is_available():
    device_and_dtype.append((torch.device("mps"), torch.float32))


patch_forward_param_list = list(
    product(
        modules,
        base_module_and_input_adn_weight,
        device_and_dtype,
    )
)


class LycorisFunctionalTests(unittest.TestCase):
    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_functional(self, module, base, device_dtype):
        func, test_weight, test_input = base(16)
        device, dtype = device_dtype
        print(
            f"{module.__name__: <27}",
            f"{func.__name__: <7}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            sep="||",
        )

        w = test_weight.to(device, dtype)
        x = test_input.to(device, dtype)
        y = func(x, w)

        params = list(module.weight_gen(w, 4))
        for idx, param in enumerate(params):
            if param is not None:
                param = param.to(device, dtype)
                params[idx] = param + torch.randn_like(param) * 0.01

        if module.__name__ == "boft":
            # boft.bypass_forward_diff(org_out, *weights, constraint=None, need_transpose=False)
            diff_w = module.diff_weight(w, *params)
            diff_y = module.bypass_forward_diff(y, *params, need_transpose=w.ndim > 2)
        elif module.__name__ == "diag_oft":
            # diag_oft.bypass_forward_diff(x, org_out, *weights, constraint=None, need_transpose=False)
            diff_w = module.diff_weight(w, *params)
            diff_y = module.bypass_forward_diff(x, y, *params, need_transpose=w.ndim > 2)
        else:
            # For locon/loha/lokr, bypass_forward_diff signature is:
            #   bypass_forward_diff(x, org_out, *weights, ...)
            diff_w = module.diff_weight(*params)
            diff_y = module.bypass_forward_diff(x, y, *params)

        diff_y_from_diff_w = func(x, diff_w.to(x))
        self.assertTrue(
            F.mse_loss(diff_y, diff_y_from_diff_w).item() < EPS_DTYPE[dtype],
            f"Error: {module.__name__} {base.__name__} {device} {dtype} ||"
            f"{F.mse_loss(diff_y, diff_y_from_diff_w).item()}",
        )
