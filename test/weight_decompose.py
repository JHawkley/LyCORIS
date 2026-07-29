import unittest
from itertools import product

import torch
import torch.nn as nn
from parameterized import parameterized

from lycoris.modules import (
    LycorisBaseModule,
    LoConModule,
    LohaModule,
    LokrModule,
    get_module,
    make_module,
)
from lycoris.modules.weight_decompose import (
    WeightDecomposeOnInput,
    WeightDecomposeOnOutput,
    infer_wd_on_out,
)


wd_capable_modules = [
    LoConModule,
    LohaModule,
    LokrModule,
]
# Deliberately non-square shapes so the on-input/on-output dora_scale shapes
# are distinguishable (and the wd_on_out inference is actually exercised).
base_module_and_input = [
    lambda: (nn.Linear(16, 8), torch.randn(2, 16)),
    lambda: (nn.Conv2d(8, 16, 3, 1, 1), torch.randn(1, 8, 8, 8)),
]
wd_on_out_options = [True, False]

param_list = list(product(wd_capable_modules, base_module_and_input, wd_on_out_options))


def reference_apply_weight_decompose(weight, dora_scale, dora_norm_dims, wd_on_out, multiplier=1):
    """Self-contained copy of the pre-refactor ``apply_weight_decompose``
    implementation, used to verify numerical parity with the new
    weight-decomposition modules.
    """
    weight = weight.to(dora_scale.dtype)
    if wd_on_out:
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1)
            .reshape(weight.shape[0], *[1] * dora_norm_dims)
        ) + torch.finfo(weight.dtype).eps
    else:
        weight_norm = (
            weight.transpose(0, 1)
            .reshape(weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[1], *[1] * dora_norm_dims)
            .transpose(0, 1)
        ) + torch.finfo(weight.dtype).eps

    scale = dora_scale.to(weight.device) / weight_norm
    if multiplier != 1:
        scale = multiplier * (scale - 1) + 1

    return weight * scale


class WeightDecomposeTests(unittest.TestCase):
    @parameterized.expand(param_list)
    def test_wd_module_selection_and_scale_shape(self, module, base, wd_on_out):
        base, _ = base()
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose=True,
            wd_on_out=wd_on_out,
        )
        expected_cls = WeightDecomposeOnOutput if wd_on_out else WeightDecomposeOnInput
        self.assertIsInstance(net.wd_module, expected_cls)
        self.assertEqual(net.wd_module.on_output, wd_on_out)

        # dora_scale is a proper trainable parameter of the child module,
        # reachable through the backward-compatible property.
        self.assertIs(net.dora_scale, net.wd_module.dora_scale)
        self.assertIn("wd_module.dora_scale", dict(net.named_parameters()))

        org_weight = base.weight
        dora_norm_dims = org_weight.dim() - 1
        if wd_on_out:
            expected_shape = (org_weight.shape[0], *([1] * dora_norm_dims))
            expected_scale = torch.norm(
                org_weight.detach().float().reshape(org_weight.shape[0], -1),
                dim=1,
                keepdim=True,
            ).reshape(expected_shape)
        else:
            expected_shape = (1, org_weight.shape[1], *([1] * (dora_norm_dims - 1)))
            expected_scale = (
                torch.norm(
                    org_weight.detach().float().transpose(1, 0).reshape(org_weight.shape[1], -1),
                    dim=1,
                    keepdim=True,
                )
                .reshape(org_weight.shape[1], *[1] * dora_norm_dims)
                .transpose(1, 0)
            )
        self.assertEqual(tuple(net.dora_scale.shape), expected_shape)
        self.assertTrue(torch.allclose(net.dora_scale, expected_scale))

    @parameterized.expand(param_list)
    def test_wd_module_matches_reference(self, module, base, wd_on_out):
        base, test_input = base()
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose=True,
            wd_on_out=wd_on_out,
        )
        org_weight = base.weight.detach()
        dora_norm_dims = org_weight.dim() - 1

        for multiplier in (1.0, 0.5, 0.0):
            reference = reference_apply_weight_decompose(
                org_weight, net.dora_scale, dora_norm_dims, wd_on_out, multiplier
            )
            actual = net.wd_module(org_weight, multiplier)
            self.assertTrue(
                torch.allclose(reference, actual, atol=1e-6),
                f"multiplier={multiplier}, diff: {(reference - actual).abs().max().item()}",
            )
            # Backward-compatible method shim delegates to the submodule.
            self.assertTrue(
                torch.allclose(actual, net.apply_weight_decompose(org_weight, multiplier))
            )

        # dora_scale stays trainable through the full forward pass.
        net.apply_to()
        test_output = base(test_input)
        test_output.sum().backward()
        self.assertIsNotNone(net.wd_module.dora_scale.grad)
        net.restore()

    @parameterized.expand(param_list)
    def test_state_dict_format_compat(self, module, base, wd_on_out):
        base, _ = base()
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose=True,
            wd_on_out=wd_on_out,
        )
        state_dict = net.state_dict()

        # On-disk format is unchanged: dora_scale at the top level, no keys
        # leaking the internal wd_module submodule (third-party compat).
        self.assertIn("dora_scale", state_dict)
        self.assertFalse(any("wd_module" in key for key in state_dict))

        # Strict round-trip load exercises the dora_scale key remapping.
        net.dora_scale.data.copy_(torch.zeros_like(net.dora_scale))
        net.load_state_dict(state_dict, strict=True)
        self.assertTrue(torch.allclose(net.dora_scale, state_dict["dora_scale"]))

    @parameterized.expand(param_list)
    def test_make_module_from_state_dict_recovers_wd_on_out(self, module, base, wd_on_out):
        base, _ = base()
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose=True,
            wd_on_out=wd_on_out,
        )
        state_dict = {f"test.{k}": v for k, v in net.state_dict().items()}

        lyco_type, params = get_module(state_dict, "test")
        self.assertIs(lyco_type, module)
        net2: LycorisBaseModule = make_module(lyco_type, params, "test", base)

        # wd_on_out is recovered from the serialized dora_scale shape.
        self.assertEqual(net2.wd_on_out, wd_on_out)
        expected_cls = WeightDecomposeOnOutput if wd_on_out else WeightDecomposeOnInput
        self.assertIsInstance(net2.wd_module, expected_cls)
        self.assertTrue(torch.allclose(net2.dora_scale, net.dora_scale))

        merged, _ = net.get_merged_weight()
        merged2, _ = net2.get_merged_weight()
        self.assertTrue(
            torch.allclose(merged, merged2, atol=1e-5),
            f"diff: {(merged - merged2).abs().max().item()}",
        )

    def test_infer_wd_on_out(self):
        # Unambiguous shapes.
        linear = nn.Linear(16, 8)
        self.assertTrue(infer_wd_on_out(torch.ones(8, 1), linear.weight))
        self.assertFalse(infer_wd_on_out(torch.ones(1, 16), linear.weight))
        conv = nn.Conv2d(8, 16, 3)
        self.assertTrue(infer_wd_on_out(torch.ones(16, 1, 1, 1), conv.weight))
        self.assertFalse(infer_wd_on_out(torch.ones(1, 8, 1, 1), conv.weight))

        # Degenerate all-singleton shapes: only resolvable via the org weight.
        in_dim_one = nn.Linear(1, 8)
        self.assertFalse(infer_wd_on_out(torch.ones(1, 1), in_dim_one.weight))
        self.assertTrue(infer_wd_on_out(torch.ones(8, 1), in_dim_one.weight))
        out_dim_one = nn.Linear(16, 1)
        self.assertTrue(infer_wd_on_out(torch.ones(1, 1), out_dim_one.weight))
        self.assertFalse(infer_wd_on_out(torch.ones(1, 16), out_dim_one.weight))


if __name__ == "__main__":
    unittest.main()
