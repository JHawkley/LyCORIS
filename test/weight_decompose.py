import copy
import unittest
from itertools import product

import torch
import torch.nn as nn
from parameterized import parameterized

from lycoris.modules import (
    LycorisBaseModule,
    ButterflyOFTModule,
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
    normalize_weight_decompose_arg,
    parse_weight_decompose,
    pop_wd_for_diff_key,
)


wd_capable_modules = [
    LoConModule,
    LohaModule,
    LokrModule,
    ButterflyOFTModule,
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
        # leaking the internal wd_module submodule (third-party compat), and
        # no diff-mode marker for merged-mode checkpoints.
        self.assertIn("dora_scale", state_dict)
        self.assertNotIn("wd_for_diff", state_dict)
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


class WeightDecomposeParseTests(unittest.TestCase):
    def test_parse_valid_values(self):
        self.assertEqual(parse_weight_decompose(True), (True, False))
        self.assertEqual(parse_weight_decompose("merged"), (True, False))
        self.assertEqual(parse_weight_decompose("diff"), (True, True))
        self.assertEqual(parse_weight_decompose(False), (False, False))
        self.assertEqual(parse_weight_decompose("none"), (False, False))
        self.assertEqual(parse_weight_decompose(None), (False, False))

    def test_parse_auto_resolution(self):
        # LoCon/LoHa/LoKr compute the diff weight fastest.
        self.assertEqual(parse_weight_decompose("auto"), (True, True))
        self.assertEqual(parse_weight_decompose("auto", "diff"), (True, True))
        self.assertEqual(parse_weight_decompose("auto", "merged"), (True, False))
        with self.assertRaises(ValueError):
            parse_weight_decompose("auto", "bogus")

    def test_parse_invalid_raises(self):
        for bad in ("dora", "yes", 2, ["diff"]):
            with self.assertRaises(ValueError, msg=f"value={bad!r}"):
                parse_weight_decompose(bad)

    def test_normalize_user_arg(self):
        self.assertEqual(normalize_weight_decompose_arg("diff"), "diff")
        self.assertEqual(normalize_weight_decompose_arg("Auto"), "auto")
        self.assertEqual(normalize_weight_decompose_arg("NONE"), "none")
        self.assertIs(normalize_weight_decompose_arg(True), True)
        self.assertIs(normalize_weight_decompose_arg("true"), True)
        self.assertIs(normalize_weight_decompose_arg("false"), False)
        self.assertIs(normalize_weight_decompose_arg(False), False)

    def test_pop_wd_for_diff_key(self):
        state_dict = {"test.wd_for_diff": torch.tensor(True), "test.alpha": torch.tensor(1)}
        self.assertTrue(pop_wd_for_diff_key(state_dict, "test."))
        self.assertNotIn("test.wd_for_diff", state_dict)
        # Missing marker means merged mode (the historical default).
        self.assertFalse(pop_wd_for_diff_key(state_dict, "test."))


class DiffWeightDecomposeTests(unittest.TestCase):
    @parameterized.expand(param_list)
    def test_diff_mode_initialization(self, module, base, wd_on_out):
        base, test_input = base()
        ref = copy.deepcopy(base)
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose="diff",
            wd_on_out=wd_on_out,
        )
        self.assertTrue(net.wd)
        self.assertTrue(net.wd_for_diff)
        self.assertTrue(net.wd_module.for_diff)
        expected_cls = WeightDecomposeOnOutput if wd_on_out else WeightDecomposeOnInput
        self.assertIsInstance(net.wd_module, expected_cls)

        # dora_scale starts at zero; the raw diff starts non-zero so the
        # normalized direction (and its gradients) are well-defined.
        self.assertTrue(torch.all(net.dora_scale == 0))
        raw_diff = net.make_weight(scale=1, diff=True)
        self.assertGreater(raw_diff.abs().sum().item(), 0)

        # The module starts as an exact identity on the base model...
        net.apply_to()
        with torch.no_grad():
            expected = ref(test_input)
            actual = base(test_input)
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-6),
            f"diff: {(actual - expected).abs().max().item()}",
        )
        # ...and dora_scale receives gradients from the first step.
        base(test_input).sum().backward()
        self.assertIsNotNone(net.wd_module.dora_scale.grad)
        net.restore()

    @parameterized.expand(param_list)
    def test_diff_mode_matches_reference(self, module, base, wd_on_out):
        base, _ = base()
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose="diff",
            wd_on_out=wd_on_out,
        )
        with torch.no_grad():
            net.dora_scale.copy_(torch.rand_like(net.dora_scale) + 0.5)

        # The decomposition is applied to the raw diff weight.
        raw_diff = net.make_weight(scale=1, diff=True)
        dora_norm_dims = net.wd_module.dora_norm_dims
        reference = reference_apply_weight_decompose(
            raw_diff, net.dora_scale, dora_norm_dims, wd_on_out
        )
        diff, _ = net.get_diff_weight()
        self.assertTrue(
            torch.allclose(reference, diff, atol=1e-6),
            f"diff: {(reference - diff).abs().max().item()}",
        )

        # The merged weight is the base weight plus the decomposed diff.
        merged, _ = net.get_merged_weight()
        self.assertTrue(
            torch.allclose(merged, base.weight.detach() + reference, atol=1e-5),
            f"diff: {(merged - base.weight.detach() - reference).abs().max().item()}",
        )

        # multiplier=0 interpolates the rescaling to the identity, leaving the
        # raw diff untouched (same convention as merged mode).
        diff0, _ = net.get_diff_weight(multiplier=0)
        self.assertTrue(torch.allclose(diff0, raw_diff, atol=1e-6))

    @parameterized.expand(param_list)
    def test_diff_mode_state_dict_marker(self, module, base, wd_on_out):
        base, _ = base()
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose="diff",
            wd_on_out=wd_on_out,
        )
        state_dict = net.state_dict()

        # Diff mode adds exactly one key: the marker for third-party consumers.
        self.assertIn("wd_for_diff", state_dict)
        self.assertTrue(bool(state_dict["wd_for_diff"]))
        self.assertFalse(any("wd_module" in key for key in state_dict))

        # Strict round-trip load exercises the marker popping in the prehook.
        net.dora_scale.data.copy_(torch.ones_like(net.dora_scale))
        net.load_state_dict(state_dict, strict=True)
        self.assertTrue(torch.allclose(net.dora_scale, state_dict["dora_scale"]))

        # make_module recovers the diff mode from the marker.
        prefixed = {f"test.{k}": v for k, v in state_dict.items()}
        lyco_type, params = get_module(prefixed, "test")
        self.assertIs(lyco_type, module)
        net2: LycorisBaseModule = make_module(lyco_type, params, "test", base)
        self.assertTrue(net2.wd)
        self.assertTrue(net2.wd_for_diff)
        self.assertTrue(net2.wd_module.for_diff)
        self.assertEqual(net2.wd_on_out, wd_on_out)
        self.assertTrue(torch.allclose(net2.dora_scale, net.dora_scale))

        merged, _ = net.get_merged_weight()
        merged2, _ = net2.get_merged_weight()
        self.assertTrue(
            torch.allclose(merged, merged2, atol=1e-5),
            f"diff: {(merged - merged2).abs().max().item()}",
        )

    @parameterized.expand(param_list)
    def test_merged_mode_state_dict_has_no_marker(self, module, base, wd_on_out):
        base, _ = base()
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose="merged",
            wd_on_out=wd_on_out,
        )
        self.assertTrue(net.wd)
        self.assertFalse(net.wd_for_diff)
        state_dict = net.state_dict()
        self.assertNotIn("wd_for_diff", state_dict)

        # make_module from a merged-mode checkpoint recovers merged mode.
        prefixed = {f"test.{k}": v for k, v in state_dict.items()}
        lyco_type, params = get_module(prefixed, "test")
        net2: LycorisBaseModule = make_module(lyco_type, params, "test", base)
        self.assertTrue(net2.wd)
        self.assertFalse(net2.wd_for_diff)

    @parameterized.expand(wd_capable_modules)
    def test_auto_resolves_by_algorithm(self, module):
        # "auto" defers to the algorithm's wd_auto_mode — most algorithms
        # compute the diff weight fastest ("diff"), but BOFT computes the
        # merged weight fastest ("merged").
        net: LycorisBaseModule = module(
            "test",
            nn.Linear(16, 8),
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose="auto",
        )
        expected_for_diff = net.wd_auto_mode == "diff"
        self.assertTrue(net.wd)
        self.assertEqual(net.wd_for_diff, expected_for_diff)

    @parameterized.expand(wd_capable_modules)
    def test_none_disables_decomposition(self, module):
        net: LycorisBaseModule = module(
            "test",
            nn.Linear(16, 8),
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose="none",
        )
        self.assertFalse(net.wd)
        self.assertFalse(net.wd_for_diff)
        self.assertIsNone(net.wd_module)


if __name__ == "__main__":
    unittest.main()
