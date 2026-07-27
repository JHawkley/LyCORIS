import unittest

import torch
import torch.nn as nn

try:
    from torchao.quantization.quant_api import (
        int8_weight_only,
        intx_quantization_aware_training,
        float8_dynamic_activation_float8_weight,
        Float8Linear,
        quantize_,
    )
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


@unittest.skipIf(not TORCHAO_AVAILABLE, "torchao is required for compile tests")
class LycorisCompileSwapErrorTests(unittest.TestCase):
    def test_compile_device_swap(self):
        net = nn.Linear(16, 16).cuda()
        quantize_(net, float8_dynamic_activation_float8_weight())
        test_x = torch.randn(1, 16).cuda()

        compiled_net = torch.compile(net)
        compiled_net(test_x)
        compiled_net.cpu()
        compiled_net.cuda()


if __name__ == "__main__":
    unittest.main()
