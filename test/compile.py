import sys
import unittest

sys.setrecursionlimit(10000000)

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris import create_lycoris

try:
    from torchao.quantization.quant_api import int8_weight_only, quantize_
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(8, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


@unittest.skipIf(not TORCHAO_AVAILABLE, "torchao is required for compile tests")
class LycorisCompileTests(unittest.TestCase):
    def test_lycoris_compile(self):
        net = TestNet().cuda()

        test_x = torch.randn(1, 8).cuda()
        out = net(test_x)
        print("\nOriginal\n", out)

        quantize_(net, int8_weight_only())
        out = net(test_x)
        print("\nQuantized\n", out)
        compiled_net = torch.compile(net)
        out = compiled_net(test_x)
        print("\nQuantized compiled\n", out)

        compiled_net._orig_mod.cpu()
        compiled_net._orig_mod.cuda()

        compiled_net.cpu()
        compiled_net.cuda()

        lycoris = create_lycoris(
            net, algo="lokr", factor=2, bypass_mode=True, full_matrix=True
        )
        lycoris.apply_to()
        lycoris.cuda()
        out = net(test_x)
        print("\nBypass Lokr Applied\n", out)

        compiled_net = torch.compile(net)
        compiled_out = compiled_net(test_x)
        print("\nLyCORIS+torchao quantize_ compiled\n", compiled_out)
        torch.sum(compiled_out).backward()

        diff = torch.sum(torch.abs(out - compiled_out))
        print("\nDiff: ", diff)
        net.cpu()
        lycoris.cpu()

        self.assertTrue(
            torch.allclose(out, compiled_out, atol=1e-6),
            f"Output mismatch: diff={diff}",
        )


if __name__ == "__main__":
    unittest.main()
