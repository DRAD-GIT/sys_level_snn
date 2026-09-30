"""literature_macros.py: every macro reproduces its paper at the paper's test
condition, and runs a convolution layer."""
import unittest

import torch

import literature_macros
from hardware import evaluate_layer


class LiteratureMacroTests(unittest.TestCase):
    def test_calibration_matches_papers(self):
        for arch, what, engine, paper in literature_macros.check():
            with self.subTest(arch=arch.name, what=what):
                # A-SSCC'25 and ESSERC'24 are straight-line fits over four measured
                # parallelisms (within 2.6% and 0.3%); the rest match within 0.1%.
                self.assertLess(abs(engine / paper - 1), 0.03)

    def test_conv_layer(self):
        g = torch.Generator().manual_seed(0)
        for arch in literature_macros.LITERATURE_MACROS:
            with self.subTest(arch=arch.name):
                top = 2 ** (arch.mapping.weight_bits - 1) - 1
                weights = torch.randint(-top, top + 1, (16, 8, 3, 3), generator=g)
                spikes = (torch.rand(2, 8, 12, 12, 5, generator=g) < 0.1).float()
                cost = evaluate_layer(arch, spikes, weights, padding=1)
                self.assertGreater(cost.energy_nj, 0)
                self.assertGreater(cost.latency_ns, 0)


if __name__ == "__main__":
    unittest.main()
