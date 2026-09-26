"""SRMLayer against recorded runs of the original slayerSNN framework.

Skipped until reference files exist: record them on a machine with slayerSNN
using tools/export_slayer_reference.py and commit them under reference/.
"""
import glob
import os
import unittest

import torch

from tools.slayer_reference import pack, unpack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCES = sorted(glob.glob(os.path.join(ROOT, "reference", "*_slayer.pt")))

# CUDA and PyTorch round float32 sums differently, so a membrane potential
# within rounding of the threshold may flip a spike. Allow a tiny fraction.
MAX_DIFFERING_FRACTION = 1e-3


class PackingTests(unittest.TestCase):
    def test_pack_roundtrip(self):
        x = (torch.rand(2, 3, 5, 7, generator=torch.Generator().manual_seed(0)) > 0.7).float() / 4
        self.assertTrue(torch.equal(unpack(pack(x)), x))
        with self.assertRaises(ValueError):
            pack(torch.tensor([0.0, 1.0, 2.0]))


@unittest.skipUnless(REFERENCES, "no slayerSNN reference files in reference/")
class SlayerReferenceTests(unittest.TestCase):
    def test_matches_slayer(self):
        from tools.compare_slayer_reference import compare
        for path in REFERENCES:
            with self.subTest(reference=os.path.basename(path)):
                report = compare(path, log=lambda *_: None)
                self.assertEqual(report["reader_mismatches"], 0)
                self.assertEqual(report["prediction_mismatches"], 0)
                for name, entry in report["layers"].items():
                    self.assertLessEqual(entry["differ"] / entry["total"], MAX_DIFFERING_FRACTION, name)


if __name__ == "__main__":
    unittest.main()
