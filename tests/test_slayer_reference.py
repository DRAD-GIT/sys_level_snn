"""SRMLayer against recorded runs of the original slayerSNN framework.

Skipped until reference files exist: record them on a machine with slayerSNN
using tools/export_slayer_reference.py and commit them under reference/.
"""
import glob
import os
import tempfile
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


class CompareToolTests(unittest.TestCase):
    def test_compare_on_a_recording_of_srmlayer_itself(self):
        """A reference recorded in export's format, but from SRMLayer, so the
        comparison runs end to end here and must report no differences."""
        import models
        from evaluation.probes import LayerProbe
        from evaluation.software import predict_class
        from hardware import Component, Precision, Stage, compose
        from architectures import crossbars
        from tools.compare_slayer_reference import compare
        from tools.slayer_reference import FORMAT

        spec = models.get_spec("nmnist")
        net = models.load_pretrained(spec).eval()
        probe = LayerProbe(net, spec.layers)
        spikes = (torch.rand(2, 34, 34, 300, generator=torch.Generator().manual_seed(0)) > 0.98).float()
        with torch.no_grad():
            output = net(spikes[None])
        sample = {"index": 0, "label": 0, "slayer_input": pack(spikes), "our_input": pack(spikes),
                  "layer_inputs": {name: pack(probe.inputs[name]) for name in spec.layers},
                  "output_spikes": dict(probe.output_spikes), "output": pack(output),
                  "predicted": int(predict_class(output)[0])}
        probe.remove()
        # An architecture charging every LIF output spike.
        arch = compose("spike_events", Precision(4), [
            crossbars.conv_xbar(r_on=2e4, r_off=2e5),
            Stage("fire", 2.0, level="timestep"),
            Component("lif", count="outputs", during="fire", static_ua=1.0,
                      event_pj=1.0, events="output_spike")])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "nmnist_slayer.pt")
            torch.save({"format": FORMAT, "model": "nmnist", "device": "cpu", "samples": [sample],
                        "full": None}, path)
            report = compare(path, architectures=[arch], log=lambda *_: None)
        self.assertEqual((report["reader_mismatches"], report["prediction_mismatches"],
                          report["output_differ"]), (0, 0, 0))
        for name, entry in report["layers"].items():
            self.assertEqual(entry["differ"], 0, name)
        self.assertGreater(report["energy_ours"]["spike_events"], 0)
        self.assertEqual(report["energy_ours"], report["energy_slayer"])


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
