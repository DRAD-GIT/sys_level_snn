"""SRMLayer against recorded runs of the original slayerSNN framework.

Skipped until reference files exist: record them on a machine with slayerSNN
using tools/export_slayer_reference.py and commit them under reference/.
"""
import dataclasses
import glob
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
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
            # A 5-sample test set (random events) and the predictions SRMLayer
            # makes one sample at a time, as a --full recording would hold.
            data = os.path.join(folder, "N_MNIST")
            os.makedirs(os.path.join(data, "Test"))
            rng = np.random.default_rng(0)
            for i in range(5):
                n, ts = 3000, np.sort(rng.integers(0, 300000, 3000))
                x, y, p = rng.integers(0, 34, n), rng.integers(0, 34, n), rng.integers(0, 2, n)
                np.stack([x, y, (p << 7) | (ts >> 16), (ts >> 8) & 255, ts & 255], 1) \
                    .astype(np.uint8).tofile(os.path.join(data, "Test", f"{i:05}.bin"))
            with open(os.path.join(data, "Test.txt"), "w") as file:
                file.write("".join(f"{i} {i % 10}\n" for i in range(5)))
            dataset = models.test_dataset(spec, models.load_params(spec.path(spec.params_yaml)), data)
            with torch.no_grad():
                predictions = [int(predict_class(net(dataset[i][1][None]))[0]) for i in range(5)]
            path = os.path.join(folder, "nmnist_slayer.pt")
            torch.save({"format": FORMAT, "model": "nmnist", "device": "cpu", "samples": [sample],
                        "full": {"predictions": predictions, "labels": [i % 10 for i in range(5)]}},
                       path)
            small_batches = dataclasses.replace(spec, batch_size=2)       # batches of 2, 2, 1
            with patch.object(models.nmnist, "SPEC", small_batches):
                report = compare(path, full=True, data_dir=data, architectures=[arch],
                                 num_workers=0, log=lambda *_: None)
        self.assertEqual(report["full_agreement"], 1.0)
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
