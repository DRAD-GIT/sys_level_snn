"""End-to-end pipeline: probes, runner, report switches and export."""
import dataclasses
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

import models
import models.nmnist
from architectures import c3cim, memories, ota_cim

ARCHITECTURES = [ota_cim.build("ota", memories.RRAM_1BIT), c3cim.build("c3cim", memories.RRAM_C3)]
from evaluation.probes import LayerProbe
from evaluation.report import export, format_results
from evaluation.runner import evaluate


class RandomSpikes(torch.utils.data.Dataset):
    def __init__(self, **_):
        self.gen = torch.Generator().manual_seed(0)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        target = torch.zeros(10, 1, 1, 1)
        target[index] = 1
        return index, (torch.rand(2, 34, 34, 300, generator=self.gen) > 0.97).float(), target, index


class ProbeTests(unittest.TestCase):
    def test_inputs_and_output_spikes_per_layer(self):
        spec = models.get_spec("nmnist")
        net = models.load_pretrained(spec).eval()
        x = (torch.rand(1, 2, 34, 34, 50, generator=torch.Generator().manual_seed(1)) > 0.9).float()
        with torch.no_grad():  # expected counts from an unprobed forward pass
            sc1_spikes = net.slayer.spike(net.slayer.psp(net.SC1(x)))
            output = net(x)
        probe = LayerProbe(net, spec.layers)
        with torch.no_grad():
            net(x)
        self.assertTrue(torch.equal(probe.inputs["SC1"], x))
        self.assertEqual(probe.output_spikes["SC1"], int(torch.count_nonzero(sc1_spikes)))
        self.assertEqual(probe.output_spikes["SF2"], int(torch.count_nonzero(output)))
        probe.remove()
        self.assertNotIn("spike", vars(net.slayer))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        spec = dataclasses.replace(models.get_spec("nmnist"), dataset_class=RandomSpikes)
        self.spec_patch = patch.object(models.nmnist, "SPEC", spec)
        self.spec_patch.start()
        self.results = evaluate("nmnist", ARCHITECTURES, batch_size=2,
                                max_batches=1, num_workers=0, log=lambda *_: None)

    def tearDown(self):
        self.spec_patch.stop()

    def test_results_per_architecture_and_layer(self):
        self.assertEqual(set(self.results), {"ota", "c3cim"})
        for accuracy, costs in self.results.values():
            self.assertEqual(list(costs), list(models.get_spec("nmnist").layers))
            self.assertTrue(0 <= accuracy <= 100)
            for cost in costs.values():
                self.assertEqual(cost.inferences, 2)
                self.assertGreater(cost.energy_nj, 0)
        sc1 = self.results["ota"][1]["SC1"]
        self.assertEqual(sc1.geometry.windows, 28 * 28)          # 34x34 input, 7x7 kernel
        self.assertGreater(sc1.output_spikes, 0)

    def test_metric_switches(self):
        metrics = {"energy": True, "latency": True, "area": False, "layers": True, "components": False}
        text = format_results(self.results, metrics)
        self.assertIn("energy/inference", text)
        self.assertNotIn("area", text)
        self.assertNotIn("component", text)
        self.assertNotIn("TOPS/W", text)
        with tempfile.TemporaryDirectory() as folder:
            rows = export(self.results, ARCHITECTURES, "nmnist", metrics, folder)
            rows = export(self.results, ARCHITECTURES, "nmnist", metrics, folder)
            with open(rows[0]["result_json"]) as file:
                report = json.load(file)
            self.assertEqual(set(report["network"]), {"energy", "latency"})
            self.assertNotIn("components", report["layers"]["SC1"])
            with open(os.path.join(folder, "comparison_summary.csv")) as file:
                self.assertEqual(len(file.read().strip().splitlines()), 3)   # header + 2 (upserted)


if __name__ == "__main__":
    unittest.main()
